#!/usr/bin/env python3
"""
NLD 训练脚本 (Interlude)

使用方法:
    # 单卡
    python scripts/train_nld.py --config configs/local_train.yaml
    
    # 多卡 (FSDP, 推荐)
    torchrun --nproc_per_node=8 scripts/train_nld.py --config configs/local_train.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import json
from pathlib import Path

# NCCL 超时设置: segment-wise forward 各 rank 耗时差异大, 需要更长的超时
os.environ.setdefault('NCCL_TIMEOUT', '3600')  # 60 分钟 (与 ddp_timeout 一致)
os.environ.setdefault('TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC', '3600')  # watchdog 超时

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    TrainingArguments,
    AutoProcessor,
    set_seed,
)

from rld.model_v2 import NLDModel
from rld.data import NLDDataset, NLDCollator, LATENT_TOKEN, LATENT_END_TOKEN
from rld.trainer_nld import NLDTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="NLD Training Script")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--output_dir", type=str, default=None, help="覆盖输出目录")
    parser.add_argument("--learning_rate", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--num_train_epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从 checkpoint 恢复")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _is_main_process():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    return local_rank in (-1, 0)


def main():
    args = parse_args()
    config = load_config(args.config)

    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    nld_config = config.get('nld', {})
    seed = config.get('seed', 42)

    set_seed(seed)
    is_main = _is_main_process()

    # ====== 1. 加载 Processor ======
    if is_main:
        print("\n[1/5] 加载 Processor...")
    processor = AutoProcessor.from_pretrained(model_config['model_path'])

    _min_pixels = data_config.get('min_pixels', 3136)
    _max_pixels = data_config.get('max_pixels', 1003520)
    processor.image_processor.min_pixels = _min_pixels
    processor.image_processor.max_pixels = _max_pixels
    if is_main:
        print(f"  图片分辨率限制: min_pixels={_min_pixels}, max_pixels={_max_pixels}")

    # ====== 1.5 注册 <|latent|> 和 <|/latent|> 特殊 token ======
    num_added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [LATENT_TOKEN, LATENT_END_TOKEN]}
    )
    latent_token_id = processor.tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    latent_end_token_id = processor.tokenizer.convert_tokens_to_ids(LATENT_END_TOKEN)
    if is_main:
        print(f"  注册特殊 token: {LATENT_TOKEN} → id={latent_token_id}, "
              f"{LATENT_END_TOKEN} → id={latent_end_token_id} (新增 {num_added} 个)")

    # ====== 2. 创建数据集 ======
    if is_main:
        print("\n[2/5] 创建数据集...")
    train_dataset = NLDDataset(
        json_path=data_config['train_json'],
        processor=processor,
        image_base_dir=data_config.get('image_base_dir', None),
        use_system_prompt=data_config.get('use_system_prompt', True),
        system_prompt=data_config.get('system_prompt', None),
        max_seq_len=data_config.get('max_seq_len', 4096),
        skip_image_check=data_config.get('skip_image_check', False),
    )
    if is_main:
        print(f"训练样本数: {len(train_dataset)}")

    eval_dataset = None
    if data_config.get('eval_json'):
        eval_dataset = NLDDataset(
            json_path=data_config['eval_json'],
            processor=processor,
            image_base_dir=data_config.get('image_base_dir', None),
            use_system_prompt=data_config.get('use_system_prompt', True),
            max_seq_len=data_config.get('max_seq_len', 4096),
            skip_image_check=data_config.get('skip_image_check', False),
        )
        if is_main:
            print(f"评估样本数: {len(eval_dataset)}")

    # ====== 3. 创建模型 ======
    if is_main:
        print("\n[3/5] 创建 NLD 模型...")

    # 将 Key Token Decoding Loss 权重设置为环境变量 (NLDModel 通过环境变量读取)
    if 'key_token_weight' in nld_config:
        os.environ['NLD_KEY_TOKEN_WEIGHT'] = str(nld_config['key_token_weight'])

    # ====== 3 层 anti-collapse 修复超参注入 (yaml → env vars) ======
    # 这些超参既可在 yaml 中显式设置, 也可在命令行用环境变量覆盖
    _antic_env_map = {
        'exit_margin':                       'NLD_EXIT_MARGIN',
        'exit_margin_weight':                'NLD_EXIT_MARGIN_WEIGHT',
        'exit_margin_loss_weight':           'NLD_EXIT_MARGIN_LOSS_WEIGHT',
        'sw_srs_alpha':                      'NLD_SW_SRS_ALPHA',
        'swsrs_anti_collapse_margin':        'NLD_SWSRS_ANTI_COLLAPSE_MARGIN',
        'swsrs_anti_collapse_weight':        'NLD_SWSRS_ANTI_COLLAPSE_WEIGHT',
        'swsrs_anti_collapse_loss_weight':   'NLD_SWSRS_ANTI_COLLAPSE_LOSS_WEIGHT',
        'diversity_threshold':               'NLD_DIVERSITY_THRESHOLD',
        'diversity_weight':                  'NLD_DIVERSITY_WEIGHT',
        'diversity_loss_weight':             'NLD_DIVERSITY_LOSS_WEIGHT',
        # ====== Phase 1 ======
        'loss_mode':                         'NLD_LOSS_MODE',
        # ====== path B2 视觉侧 vision_loss (过渡模态几何约束: SLERP + 双 anchor margin) ======
        # 默认全部为 0 (向后兼容), yaml 显式设置后会被注入到 env vars 启用.
        'vision_loss_weight':                'NLD_VISION_LOSS_WEIGHT',         # latent_thinker 内部本地权重
        'vision_loss_total_weight':          'NLD_VISION_LOSS_TOTAL_WEIGHT',   # model_v2 顶层权重 (与 local_w 相乘)
        'vision_top_k':                      'NLD_VISION_TOP_K',               # top-K 视觉 token 检索 (默认 6)
        'vision_margin':                     'NLD_VISION_MARGIN',              # 双 anchor margin 的 δ (默认 0.05)
    }
    for _ykey, _envkey in _antic_env_map.items():
        if _ykey in nld_config:
            os.environ[_envkey] = str(nld_config[_ykey])

    model = NLDModel(
        model_path=model_config['model_path'],
        hidden_size=model_config.get('hidden_size', 4096),
        total_layers=model_config.get('total_layers', 36),
        torch_dtype=getattr(torch, model_config.get('torch_dtype', 'bfloat16')),
        attn_implementation=model_config.get('attn_implementation', 'flash_attention_2'),
        max_think_steps=nld_config.get('max_think_steps', 5),
        think_layer_start=nld_config.get('think_layer_start', 0),
        think_layer_end=nld_config.get('think_layer_end', 36),
        saturation_exit_threshold=nld_config.get('saturation_exit_threshold', 0.99),
    )
    
    # ============================================================
    # 调整 embedding 大小以适配新增的特殊 token
    # ----------------------------------------------------------
    # ⚠️ 关键修复: Qwen3-VL 原生 embed_tokens.shape[0] = 151936 (含 ~290 个预留
    #    特殊 token 槽位, 如 <|image_pad|>, <|vision_start|>, <|object_ref_*|>,
    #    <|box_*|>, <|quad_*|> 等), 但 len(tokenizer) 只有 ~151669.
    #    直接 resize_token_embeddings(len(tokenizer)) 会 **缩小** embedding 到
    #    151671, 截断最后 265 行已训练好的预留特殊 token 权重;
    #    若 config.tie_word_embeddings=True, 还会让 lm_head 被重建→数值异常,
    #    最终导致 forward 时 logits 整体 NaN (与 attention 实现/精度无关).
    #
    #    正确做法: 取 len(tokenizer) 和原始 embed 大小的 **最大值** 作为目标.
    #    - 若 len(tokenizer) ≤ _V_old: 保持不变 (形状不动, 预留位完全保留).
    #      新增的 <|latent|>/<|/latent|> 会复用 Qwen 的预留槽位 (id 151669/151670),
    #      这些位置原本就是 Qwen 预留的"近高斯"初始化权重, 训练时会被塑形,
    #      无需额外重新初始化.
    #    - 若 len(tokenizer) > _V_old: 扩展, 新行用现有词表均值 + 小噪声初始化.
    # ============================================================
    _V_old = model.base_model.get_input_embeddings().weight.shape[0]
    _target_vocab = max(len(processor.tokenizer), _V_old)
    if _target_vocab != _V_old:
        model.base_model.resize_token_embeddings(_target_vocab)
    _V_new = model.base_model.get_input_embeddings().weight.shape[0]
    if is_main:
        print(
            f"[NLD] Embedding 检查: tokenizer 词表 {len(processor.tokenizer)} tokens, "
            f"模型 embed {_V_old} → {_V_new} (target={_target_vocab})"
        )
        print(
            f"[NLD] 新 token ids: {LATENT_TOKEN}={latent_token_id}, "
            f"{LATENT_END_TOKEN}={latent_end_token_id}; "
            f"落入原 embed 范围 [0, {_V_old}) → "
            f"{'✅ 复用 Qwen 预留槽位' if (latent_token_id < _V_old and latent_end_token_id < _V_old) else '⚠️ 越界, 必须扩展 embedding'}"
        )

    # ====== 1.6 新 token embedding 初始化 (仅在 embedding 确实扩展时触发) ======
    # 问题: resize_token_embeddings 默认用 std=0.02 噪声初始化新行,
    #       cold-start 时 lm_head 对应行也是噪声, logit 极不稳定 → softmax 数值崩 → NaN
    # 解决: 用现有词表 embedding 的均值 + 小噪声初始化新 token (EleutherAI / Llama 标准做法)
    # 注意: 若 _V_new == _V_old (常见于 Qwen3-VL), 新 token 直接复用预留槽位,
    #       预留槽位已有 Qwen 初始化的权重, 无需重设.
    if _V_new > _V_old:
        with torch.no_grad():
            input_emb = model.base_model.get_input_embeddings()
            output_emb = model.base_model.get_output_embeddings()  # lm_head (若未 tied)
            # 用原有词表的均值作为新 token 初值
            _old_input_mean = input_emb.weight[:_V_old].mean(dim=0, keepdim=True)  # [1, H]
            _old_input_std = input_emb.weight[:_V_old].std(dim=0, keepdim=True).clamp(min=1e-6) * 0.1
            for _tid in range(_V_old, _V_new):
                input_emb.weight[_tid] = _old_input_mean.squeeze(0) + \
                    torch.randn_like(_old_input_mean.squeeze(0)) * _old_input_std.squeeze(0)
            if output_emb is not None and output_emb.weight.data_ptr() != input_emb.weight.data_ptr():
                # lm_head 未与 input embedding tied, 单独初始化
                _old_output_mean = output_emb.weight[:_V_old].mean(dim=0, keepdim=True)
                _old_output_std = output_emb.weight[:_V_old].std(dim=0, keepdim=True).clamp(min=1e-6) * 0.1
                for _tid in range(_V_old, _V_new):
                    output_emb.weight[_tid] = _old_output_mean.squeeze(0) + \
                        torch.randn_like(_old_output_mean.squeeze(0)) * _old_output_std.squeeze(0)
                _tied_status = "分离 (lm_head 独立初始化)"
            else:
                _tied_status = "tied (lm_head 共享 input embedding)"
            if is_main:
                print(f"[NLD] 新 token 权重初始化: mean-based + 0.1*std noise, "
                      f"lm_head {_tied_status}")

    model.set_processor(processor)

    # ====== 加载预训练权重 (如果有) ======
    resume_nld = nld_config.get('resume_from_checkpoint', None)
    if resume_nld and os.path.exists(resume_nld):
        model.load_pretrained(resume_nld)
        if is_main:
            print(f"[NLD] ✅ 从 checkpoint 加载 (save_pretrained 格式): {resume_nld}")

    # ============================================================
    # 仅加载 HF Trainer 的 model.safetensors 作为初始权重 (支持 shape 不匹配)
    # ----------------------------------------------------------
    # 用途: 从老 checkpoint (如 rld_stage2/checkpoint-400) 继承模型权重作为
    #      "预训练起点", 但 Trainer 的 step_counter / optimizer / scheduler
    #      全部从 0 开始, 避开 HF Trainer 原生 resume 在以下场景下的硬错误:
    #        1. checkpoint 的 embed_tokens 形状 [V_ckpt, H] 与当前模型 [V_cur, H]
    #           不一致 (老 bug 代码 shrink 到 151671, 新代码保留 151936)
    #        2. checkpoint 含当前代码已删除的子模块参数 (如 visual_alignment.*)
    #
    # 实现细节:
    #   - 用 safetensors 读取 model.safetensors
    #   - 对 embed_tokens.weight / lm_head.weight: 若 shape 不匹配但 hidden dim
    #     一致, 把 ckpt 的前 V_ckpt 行拷进当前模型, 其余行保留当前初始化
    #   - 其他参数: 走 load_state_dict(strict=False), 自动忽略 Unexpected keys
    #   - 仅 rank0 加载, 然后 FSDP 的 sync_module_states 会把权重广播给其他 rank
    # ============================================================
    resume_model_only = nld_config.get('resume_from_model_only', None)
    if resume_model_only and os.path.isdir(resume_model_only):
        import safetensors.torch as _st
        _safetensor_files = sorted([
            os.path.join(resume_model_only, f)
            for f in os.listdir(resume_model_only)
            if f.endswith('.safetensors')
        ])
        if not _safetensor_files:
            if is_main:
                print(f"[NLD] ⚠️ {resume_model_only} 下未找到 .safetensors, 跳过权重加载")
        else:
            # 聚合所有 safetensors shard
            _ckpt_state = {}
            for _sf in _safetensor_files:
                _ckpt_state.update(_st.load_file(_sf, device='cpu'))

            # HF Trainer 保存的 model.safetensors 通常不带 `_fsdp_wrapped_module.` 前缀
            # 但 NLDModel 的 state_dict key 以 `base_model.` / `latent_thinker.` 开头
            # 若 ckpt 是 NLDModel.save_pretrained 产物, key 已对齐
            # 若 ckpt 是 HF Trainer FSDP 产物, key 同样以 `base_model.` / `latent_thinker.` 开头 (已聚合)
            # → 直接用 model.state_dict().keys() 做 shape 适配

            _cur_state = model.state_dict()
            _adapted_state = {}
            _n_skip_shape = 0
            _n_pad_rows = 0
            _n_unexpected = 0
            _n_match = 0
            for _k, _v in _ckpt_state.items():
                if _k not in _cur_state:
                    _n_unexpected += 1
                    continue
                _cur_v = _cur_state[_k]
                if _v.shape == _cur_v.shape:
                    _adapted_state[_k] = _v
                    _n_match += 1
                elif (
                    _v.dim() == _cur_v.dim()
                    and _v.shape[1:] == _cur_v.shape[1:]
                    and _v.shape[0] < _cur_v.shape[0]
                ):
                    # vocab-like 维度: ckpt 比 current 小, 把 ckpt 内容拷到前 V_ckpt 行
                    # 例: embed_tokens.weight [151671, 4096] → [151936, 4096]
                    #     前 151671 行用 ckpt, 后 265 行保留当前初始化 (Qwen 预留槽位)
                    _new_v = _cur_v.clone()
                    _new_v[:_v.shape[0]] = _v.to(_cur_v.dtype)
                    _adapted_state[_k] = _new_v
                    _n_pad_rows += 1
                    if is_main:
                        print(f"[NLD] 🔧 shape 适配: {_k} {tuple(_v.shape)} → {tuple(_cur_v.shape)} (前 {_v.shape[0]} 行用 ckpt)")
                else:
                    _n_skip_shape += 1
                    if is_main:
                        print(f"[NLD] ⚠️ shape 不兼容跳过: {_k} ckpt={tuple(_v.shape)} current={tuple(_cur_v.shape)}")

            # 仅 rank0 做一次 load, 其他 rank 走 FSDP sync_module_states 广播
            _missing, _unexpected = model.load_state_dict(_adapted_state, strict=False)
            if is_main:
                print(f"[NLD] ✅ 从 model.safetensors 加载权重: {resume_model_only}")
                print(f"[NLD]    shard 数: {len(_safetensor_files)}, ckpt 参数数: {len(_ckpt_state)}")
                print(f"[NLD]    ├─ 完全匹配: {_n_match} 个")
                print(f"[NLD]    ├─ shape 适配 (pad rows): {_n_pad_rows} 个")
                print(f"[NLD]    ├─ shape 不兼容跳过: {_n_skip_shape} 个")
                print(f"[NLD]    ├─ ckpt 多余 key: {_n_unexpected} 个 (已丢弃)")
                print(f"[NLD]    ├─ 当前模型缺失 (保留初始化): {len(_missing)} 个")
                print(f"[NLD]    └─ load_state_dict 多余 (应为 0): {len(_unexpected)} 个")
                if _missing:
                    print(f"[NLD]    missing 样本 (前 5): {_missing[:5]}")
            del _ckpt_state, _adapted_state

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"可训练参数: {trainable / 1e6:.2f}M / 总参数: {total / 1e9:.2f}B "
              f"({trainable / total * 100:.2f}%)")

    # ============================================================
    # Phase 1 冻结早期层 (保护 base CoT 能力，避免 catastrophic forgetting)
    # ----------------------------------------------------------
    # 动机:
    #   1. base 模型已经学会纯文本 CoT, 冻结早期层防止 latent 训练破坏该能力;
    #   2. 仅允许后期 layers + lm_head + embed_tokens(<|latent|>/<|/latent|>) +
    #      latent_thinker 参与训练, 【全局 footprint 下降~70%】;
    #   3. 冻结范围由 yaml 字段 freeze_text_layers_below 控制
    #      (设为 0 = 不冻结, 设为 N = 冻结 layers[0..N-1]).
    #
    # 路径:
    #   model.base_model.model.language_model.layers[i]
    #   model.base_model.model.visual                      ← 可选冻结 (vision tower)
    #   model.base_model.model.embed_tokens                ← 默认 仅训练两个特殊 token 行
    #   model.base_model.lm_head                           ← 默认保持可训练 (输出头要适应 latent)
    # ============================================================
    _phase1_cfg = nld_config.get('phase1', {}) or {}
    _freeze_below = int(_phase1_cfg.get('freeze_text_layers_below', 0))
    _freeze_visual = bool(_phase1_cfg.get('freeze_visual', False))
    _freeze_embed_except_special = bool(_phase1_cfg.get('freeze_embed_except_special', False))
    if _freeze_below > 0 or _freeze_visual or _freeze_embed_except_special:
        if is_main:
            print(f"\n🧊 [Phase 1] 冻结配置:")
            print(f"  freeze_text_layers_below = {_freeze_below}")
            print(f"  freeze_visual            = {_freeze_visual}")
            print(f"  freeze_embed_except_special = {_freeze_embed_except_special}")

        _frozen, _kept = 0, 0
        # ---- 1. 冻结文本模型早期 layers ----
        try:
            _lang = model.base_model.model.language_model
            _layers = _lang.layers if hasattr(_lang, 'layers') else None
        except AttributeError:
            _layers = None
        if _layers is not None and _freeze_below > 0:
            _N = len(_layers)
            _freeze_below_clipped = min(_freeze_below, _N)
            for _i in range(_freeze_below_clipped):
                for _p in _layers[_i].parameters():
                    if _p.requires_grad:
                        _p.requires_grad = False
                        _frozen += _p.numel()
            if is_main:
                print(f"  ✅ 冻结 language_model.layers[0..{_freeze_below_clipped - 1}] "
                      f"({_freeze_below_clipped}/{_N} 层)")

        # ---- 2. 可选: 冻结 vision tower ----
        if _freeze_visual:
            try:
                _vis = model.base_model.model.visual
            except AttributeError:
                _vis = None
            if _vis is not None:
                for _p in _vis.parameters():
                    if _p.requires_grad:
                        _p.requires_grad = False
                        _frozen += _p.numel()
                if is_main:
                    print(f"  ✅ 冻结 visual (vision tower)")

        # ---- 3. 可选: 冻结 embed_tokens 除两个特殊 token 外的所有行 ----
        # 实现思路: 用 hook 拦截梯度, 只让 latent_id / latent_end_id 两行可更新
        if _freeze_embed_except_special and latent_token_id is not None:
            _emb = model.base_model.model.embed_tokens
            _ids_keep = [latent_token_id]
            _latent_end_id = processor.tokenizer.convert_tokens_to_ids('<|/latent|>')
            if _latent_end_id is not None and _latent_end_id != processor.tokenizer.unk_token_id:
                _ids_keep.append(_latent_end_id)
            _ids_keep_set = set(_ids_keep)

            def _embed_grad_mask(grad):
                # grad: [V, H]; 只保留 _ids_keep 行的梯度, 其余与 0
                _mask = torch.zeros(grad.shape[0], 1, device=grad.device, dtype=grad.dtype)
                for _kid in _ids_keep_set:
                    if 0 <= _kid < grad.shape[0]:
                        _mask[_kid, 0] = 1.0
                return grad * _mask

            _emb.weight.register_hook(_embed_grad_mask)
            if is_main:
                print(f"  ✅ embed_tokens 梯度掩蔓: 仅允许更新 ids={_ids_keep_set}")

        # ---- 冻结后参数量重算 ----
        if is_main:
            _new_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  📊 冻结后可训练参数: {_new_trainable / 1e6:.2f}M "
                  f"({_new_trainable / total * 100:.2f}%, 下降 {(trainable - _new_trainable) / 1e6:.2f}M)")

    # ====== 4. 创建 Collator ======
    collator = NLDCollator(
        processor=processor,
        latent_token_id=latent_token_id,
    )

    # ====== 5. 训练参数 ======
    if is_main:
        print("\n[4/5] 配置训练参数...")

    output_dir = args.output_dir or training_config['output_dir']
    
    # FSDP 配置 (替代 DeepSpeed, 完全兼容 KV cache + 多次 forward)
    fsdp_strategy = training_config.get('fsdp', '')
    fsdp_config_path = training_config.get('fsdp_config', None)
    fsdp_config = {}
    if fsdp_config_path and os.path.exists(fsdp_config_path):
        with open(fsdp_config_path, 'r') as f:
            fsdp_config = json.load(f)
        if is_main:
            print(f"[NLD] FSDP 配置: {fsdp_config_path}")
    elif fsdp_strategy:
        if is_main:
            print(f"[NLD] FSDP 策略: {fsdp_strategy}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs or training_config['num_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=args.learning_rate or training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_steps=training_config.get('warmup_steps', 100),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        bf16=model_config.get('torch_dtype', 'bfloat16') == 'bfloat16',
        fp16=model_config.get('torch_dtype', '') == 'float16',
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=training_config.get('logging_steps', 20),
        logging_first_step=True,
        save_strategy="steps",
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 3),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=training_config.get('eval_steps', 500) if eval_dataset else None,
        dataloader_num_workers=training_config.get('num_workers', 4),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,  # FSDP 模式下不需要且不兼容
        fsdp=fsdp_strategy if fsdp_strategy else "",
        fsdp_config=fsdp_config if fsdp_config else None,
        report_to=["tensorboard"],
        seed=seed,
        ddp_timeout=3600,  # 增大超时: segment-wise forward 各 rank 耗时差异大
        gradient_checkpointing=training_config.get('gradient_checkpointing', False),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
    )

    # ====== 6. 创建 Trainer ======
    if is_main:
        print("\n[5/5] 创建 Trainer...")
    
    _thinker_lr = training_config.get('thinker_lr', 1e-4)
    _vlm_lr = training_config.get('vlm_lr', None)
    
    trainer = NLDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        thinker_lr=_thinker_lr,
        vlm_lr=_vlm_lr,
        monitor_every_n_steps=training_config.get('monitor_every_n_steps', 50),
    )

    # ====== 7. 训练 ======
    if is_main:
        print("\n" + "=" * 80)
        print("🚀 开始 NLD 训练 (Interlude)...")
        print("=" * 80 + "\n")

    resume_from = args.resume_from_checkpoint or training_config.get('resume_from_checkpoint')
    if resume_from and not os.path.exists(resume_from):
        resume_from = None

    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # ====== 8. 保存 ======
    if is_main:
        print("\n" + "=" * 80)
        print("💾 保存模型...")
        print("=" * 80)

    final_dir = os.path.join(output_dir, 'model')

    # 保存前同步: 所有 rank 必须先离开 trainer.train 的最后一个 step,
    # 才能进入 FSDP FULL_STATE_DICT gather, 否则会发生 collective 错位.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # NLDModel.save_pretrained 内部已做 FSDP 安全保存:
    #   - 用 FullStateDictConfig(offload_to_cpu=True, rank0_only=True) 把参数
    #     gather 到 rank0; 仅 rank0 写盘 (thinker + vlm_full); 其他 rank barrier 等待.
    # 这里外部不再需要任何 rank gating, 直接调用即可.
    model.save_pretrained(final_dir)

    # 保存后同步, 防止 rank0 还在写盘时其他 rank 提前结束进程.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if _is_main_process():
        processor.save_pretrained(final_dir)

    # ---- trainer 自带的 metrics / state 落盘 (仅 rank0, 避免多 rank 写同一文件) ----
    if _is_main_process():
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if is_main:
        print(f"\n✅ 模型已保存到 {final_dir}")
        print("\n🎉 训练完成！")


if __name__ == "__main__":
    main()
