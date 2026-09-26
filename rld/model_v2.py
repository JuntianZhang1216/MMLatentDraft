"""NLD Model v2: Interlude — 基于 VLM 原生隐空间的双流推理解码

核心思想 (COCONUT 式隐空间推理 + CoT 统一序列架构):
  主线: VLM 生成自然语言 CoT (可解释)
  辅助: 在 step boundary 处插入隐空间思考步骤 (信息增益)
  统一: 隐式推理的 hidden states 作为特殊 token 永久存在于历史序列中，从不丢弃
       隐式推理 + 自然语言 CoT 构成完整的、连续的推理过程

与 RLD v1 的根本区别:
  NLD v2: VLM 原生空间思考 → thought KV 直接追加到全局 cache (无瓶颈, 信息增益)

统一 forward 架构:
  训练和推理都使用 text_model() 整体调用 (而非手动逐层 forward)
  - KV cache 由 DynamicCache 自动管理
  - gradient checkpointing 由 text_model 内部正确处理
  - attention mask 由 text_model 内部自动构建
  - thinker 内部也使用 text_model() 调用，thought KV 直接追加到全局 cache
  - 训推一致: 训练和推理使用完全相同的 forward 路径

训练流程 (Segment-wise + Latent Thinking, 统一序列):
  1. 视觉 encoder → visual_embeddings (no_grad)
  2. 按 step boundary 切分序列为 segments
  3. 对每个 segment:
     a. text_model() 整体调用 (全量微调, 所有层有梯度)
        KV cache 自动管理，如果之前有 thought tokens，它们的 KV 已在 cache 中
        当前 segment 的 attention 自然能看到 thought tokens
     b. 如果是 step boundary:
        - NativeLatentThinker: last_hidden → text_model() × steps → thought_output
        - 方案 B: thinker 内部使用局部 cache (训练时不修改 global_cache)
        - thought_output 作为 prefix embedding 注入下一个 segment 的 inputs_embeds
        - 下一个 segment 的 text_model() 把 thought prefix + segment tokens 的 KV 一起写入 global_cache
        - 梯度路径: CE loss → Segment hidden → inputs_embeds → thought_output → thinker → VLM
  4. concat 所有 segment hidden (含 thought prefix) → logits → CE loss
  5. 梯度路径: CE loss → logits → segment hidden → inputs_embeds → thought_output → thinker → VLM 全部层

监督信号:
  不需要对隐空间步骤本身标注！
  CE loss 的梯度通过 thought hidden states 自动回传到 NativeLatentThinker。

参数量:
NLD v2: NativeLatentThinker ~2M + 全量微调 VLM 8.3B (FSDP 分布式)
"""
import os
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
from transformers.cache_utils import DynamicCache as _RawDynamicCache

# Qwen3-VL
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .latent_thinker import NativeLatentThinker




# ============================================================
# DynamicCache 兼容层 (复用 model.py 的实现)
# ============================================================

class _CacheProxy:
    """代理 DynamicCache.layers 中每个 layer 的 keys 或 values 属性"""
    def __init__(self, cache: "_RawDynamicCache", attr: str):
        self._cache = cache
        self._attr = attr

    def __len__(self):
        return len(self._cache.layers)

    def __getitem__(self, idx):
        layer = self._cache.layers[idx]
        return getattr(layer, self._attr, None)

    def __setitem__(self, idx, value):
        while len(self._cache.layers) <= idx:
            from transformers.cache_utils import DynamicLayer
            self._cache.layers.append(DynamicLayer())
        setattr(self._cache.layers[idx], self._attr, value)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def append(self, value):
        from transformers.cache_utils import DynamicLayer
        idx = len(self._cache.layers)
        self._cache.layers.append(DynamicLayer())
        setattr(self._cache.layers[idx], self._attr, value)


def _patch_dynamic_cache():
    """为 DynamicCache 添加 key_cache / value_cache 兼容属性 (幂等)"""
    if hasattr(_RawDynamicCache, '_compat_patched'):
        return

    @property
    def _key_cache_compat(self):
        return _CacheProxy(self, "keys")

    @property
    def _value_cache_compat(self):
        return _CacheProxy(self, "values")

    if not hasattr(_RawDynamicCache, 'key_cache') or isinstance(
        getattr(_RawDynamicCache, 'key_cache', None), property
    ):
        _RawDynamicCache.key_cache = _key_cache_compat

    if not hasattr(_RawDynamicCache, 'value_cache') or isinstance(
        getattr(_RawDynamicCache, 'value_cache', None), property
    ):
        _RawDynamicCache.value_cache = _value_cache_compat

    _RawDynamicCache._compat_patched = True

_patch_dynamic_cache()

from transformers.cache_utils import DynamicCache


def _extract_visual_output(vision_output):
    """从视觉编码器输出中提取 merged_hidden_states 和 deepstack_features"""
    if isinstance(vision_output, (tuple, list)) and len(vision_output) == 2:
        hidden_states = vision_output[0]
        deepstack_features = vision_output[1]
    elif isinstance(vision_output, (tuple, list)):
        hidden_states = vision_output[0]
        deepstack_features = None
    else:
        hidden_states = vision_output
        deepstack_features = None
    return hidden_states, deepstack_features


def _is_main_process():
    """判断当前进程是否为主进程"""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_rank() == 0
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', -1)))
    return local_rank in (-1, 0)


class NLDModel(nn.Module):
    """
    NLD Model v2: Interlude
    
    Qwen3-VL + NativeLatentThinker 双流推理解码
    
    统一 forward 架构:
    ┌──────────────────────────────────────────────────────────────┐
    │  训练流程 (Segment-wise + Latent Thinking):                   │
    │  1. 视觉 encoder → visual_embeddings (no_grad)               │
    │  2. 按 step boundary 切分序列为 segments                      │
    │  3. 对每个 segment:                                           │
    │     a. text_model() 整体调用 (全量微调, 所有层有梯度)         │
    │        KV cache 自动管理, thought KV 已在 cache 中            │
    │     b. 如果是 step boundary:                                  │
    │        NativeLatentThinker → text_model() × steps             │
    │        thought KV 直接追加到全局 cache                        │
    │        thought hidden states 收集到输出序列                   │
    │  4. concat 所有 hidden (含 thought 占位) → logits             │
    │     thought 位置 labels=-100 (不计算 loss)                    │
    │  5. CE loss → 梯度回传到 NativeLatentThinker + VLM 全部层     │
    └──────────────────────────────────────────────────────────────┘
    """

    # Step boundary 检测: 使用 <|latent|> token id
    # <|latent|> 是注册的特殊 token，boundary 检测通过 token id 比较完成 (O(1))
    LATENT_TOKEN = "<|latent|>"

    def __init__(
        self,
        model_path: str,
        hidden_size: int = 4096,
        total_layers: int = 36,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        # NativeLatentThinker 参数
        max_think_steps: int = 5,
        think_layer_start: int = 0,
        think_layer_end: int = 36,
        saturation_exit_threshold: float = 0.99,
        # 以下参数保留接口兼容性，但不再使用
        adaptive_exit: bool = True,
        exit_threshold: float = 0.5,
        ponder_cost_weight: float = 0.01,
        saturation_aux_weight: float = 0.005,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_layers = total_layers
        self._verbose = _is_main_process()
        self._debug = self._verbose and os.environ.get('RLD_DEBUG', '0') == '1'

        # 注意力实现
        env_attn = os.environ.get('RLD_ATTN_IMPL', None)
        if env_attn is not None and env_attn != attn_implementation:
            if self._verbose:
                print(f"[NLD] ⚠️ 注意力实现: {attn_implementation} → {env_attn} (通过 RLD_ATTN_IMPL 覆盖)")
            attn_implementation = env_attn
        self.attn_implementation = attn_implementation

        # ====== 1. 加载并冻结 Qwen3-VL ======
        if self._verbose:
            print(f"[NLD] 加载 Qwen3-VL: {model_path}")
            print(f"[NLD] 注意力实现: {attn_implementation}")

        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        # 全量微调: 基座参数全部可训练 (使用 FSDP 分布式)
        # 不再冻结基座参数
        for param in self.base_model.parameters():
            param.requires_grad = True

        if self._verbose:
            total_params = sum(p.numel() for p in self.base_model.parameters())
            print(f"[NLD] Qwen3-VL 全量微调: {total_params / 1e9:.2f}B 参数 (全部可训练)")

        # ====== 2. 创建 NativeLatentThinker ======
        # 从基座 config 获取实际 hidden_size
        text_config = self.base_model.config.text_config if hasattr(self.base_model.config, 'text_config') else self.base_model.config
        actual_hidden_size = getattr(text_config, 'hidden_size', hidden_size)
        num_attention_heads = getattr(text_config, 'num_attention_heads', 28)
        
        # think_layers: 隐空间思考使用全部 VLM 层 (与 COCONUT 一致)
        # COCONUT: 每步 recurrence 过完整 Transformer 全部层
        self.think_layer_start = think_layer_start
        self.think_layer_end = min(think_layer_end, total_layers)
        self.think_layer_indices = list(range(total_layers))  # 全部层
        
        self.latent_thinker = NativeLatentThinker(
            hidden_size=actual_hidden_size,
            max_think_steps=max_think_steps,
            saturation_exit_threshold=saturation_exit_threshold,
            latent_end_token_id=None,  # 在 set_processor 中设置
            # ====== 3 层 anti-collapse 修复超参 (从环境变量读取, 默认值即"温和起步") ======
            exit_margin=float(os.environ.get('NLD_EXIT_MARGIN', '2.0')),
            exit_margin_weight=float(os.environ.get('NLD_EXIT_MARGIN_WEIGHT', '0.5')),
            sw_srs_alpha=float(os.environ.get('NLD_SW_SRS_ALPHA', '0.3')),
            swsrs_anti_collapse_weight=float(os.environ.get('NLD_SWSRS_ANTI_COLLAPSE_WEIGHT', '0.5')),
            swsrs_anti_collapse_margin=float(os.environ.get('NLD_SWSRS_ANTI_COLLAPSE_MARGIN', '1.0')),
            diversity_threshold=float(os.environ.get('NLD_DIVERSITY_THRESHOLD', '0.95')),
            diversity_weight=float(os.environ.get('NLD_DIVERSITY_WEIGHT', '1.0')),
            # ====== 语义侧监督 latent loss mode ======
            #   'laser_dwal' (默认, Laser 官方): student 全词表 log_softmax + teacher 子集 weighted-CE,
            #                                   严格对齐 forward_dwal.py, 不退化.
            #   'stage_kw'                    : teacher = K_s 上外部均匀 hard anchor + KL on W_s, 不退化.
            #   'sw_srs'                      : 旧 self-distill KL on W_s (KL=0 退化, 仅向后兼容).
            loss_mode=str(os.environ.get('NLD_LOSS_MODE', 'laser_dwal')),
            # ====== path B2 视觉侧 vision loss (默认关闭, 向后兼容) ======
            vision_loss_weight=float(os.environ.get('NLD_VISION_LOSS_WEIGHT', '0.0')),
            vision_top_k=int(os.environ.get('NLD_VISION_TOP_K', '6')),
            vision_margin=float(os.environ.get('NLD_VISION_MARGIN', '0.05')),
        )
        self.latent_thinker = self.latent_thinker.to(dtype=torch_dtype)

        # ====== 2.5 Latent 监督权重 ======
        # 当前仅使用 SW-SRS (Stage-Windowed Self-Refined Supervision, 参考 Laser arxiv 2601.06803).
        # 通过 key_token_weight 控制 (复用旧字段名以兼容已有脚本/配置).
        # 不引入任何视觉侧监督: 视觉对齐由下游 next-token CE 自然反传提供.
        self.key_token_weight = float(os.environ.get('NLD_KEY_TOKEN_WEIGHT', '1.0'))

        # ====== 3 层 anti-collapse 顶层权重 (与 latent_thinker 内部权重独立, 用于 model_v2 处再加权) ======
        # latent_thinker 内部已用 exit_margin_weight 等参与本地 loss 强度调节;
        # 这里的权重用于在 total_loss 中再做一次全局加权, 与 key_token_weight 同级.
        self.exit_margin_loss_weight = float(os.environ.get('NLD_EXIT_MARGIN_LOSS_WEIGHT', '1.0'))
        self.swsrs_anti_collapse_loss_weight = float(os.environ.get('NLD_SWSRS_ANTI_COLLAPSE_LOSS_WEIGHT', '1.0'))
        self.diversity_loss_weight = float(os.environ.get('NLD_DIVERSITY_LOSS_WEIGHT', '1.0'))
        # path B2 视觉侧 vision_loss 顶层权重 (与 latent_thinker.vision_loss_weight 同级, 两级相乘)
        self.vision_loss_weight = float(os.environ.get('NLD_VISION_LOSS_TOTAL_WEIGHT', '1.0'))

        if self._verbose:
            print(f"[NLD] Latent 监督: SW-SRS (weight={self.key_token_weight}), "
                  f"tau=1.0, eta=0.6, alpha={self.latent_thinker.sw_srs_alpha} [Laser-style self-refined]")
            print(f"[NLD] Anti-Collapse 修复 (3 层):")
            print(f"      Layer 1 (Soft Anti-Premature-Exit): margin={self.latent_thinker.exit_margin}, "
                  f"local_w={self.latent_thinker.exit_margin_weight}, total_w={self.exit_margin_loss_weight}")
            print(f"      Layer 2 (SW-SRS Anti-Collapse):     margin={self.latent_thinker.swsrs_anti_collapse_margin}, "
                  f"local_w={self.latent_thinker.swsrs_anti_collapse_weight}, total_w={self.swsrs_anti_collapse_loss_weight}")
            print(f"      Layer 3 (Stage-Diversity):          threshold={self.latent_thinker.diversity_threshold}, "
                  f"local_w={self.latent_thinker.diversity_weight}, total_w={self.diversity_loss_weight}")
            print(f"[NLD] path B2 Vision Loss: local_w={self.latent_thinker.vision_loss_weight}, "
                  f"total_w={self.vision_loss_weight}, top_k={self.latent_thinker.vision_top_k}, "
                  f"margin={self.latent_thinker.vision_margin} "
                  f"({'OFF' if (self.latent_thinker.vision_loss_weight * self.vision_loss_weight) <= 0 else 'ON'})")

        # GQA 参数        
        # GQA 参数 (推理时需要)
        self.num_kv_heads = getattr(text_config, 'num_key_value_heads', 8)
        self.head_dim = getattr(text_config, 'head_dim', actual_hidden_size // num_attention_heads)

        if self._verbose:
            self.latent_thinker.print_summary()
            print(f"[NLD] Think layers: 全部 {len(self.think_layer_indices)} 层 (L0~L{total_layers-1}, 与 COCONUT 一致)")
            print(f"[NLD] 全量微调模式: 所有层都有梯度 (FSDP 分布式)")
            print(f"[NLD] GQA: num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}")

        # ====== 3. Step delimiter token ids ======
        self.step_delimiter_ids = None
        self.step_delimiter_id = None

    @property
    def _inner_model(self):
        """获取内部 Qwen3VL 模型"""
        return self.base_model.model

    @property
    def _lm_head(self):
        """获取 lm_head"""
        return self.base_model.lm_head

    def set_processor(self, processor):
        """设置 processor (用于推理时的 tokenizer)"""
        self.processor = processor
        tokenizer = processor.tokenizer
        # 获取 <|latent|> token id (必须已注册为特殊 token)
        self.latent_token_id = tokenizer.convert_tokens_to_ids(self.LATENT_TOKEN)
        if self.latent_token_id == tokenizer.unk_token_id:
            if self._verbose:
                print(f"[NLD] ⚠️ {self.LATENT_TOKEN} 未在 tokenizer 中注册, boundary 检测将使用 fallback")
            self.latent_token_id = None
        else:
            if self._verbose:
                print(f"[NLD] ✅ <|latent|> token id = {self.latent_token_id}")
        
        # 获取 <|/latent|> token id (退出 token)
        LATENT_END_TOKEN = "<|/latent|>"
        self.latent_end_token_id = tokenizer.convert_tokens_to_ids(LATENT_END_TOKEN)
        if self.latent_end_token_id == tokenizer.unk_token_id:
            if self._verbose:
                print(f"[NLD] ⚠️ {LATENT_END_TOKEN} 未在 tokenizer 中注册, 退出 token 预测将禁用")
            self.latent_end_token_id = None
        else:
            if self._verbose:
                print(f"[NLD] ✅ <|/latent|> token id = {self.latent_end_token_id}")
            # 将 latent_end_token_id 传递给 NativeLatentThinker
            self.latent_thinker.latent_end_token_id = self.latent_end_token_id
        
        # 兼容旧接口
        self.step_delimiter_ids = None
        self.step_delimiter_id = self.latent_token_id

        # ====== T2' (LASER 贴合): 注入 SW-SRS / stage_kw 的 W_t 排除集合 ======
        # LASER paper (arxiv 2601.06803, Sec 3.3) 显式排除 <laser_end> 不进 W_t.
        # 我们的对应物: 所有特殊 token (latent / pause / bos / eos / pad / unk /
        # im_start / im_end / image_token_id 等), 防止 hidden 把概率塞给特殊 token.
        try:
            _excluded: List[int] = []
            # 1) latent / pause / latent_end
            for tok_str in ("<|latent|>", "<|/latent|>", "<|pause|>"):
                _tid = tokenizer.convert_tokens_to_ids(tok_str)
                if _tid is not None and _tid != tokenizer.unk_token_id and _tid >= 0:
                    _excluded.append(int(_tid))
            # 2) tokenizer 标准 specials
            for _attr in ("bos_token_id", "eos_token_id", "pad_token_id",
                          "unk_token_id", "sep_token_id", "cls_token_id", "mask_token_id"):
                _tid = getattr(tokenizer, _attr, None)
                if _tid is not None and _tid >= 0:
                    _excluded.append(int(_tid))
            # 3) chat template specials
            for tok_str in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
                try:
                    _tid = tokenizer.convert_tokens_to_ids(tok_str)
                    if _tid is not None and _tid != tokenizer.unk_token_id and _tid >= 0:
                        _excluded.append(int(_tid))
                except Exception:
                    pass
            # 4) image_token_id (来自 base_model.config.image_token_id)
            try:
                _img_tid = getattr(self.base_model.config, "image_token_id", None)
                if _img_tid is not None and int(_img_tid) >= 0:
                    _excluded.append(int(_img_tid))
            except Exception:
                pass
            # 5) tokenizer.additional_special_tokens_ids 兜底
            try:
                _addtl = getattr(tokenizer, "additional_special_tokens_ids", None) or []
                for _tid in _addtl:
                    if _tid is not None and int(_tid) >= 0:
                        _excluded.append(int(_tid))
            except Exception:
                pass

            # ====== A 方案 (v6 LASER-vsp 适配): 排除英文 BPE 高频功能词 ======
            # 背景: visual_scanpath 数据中的 concept 是 1-3 词短语 (如 "thermal night plaza"),
            #       经 BPE 拆分后会引入大量功能词 token (如 ' of', ' the', ' to', ' a' ...).
            #       这些 token 没有视觉/概念意义, 若进入 W_t 会让 hidden 朝功能词飘.
            # 处理: 用 tokenize 实测 ' word' 和 'word' 两种形式, 取它们的全部 BPE id 加入排除集.
            #       注意只排除 "纯功能词"; 实词 (名词/动词/形容词) 绝不排除.
            _STOP_WORDS = (
                # 冠词
                "a", "an", "the",
                # 介词 (高频)
                "of", "in", "on", "at", "to", "for", "with", "by", "from",
                "into", "onto", "upon", "over", "under", "above", "below",
                "near", "off", "out",
                # 连词
                "and", "or", "but", "nor", "so", "yet", "if",
                # 代词
                "it", "its", "this", "that", "these", "those",
                "he", "she", "they", "them", "his", "her", "their",
                # be 动词 / 助动词
                "is", "are", "was", "were", "be", "been", "being",
                "has", "have", "had", "do", "does", "did",
                # 其他高频虚词
                "as", "than", "then", "there", "here",
                # 标点单字
                "-", ",", ".", ":", ";", "/",
            )
            _filler_ids: List[int] = []
            for _w in _STOP_WORDS:
                for _form in (_w, " " + _w):
                    try:
                        _ids = tokenizer(_form, add_special_tokens=False)["input_ids"]
                        if _ids and len(_ids) == 1:
                            # 只排除"单 BPE token 即可表达"的功能词;
                            # 多 BPE 拼出的形式不排除 (避免误伤实词的 sub-piece)
                            _filler_ids.append(int(_ids[0]))
                    except Exception:
                        pass
            if _filler_ids:
                _excluded.extend(_filler_ids)
                if self._verbose:
                    print(f"[NLD] ✅ SW-SRS A方案: 排除 {len(set(_filler_ids))} 个英文功能词 BPE id")

            _excluded_unique = sorted(set(_excluded))
            if _excluded_unique:
                _buf = torch.tensor(_excluded_unique, dtype=torch.long)
                # register_buffer 已在 __init__ 创建; 这里直接覆盖
                self.latent_thinker._sw_srs_excluded_ids = _buf.to(
                    device=next(self.latent_thinker.parameters()).device
                )
                if self._verbose:
                    print(f"[NLD] ✅ SW-SRS W_t 排除集合 ({len(_excluded_unique)} ids): {_excluded_unique}")
            else:
                if self._verbose:
                    print(f"[NLD] ⚠️ SW-SRS W_t 排除集合为空 (未找到任何 specials)")
        except Exception as _e:
            print(f"[NLD] ⚠️ SW-SRS specials 注入失败: {_e}; W_t 不做排除 (向后兼容)")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用梯度检查点 (代理到 base_model, DeepSpeed/Trainer 会调用此方法)"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            if self._verbose:
                print("[NLD] ✅ 梯度检查点已启用 (节省显存, 以计算换内存)")

    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self):
        """检查是否启用了梯度检查点"""
        if hasattr(self.base_model, 'is_gradient_checkpointing'):
            return self.base_model.is_gradient_checkpointing
        return False

    @staticmethod
    def _detach_cache(cache: DynamicCache) -> DynamicCache:
        """Detach KV cache 中所有 tensor 的梯度，截断跨 segment 的梯度传播。
        
        Args:
            cache: 需要 detach 的 DynamicCache
        
        Returns:
            新的 DynamicCache，所有 KV tensor 已 detach
        """
        from transformers.cache_utils import DynamicLayer as _DL
        detached = DynamicCache()
        for layer in cache.layers:
            new_layer = _DL()
            if hasattr(layer, 'keys') and layer.keys is not None:
                new_layer.keys = layer.keys.detach()
                new_layer.values = layer.values.detach()
                new_layer.is_initialized = True
            detached.layers.append(new_layer)
        return detached

    def _find_delimiter_positions(self, input_ids_1d: torch.Tensor) -> List[int]:
        """在 token 序列中查找 step boundary 位置
        
        仅使用 <|latent|> token id 直接定位 (O(n) 扫描, 无需文本解码)
        注意: Final Answer 后不触发隐空间推理，因此不作为 boundary
        """
        positions = []
        
        # 通过 token id 查找 <|latent|> 位置 (精确、高效)
        if hasattr(self, 'latent_token_id') and self.latent_token_id is not None:
            latent_positions = (input_ids_1d == self.latent_token_id).nonzero(as_tuple=True)[0].tolist()
            positions.extend(latent_positions)
        
        return sorted(set(positions))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        step_boundaries: Optional[List[List[int]]] = None,
        prompt_lens: Optional[List[int]] = None,
        loss_weight_mask: Optional[torch.Tensor] = None,
        latent_think_steps: Optional[List[List[int]]] = None,
        stage_concept_token_ids: Optional[List[List[List[int]]]] = None,
        # path B2 视觉侧: 与 stage_concept_token_ids 同构的 role 字符串嵌套
        stage_concept_roles: Optional[List[List[List[str]]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播 (Chunkwise + Latent Thinking)
        
        核心流程 (统一序列: thought tokens 永久保留):
          1. 视觉 encoder → visual_embeddings (no_grad)
          2. 按 step boundary 切分序列为 chunks
          3. 对每个 chunk:
             a. VLM 36 层 forward (有梯度)
             b. 如果是 step boundary:
                NativeLatentThinker → thought_output
                thought_output 作为下一个 chunk 的 prefix embeddings
                thought KV 永久保留在历史 cache 中
          4. concat 所有 chunk hidden (含 thought 占位, labels=-100) → logits → CE loss
        
        梯度路径:
          CE loss → logits → chunk_{c+1} hidden → thought_prefix → NativeLatentThinker
          → VLM 全部层 (全量微调) → chunk_c hidden
        
        统一序列设计:
          thought tokens 的 KV 不从 cache 中剥离，后续所有 chunk/token 都能通过
          attention 看到之前所有的 thought tokens。隐式推理 + 自然语言 CoT 构成
          完整的、连续的推理序列。
        """
        B = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device

        # 重置 accumulated losses 累积器
        self._accumulated_exit_token_loss = None
        self._accumulated_key_token_loss = None
        # ====== 3 层 anti-collapse 新增 loss 累积器 ======
        self._accumulated_exit_margin_loss = None
        self._accumulated_swsrs_anti_collapse_loss = None
        self._accumulated_diversity_loss = None
        self._accumulated_vision_loss = None
        self._last_exit_stats = None
        # Anti-collapse 监控累积 (每次 latent 触发各 append 一个 dict)
        self._accumulated_hidden_stats = []
        self._accumulated_sw_srs_stats = []
        self._accumulated_vision_stats = []

        import time as _time
        _fwd_start = _time.time()
        if not hasattr(self, '_fwd_count'):
            self._fwd_count = 0
        self._fwd_count += 1
        _rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
        _is_rank0 = (_rank == 0)

        # ====== dtype 对齐 ======
        if not hasattr(self, '_base_model_dtype'):
            self._base_model_dtype = torch.bfloat16
            for _n, _p in self.base_model.named_parameters():
                self._base_model_dtype = _p.dtype
                break
            if self._verbose:
                thinker_dtype = next(self.latent_thinker.parameters()).dtype
                print(f"[NLD] dtype 检测: 基座模型={self._base_model_dtype}, "
                      f"latent_thinker={thinker_dtype}")

        _base_dtype = self._base_model_dtype

        # ====== 1. 视觉 encoder (no_grad) ======
        _t_vision_start = _time.time()
        with torch.no_grad():
            cached_visual_embeds = None
            deepstack_features = None
            # TGVR 档 A: 保留 per-sample 视觉 embedding (按 sample 切分),
            # 用于 latent_thinker 内的 stage key tokens 召回 v_pos_s 诊断.
            _image_embeds_per_sample: Optional[List[torch.Tensor]] = None

            if pixel_values is not None:
                inner_model = self._inner_model
                _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
                vision_output = inner_model.visual(
                    _pixel_values, grid_thw=image_grid_thw
                )
                merged_hidden_states, deepstack_features = _extract_visual_output(vision_output)
                split_sizes = (
                    image_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size ** 2
                ).tolist()
                image_embeds_list = list(torch.split(merged_hidden_states, split_sizes))

                cached_visual_embeds = torch.cat(image_embeds_list, dim=0)
                # 保留按 sample 切分的视觉 embedding (TGVR 诊断用, 已 detach 因为 no_grad)
                _image_embeds_per_sample = [t.detach() for t in image_embeds_list]

        _t_vision_end = _time.time()

        # ====== 2. Embedding + 视觉融合 ======
        inner_model = self._inner_model
        text_model = inner_model.language_model
        lm_head = self._lm_head

        with torch.no_grad():
            inputs_embeds = inner_model.get_input_embeddings()(input_ids)

            if cached_visual_embeds is not None:
                image_token_id = inner_model.config.image_token_id
                image_mask = (input_ids == image_token_id)
                num_image_tokens = image_mask.sum().item()
                if num_image_tokens > 0:
                    image_embeds = cached_visual_embeds[:num_image_tokens].to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    image_mask_3d = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask_3d, image_embeds)

        # 确保 inputs_embeds 有梯度 (gradient checkpointing 需要至少一个输入有 requires_grad)
        if not inputs_embeds.requires_grad:
            inputs_embeds = inputs_embeds.detach().requires_grad_(True)

        # ---- 早期 NaN 探针 A: vision → embed 融合 (每次 forward 都检查, 只在异常时打印) ----
        with torch.no_grad():
            _ie_nan = torch.isnan(inputs_embeds).any().item()
            _ie_inf = torch.isinf(inputs_embeds).any().item()
            _vis_nan = False
            if cached_visual_embeds is not None:
                _vis_nan = (
                    torch.isnan(cached_visual_embeds).any().item()
                    or torch.isinf(cached_visual_embeds).any().item()
                )
            if _ie_nan or _ie_inf or _vis_nan:
                _ie_max = -1.0 if (_ie_nan or _ie_inf) else inputs_embeds.abs().max().item()
                print(
                    f"  🚨A [rank {_rank}] fwd#{self._fwd_count} STAGE=A(vision+embed) "
                    f"inputs_embeds(nan={_ie_nan}, inf={_ie_inf}, max={_ie_max:.1f})  "
                    f"vision_embed_bad={_vis_nan}",
                    flush=True,
                )

        # visual_pos_masks (DeepStack 需要)
        visual_pos_masks = None
        deepstack_for_fwd = None
        if pixel_values is not None and image_grid_thw is not None:
            image_token_id = inner_model.config.image_token_id
            visual_pos_masks = (input_ids == image_token_id)
            deepstack_for_fwd = deepstack_features

        # ====== 3. mRoPE position_ids ======
        with torch.no_grad():
            seg_token_mask = torch.ones(B, seq_len, device=device, dtype=torch.long)
            inner_model.rope_deltas = None
            position_ids, rope_deltas = inner_model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw if pixel_values is not None else None,
                video_grid_thw=None,
                attention_mask=seg_token_mask,
            )
            inner_model.rope_deltas = rope_deltas

            if position_ids is not None and position_ids.dim() == 3:
                text_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
                position_ids = torch.cat([text_pos.unsqueeze(0), position_ids], dim=0)

        # position_ids 处理
        _position_ids = position_ids
        if _position_ids is not None and _position_ids.ndim == 3 and _position_ids.shape[0] == 4:
            _text_position_ids = _position_ids[0]
            _mrope_position_ids = _position_ids[1:]
        elif _position_ids is not None:
            _text_position_ids = _position_ids[0]
            _mrope_position_ids = _position_ids
        else:
            cache_position = torch.arange(seq_len, device=device)
            _text_position_ids = cache_position.view(1, 1, -1).expand(3, B, -1)[0]
            _mrope_position_ids = cache_position.view(1, 1, -1).expand(3, B, -1)

        cache_position = torch.arange(seq_len, device=device)

        # ====== 4. 检测 step 边界 ======
        if prompt_lens is not None:
            gen_start = max(prompt_lens)
        else:
            gen_start = 0

        if step_boundaries is None and hasattr(self, 'processor') and self.processor is not None:
            step_boundaries = []
            for b in range(B):
                positions = self._find_delimiter_positions(input_ids[b])
                positions = [p for p in positions if p >= gen_start]
                step_boundaries.append(positions)

        # 收集所有 step 边界
        all_boundaries = set()
        if step_boundaries:
            for b_list in step_boundaries:
                all_boundaries.update(b_list)
        split_points = sorted([p for p in all_boundaries if gen_start < p < seq_len])

        per_sample_sets = [set(bl) for bl in step_boundaries] if step_boundaries else [set() for _ in range(B)]

        MAX_STEPS = 14
        if len(split_points) > MAX_STEPS:
            split_points = split_points[:MAX_STEPS]

        # ====== 构建 chunks ======
        chunk_ranges = []
        if len(split_points) == 0:
            chunk_ranges.append((0, seq_len))
        else:
            chunk_ranges.append((0, split_points[0] + 1))
            for i in range(1, len(split_points)):
                chunk_ranges.append((split_points[i - 1] + 1, split_points[i] + 1))
            if split_points[-1] + 1 < seq_len:
                chunk_ranges.append((split_points[-1] + 1, seq_len))

        # ====== FSDP 预同步: 在循环之前同步所有 rank 的 text_model() 调用次数 ======
        # FSDP full_shard 模式下, 每次 text_model() 调用都触发 AllGather 集合通信,
        # 要求所有 rank 同步参与。不同样本的 chunks 数和 thinker steps 数不同,
        # 导致各 rank 的 text_model() 调用次数不同 → 集合通信不匹配 → NCCL 死锁。
        # 解决方案: 在循环之前预计算本 rank 的调用次数, 通过 all_reduce 获取全局最大值,
        # 循环结束后不足的 rank 用 dummy forward 补齐。
        # 注意: all_reduce 必须在所有 text_model() 调用之前执行, 否则会与 FSDP 的
        # AllGather 交叉导致死锁 (不同 rank 在同一 SeqNum 执行不同类型的集合操作)。
        import torch.distributed as dist
        _fsdp_sync_needed = dist.is_initialized() and self.training
        _fsdp_max_calls = 0
        _fsdp_my_calls = 0
        if _fsdp_sync_needed:
            # 预计算本 rank 的 text_model() 调用次数:
            # - 每个 chunk 调用 1 次 text_model()
            # - 每个 boundary 处的 thinker 调用 target_steps 次 text_model()
            # 训练时不再受 max_think_steps 硬上限约束，使用实际的 latent_think_steps
            _num_boundaries = len(split_points)
            # 计算每个 boundary 的实际 target_steps (取 batch 中最大值)
            _predicted_think_steps_per_boundary = self.latent_thinker.max_think_steps  # 默认回退值
            if latent_think_steps is not None:
                _all_steps = []
                for b_idx in range(B):
                    if b_idx < len(latent_think_steps):
                        for s in latent_think_steps[b_idx]:
                            if s > 0:
                                _all_steps.append(s)
                if _all_steps:
                    _predicted_think_steps_per_boundary = max(_all_steps)
            _my_predicted_calls = len(chunk_ranges) + _num_boundaries * _predicted_think_steps_per_boundary
            _calls_tensor = torch.tensor([_my_predicted_calls], device=device, dtype=torch.long)
            dist.all_reduce(_calls_tensor, op=dist.ReduceOp.MAX)
            _fsdp_max_calls = _calls_tensor.item()
            _fsdp_my_calls = 0  # 实际调用计数器, 在循环中递增
            
            if self._verbose and self._fwd_count <= 2:
                print(f"[rank {_rank}] FSDP 预同步: 预计 {_my_predicted_calls} 次 text_model() 调用, "
                      f"全局最大 {_fsdp_max_calls} 次", flush=True)

        # ====== 5. Segment-wise Forward + Latent Thinking ======
        # 统一 forward 架构: 使用 text_model() 整体调用替代手动逐层 forward
        # - KV cache 由 DynamicCache 自动管理
        # - attention mask 由 text_model 内部自动构建
        # - 方案 B (梯度可回传): thinker 内部使用局部 cache (不修改 global_cache)
        #   thinker 完成后, thought_output 作为 prefix embedding 注入下一个 segment 的 inputs_embeds
        #   梯度路径: CE loss → Segment 2 hidden → inputs_embeds → thought_output → thinker → VLM
        #   下一个 segment 的 text_model() 会把 thought prefix + segment tokens 的 KV 一起写入 global_cache
        #   这样后续所有 segment 都能通过 KV cache 看到 thought tokens

        # ---- 临时禁用 gradient checkpointing ----
        # activation checkpointing 与有状态 KV cache 不兼容:
        # backward 重计算 layer forward 时, DynamicCache 已被后续 segment/thinker 步骤修改,
        # 导致 KV shape 不一致 (forward 时 410, recompute 时 420)。
        # 解决方案: 在 segment-wise forward 循环中禁用 gradient checkpointing,
        # 仍保留 FSDP 参数分片来节省显存。循环结束后恢复。
        _gc_was_enabled = getattr(text_model, 'gradient_checkpointing', False)
        if _gc_was_enabled:
            text_model.gradient_checkpointing = False
            if self._verbose and self._fwd_count == 0:
                print("[NLD] ⚠️ 临时禁用 gradient checkpointing (segment-wise forward 与 KV cache 不兼容)")

        all_chunk_hidden = []  # 收集每个 segment 的 hidden states (含 thought prefix)
        thought_count = 0  # 实际触发的隐空间思考次数
        kv_offset = 0  # thought tokens 在 KV cache 中占据的额外位置数
        thought_insertions = []  # [(insert_pos_in_output_seq, num_thought_tokens), ...]
        cumulative_thought_tokens = 0  # 已插入的 thought token 总数
        
        # 暂存上一个 boundary 产生的 thought_output, 注入到下一个 segment
        pending_thought_prefix = None  # (thought_output, num_thought_steps, thinker_result)

        # 全局 KV cache (由 DynamicCache 自动管理)
        global_cache = DynamicCache()

        _t_base_fwd_start = _time.time()

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_ranges):
            chunk_len = chunk_end - chunk_start

            # ---- 当前 segment 的输入切片 ----
            chunk_embeds = inputs_embeds[:, chunk_start:chunk_end, :]  # [B, chunk_len, H]
            # cache_position 需要加上 kv_offset (之前所有 thought tokens 占据的位置)
            chunk_cache_position = cache_position[chunk_start:chunk_end] + kv_offset
            chunk_mrope_pos = _mrope_position_ids[:, :, chunk_start:chunk_end]

            # ---- position_ids: [4, B, chunk_len] ----
            chunk_text_pos = (_text_position_ids[:, chunk_start:chunk_end] + kv_offset
                              if _text_position_ids.dim() == 2
                              else _text_position_ids[:, chunk_start:chunk_end])
            actual_position_ids = torch.cat([chunk_text_pos.unsqueeze(0), chunk_mrope_pos], dim=0)

            # ---- 方案 B: 如果有 pending thought prefix, 注入到当前 segment 的 inputs_embeds 前面 ----
            if pending_thought_prefix is not None:
                thought_output_prefix, n_thought_prefix, _ = pending_thought_prefix
                
                # thought prefix 的 position_ids: [4, B, n_thought]
                # thought tokens 的位置紧跟在上一个 segment 末尾之后
                thought_base_pos = chunk_cache_position[0].item() - n_thought_prefix
                thought_cache_positions = torch.arange(
                    thought_base_pos, thought_base_pos + n_thought_prefix, device=device
                )
                thought_mrope_positions = thought_cache_positions.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                thought_text_positions = torch.full(
                    (B, n_thought_prefix), thought_base_pos, device=device, dtype=torch.long
                )
                # 多步 thought 的 text_pos 递增
                for t_idx in range(n_thought_prefix):
                    thought_text_positions[:, t_idx] = thought_base_pos + t_idx
                thought_pos_ids = torch.cat([
                    thought_text_positions.unsqueeze(0),  # [1, B, n_thought]
                    thought_mrope_positions,  # [3, B, n_thought]
                ], dim=0)  # [4, B, n_thought]
                
                # 拼接: thought prefix + segment embeds
                chunk_embeds = torch.cat([
                    thought_output_prefix.to(dtype=chunk_embeds.dtype),
                    chunk_embeds,
                ], dim=1)  # [B, n_thought + chunk_len, H]
                
                # 拼接 cache_position
                chunk_cache_position = torch.cat([thought_cache_positions, chunk_cache_position])
                
                # 拼接 position_ids
                actual_position_ids = torch.cat([thought_pos_ids, actual_position_ids], dim=2)
                
                pending_thought_prefix = None  # 已消费

            # ---- attention_mask ----
            total_kv_len = chunk_cache_position[-1].item() + 1
            actual_attn_mask = torch.ones(B, total_kv_len, device=device, dtype=attention_mask.dtype)

            # ---- visual_pos_masks 处理 ----
            # 如果有 thought prefix, visual_pos_masks 需要在前面补 False
            seg_visual_pos_masks = (visual_pos_masks[:, chunk_start:chunk_end] if visual_pos_masks is not None else None)
            if seg_visual_pos_masks is not None and chunk_embeds.shape[1] > chunk_len:
                # thought prefix 不是视觉 token, 补 False
                n_prefix = chunk_embeds.shape[1] - chunk_len
                prefix_mask = torch.zeros(B, n_prefix, device=device, dtype=seg_visual_pos_masks.dtype)
                seg_visual_pos_masks = torch.cat([prefix_mask, seg_visual_pos_masks], dim=1)

            # ---- DeepStack: 只在包含视觉 token 的 segment 传递 ----
            # 视觉 token 全部在 prompt 中 (第一个 chunk), 后续 chunk 只有纯文本 CoT 步骤
            # _deepstack_process 要求 visual_embeds 数量 == visual_pos_masks 中 True 的数量
            # 因此: 无视觉 token 的 segment 必须不传 deepstack, 否则尺寸不匹配
            seg_deepstack = None
            if seg_visual_pos_masks is not None and deepstack_for_fwd is not None:
                num_vis_in_seg = seg_visual_pos_masks.sum().item()
                if num_vis_in_seg > 0:
                    # 第一个 chunk 包含所有视觉 token, deepstack_for_fwd 完整传入即可
                    seg_deepstack = deepstack_for_fwd
                else:
                    # 后续 chunk 无视觉 token, 不传 deepstack
                    seg_visual_pos_masks = None

            # ---- text_model() 整体调用 ----
            # KV cache 自动管理: 当前 segment (含 thought prefix) 的 KV 自动追加到 global_cache
            # thought prefix 的 KV 通过这里写入 global_cache (而非 thinker 内部)
            # 梯度路径: CE loss → hidden → inputs_embeds → thought_output → thinker ✅
            seg_outputs = text_model(
                inputs_embeds=chunk_embeds,
                attention_mask=actual_attn_mask,
                position_ids=actual_position_ids,
                past_key_values=global_cache,
                cache_position=chunk_cache_position,
                use_cache=True,
                visual_pos_masks=seg_visual_pos_masks,
                deepstack_visual_embeds=seg_deepstack,
            )
            global_cache = seg_outputs.past_key_values
            if _fsdp_sync_needed:
                _fsdp_my_calls += 1
            
            # last_hidden_state 已过 final norm
            chunk_normed = seg_outputs.last_hidden_state  # [B, actual_input_len, H]

            # ---- 早期 NaN 探针 B: text_model 输出 (每次都检查, 仅异常时打印) ----
            with torch.no_grad():
                _cn_nan = torch.isnan(chunk_normed).any().item()
                _cn_inf = torch.isinf(chunk_normed).any().item()
                if _cn_nan or _cn_inf:
                    _cn_max = float('nan') if _cn_nan else chunk_normed.abs().max().item()
                    # 同时检查输入 embed 是否干净, 判断是 text_model 自己产生 NaN 还是上游传染
                    _chunk_in_nan = torch.isnan(chunk_embeds).any().item()
                    _chunk_in_inf = torch.isinf(chunk_embeds).any().item()
                    print(
                        f"  🚨B [rank {_rank}] fwd#{self._fwd_count} STAGE=B(text_model) "
                        f"chunk#{chunk_idx}/{len(chunk_ranges)} "
                        f"out_nan={_cn_nan} out_inf={_cn_inf} out_max={_cn_max} "
                        f"in_nan={_chunk_in_nan} in_inf={_chunk_in_inf} "
                        f"chunk_shape={tuple(chunk_embeds.shape)} kv_len={total_kv_len}",
                        flush=True,
                    )

            # 收集 segment 的 hidden states (含 thought prefix 部分)
            all_chunk_hidden.append(chunk_normed)

            # ---- 跨 segment 梯度截断: detach KV cache ----
            # 每个 segment 的计算图独立，通过 detach KV cache 截断跨 segment 梯度
            # 梯度通过 thought prefix → inputs_embeds → thinker 回传
            global_cache = self._detach_cache(global_cache)

            # ---- 如果是 step boundary, 触发 NativeLatentThinker ----
            chunk_end_token = chunk_end - 1
            is_at_boundary = chunk_end_token in set(split_points)

            if is_at_boundary and chunk_idx < len(chunk_ranges) - 1:
                # 取当前 segment 最后 1 个 token 的 hidden state 作为 thought token
                # 注意: 如果当前 segment 有 thought prefix, 最后一个 token 仍然是原始 segment 的最后一个
                last_hidden = chunk_normed[:, -1:, :]  # [B, 1, H]
                
                # 第一步 thought 的位置 (放在当前 segment 末尾之后)
                base_thought_pos = chunk_end + kv_offset
                thought_positions = torch.arange(
                    base_thought_pos, base_thought_pos + 1, device=device
                )
                thought_mrope_pos = thought_positions.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                
                # 第一步的 position_ids: [4, B, 1]
                thought_text_pos_t = torch.full((B, 1), base_thought_pos, device=device, dtype=torch.long)
                thought_position_ids = torch.cat([thought_text_pos_t.unsqueeze(0), thought_mrope_pos], dim=0)
                
                # 第一步的 attention_mask
                think_attn_mask = torch.ones(B, base_thought_pos + 1, device=device, dtype=torch.long)
                
                # NativeLatentThinker: Coconut 等价的隐空间推理
                # 方案 B: thinker 内部使用局部 cache (训练时不修改 global_cache)
                # thought_output 将作为 prefix 注入下一个 segment
                
                # 确定当前 boundary 的 target_steps (per-sample 的隐空间迭代步数)
                # latent_think_steps: List[List[int]], 每个样本的每个 <|latent|> 对应的步数
                # thought_count 是当前已触发的 boundary 索引
                _target_steps = None
                if latent_think_steps is not None:
                    # 取 batch 中所有样本在当前 boundary 的 target_steps 的最大值
                    # (因为 thinker 对整个 batch 统一执行，步数取 max)
                    _batch_steps = []
                    for b_idx in range(B):
                        if b_idx < len(latent_think_steps) and thought_count < len(latent_think_steps[b_idx]):
                            _s = latent_think_steps[b_idx][thought_count]
                            if _s > 0:
                                _batch_steps.append(_s)
                    if _batch_steps:
                        _target_steps = max(_batch_steps)

                # ---- per-boundary 切片 stage_concept_token_ids ----
                # 新数据契约 (v6.2): stage_concept_token_ids 是 4 维
                #   [B][boundary][stage][token_ids], 不同 boundary 配独立 stages.
                # 旧数据契约 (v6.1 之前): 3 维 [B][stage][token_ids], 所有 boundary 共享同一组 stages.
                # 我们在每次 boundary 循环中按 thought_count 切片到 thinker 期望的 3 维 [B][stage][token_ids].
                _per_boundary_stages = stage_concept_token_ids
                if stage_concept_token_ids is not None and len(stage_concept_token_ids) > 0:
                    _first = stage_concept_token_ids[0]
                    # _first 形态判定: 4 维样本的 _first 是 List[List[List[int]]] (boundary -> stage -> ids),
                    #   其首元素是 list (stage), 内部又是 list (token_ids);
                    #   3 维样本的 _first 是 List[List[int]] (stage -> ids), 其首元素直接是 list of int.
                    _is_4d = (
                        isinstance(_first, list)
                        and len(_first) > 0
                        and isinstance(_first[0], list)
                        and len(_first[0]) > 0
                        and isinstance(_first[0][0], list)
                    )
                    if _is_4d:
                        # 4 维: 按 thought_count 取每个样本的当前 boundary 的 stages
                        _sliced = []
                        for b_idx in range(B):
                            sample_boundaries = stage_concept_token_ids[b_idx] if b_idx < len(stage_concept_token_ids) else []
                            if thought_count < len(sample_boundaries):
                                _sliced.append(sample_boundaries[thought_count])
                            else:
                                _sliced.append([])
                        _per_boundary_stages = _sliced
                    # else: 3 维, 保持向后兼容直接传

                # ---- per-boundary 切片 stage_concept_roles (与 stages 同步) ----
                _per_boundary_roles: Optional[List[List[str]]] = None
                if stage_concept_roles is not None and len(stage_concept_roles) > 0:
                    _sliced_roles: List[List[str]] = []
                    for b_idx in range(B):
                        sample_role_boundaries = (
                            stage_concept_roles[b_idx]
                            if b_idx < len(stage_concept_roles) else []
                        )
                        if thought_count < len(sample_role_boundaries):
                            _sliced_roles.append(
                                list(sample_role_boundaries[thought_count])
                            )
                        else:
                            _sliced_roles.append([])
                    _per_boundary_roles = _sliced_roles

                thinker_result = self.latent_thinker(
                    last_hidden=last_hidden,
                    text_model=text_model,
                    past_key_values=global_cache,
                    attention_mask=think_attn_mask,
                    cache_position=thought_positions,
                    position_ids=thought_position_ids,
                    rotary_emb_fn=text_model.rotary_emb,
                    mrope_position_ids=thought_mrope_pos,
                    base_thought_pos=base_thought_pos,
                    B=B,
                    target_steps=_target_steps,
                    lm_head=lm_head,
                    embed_tokens=inner_model.get_input_embeddings(),
                    stage_key_token_ids=_per_boundary_stages,
                    # TGVR 档 A 诊断 (无 loss): 传入 per-sample 视觉 embedding
                    image_embeds_per_sample=_image_embeds_per_sample,
                    # path B2 视觉侧 vision loss 需要 每 stage 的 role
                    stage_roles=_per_boundary_roles,
                )
                
                thought_output = thinker_result['thought_output']  # [B, num_steps, H]
                num_thought_steps = thinker_result.get('num_thought_steps', 1)
                if _fsdp_sync_needed:
                    # 实际 thinker 内部执行了 num_thought_steps 次 text_model() 调用
                    # 预同步时按实际 target_steps 预计，dummy forward 会补齐差额
                    _fsdp_my_calls += num_thought_steps
                
                # 记录 thought token 在输出序列中的插入位置
                insert_pos = chunk_end + cumulative_thought_tokens
                thought_insertions.append((insert_pos, num_thought_steps))
                cumulative_thought_tokens += num_thought_steps
                
                # 方案 B: 暂存 thought_output, 作为下一个 segment 的 prefix
                pending_thought_prefix = (thought_output, num_thought_steps, thinker_result)
                
                # 累积 exit_token_loss (退出 token 预测 loss)
                if 'exit_token_loss' in thinker_result:
                    if self._accumulated_exit_token_loss is None:
                        self._accumulated_exit_token_loss = thinker_result['exit_token_loss']
                    else:
                        self._accumulated_exit_token_loss = self._accumulated_exit_token_loss + thinker_result['exit_token_loss']
                
                # 累积 key_token_loss (Key Token Decoding Loss)
                if 'key_token_loss' in thinker_result:
                    _kt_loss = self.key_token_weight * thinker_result['key_token_loss']
                    if self._accumulated_key_token_loss is None:
                        self._accumulated_key_token_loss = _kt_loss
                    else:
                        self._accumulated_key_token_loss = self._accumulated_key_token_loss + _kt_loss

                # ====== 3 层 anti-collapse 新增 loss 累积 ======
                # Layer 1: exit_margin_loss (中间步软约束)
                if 'exit_margin_loss' in thinker_result:
                    _em = self.exit_margin_loss_weight * thinker_result['exit_margin_loss']
                    if self._accumulated_exit_margin_loss is None:
                        self._accumulated_exit_margin_loss = _em
                    else:
                        self._accumulated_exit_margin_loss = self._accumulated_exit_margin_loss + _em
                # Layer 2: swsrs_anti_collapse_loss (强制 stage keys mass > exit logit)
                if 'swsrs_anti_collapse_loss' in thinker_result:
                    _sac = self.swsrs_anti_collapse_loss_weight * thinker_result['swsrs_anti_collapse_loss']
                    if self._accumulated_swsrs_anti_collapse_loss is None:
                        self._accumulated_swsrs_anti_collapse_loss = _sac
                    else:
                        self._accumulated_swsrs_anti_collapse_loss = self._accumulated_swsrs_anti_collapse_loss + _sac
                # Layer 3: diversity_loss (跨步 hidden 不能复读)
                if 'diversity_loss' in thinker_result:
                    _dv = self.diversity_loss_weight * thinker_result['diversity_loss']
                    if self._accumulated_diversity_loss is None:
                        self._accumulated_diversity_loss = _dv
                    else:
                        self._accumulated_diversity_loss = self._accumulated_diversity_loss + _dv

                # ====== path B2 视觉侧 vision_loss 累积 ======
                if 'vision_loss' in thinker_result:
                    _vl = self.vision_loss_weight * thinker_result['vision_loss']
                    if self._accumulated_vision_loss is None:
                        self._accumulated_vision_loss = _vl
                    else:
                        self._accumulated_vision_loss = self._accumulated_vision_loss + _vl
                if 'vision_stats' in thinker_result and thinker_result['vision_stats']:
                    self._accumulated_vision_stats.append(thinker_result['vision_stats'])
                
                # 收集 exit_stats
                if 'exit_stats' in thinker_result:
                    self._last_exit_stats = thinker_result['exit_stats']
                
                # 收集 anti-collapse 统计 (hidden_stats + sw_srs_stats)
                if 'hidden_stats' in thinker_result and thinker_result['hidden_stats']:
                    self._accumulated_hidden_stats.append(thinker_result['hidden_stats'])
                if 'sw_srs_stats' in thinker_result and thinker_result['sw_srs_stats']:
                    self._accumulated_sw_srs_stats.append(thinker_result['sw_srs_stats'])

                thought_count += 1
                kv_offset += num_thought_steps
                
                # 方案 B: 不需要 detach thinker 的 KV cache
                # 因为 thinker 训练时使用局部 cache, 不修改 global_cache
                # thought KV 将在下一个 segment forward 时通过 prefix 注入写入 global_cache

        # ====== FSDP Dummy Forward 对齐 ======
        # 根据循环前预同步的 _fsdp_max_calls, 补齐不足的 text_model() 调用次数。
        # 此时所有 rank 都已完成 segment-wise forward 循环, 不会与 FSDP AllGather 交叉。
        if _fsdp_sync_needed:
            _dummy_needed = _fsdp_max_calls - _fsdp_my_calls
            
            if _dummy_needed > 0:
                # 用 1 个 dummy token 做 forward, 不使用 global_cache (避免污染)
                # dummy 输出不参与 loss 计算
                _dummy_embeds = torch.zeros(B, 1, inputs_embeds.shape[-1], device=device, dtype=inputs_embeds.dtype)
                _dummy_embeds.requires_grad_(True)
                _dummy_cache_pos = torch.zeros(1, device=device, dtype=torch.long)
                _dummy_attn_mask = torch.ones(B, 1, device=device, dtype=torch.long)
                _dummy_pos_ids = torch.zeros(4, B, 1, device=device, dtype=torch.long)
                
                for _di in range(_dummy_needed):
                    _dummy_cache = DynamicCache()
                    _dummy_out = text_model(
                        inputs_embeds=_dummy_embeds,
                        attention_mask=_dummy_attn_mask,
                        position_ids=_dummy_pos_ids,
                        past_key_values=_dummy_cache,
                        cache_position=_dummy_cache_pos,
                        use_cache=True,
                    )
                    del _dummy_cache, _dummy_out
                
                if self._verbose and self._fwd_count <= 2:
                    print(f"[rank {_rank}] FSDP 对齐: 补齐 {_dummy_needed} 次 dummy forward "
                          f"(本 rank {_fsdp_my_calls} 次, 全局最大 {_fsdp_max_calls} 次)", flush=True)

        # 释放 KV cache
        del global_cache

        # ---- 恢复 gradient checkpointing ----
        if _gc_was_enabled:
            text_model.gradient_checkpointing = True

        _t_base_fwd_end = _time.time()

        # ====== 6. Concat 所有 chunk hidden → 完整序列 (含 thought tokens) ======
        if len(all_chunk_hidden) == 1:
            adapted_hidden = all_chunk_hidden[0]
        else:
            adapted_hidden = torch.cat(all_chunk_hidden, dim=1)  # [B, seq_len + total_thought_tokens, H]

        # ====== 7. 计算 logits ======
        # ---- NaN 探针 C: adapted_hidden → logits 前后 ----
        with torch.no_grad():
            _ah_nan = torch.isnan(adapted_hidden).any().item()
            _ah_inf = torch.isinf(adapted_hidden).any().item()
            if _ah_nan or _ah_inf:
                _ah_max = float('nan') if _ah_nan else adapted_hidden.abs().max().item()
                print(
                    f"  🚨C [rank {_rank}] fwd#{self._fwd_count} STAGE=C(before_lm_head) "
                    f"adapted_hidden nan={_ah_nan} inf={_ah_inf} max={_ah_max} "
                    f"shape={tuple(adapted_hidden.shape)} num_chunks={len(all_chunk_hidden)}",
                    flush=True,
                )

        logits = lm_head(adapted_hidden)  # [B, extended_seq_len, V]

        # ---- NaN 探针 D: logits 输出 (lm_head 本身是否产生 NaN) ----
        with torch.no_grad():
            _lg_nan = torch.isnan(logits).any().item()
            _lg_inf = torch.isinf(logits).any().item()
            if _lg_nan or _lg_inf:
                # 若 adapted_hidden 干净但 logits 炸 → 一定是 lm_head.weight 本身有问题
                _upstream_ok = (not _ah_nan) and (not _ah_inf)
                _lmw = lm_head.weight if hasattr(lm_head, 'weight') else None
                if _lmw is not None:
                    _lmw_nan = torch.isnan(_lmw).any().item()
                    _lmw_inf = torch.isinf(_lmw).any().item()
                    _lmw_max = float('nan') if (_lmw_nan or _lmw_inf) else _lmw.abs().max().item()
                else:
                    _lmw_nan, _lmw_inf, _lmw_max = 'NA', 'NA', 'NA'
                print(
                    f"  🚨D [rank {_rank}] fwd#{self._fwd_count} STAGE=D(after_lm_head) "
                    f"logits nan={_lg_nan} inf={_lg_inf}  "
                    f"upstream_hidden_clean={_upstream_ok}  "
                    f"lm_head.weight(nan={_lmw_nan}, inf={_lmw_inf}, max={_lmw_max})",
                    flush=True,
                )

        # ====== 7.5 对齐 labels 和 loss_weight_mask ======
        # 由于 thought tokens 被保留在序列中，adapted_hidden 的长度 > 原始 seq_len
        # 需要在 labels 和 loss_weight_mask 的 thought 位置插入 -100 / 0.0 占位
        if thought_insertions and labels is not None:
            # 构建扩展后的 labels: 在 thought 位置插入 -100
            extended_len = adapted_hidden.shape[1]
            extended_labels = torch.full((B, extended_len), -100, dtype=labels.dtype, device=device)
            
            # 构建映射: 原始序列中的每个位置 → 扩展序列中的位置
            # thought_insertions: [(insert_pos, num_tokens), ...]
            # insert_pos 是在扩展序列中的位置 (已考虑之前的 thought tokens)
            src_pos = 0  # 原始序列中的位置
            dst_pos = 0  # 扩展序列中的位置
            for insert_pos, n_thought in thought_insertions:
                # 复制 thought 之前的原始 token labels
                copy_len = insert_pos - dst_pos  # 这段是原始 token
                if copy_len > 0:
                    # 从原始 labels 中复制
                    orig_start = src_pos
                    orig_end = src_pos + copy_len
                    extended_labels[:, dst_pos:dst_pos + copy_len] = labels[:, orig_start:orig_end]
                    src_pos += copy_len
                    dst_pos += copy_len
                # thought token 位置: 保持 -100 (已初始化)
                dst_pos += n_thought
            # 复制剩余的原始 token labels
            remaining = seq_len - src_pos
            if remaining > 0:
                extended_labels[:, dst_pos:dst_pos + remaining] = labels[:, src_pos:src_pos + remaining]
            
            labels = extended_labels
            
            # 同样扩展 loss_weight_mask
            if loss_weight_mask is not None:
                extended_weights = torch.zeros((B, extended_len), dtype=loss_weight_mask.dtype, device=device)
                src_pos = 0
                dst_pos = 0
                for insert_pos, n_thought in thought_insertions:
                    copy_len = insert_pos - dst_pos
                    if copy_len > 0:
                        orig_start = src_pos
                        orig_end = src_pos + copy_len
                        extended_weights[:, dst_pos:dst_pos + copy_len] = loss_weight_mask[:, orig_start:orig_end]
                        src_pos += copy_len
                        dst_pos += copy_len
                    # thought token 位置: 权重为 0 (不计算 loss)
                    dst_pos += n_thought
                remaining = seq_len - src_pos
                if remaining > 0:
                    extended_weights[:, dst_pos:dst_pos + remaining] = loss_weight_mask[:, src_pos:src_pos + remaining]
                loss_weight_mask = extended_weights

        # ====== 8. 计算 loss ======
        ctrl_dtype = next(self.latent_thinker.parameters()).dtype
        total_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)
        main_loss = torch.tensor(0.0, device=device, dtype=ctrl_dtype)

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()

            if loss_weight_mask is not None:
                shift_weights = loss_weight_mask[:, 1:].contiguous().to(shift_logits.dtype)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                per_token_loss = per_token_loss.view(shift_labels.shape)
                valid_mask = (shift_labels != -100).float()
                weighted_loss = per_token_loss * shift_weights * valid_mask
                ce_loss = weighted_loss.sum() / (shift_weights * valid_mask).sum().clamp(min=1.0)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            main_loss = ce_loss.detach()
            total_loss = ce_loss

            # 退出 token 预测 loss (模型学习自适应退出 latent 推理)
            if hasattr(self, '_accumulated_exit_token_loss') and self._accumulated_exit_token_loss is not None:
                total_loss = total_loss + self._accumulated_exit_token_loss

            # SW-SRS Loss (Stage-Windowed Self-Refined Supervision, 参考 Laser arxiv 2601.06803)
            if hasattr(self, '_accumulated_key_token_loss') and self._accumulated_key_token_loss is not None:
                total_loss = total_loss + self._accumulated_key_token_loss

            # ====== 3 层 anti-collapse loss 累加 (含 per-loss NaN/超大值守卫) ======
            # 防御性策略: 任何 anti-collapse loss 分量出现 NaN / Inf / 超大值时跳过累加,
            #   避免单个异常 batch 污染 total_loss → 反向梯度爆炸 → bf16 权重 NaN.
            #   阈值 30.0: 远高于正常工作点 (~2-5), 但低于会撕裂训练的 50+ 量级.
            #   2026-05-18 调整: 20.0 → 30.0 配合 latent_thinker per_term_cap 10→6
            #   两侧夹逼后, 多数 batch 的 swsrs_anti_collapse loss 应稳定 < 30,
            #   anti-collapse 信号能进 total_loss 而非被守卫吃掉.
            _ac_loss_cap = 30.0
            _rank_dbg = int(os.environ.get('RANK', '0'))

            def _safe_add_ac_loss(_total, _comp_loss, _name):
                """安全累加 anti-collapse loss 分量, 异常时跳过 + 打印警告

                跳过时若 latent_thinker 提供了 _last_swsrs_anti_collapse_diag,
                顺带打印 end_logit / window_logsumexp / hit_cap_ratio,
                便于定位是 lm_head 在 <|/latent|> 上偏置漂移, 还是 hidden 塌缩.
                """
                if _comp_loss is None:
                    return _total
                _v = _comp_loss.detach().float()
                if torch.isnan(_v).any() or torch.isinf(_v).any():
                    print(f"  ⚠️ [rank {_rank_dbg}] anti-collapse {_name} = NaN/Inf, 跳过累加",
                          flush=True)
                    return _total
                if _v.abs().max().item() > _ac_loss_cap:
                    _diag_str = ""
                    if _name == 'swsrs_anti_collapse':
                        _diag = getattr(self.latent_thinker, '_last_swsrs_anti_collapse_diag', None)
                        if _diag is not None:
                            _diag_str = (
                                f"  diag: end_logit_mean={_diag['end_logit_mean']:+.2f} "
                                f"(max={_diag['end_logit_max']:+.2f})  "
                                f"window_lse_mean={_diag['window_lse_mean']:+.2f} "
                                f"(max={_diag['window_lse_max']:+.2f})  "
                                f"gap_mean={_diag['gap_mean']:+.2f}  "
                                f"hit_cap={_diag['hit_cap_ratio']*100:.1f}%  "
                                f"n_terms={_diag['n_terms']}"
                            )
                    print(f"  ⚠️ [rank {_rank_dbg}] anti-collapse {_name} = "
                          f"{_v.abs().max().item():.2f} (> {_ac_loss_cap}), 跳过累加"
                          f"{_diag_str}",
                          flush=True)
                    return _total
                return _total + _comp_loss

            # Layer 1: 中间步 exit margin (软约束, 避免提前预测 <|/latent|>)
            if hasattr(self, '_accumulated_exit_margin_loss'):
                total_loss = _safe_add_ac_loss(
                    total_loss, self._accumulated_exit_margin_loss, 'exit_margin'
                )
            # Layer 2: SW-SRS anti-collapse (强制 stage keys mass > exit logit, 拦截 hidden 塌缩到 exit)
            if hasattr(self, '_accumulated_swsrs_anti_collapse_loss'):
                total_loss = _safe_add_ac_loss(
                    total_loss, self._accumulated_swsrs_anti_collapse_loss, 'swsrs_anti_collapse'
                )
            # Layer 3: 跨步 hidden 多样性 (强制 hidden 演化, 不能复读)
            if hasattr(self, '_accumulated_diversity_loss'):
                total_loss = _safe_add_ac_loss(
                    total_loss, self._accumulated_diversity_loss, 'diversity'
                )
            # path B2 视觉侧 vision_loss (走同一套 NaN/大值守卫, 默认权重 0 时为 0 张量)
            if hasattr(self, '_accumulated_vision_loss'):
                total_loss = _safe_add_ac_loss(
                    total_loss, self._accumulated_vision_loss, 'vision'
                )

        # 确保 total_loss 有 grad_fn
        if total_loss.grad_fn is None and self.training:
            anchor = next(self.latent_thinker.parameters())
            total_loss = total_loss + (anchor.sum() * 0.0).to(total_loss.dtype)

        # NaN/Inf 检测 (增强诊断: 打印各 loss 分量 + logits 数值范围, 便于定位根因)
        if torch.isnan(total_loss).item() or torch.isinf(total_loss).item():
            _loss_type = "NaN" if torch.isnan(total_loss).item() else "Inf"
            _ce_v = ce_loss.item() if (labels is not None and torch.is_tensor(ce_loss)) else None
            _exit_v = self._accumulated_exit_token_loss.item() if (
                hasattr(self, '_accumulated_exit_token_loss') and
                self._accumulated_exit_token_loss is not None and
                torch.is_tensor(self._accumulated_exit_token_loss)
            ) else None
            _srs_v = self._accumulated_key_token_loss.item() if (
                hasattr(self, '_accumulated_key_token_loss') and
                self._accumulated_key_token_loss is not None and
                torch.is_tensor(self._accumulated_key_token_loss)
            ) else None
            try:
                _logit_max = logits.abs().max().item()
                _logit_mean = logits.float().mean().item()
                _logit_nan_cnt = torch.isnan(logits).sum().item()
            except Exception:
                _logit_max, _logit_mean, _logit_nan_cnt = -1.0, -1.0, -1
            print(
                f"  ❌ [rank {_rank}] loss 为 {_loss_type}! total={total_loss.item()}  "
                f"ce={_ce_v}  exit={_exit_v}  sw_srs={_srs_v}  "
                f"logit|max|={_logit_max:.2f} mean={_logit_mean:.2f} nan_cnt={_logit_nan_cnt}",
                flush=True
            )
            total_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
            anchor = next(self.latent_thinker.parameters())
            total_loss = total_loss + (anchor.sum() * 0.0).to(total_loss.dtype)

        # ====== 监控指标 (精简: 仅保留核心信号) ======
        _draft_metrics = {}
        with torch.no_grad():
            # 结构: 仅 thought_count (本 batch latent 触发次数)
            _draft_metrics['nld/thought_count'] = float(thought_count)
            
            # Latent 监督 loss (核心)
            if hasattr(self, '_accumulated_exit_token_loss') and self._accumulated_exit_token_loss is not None:
                _draft_metrics['nld/exit_token_loss'] = self._accumulated_exit_token_loss.item() if isinstance(self._accumulated_exit_token_loss, torch.Tensor) else self._accumulated_exit_token_loss
            if hasattr(self, '_accumulated_key_token_loss') and self._accumulated_key_token_loss is not None:
                _draft_metrics['nld/sw_srs_loss'] = self._accumulated_key_token_loss.item() if isinstance(self._accumulated_key_token_loss, torch.Tensor) else self._accumulated_key_token_loss
            # ====== 3 层 anti-collapse 新增 loss 指标透传 ======
            if hasattr(self, '_accumulated_exit_margin_loss') and self._accumulated_exit_margin_loss is not None:
                _draft_metrics['nld/exit_margin_loss'] = self._accumulated_exit_margin_loss.item() if isinstance(self._accumulated_exit_margin_loss, torch.Tensor) else self._accumulated_exit_margin_loss
            if hasattr(self, '_accumulated_swsrs_anti_collapse_loss') and self._accumulated_swsrs_anti_collapse_loss is not None:
                _draft_metrics['nld/swsrs_anti_collapse_loss'] = self._accumulated_swsrs_anti_collapse_loss.item() if isinstance(self._accumulated_swsrs_anti_collapse_loss, torch.Tensor) else self._accumulated_swsrs_anti_collapse_loss
            if hasattr(self, '_accumulated_diversity_loss') and self._accumulated_diversity_loss is not None:
                _draft_metrics['nld/diversity_loss'] = self._accumulated_diversity_loss.item() if isinstance(self._accumulated_diversity_loss, torch.Tensor) else self._accumulated_diversity_loss
            # path B2 视觉侧 vision loss + 诊断
            if hasattr(self, '_accumulated_vision_loss') and self._accumulated_vision_loss is not None:
                _draft_metrics['nld/vision_loss'] = (
                    self._accumulated_vision_loss.item()
                    if isinstance(self._accumulated_vision_loss, torch.Tensor)
                    else self._accumulated_vision_loss
                )
            if getattr(self, '_accumulated_vision_stats', None):
                # 聚合诊断: 对多个 boundary 的同名项取均; vis_count / vis_loss 取和
                _vis_acc: Dict[str, list] = {}
                _vis_sum_keys = ('vis_loss', 'vis_count')
                for _vs in self._accumulated_vision_stats:
                    if not isinstance(_vs, dict):
                        continue
                    for _k, _v in _vs.items():
                        if not isinstance(_v, (int, float)):
                            continue
                        _vis_acc.setdefault(_k, []).append(float(_v))
                for _k, _vals in _vis_acc.items():
                    if not _vals:
                        continue
                    _val = sum(_vals) if _k in _vis_sum_keys else (sum(_vals) / len(_vals))
                    _draft_metrics[f'nld/{_k}'] = float(_val)

            # ================ Anti-Collapse 监控 (核心信号) ================
            # (1) Hidden 几何: 13 项
            #     原有 4 项: h_norm_mean / h_batch_cos_mean / h_effective_rank / h_first_last_cos
            #     adj_cos 3 项: h_adj_cos_mean / h_adj_cos_min / h_adj_cos_max (相邻步 cos 序列)
            #     SAM 3 项: h_stage_diag_score / h_stage_monotonic / h_stage_shift_kl
            #              (Stage Alignment Matrix, 真正反映 hidden 推理语义)
            #     Saturation 3 项: h_sat_step1 / h_sat_step_last / h_sat_early_exit_ratio
            #                     (推理期 early-exit 病态预警)
            if self._accumulated_hidden_stats:
                for _k in [
                    'h_norm_mean', 'h_batch_cos_mean', 'h_effective_rank',
                    'h_first_last_cos',
                    'h_adj_cos_mean', 'h_adj_cos_min', 'h_adj_cos_max',
                    'h_stage_diag_score', 'h_stage_monotonic', 'h_stage_shift_kl',
                    'h_sat_step1', 'h_sat_step_last', 'h_sat_early_exit_ratio',
                    # TGVR 档 A 诊断 (Text-Guided Visual Recall, 无 loss)
                    'tgvr_cos_h_v_mean', 'tgvr_cos_h_v_first', 'tgvr_cos_h_v_last',
                    'tgvr_v_pos_norm_mean', 'tgvr_attn_entropy_mean',
                    'tgvr_topk_recall_mean',
                ]:
                    _vals = [d.get(_k) for d in self._accumulated_hidden_stats if _k in d]
                    if _vals:
                        _draft_metrics[f'collapse/{_k}'] = float(sum(_vals) / len(_vals))
            # (2) SW-SRS 内部: 4 项
            #     原有 2 项: q_entropy_mean / topk_hit_ratio
            #     新增 2 项: q_entropy_first_stage / q_entropy_last_stage
            #              (用于观察 SW-SRS "先宽后窄" 是否生效, late << early 是好信号)
            if self._accumulated_sw_srs_stats:
                for _k in [
                    'q_entropy_mean', 'q_entropy_first_stage', 'q_entropy_last_stage',
                    'topk_hit_ratio',
                ]:
                    _vals = [d.get(_k) for d in self._accumulated_sw_srs_stats if _k in d]
                    if _vals:
                        _draft_metrics[f'collapse/sw_srs_{_k}'] = float(sum(_vals) / len(_vals))
            # (3) num_thought_steps 分布 (诊断"固定第 N 步 exit")
            #     训练期由数据 num_stages 决定, 正常应分布在 2-9; 若实测均值远低于数据均值,
            #     说明模型即使被强制 force, internal 结构已经"想"早 exit (推理期会 early-exit)
            # 从 hidden_stats 中读 num_thought_steps 字段 (latent_thinker.py 注入)
            _ts = [
                int(d['num_thought_steps'])
                for d in self._accumulated_hidden_stats
                if 'num_thought_steps' in d and d['num_thought_steps'] > 0
            ]
            if _ts:
                _draft_metrics['nld/num_thought_steps_mean'] = float(sum(_ts) / len(_ts))
                _draft_metrics['nld/num_thought_steps_min'] = float(min(_ts))
                _draft_metrics['nld/num_thought_steps_max'] = float(max(_ts))
            # ================ Anti-Collapse 监控结束 ================

            # Per-token loss 分解 (think / answer 分段 CE + top-1 acc)
            if loss_weight_mask is not None and labels is not None:
                _shift_labels = labels[:, 1:].contiguous()
                _shift_weights = loss_weight_mask[:, 1:].contiguous().float()
                _valid = (_shift_labels != -100)
                _think_mask = _valid & (_shift_weights <= 1.01)
                _answer_mask = _valid & (_shift_weights > 1.01)

                _loss_fct_none = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                _per_token_ce = _loss_fct_none(
                    logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                    _shift_labels.view(-1)
                ).view(_shift_labels.shape)

                if _think_mask.any():
                    _draft_metrics['loss/think_ce'] = (_per_token_ce * _think_mask.float()).sum().item() / _think_mask.float().sum().item()
                if _answer_mask.any():
                    _draft_metrics['loss/answer_ce'] = (_per_token_ce * _answer_mask.float()).sum().item() / _answer_mask.float().sum().item()

                # Top-1 准确率 (仅 answer 段, overall 版本信息冗余已删除)
                if _answer_mask.any():
                    _pred_top1 = logits[:, :-1, :].contiguous().argmax(dim=-1)
                    _top1_correct = (_pred_top1 == _shift_labels) & _answer_mask
                    _draft_metrics['acc/top1_answer'] = _top1_correct.float().sum().item() / _answer_mask.float().sum().item()

        # ====== 耗时汇总 ======
        _fwd_end = _time.time()
        _fwd_total = _fwd_end - _fwd_start
        _should_profile = (_is_rank0 and self._fwd_count % 50 == 0)
        if _should_profile or _fwd_total > 30:
            _t_vision = _t_vision_end - _t_vision_start
            _t_base = _t_base_fwd_end - _t_base_fwd_start
            print(f"[rank {_rank}] fwd#{self._fwd_count} 耗时: "
                  f"total={_fwd_total:.1f}s (vision={_t_vision:.1f}s, "
                  f"chunkwise={_t_base:.1f}s) "
                  f"chunks={len(chunk_ranges)} thoughts={thought_count}", flush=True)

        return {
            'loss': total_loss,
            'main_loss': main_loss,
            'div_loss': torch.tensor(0.0, device=device),  # NLD 无 diversity loss
            'commit_loss': torch.tensor(0.0, device=device),
            'grounding_loss': torch.tensor(0.0, device=device),
            'logits': None,
            'draft_metrics': _draft_metrics,
        }

    # ================================================================
    # 推理流程
    # ================================================================

    @torch.no_grad()
    def _run_directly_answer_fallback(
        self,
        original_input_ids: torch.Tensor,
        original_attention_mask: torch.Tensor,
        original_pixel_values: torch.Tensor,
        original_image_grid_thw: torch.Tensor,
        cached_visual_embeds: torch.Tensor,
        gen_deepstack_features,
        fallback_input_ids: Optional[torch.Tensor] = None,
        fallback_attention_mask: Optional[torch.Tensor] = None,
        fallback_assistant_prefix_len: int = 0,
        fallback_max_new_tokens: int = 32,
    ) -> Tuple[torch.Tensor, int]:
        """Directly-answer fallback 推理 (单 batch).

        设计:
          1) 优先使用调用方提供的 fallback_input_ids (已 tokenize 好的
             directly-answer prompt + 'Final Answer: ' assistant prefix).
          2) 否则在原 input_ids 末尾 append 一段 hint + prefix token
             (调用方零改动的兜底路径; 完全 token-level, 不走 chat template).
          3) 全程 greedy (do_sample=False), 不触发 NativeLatentThinker
             (检测到 <|latent|> 也只当普通 token 继续解码), 默认 32 token 即停.
          4) 视觉编码完全复用第一次 generate 已经算好的
             cached_visual_embeds / gen_deepstack_features, 不再走一次 ViT.

        Returns:
            (fb_generated_ids: [B, prompt_len_eff + new_len], num_new_tokens: int)
            fb_generated_ids 的 prompt_len_eff = fb_input_ids.shape[1] -
                fallback_assistant_prefix_len, 即把 'Final Answer: ' 这段
                assistant prefix 也视为"新生成"的一部分, 让下游 decode 时
                'Final Answer: <answer>' 字面量自然出现.
        """
        from transformers.cache_utils import DynamicCache

        device = original_input_ids.device
        inner_model = self._inner_model
        text_model = inner_model.language_model
        lm_head = self._lm_head

        # ---- 1) 选择 fallback input_ids ----
        if fallback_input_ids is not None:
            fb_input_ids = fallback_input_ids.to(device)
            if fallback_attention_mask is not None:
                fb_attn_mask = fallback_attention_mask.to(device)
            else:
                fb_attn_mask = torch.ones_like(fb_input_ids)
            prefix_len_for_decode = int(fallback_assistant_prefix_len)
            # 调用方传入的 fb_input_ids: 不假定其与 original_input_ids 头部对齐,
            # 且其形态由调用方完全负责 (例如已在 chat-template 层把 hint 写进 user
            # 文本里). 这里不做任何中间段裁剪.
            hint_start_in_fb = 0
            hint_end_in_fb = 0
        else:
            # 兜底: 把 hint + 'Final Answer: ' 直接 token append 到原 input_ids.
            # hint 让模型遵循 query 自身的 answer-format 要求, prefix 强制注入 'Final Answer: '
            # 让模型只能从冒号后续写一个 token 级别的答案.
            if not (hasattr(self, 'processor') and self.processor is not None):
                # 没 processor 没法 token append, 直接复用原 input_ids 重跑
                # 大概率仍会病态, 但至少不崩.
                fb_input_ids = original_input_ids
                fb_attn_mask = original_attention_mask
                prefix_len_for_decode = 0
                # 该分支没有 hint append, 不需要裁剪.
                hint_start_in_fb = 0
                hint_end_in_fb = 0
            else:
                tok = self.processor.tokenizer
                hint_text = (
                    "\n\nIMPORTANT: Stop reasoning and answer the question "
                    "above NOW. Do NOT explain, do NOT describe what you "
                    "see, do NOT plan, do NOT use any chain-of-thought or "
                    "latent thoughts. Follow the question's own answer-format "
                    "requirement (e.g. a letter for multiple-choice, a number "
                    "for counting, a short phrase for open-ended) exactly. "
                    "Output ONLY the final answer and nothing else."
                )
                prefix_text = "\n\nFinal Answer: "
                hint_ids = tok(hint_text, add_special_tokens=False)['input_ids']
                prefix_ids = tok(prefix_text, add_special_tokens=False)['input_ids']
                append_ids = torch.tensor(
                    [hint_ids + prefix_ids], device=device,
                    dtype=original_input_ids.dtype,
                )
                fb_input_ids = torch.cat([original_input_ids, append_ids], dim=1)
                fb_attn_mask = torch.cat(
                    [original_attention_mask,
                     torch.ones_like(append_ids, dtype=original_attention_mask.dtype)],
                    dim=1,
                )
                prefix_len_for_decode = len(prefix_ids)
                # 记录 hint 段在 fb_input_ids 中的 [start, end) 区间, 便于返回前
                # 把 hint 从 generated_ids 中精确裁掉, 保证调用方按
                # original_input_ids.shape[1] 切片时不会解码出 hint 文本.
                hint_token_len = len(hint_ids)
                hint_start_in_fb = original_input_ids.shape[1]
                hint_end_in_fb = hint_start_in_fb + hint_token_len

        B = fb_input_ids.shape[0]
        # ---- 2) Prefill (复用 cached_visual_embeds, 不再过 ViT) ----
        prompt_embeds = inner_model.get_input_embeddings()(fb_input_ids)
        image_token_id = inner_model.config.image_token_id
        image_mask = (fb_input_ids == image_token_id)
        num_image_tokens = image_mask.sum().item()
        if num_image_tokens > 0 and cached_visual_embeds is not None:
            n_avail = cached_visual_embeds.shape[0]
            if num_image_tokens <= n_avail:
                img_emb = cached_visual_embeds[:num_image_tokens].to(
                    prompt_embeds.device, prompt_embeds.dtype,
                )
                image_mask_3d = image_mask.unsqueeze(-1).expand_as(prompt_embeds)
                prompt_embeds = prompt_embeds.masked_scatter(image_mask_3d, img_emb)
            # 若 image token 数对不上 (理论上不应该, 因为 fallback prompt 仍
            # 用同一张图像), 跳过 image embed 注入, 让模型按文本继续.

        seg_token_mask = torch.ones(
            B, fb_input_ids.shape[1], device=device, dtype=torch.long,
        )
        inner_model.rope_deltas = None
        position_ids, rope_deltas = inner_model.get_rope_index(
            fb_input_ids,
            image_grid_thw=original_image_grid_thw if original_pixel_values is not None else None,
            video_grid_thw=None,
            attention_mask=seg_token_mask,
        )
        inner_model.rope_deltas = rope_deltas

        prompt_len = fb_input_ids.shape[1]
        if position_ids is not None and position_ids.dim() == 3:
            text_pos = torch.arange(prompt_len, device=device).unsqueeze(0).expand(B, -1)
            position_ids = torch.cat([text_pos.unsqueeze(0), position_ids], dim=0)

        visual_pos_masks = (
            (fb_input_ids == image_token_id)
            if original_pixel_values is not None else None
        )
        cache_position = torch.arange(prompt_len, device=device)

        cache = DynamicCache()
        prefill_out = text_model(
            inputs_embeds=prompt_embeds,
            attention_mask=fb_attn_mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=gen_deepstack_features,
        )

        past_key_values = prefill_out.past_key_values
        mrope_last_pos = position_ids[1:, :, -1:] if position_ids is not None else None

        prefill_hidden = prefill_out.last_hidden_state
        next_token_logits = lm_head(prefill_hidden[:, -1, :])

        generated_ids = fb_input_ids.clone()
        eos_token_id = (
            self.processor.tokenizer.eos_token_id
            if hasattr(self, 'processor') and self.processor is not None else None
        )
        current_pos = prompt_len
        num_new = 0

        # ---- 3) Greedy 解码 (无 latent 触发, do_sample=False) ----
        for _ in range(max(1, int(fallback_max_new_tokens))):
            next_token = next_token_logits.argmax(dim=-1, keepdim=True).squeeze(-1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
            num_new += 1

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            new_mrope_pos = mrope_last_pos + 1 if mrope_last_pos is not None else None
            text_pos_gen = torch.full((B, 1), current_pos, device=device, dtype=torch.long)
            if new_mrope_pos is not None:
                position_ids_4d = torch.cat([text_pos_gen.unsqueeze(0), new_mrope_pos], dim=0)
            else:
                position_ids_4d = text_pos_gen

            token_embeds = inner_model.get_input_embeddings()(next_token.unsqueeze(-1))
            current_total_len = current_pos + 1
            new_attention_mask = torch.ones(
                B, current_total_len, device=device, dtype=fb_attn_mask.dtype,
            )
            cache_position_gen = torch.tensor([current_pos], device=device, dtype=torch.long)

            text_outputs = text_model(
                inputs_embeds=token_embeds,
                attention_mask=new_attention_mask,
                position_ids=position_ids_4d,
                past_key_values=past_key_values,
                cache_position=cache_position_gen,
                use_cache=True,
            )
            past_key_values = text_outputs.past_key_values
            new_hidden = text_outputs.last_hidden_state
            next_token_logits = lm_head(new_hidden[:, -1, :])
            if mrope_last_pos is not None:
                mrope_last_pos = new_mrope_pos
            current_pos += 1

        # ---- 4) 把 'Final Answer: ' assistant prefix 也视为新生成的一部分,
        # 这样下游 decode(generated_ids[:, prompt_len_eff:]) 时
        # 'Final Answer: <answer>' 字面量自然出现, 无需调用方再做特殊处理.
        # 关键修复 (方案B): 调用方 (eval / inference 脚本) 普遍按
        # generated_ids[:, original_input_ids.shape[1]:] 切片解码, 因此返回
        # 的 generated_ids 必须满足: [original_input_ids ; prefix_ids ; new_ids].
        # 而内部用的 fb_input_ids 形态是 [original_input_ids ; hint_ids ; prefix_ids],
        # 于是返回前把中间的 hint_ids 段裁掉, 让 generated_ids 与原 prompt_len
        # 严格对齐. 调用方按 input_ids.shape[1] 切就拿到
        # 'Final Answer: <answer>' 字面量, 不会再解码出 hint 文本.
        if hint_end_in_fb > hint_start_in_fb:
            # 切片: [:, :hint_start] + [:, hint_end:] (沿 dim=1 拼接)
            generated_ids = torch.cat(
                [generated_ids[:, :hint_start_in_fb],
                 generated_ids[:, hint_end_in_fb:]],
                dim=1,
            )
        # num_new 这里只算 greedy 解的部分, 加上 prefix_len_for_decode 后正好是
        # "调用方按 original_input_ids.shape[1] 切片后能拿到的新生成 token 数".
        return generated_ids, num_new + prefix_len_for_decode

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        return_diagnostics: bool = False,
        # ---- Two-level fallback (推理时默认机制, 默认全 ON) ----
        # Level-0 (默认自由生成):
        #   完整 latent + CoT, 不做任何 logit 干预. 每一题首次都走这一路.
        # Level-1 (post-hoc retry, 抑制 latent):
        #   首次生成事后命中病态 (hit_max / no_final_answer / ngram_repeat) 时,
        #   彻底丢弃首次输出, 沿用同一份 prompt + 视觉缓存 重 prefill + 重新自回归
        #   生成一次. 此次解码循环全程把 <|latent|> 的 logit 置为 -inf, 且禁止
        #   NativeLatentThinker 被调用 -> 模型只能走纯文本 CoT 直到 EOS / max_new_tokens.
        # Level-2 (post-hoc directly-answer rewrite):
        #   L1 重新生成后再次事后检测; 仍命中病态则丢弃 L1 输出, 走 fallback prompt
        #   + 'Final Answer: ' assistant prefix 强制 greedy 极短解码, 写回 generated_ids.
        # 全关传 enable_fallback=False; 仅关 L1 传 level1_fallback_enabled=False
        # (此时跳过 L1, 命中病态直接走 L2).
        enable_fallback: bool = True,
        # ---- Level-1 参数 ----
        level1_fallback_enabled: bool = True,
        # 兼容保留: 旧版 in-loop 切换的阈值参数, 新版 post-hoc retry 不再使用,
        # 仅为调用方零改动而保留签名.
        level1_max_latent_events: int = 8,
        level1_min_text_tokens_progress: int = 0,
        level1_force_eos_after_n_blocked: Optional[int] = None,
        # ---- Level-2 参数 (即原 directly-answer fallback) ----
        fallback_input_ids: Optional[torch.Tensor] = None,
        fallback_attention_mask: Optional[torch.Tensor] = None,
        fallback_assistant_prefix_len: int = 0,
        fallback_max_new_tokens: int = 32,
        fallback_on_hit_max: bool = True,
        fallback_on_no_final_answer: bool = True,
        fallback_on_ngram_repeat: bool = True,
        fallback_final_answer_marker: str = "Final Answer:",
        fallback_ngram_size: int = 8,
        fallback_ngram_window: int = 256,
        fallback_ngram_repeat_thresh: int = 4,
        # ---- 外部强制拑制 latent (供上层封装当 "L1" 使用) ----
        # 为 True 时, 首次推理即走 suppress_latent=True 路径 (纯文本 CoT,
        # 无 latent), 等同于外层调用方以 "L1 重生成" 形态进入 generate.
        suppress_latent: bool = False,
    ) -> torch.Tensor:
        """
        推理时的生成
        
        核心设计 (统一序列, 训推一致):
        - 正常自回归生成 CoT
        - 检测到 step boundary 时触发 NativeLatentThinker
        - 方案 B (训推一致): thinker 使用局部 cache，thought_output 作为 prefix
          再过一遍 text_model() 写入主 cache (与训练时行为完全一致)
        - thought tokens 的 KV 永久保留在历史 cache 中 (不剥离)
        - 后续所有 token 的 attention 自然看到 thought 信息
        - 隐式推理 + 自然语言 CoT 构成完整的推理序列

        Args:
            return_diagnostics: 若 True, 返回 (generated_ids, diagnostics_list)
                                diagnostics_list 是每次 latent 触发的诊断 dict 列表
                                包含: exit_reason, num_thought_steps, hidden_stats, sw_srs_stats
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # 1. 视觉 encoder (无论跑几次主推理 (L0/L1) 都只过一次 ViT, 之后整轮复用)
        inner_model = self._inner_model
        _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
        vision_output = inner_model.visual(_pixel_values, grid_thw=image_grid_thw)
        merged_hidden_states, gen_deepstack_features = _extract_visual_output(vision_output)
        split_sizes = (
            image_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size ** 2
        ).tolist()
        image_embeds_list = list(torch.split(merged_hidden_states, split_sizes))
        cached_visual_embeds = torch.cat(image_embeds_list, dim=0)

        text_model = inner_model.language_model
        lm_head = self._lm_head

        # ============================================================
        # 内嵌函数 _run_one_pass: 完成 "prefill + 自回归循环" 全过程.
        # 通过参数 suppress_latent 控制本次推理是否屏蔽 <|latent|> token:
        #   - suppress_latent=False (L0 默认): 完整 latent + CoT, 自由生成.
        #   - suppress_latent=True  (L1 retry): 全程 logit 屏蔽 + 禁用 thinker,
        #     模型只能走纯文本 CoT 直到 EOS / max_new_tokens.
        # 视觉特征 (cached_visual_embeds, gen_deepstack_features) 由外层算一次后
        # 通过闭包共享, 不重复过 ViT.
        # 返回: (generated_ids, latent_diagnostics_list, prompt_len)
        # ============================================================
        def _run_one_pass(suppress_latent: bool):
            latent_diagnostics: list = []
            prompt_embeds = inner_model.get_input_embeddings()(input_ids)
            image_token_id = inner_model.config.image_token_id
            image_mask = (input_ids == image_token_id)
            num_image_tokens = image_mask.sum().item()
            if num_image_tokens > 0:
                img_emb = cached_visual_embeds[:num_image_tokens].to(prompt_embeds.device, prompt_embeds.dtype)
                image_mask_3d = image_mask.unsqueeze(-1).expand_as(prompt_embeds)
                prompt_embeds = prompt_embeds.masked_scatter(image_mask_3d, img_emb)

            seg_token_mask = torch.ones(B, input_ids.shape[1], device=device, dtype=torch.long)
            inner_model.rope_deltas = None
            position_ids, rope_deltas = inner_model.get_rope_index(
                input_ids, image_grid_thw=image_grid_thw, video_grid_thw=None,
                attention_mask=seg_token_mask,
            )
            inner_model.rope_deltas = rope_deltas

            prompt_len = input_ids.shape[1]
            if position_ids is not None and position_ids.dim() == 3:
                text_pos = torch.arange(prompt_len, device=device).unsqueeze(0).expand(B, -1)
                position_ids = torch.cat([text_pos.unsqueeze(0), position_ids], dim=0)

            visual_pos_masks = (input_ids == image_token_id) if pixel_values is not None else None
            cache_position = torch.arange(prompt_len, device=device)

            cache = DynamicCache()
            prefill_out = text_model(
                inputs_embeds=prompt_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=gen_deepstack_features,
            )
            past_key_values = prefill_out.past_key_values
            mrope_last_pos = position_ids[1:, :, -1:] if position_ids is not None else None

            prefill_hidden = prefill_out.last_hidden_state
            next_token_logits = lm_head(prefill_hidden[:, -1, :])

            # 3. 自回归生成
            generated_ids = input_ids.clone()
            eos_token_id = self.processor.tokenizer.eos_token_id if hasattr(self, 'processor') else None

            # <|latent|> 触发计数 (诊断用)
            latent_trigger_count = [0 for _ in range(B)]

            # 保留最近的 hidden state (用于 NativeLatentThinker)
            recent_last_hidden = prefill_hidden[:, -1:, :]  # [B, 1, H] 最后一个 token

            current_pos = prompt_len

            for gen_step in range(max_new_tokens):
                # ---- L1 retry 模式: 全程屏蔽 <|latent|> 的 logit ----
                # suppress_latent=True 由 generate() 在第一次推理触发病态后, 重新
                # 调用本闭包时显式传入. 模型采样时永远看不到 <|latent|>, 完全退化
                # 为纯文本 CoT.
                if (suppress_latent
                        and hasattr(self, 'latent_token_id')
                        and self.latent_token_id is not None):
                    next_token_logits[..., self.latent_token_id] = float('-inf')

                # 采样
                if do_sample and temperature > 0:
                    scaled_logits = next_token_logits / temperature
                    probs = F.softmax(scaled_logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_indices.gather(-1, next_token)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)

                next_token = next_token.squeeze(-1)
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)

                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

                # ---- 标准自回归 forward ----
                new_mrope_pos = mrope_last_pos + 1
                text_pos_gen = torch.full((B, 1), current_pos, device=device, dtype=torch.long)
                position_ids_4d = torch.cat([text_pos_gen.unsqueeze(0), new_mrope_pos], dim=0)

                token_embeds = inner_model.get_input_embeddings()(next_token.unsqueeze(-1))

                current_total_len = current_pos + 1
                new_attention_mask = torch.ones(B, current_total_len, device=device, dtype=attention_mask.dtype)
                cache_position_gen = torch.tensor([current_pos], device=device, dtype=torch.long)

                text_outputs = text_model(
                    inputs_embeds=token_embeds,
                    attention_mask=new_attention_mask,
                    position_ids=position_ids_4d,
                    past_key_values=past_key_values,
                    cache_position=cache_position_gen,
                    use_cache=True,
                )
                past_key_values = text_outputs.past_key_values
                new_hidden = text_outputs.last_hidden_state  # [B, 1, H]
                next_token_logits = lm_head(new_hidden[:, -1, :])

                recent_last_hidden = new_hidden  # [B, 1, H]

                mrope_last_pos = new_mrope_pos
                current_pos += 1

                # ---- Step boundary 检测 (通过 <|latent|> token id) ----
                # L1 retry (suppress_latent=True) 时, 即使采到了 <|latent|>
                # (理论上不可能, 因为已经 -inf 屏蔽), 也强制不触发 thinker.
                trigger_mask = torch.zeros(B, dtype=torch.bool, device=device)
                if not suppress_latent:
                    for b in range(B):
                        token_id = next_token[b].item()
                        # 检测 <|latent|> token: 直接比较 token id (精确、高效)
                        if hasattr(self, 'latent_token_id') and self.latent_token_id is not None:
                            if token_id == self.latent_token_id:
                                trigger_mask[b] = True

                if trigger_mask.any():
                    # 触发 NativeLatentThinker (Coconut 等价的隐空间推理)
                    last_hidden = recent_last_hidden  # [B, 1, H]

                    base_thought_pos = current_pos
                    thought_pos = torch.arange(current_pos, current_pos + 1, device=device)
                    thought_mrope_pos = thought_pos.unsqueeze(0).unsqueeze(0).expand(3, B, -1)

                    thought_text_pos = torch.full((B, 1), base_thought_pos, device=device, dtype=torch.long)
                    thought_position_ids = torch.cat([thought_text_pos.unsqueeze(0), thought_mrope_pos], dim=0)

                    think_attn_mask = torch.ones(B, base_thought_pos + 1, device=device, dtype=torch.long)

                    thinker_result = self.latent_thinker(
                        last_hidden=last_hidden,
                        text_model=text_model,
                        past_key_values=past_key_values,
                        attention_mask=think_attn_mask,
                        cache_position=thought_pos,
                        position_ids=thought_position_ids,
                        rotary_emb_fn=text_model.rotary_emb,
                        mrope_position_ids=thought_mrope_pos,
                        base_thought_pos=base_thought_pos,
                        B=B,
                        lm_head=lm_head,
                    )
                    thought_output = thinker_result['thought_output']  # [B, num_steps, H]
                    num_thought_steps = thinker_result.get('num_thought_steps', 1)

                    if return_diagnostics:
                        _exit_stats = thinker_result.get('exit_stats', {}) or {}
                        _hidden_stats = thinker_result.get('hidden_stats', {}) or {}
                        _sw_srs_stats = thinker_result.get('sw_srs_stats', {}) or {}
                        _hs_clean = {}
                        for k, v in _hidden_stats.items():
                            if isinstance(v, torch.Tensor):
                                try:
                                    _hs_clean[k] = float(v.float().mean().item())
                                except Exception:
                                    pass
                            elif isinstance(v, (int, float)):
                                _hs_clean[k] = float(v)
                        _sw_clean = {}
                        for k, v in _sw_srs_stats.items():
                            if isinstance(v, torch.Tensor):
                                try:
                                    _sw_clean[k] = float(v.float().mean().item())
                                except Exception:
                                    pass
                            elif isinstance(v, (int, float)):
                                _sw_clean[k] = float(v)
                        latent_diagnostics.append({
                            'gen_step': int(gen_step),
                            'trigger_idx': int(latent_trigger_count[0]),
                            'num_thought_steps': int(num_thought_steps),
                            'exit_reason': _exit_stats.get('exit_reason', 'unknown'),
                            'hidden_stats': _hs_clean,
                            'sw_srs_stats': _sw_clean,
                        })

                    thought_prefix_cache_pos = torch.arange(
                        current_pos, current_pos + num_thought_steps, device=device
                    )
                    thought_prefix_mrope = thought_prefix_cache_pos.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                    thought_prefix_text_pos = torch.arange(
                        current_pos, current_pos + num_thought_steps, device=device
                    ).unsqueeze(0).expand(B, -1)
                    thought_prefix_pos_ids = torch.cat([
                        thought_prefix_text_pos.unsqueeze(0),
                        thought_prefix_mrope,
                    ], dim=0)  # [4, B, num_steps]

                    thought_prefix_attn_mask = torch.ones(
                        B, current_pos + num_thought_steps, device=device, dtype=attention_mask.dtype
                    )

                    prefix_outputs = text_model(
                        inputs_embeds=thought_output,
                        attention_mask=thought_prefix_attn_mask,
                        position_ids=thought_prefix_pos_ids,
                        past_key_values=past_key_values,
                        cache_position=thought_prefix_cache_pos,
                        use_cache=True,
                    )
                    past_key_values = prefix_outputs.past_key_values

                    current_pos += num_thought_steps
                    mrope_last_pos = mrope_last_pos + num_thought_steps

                    for b in range(B):
                        if trigger_mask[b]:
                            latent_trigger_count[b] += 1

            return generated_ids, latent_diagnostics, prompt_len

        # ============================================================
        # 三级 fallback 调度: L0 自由生成 -> [病态? L1 重生成抑制 latent ->
        # [仍病态? L2 directly-answer]]
        # 如果调用方显式传入 suppress_latent=True (例如上层 generate_with_fallback
        # 走到外部 L1 阶段), 这里首次推理即走 suppress_latent=True.
        # ============================================================
        # 先做 L0 (或被外层强制为 L1 形态): 默认自由生成 (latent + CoT)
        generated_ids, latent_diagnostics, prompt_len = _run_one_pass(
            suppress_latent=bool(suppress_latent)
        )

        # ============================================================
        # Directly-answer fallback (推理时默认机制)
        # ============================================================
        # 事后扫描: 检测 hit_max / no_final_answer / ngram_repeat 三类病态模式.
        # 命中即用 fallback_input_ids (调用方提供) 或就地构造的 directly-answer
        # 输入做一次 32-token greedy 解码 (无 latent, do_sample=False),
        # 把结果替换回 generated_ids 返回.
        fb_meta: Dict = {
            'fallback_triggered': False,
            'fallback_reason': None,
            'fallback_detail': '',
            'fallback_num_new_tokens': 0,
            'first_num_new_tokens': int(generated_ids.shape[1] - prompt_len),
            # 新增: 标识本次最终走到了哪一级 fallback (None / 'L1' / 'L2'),
            # 以及 L1 retry 完成后的产物长度 / 二次病态检测结果.
            'fallback_level': None,
            'l1_retry_triggered': False,
            'l1_num_new_tokens': 0,
            'l1_residual_reason': None,
            'l1_residual_detail': '',
        }

        if enable_fallback:
            # ---- 局部函数: 病态检测 (返回 (reason, detail)) ----
            def _detect_pathology(_generated_ids):
                _new_token_ids_list = _generated_ids[0, prompt_len:].tolist()
                _num_new = len(_new_token_ids_list)
                _reason = None
                _detail = ""
                # (a) hit_max
                if fallback_on_hit_max and _num_new >= max_new_tokens - 2:
                    return 'hit_max', f"num_new_tokens={_num_new}/{max_new_tokens}"
                # (b) no_final_answer
                if fallback_on_no_final_answer:
                    if hasattr(self, 'processor') and self.processor is not None:
                        try:
                            _clean_text = self.processor.tokenizer.decode(
                                _new_token_ids_list, skip_special_tokens=True
                            )
                        except Exception:
                            _clean_text = ""
                        if fallback_final_answer_marker not in _clean_text:
                            return ('no_final_answer',
                                    f"missing '{fallback_final_answer_marker}' in decoded output")
                # (c) ngram_repeat
                if fallback_on_ngram_repeat:
                    _n = fallback_ngram_size
                    _k = fallback_ngram_repeat_thresh
                    _W = fallback_ngram_window
                    if _n > 0 and _k > 0 and _num_new >= _n + _k - 1:
                        _window = _new_token_ids_list[-_W:] if _W > 0 else _new_token_ids_list
                        if len(_window) >= _n:
                            _counts: Dict[Tuple[int, ...], int] = {}
                            for _i in range(0, len(_window) - _n + 1):
                                _key = tuple(_window[_i:_i + _n])
                                _counts[_key] = _counts.get(_key, 0) + 1
                            if _counts:
                                _top_count = max(_counts.values())
                                if _top_count >= _k:
                                    return ('ngram_repeat',
                                            f"{_n}-gram repeated {_top_count}x in last {len(_window)} tokens")
                return _reason, _detail

            # ---- 1) L0 事后病态检测 ----
            fb_reason, fb_detail = _detect_pathology(generated_ids)

            # ---- 2) 命中则先走 L1: 丢弃 L0 输出, 重新调 _run_one_pass 但屏蔽 latent ----
            if fb_reason is not None and level1_fallback_enabled:
                try:
                    l1_generated_ids, l1_diagnostics, _ = _run_one_pass(suppress_latent=True)
                    fb_meta['l1_retry_triggered'] = True
                    fb_meta['l1_num_new_tokens'] = int(l1_generated_ids.shape[1] - prompt_len)
                    # L1 重生成完成 -> 用 L1 结果替换 L0 结果
                    generated_ids = l1_generated_ids
                    # 把 L1 阶段的 latent 诊断也并入 (理论上为空, 因为 latent 已被屏蔽)
                    if return_diagnostics and l1_diagnostics:
                        latent_diagnostics.extend(l1_diagnostics)
                    # ---- 二次病态检测 ----
                    l1_reason, l1_detail = _detect_pathology(generated_ids)
                    fb_meta['l1_residual_reason'] = l1_reason
                    fb_meta['l1_residual_detail'] = l1_detail
                    if l1_reason is None:
                        # L1 救回: 标记 fallback 触发, level=L1, 主 reason 改为
                        # "{原因}_resolved_by_l1"
                        fb_meta['fallback_triggered'] = True
                        fb_meta['fallback_reason'] = f"{fb_reason}_resolved_by_l1"
                        fb_meta['fallback_detail'] = (
                            f"L0 reason={fb_reason} ({fb_detail}); L1 ok, "
                            f"l1_num_new_tokens={fb_meta['l1_num_new_tokens']}"
                        )
                        fb_meta['fallback_level'] = 'L1'
                        fb_meta['fallback_num_new_tokens'] = fb_meta['l1_num_new_tokens']
                        # 标记 L1 已成功救回, 不再走 L2
                        fb_reason = None
                    else:
                        # L1 仍病态 -> 改写 fb_reason / fb_detail 让下游 L2 接手
                        fb_reason = l1_reason
                        fb_detail = (f"L0={fb_meta.get('fallback_reason') or fb_reason}, "
                                     f"L1 still pathology: {l1_reason} ({l1_detail})")
                except Exception as l1_err:
                    # L1 重生成本身出错 -> 不改 generated_ids, 让下游 L2 仍按
                    # 原 fb_reason 接手
                    fb_meta['l1_retry_triggered'] = False
                    fb_meta['l1_residual_reason'] = f'l1_failed:{type(l1_err).__name__}'
                    fb_meta['l1_residual_detail'] = str(l1_err)

            # ---- 3) L1 仍未救回 (或 L1 被禁用) -> 走 L2 directly-answer fallback ----
            if fb_reason is not None:
                try:
                    fb_generated_ids, fb_num_new = self._run_directly_answer_fallback(
                        original_input_ids=input_ids,
                        original_attention_mask=attention_mask,
                        original_pixel_values=pixel_values,
                        original_image_grid_thw=image_grid_thw,
                        cached_visual_embeds=cached_visual_embeds,
                        gen_deepstack_features=gen_deepstack_features,
                        fallback_input_ids=fallback_input_ids,
                        fallback_attention_mask=fallback_attention_mask,
                        fallback_assistant_prefix_len=fallback_assistant_prefix_len,
                        fallback_max_new_tokens=fallback_max_new_tokens,
                    )
                    fb_meta['fallback_triggered'] = True
                    fb_meta['fallback_reason'] = fb_reason
                    fb_meta['fallback_detail'] = fb_detail
                    fb_meta['fallback_num_new_tokens'] = int(fb_num_new)
                    fb_meta['fallback_level'] = 'L2'
                    generated_ids = fb_generated_ids
                except Exception as fb_err:
                    # L2 也出错: 保留 L1 (若有) 或 L0 的 generated_ids
                    fb_meta['fallback_triggered'] = False
                    fb_meta['fallback_reason'] = f'{fb_reason}_but_l2_failed'
                    fb_meta['fallback_detail'] = f"{fb_detail}; l2 err: {fb_err}"
                    fb_meta['fallback_level'] = None

        # 把 fb_meta 暴露为实例属性, 调用方可读取最近一次 generate 的 fallback 状态.
        # 这样保持 generate 返回值签名 100% 向后兼容 (老调用方零改动).
        self._last_fallback_meta = fb_meta

        if return_diagnostics:
            return generated_ids, latent_diagnostics
        return generated_ids

    # ================================================================
    # 保存/加载
    # ================================================================

    def save_pretrained(self, save_dir: str):
        """保存全量模型: NativeLatentThinker + VLM 全量参数.

        FSDP 分布式安全保存:
          - 用 FullStateDictConfig(offload_to_cpu=True, rank0_only=True) 把
            base_model 的分片参数 gather 到 rank0 的 CPU 内存, 然后 *仅 rank0*
            调用 base_model.save_pretrained(state_dict=...) 写盘. 其他 rank 拿到
            空 state_dict, 不进行任何文件 IO, 从根上避免 cephfs 上多 rank 并发
            写同一组 safetensors shard 导致 header / data 区间错位的问题
            (典型现象: SafetensorError: incomplete metadata, file not fully covered).
          - 同样地, latent_thinker (含 step_embedding 等子模块) 也仅由 rank0
            torch.save 一次, 避免 .pt 文件被并发覆盖.
          - 保存前后用 dist.barrier() 同步, 防止 rank0 写盘期间其他 rank 进入
            后续 forward / backward.

        非分布式环境下 (单卡 / 推理) 保持原有行为.
        """
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        is_main = (not is_dist) or (torch.distributed.get_rank() == 0)

        if is_main:
            os.makedirs(save_dir, exist_ok=True)
        if is_dist:
            torch.distributed.barrier()

        # ================================================================
        # 1. 保存 NativeLatentThinker (rank0 only)
        # ================================================================
        # 在 FSDP 下, latent_thinker 的 state_dict 也需要先 gather 到 rank0
        # 否则各 rank 拿到的只是 sharded 参数. 复用 FullStateDictConfig 上下文.
        thinker_state = None
        if is_dist:
            try:
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel as _FSDP,
                    FullStateDictConfig,
                    StateDictType,
                )
                _cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with _FSDP.state_dict_type(self, StateDictType.FULL_STATE_DICT, _cfg):
                    # latent_thinker 在 NLDModel 内部, 这里直接拿整个 NLDModel
                    # 的 full state_dict, 后面按 prefix 拆分 thinker / base_model.
                    # 这种方式能确保 rank0 拿到完整、未 flatten 的 2D 参数,
                    # 避免 step_embedding [32, 4096] 被 flatten 成 [131072].
                    full_state = self.state_dict()
                    if is_main:
                        thinker_state = {
                            k[len("latent_thinker."):]: v
                            for k, v in full_state.items()
                            if k.startswith("latent_thinker.")
                        }
                        base_state = {
                            k[len("base_model."):]: v
                            for k, v in full_state.items()
                            if k.startswith("base_model.")
                        }
                    else:
                        thinker_state = None
                        base_state = None
                    del full_state
            except Exception as _e:
                # 退化到非 FSDP 路径: 直接取本地 state_dict (可能是 sharded)
                if self._verbose and is_main:
                    print(f"[NLD] ⚠️ FSDP FULL_STATE_DICT 不可用 ({type(_e).__name__}: {_e}), "
                          f"回退到本地 state_dict (可能 sharded, 仅在非 FSDP 时安全)")
                thinker_state = self.latent_thinker.state_dict() if is_main else None
                base_state = None  # 让下文走 base_model.save_pretrained 自身逻辑
        else:
            thinker_state = self.latent_thinker.state_dict()
            base_state = None  # 让下文走 base_model.save_pretrained 自身逻辑

        if is_main and thinker_state is not None:
            thinker_path = os.path.join(save_dir, "nld_latent_thinker.pt")
            torch.save(thinker_state, thinker_path)
            if self._verbose:
                print(f"[NLD] NativeLatentThinker 已保存到 {thinker_path} "
                      f"({len(thinker_state)} 个参数)")

        if is_dist:
            torch.distributed.barrier()

        # ================================================================
        # 2. 保存 VLM 全量参数 (rank0 only)
        # ================================================================
        vlm_dir = os.path.join(save_dir, "vlm_full")
        if is_main:
            os.makedirs(vlm_dir, exist_ok=True)
            if base_state is not None:
                # 已经在上面 gather 到 rank0, 直接传给 save_pretrained 避免重复 gather
                self.base_model.save_pretrained(
                    vlm_dir,
                    state_dict=base_state,
                    safe_serialization=True,
                )
            else:
                # 非 FSDP 路径: 由 base_model.save_pretrained 自行处理
                self.base_model.save_pretrained(vlm_dir, safe_serialization=True)
            if self._verbose:
                print(f"[NLD] VLM 全量模型已保存到 {vlm_dir}")
        else:
            # 非 rank0 不写任何文件, 但仍要等 rank0 写完再返回, 防止上层直接进入
            # processor.save_pretrained 等其它 IO 操作
            pass

        if is_dist:
            torch.distributed.barrier()

    def load_pretrained(self, save_dir: str):
        """加载全量模型: NativeLatentThinker + VLM 全量参数.

        支持三种 ckpt 布局, 按以下优先级:
          A. save_dir/vlm_full/*.safetensors  +  save_dir/nld_latent_thinker.pt
             (NLDModel.save_pretrained 的产物; 旧路径)
          B. save_dir/model.safetensors  (HF Trainer FSDP FULL_STATE_DICT 单文件)
             或 save_dir/model.safetensors.index.json + sharded shards
             该文件是整个 NLDModel 的 state_dict, key 形如 'base_model.*' /
             'latent_thinker.*'. 按 prefix 拆分后分别 load.
          C. 上述两种混合 (A 缺失/损坏, 自动回退到 B)
        """
        import safetensors.torch
        import glob

        # ---- 形状自适应 helper ----
        def _shape_adapt(state_dict, target_shapes, prefix=""):
            reshaped = []
            for k, v in list(state_dict.items()):
                tk = k[len(prefix):] if (prefix and k.startswith(prefix)) else k
                if tk in target_shapes and tuple(v.shape) != target_shapes[tk]:
                    if v.numel() == int(torch.tensor(target_shapes[tk]).prod().item()):
                        state_dict[k] = v.reshape(target_shapes[tk])
                        reshaped.append((k, tuple(v.shape), target_shapes[tk]))
            return reshaped

        # ---- 尝试发现 HF Trainer 单文件 / sharded NLDModel ckpt ----
        single_safetensors = os.path.join(save_dir, "model.safetensors")
        sharded_index = os.path.join(save_dir, "model.safetensors.index.json")
        is_hf_trainer_ckpt = (
            os.path.isfile(single_safetensors) or os.path.isfile(sharded_index)
        )

        # ---- 路径 A: 加载 NativeLatentThinker (.pt) ----
        thinker_path = os.path.join(save_dir, "nld_latent_thinker.pt")
        thinker_loaded = False
        if os.path.exists(thinker_path):
            state_dict = torch.load(thinker_path, map_location="cpu")
            try:
                target_shapes = {k: tuple(v.shape) for k, v in self.latent_thinker.state_dict().items()}
            except Exception:
                target_shapes = {}
            reshaped_keys = _shape_adapt(state_dict, target_shapes)
            if reshaped_keys and self._verbose:
                print(f"[NLD] 形状自适应 reshape: {len(reshaped_keys)} 个参数")
                for k, src, dst in reshaped_keys:
                    print(f"[NLD]   {k}: {src} -> {dst}")
            missing, unexpected = self.latent_thinker.load_state_dict(state_dict, strict=False)
            thinker_loaded = True
            if self._verbose:
                print(f"[NLD] NativeLatentThinker 已从 {thinker_path} 加载")
                if missing:
                    print(f"[NLD]   新增参数: {missing}")
                if unexpected:
                    print(f"[NLD]   旧参数: {unexpected}")
        elif not is_hf_trainer_ckpt and self._verbose:
            print(f"[NLD] ⚠️ 未找到 NativeLatentThinker 权重: {thinker_path}")

        # 对齐 dtype/device (放在 thinker 加载之后, vlm 之前 — 与历史行为一致)
        target_dtype = next(self.base_model.parameters()).dtype
        target_device = next(self.base_model.parameters()).device
        self.latent_thinker = self.latent_thinker.to(dtype=target_dtype, device=target_device)

        # ---- 路径 A: 加载 vlm_full/*.safetensors ----
        vlm_dir = os.path.join(save_dir, "vlm_full")
        vlm_loaded_from_path_a = False
        path_a_failed = False
        if os.path.exists(vlm_dir):
            safetensor_files = sorted(glob.glob(os.path.join(vlm_dir, "*.safetensors")))
            if safetensor_files:
                # ---- sanity check: index 文件 vs shard 列表 ----
                index_path = os.path.join(vlm_dir, "model.safetensors.index.json")
                if os.path.exists(index_path) and self._verbose:
                    try:
                        import json as _json
                        with open(index_path, "r", encoding="utf-8") as _f:
                            _idx = _json.load(_f)
                        wmap = _idx.get("weight_map", {}) or {}
                        declared_shards = sorted(set(wmap.values()))
                        actual_shards = sorted(os.path.basename(sf) for sf in safetensor_files)
                        missing_shards = [s for s in declared_shards if s not in actual_shards]
                        extra_shards = [s for s in actual_shards if s not in declared_shards]
                        print(f"[NLD] VLM index: 声明 {len(declared_shards)} 个 shard, 实际找到 {len(actual_shards)} 个")
                        if missing_shards:
                            print(f"[NLD]   ⚠️ index 中声明但实际缺失的 shard: {missing_shards}")
                        if extra_shards:
                            print(f"[NLD]   ⚠️ index 未声明但目录存在的 shard: {extra_shards}")
                    except Exception as _e:
                        print(f"[NLD]   ⚠️ 解析 {index_path} 失败: {type(_e).__name__}: {_e}")

                vlm_state = {}
                failed_shards = []
                for sf in safetensor_files:
                    try:
                        sz_mb = os.path.getsize(sf) / (1024 * 1024)
                        shard_state = safetensors.torch.load_file(sf)
                        vlm_state.update(shard_state)
                        if self._verbose:
                            print(f"[NLD]   ✓ {os.path.basename(sf)}  size={sz_mb:.1f} MB  tensors={len(shard_state)}")
                    except Exception as _e:
                        sz_mb = os.path.getsize(sf) / (1024 * 1024) if os.path.exists(sf) else -1.0
                        msg = (f"[NLD]   ❌ 加载失败: {sf}\n"
                               f"[NLD]      file_size={sz_mb:.1f} MB  err={type(_e).__name__}: {_e}")
                        print(msg)
                        failed_shards.append((sf, str(_e)))

                if failed_shards:
                    path_a_failed = True
                    if not is_hf_trainer_ckpt:
                        # 没有可用 fallback, 直接抛错保持历史行为
                        raise RuntimeError(
                            f"[NLD] 共有 {len(failed_shards)} 个 safetensors shard 加载失败, "
                            f"无法完整恢复 VLM 权重. 请检查训练时的写盘是否完整, 或改用 HF Trainer\n"
                            f"  checkpoint-XXXX/model.safetensors 路径 (将 --checkpoint 指向该目录).\n"
                            f"  failed: {[os.path.basename(p) for p, _ in failed_shards]}"
                        )
                    elif self._verbose:
                        print(f"[NLD] ⚠️ vlm_full/ 中 {len(failed_shards)} 个 shard 损坏, "
                              f"将自动回退到 {save_dir} 下的 HF Trainer 单文件 ckpt")
                else:
                    missing, unexpected = self.base_model.load_state_dict(vlm_state, strict=False)
                    vlm_loaded_from_path_a = True
                    if self._verbose:
                        print(f"[NLD] VLM 全量模型已加载: {vlm_dir}")
                        if missing:
                            print(f"[NLD]   缺失参数: {len(missing)} 个")
                        if unexpected:
                            print(f"[NLD]   多余参数: {len(unexpected)} 个")

        # ---- 路径 B: 从 HF Trainer 单文件 / sharded model.safetensors 加载 ----
        # 触发条件: (a) 没走通路径 A 的 vlm_full, 或路径 A 损坏需 fallback;
        #          (b) 同时本目录下存在 model.safetensors / index.json
        # 该文件是整个 NLDModel 的 state_dict, 按 prefix 同时填充 base_model + latent_thinker.
        need_path_b = is_hf_trainer_ckpt and (path_a_failed or not vlm_loaded_from_path_a)
        if need_path_b:
            if self._verbose:
                print(f"[NLD] 进入 HF Trainer 单文件 ckpt 加载路径: {save_dir}")

            # 拼装 shard 文件列表 (单文件 / sharded 都统一处理)
            if os.path.isfile(sharded_index):
                import json as _json
                with open(sharded_index, "r", encoding="utf-8") as _f:
                    _idx = _json.load(_f)
                _shard_names = sorted(set((_idx.get("weight_map", {}) or {}).values()))
                shard_paths = [os.path.join(save_dir, n) for n in _shard_names]
            else:
                shard_paths = [single_safetensors]

            # 收集 base_model.* / latent_thinker.* 两类 key
            base_state = {}
            thinker_state = {}
            other_keys_count = 0
            for sp in shard_paths:
                if not os.path.isfile(sp):
                    raise RuntimeError(f"[NLD] HF Trainer ckpt shard 不存在: {sp}")
                try:
                    shard = safetensors.torch.load_file(sp)
                except Exception as _e:
                    raise RuntimeError(
                        f"[NLD] HF Trainer ckpt shard 加载失败: {sp}\n"
                        f"  err={type(_e).__name__}: {_e}"
                    )
                if self._verbose:
                    sz_mb = os.path.getsize(sp) / (1024 * 1024)
                    print(f"[NLD]   ✓ {os.path.basename(sp)}  size={sz_mb:.1f} MB  tensors={len(shard)}")
                for k, v in shard.items():
                    if k.startswith("base_model."):
                        base_state[k[len("base_model."):]] = v
                    elif k.startswith("latent_thinker."):
                        thinker_state[k[len("latent_thinker."):]] = v
                    else:
                        other_keys_count += 1

            if self._verbose:
                print(f"[NLD] HF Trainer ckpt 拆分: base_model={len(base_state)} 个, "
                      f"latent_thinker={len(thinker_state)} 个, other={other_keys_count} 个")

            # ---- 填充 base_model ----
            if base_state and not vlm_loaded_from_path_a:
                missing_b, unexpected_b = self.base_model.load_state_dict(base_state, strict=False)
                if self._verbose:
                    print(f"[NLD] base_model 已从 HF Trainer ckpt 加载")
                    if missing_b:
                        print(f"[NLD]   缺失参数: {len(missing_b)} 个")
                    if unexpected_b:
                        print(f"[NLD]   多余参数: {len(unexpected_b)} 个")

            # ---- 填充 latent_thinker (含 shape 自适应) ----
            if thinker_state and not thinker_loaded:
                try:
                    target_shapes = {k: tuple(v.shape) for k, v in self.latent_thinker.state_dict().items()}
                except Exception:
                    target_shapes = {}
                reshaped_keys = _shape_adapt(thinker_state, target_shapes)
                if reshaped_keys and self._verbose:
                    print(f"[NLD] 形状自适应 reshape (HF Trainer 路径): {len(reshaped_keys)} 个参数")
                    for k, src, dst in reshaped_keys:
                        print(f"[NLD]   {k}: {src} -> {dst}")
                missing_t, unexpected_t = self.latent_thinker.load_state_dict(thinker_state, strict=False)
                if self._verbose:
                    print(f"[NLD] latent_thinker 已从 HF Trainer ckpt 加载")
                    if missing_t:
                        print(f"[NLD]   缺失参数: {missing_t}")
                    if unexpected_t:
                        print(f"[NLD]   旧参数: {unexpected_t}")

            # 重新对齐 dtype/device (base_model 可能已被改写)
            target_dtype = next(self.base_model.parameters()).dtype
            target_device = next(self.base_model.parameters()).device
            self.latent_thinker = self.latent_thinker.to(dtype=target_dtype, device=target_device)
