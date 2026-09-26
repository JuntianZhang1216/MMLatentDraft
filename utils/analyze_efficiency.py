#!/usr/bin/env python3
"""
FLOPs & Latency 效率分析脚本
对比三个模型配置:
  1. Qwen3-VL-8B-Instruct          (带 CoT 要求的基座)
  2. Qwen3-VL-8B-Thinking           (原生 Thinking 模型)
  3. Interlude (Ours)             (latent reasoning + CoT)

输出:
  - outputs/efficiency_analysis/
    ├── per_sample_results.json      # 每条样本的详细数据
    ├── summary_stats.json           # 汇总统计
    ├── efficiency_comparison.png    # 柱状图对比
    └── efficiency_report.txt        # 文本报告

用法:
  python analyze_efficiency.py --checkpoint <ckpt_path> --max_samples 200
"""

import os, sys, json, argparse, time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

from rld.data import LATENT_TOKEN, LATENT_END_TOKEN, NLD_SYSTEM_PROMPT
from rld.model_v2 import NLDModel, _extract_visual_output
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image


# ============================================================
# 数据加载
# ============================================================
def load_erqa_data(data_file, data_root, max_samples=0):
    """加载 ERQA 测试数据."""
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            samples.append(obj)
            if max_samples > 0 and len(samples) >= max_samples:
                break
    return samples


# ============================================================
# FLOPs 估算
# ============================================================
def _estimate_flops(model, prompt_len, num_decode_steps,
                    num_latent_events=0, num_total_thought_steps=0):
    """粗略估算推理 FLOPs.

    Returns:
        (total_flops, prefill_flops, decode_flops_per_token)
    """
    try:
        if hasattr(model, '_inner_model'):
            # NLDModel 包装
            inner = model._inner_model
            text_model = inner.language_model
        else:
            text_model = model.language_model
        config = text_model.config
        n_layers = config.num_hidden_layers
        d_model = config.hidden_size
        seq_len = max(prompt_len, 4096)

        # Prefill
        prefill_flops = 2 * prompt_len * n_layers * d_model * (1 + 8 * d_model / seq_len)
        # Decode per token
        decode_flops_per_token = 2 * n_layers * d_model * (1 + 8 * d_model / seq_len)

        # Latent Thinker 额外 FLOPs
        latent_flops = 0
        if num_latent_events > 0 and num_total_thought_steps > 0:
            try:
                thinker = model.latent_thinker
                thinker_d_model = getattr(thinker, 'hidden_size', d_model) if hasattr(thinker, 'hidden_size') else d_model
                thinker_n_layers = getattr(thinker, 'num_hidden_layers', n_layers) if hasattr(thinker, 'num_hidden_layers') else n_layers
                thought_step_flops = 2 * thinker_n_layers * thinker_d_model * (1 + 8 * thinker_d_model / seq_len)
                latent_flops = num_total_thought_steps * thought_step_flops
            except Exception:
                # 粗略 fallback
                latent_flops = num_total_thought_steps * decode_flops_per_token * 0.5

        total_flops = prefill_flops + num_decode_steps * decode_flops_per_token + latent_flops
        return total_flops, prefill_flops, decode_flops_per_token
    except Exception as e:
        print(f"[Efficiency] ⚠️ FLOPs 估算失败: {e}, 返回 0")
        return 0, 0, 0


# ============================================================
# NLDModel 推理函数
# 参考 analyze_entropy_trigger.py 中的实现,
# 直接调用内部组件, 不经过 NLDModel.generate() (避免 pixel_values/image_grid_thw 位置参数限制)
# ============================================================
@torch.no_grad()
def generate_and_measure(
    model,
    processor,
    messages,
    max_new_tokens=2048,
    device="cuda",
    disable_latent=False,
):
    """对一条样本做推理并测量 latency 和 FLOPs.

    Args:
        disable_latent: True → 不触发 latent thinker (纯文本 CoT)
                        False → 正常 latent 推理
    Returns:
        dict with keys: generated_text, num_decode_steps, prompt_len,
                        prefill_latency_s, decode_latency_s, total_latency_s,
                        total_flops, prefill_flops, decode_flops_per_token,
                        num_latent_triggers, num_thought_steps
    """
    import time

    # 1. 构造 inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages) if process_vision_info else (None, None)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    prompt_len = input_ids.shape[1]
    B = input_ids.shape[0]
    device_ = input_ids.device

    # 2. 获取模型内部组件
    inner_model = model._inner_model
    text_model = inner_model.language_model
    lm_head = model._lm_head
    latent_token_id = model.latent_token_id
    latent_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|/latent|>")
    eos_token_id = processor.tokenizer.eos_token_id

    # ---- 3. 视觉编码 (只做一次) ----
    if pixel_values is not None and image_grid_thw is not None:
        _pixel_values = pixel_values.to(dtype=inner_model.visual.dtype)
        vision_output = inner_model.visual(_pixel_values, grid_thw=image_grid_thw)
        merged_hidden_states, gen_deepstack_features = _extract_visual_output(vision_output)
        split_sizes = (
            image_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size ** 2
        ).tolist()
        image_embeds_list = list(torch.split(merged_hidden_states, split_sizes))
        cached_visual_embeds = torch.cat(image_embeds_list, dim=0)
    else:
        cached_visual_embeds = None
        gen_deepstack_features = None

    # ---- 4. Prefill ----
    prompt_embeds = inner_model.get_input_embeddings()(input_ids)
    image_token_id = inner_model.config.image_token_id
    image_mask = (input_ids == image_token_id)
    num_image_tokens = image_mask.sum().item()
    if num_image_tokens > 0 and cached_visual_embeds is not None:
        img_emb = cached_visual_embeds[:num_image_tokens].to(prompt_embeds.device, prompt_embeds.dtype)
        image_mask_3d = image_mask.unsqueeze(-1).expand_as(prompt_embeds)
        prompt_embeds = prompt_embeds.masked_scatter(image_mask_3d, img_emb)

    seg_token_mask = torch.ones(B, input_ids.shape[1], device=device_, dtype=torch.long)
    inner_model.rope_deltas = None
    position_ids, rope_deltas = inner_model.get_rope_index(
        input_ids, image_grid_thw=image_grid_thw, video_grid_thw=None,
        attention_mask=seg_token_mask,
    )
    inner_model.rope_deltas = rope_deltas

    if position_ids is not None and position_ids.dim() == 3:
        text_pos = torch.arange(input_ids.shape[1], device=device_).unsqueeze(0).expand(B, -1)
        position_ids = torch.cat([text_pos.unsqueeze(0), position_ids], dim=0)

    visual_pos_masks = (input_ids == inner_model.config.image_token_id) if pixel_values is not None and num_image_tokens > 0 else None

    torch.cuda.synchronize(device)
    prefill_start = time.perf_counter()
    prefill_out = text_model(
        inputs_embeds=prompt_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=gen_deepstack_features,
    )
    prefill_end = time.perf_counter()
    prefill_latency_s = prefill_end - prefill_start
    past_key_values = prefill_out.past_key_values
    mrope_last_pos = position_ids[1:, :, -1:] if position_ids is not None else None
    prefill_hidden = prefill_out.last_hidden_state
    next_token_logits = lm_head(prefill_hidden[:, -1, :])

    # ---- 5. Decode 循环 ----
    generated_ids = input_ids.clone()
    generated_token_ids = []
    latent_trigger_positions = []
    latent_end_positions = []
    recent_last_hidden = prefill_hidden[:, -1:, :]
    current_pos = prompt_len
    num_decode_steps = 0
    num_thought_steps = 0

    torch.cuda.synchronize(device)
    decode_start = time.perf_counter()

    for gen_step in range(max_new_tokens):
        # Greedy 解码
        next_token = next_token_logits.argmax(dim=-1, keepdim=True).squeeze(-1)
        token_id = next_token[0].item()

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
        generated_token_ids.append(token_id)
        num_decode_steps += 1

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        # 检测 <|latent|>
        is_latent_trigger = (latent_token_id is not None) and (token_id == latent_token_id)
        is_latent_end = (latent_end_token_id is not None) and (token_id == latent_end_token_id)

        if is_latent_trigger:
            latent_trigger_positions.append(len(generated_token_ids) - 1)

            if not disable_latent:
                # ---- WITH LATENT: 触发 NativeLatentThinker ----
                last_hidden = recent_last_hidden
                base_thought_pos = current_pos
                thought_pos = torch.arange(current_pos, current_pos + 1, device=device_)
                thought_mrope_pos = thought_pos.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                thought_text_pos = torch.full((B, 1), base_thought_pos, device=device_, dtype=torch.long)
                thought_position_ids = torch.cat([thought_text_pos.unsqueeze(0), thought_mrope_pos], dim=0)
                think_attn_mask = torch.ones(B, base_thought_pos + 1, device=device_, dtype=torch.long)

                thinker_result = model.latent_thinker(
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
                thought_output = thinker_result['thought_output']
                n_thought_steps = thinker_result.get('num_thought_steps', 1)
                num_thought_steps += n_thought_steps

                # 写回 thought prefix 到 KV cache
                thought_prefix_cache_pos = torch.arange(
                    current_pos, current_pos + n_thought_steps, device=device_
                )
                thought_prefix_mrope = thought_prefix_cache_pos.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
                thought_prefix_text_pos = torch.arange(
                    current_pos, current_pos + n_thought_steps, device=device_
                ).unsqueeze(0).expand(B, -1)
                thought_prefix_pos_ids = torch.cat([
                    thought_prefix_text_pos.unsqueeze(0),
                    thought_prefix_mrope,
                ], dim=0)
                thought_prefix_attn_mask = torch.ones(
                    B, current_pos + n_thought_steps, device=device_, dtype=attention_mask.dtype
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
                current_pos += n_thought_steps
                mrope_last_pos = mrope_last_pos + n_thought_steps
            else:
                # ---- NO LATENT: 跳过 thinking, 当普通 token 继续 ----
                pass

        elif is_latent_end:
            latent_end_positions.append(len(generated_token_ids) - 1)

        # 继续自回归 forward
        new_mrope_pos = (mrope_last_pos + 1) if mrope_last_pos is not None else None
        text_pos_gen = torch.full((B, 1), current_pos, device=device_, dtype=torch.long)
        if new_mrope_pos is not None:
            position_ids_4d = torch.cat([text_pos_gen.unsqueeze(0), new_mrope_pos], dim=0)
        else:
            position_ids_4d = text_pos_gen.unsqueeze(0)

        token_embeds = inner_model.get_input_embeddings()(next_token.unsqueeze(-1))
        current_total_len = current_pos + 1
        new_attention_mask = torch.ones(B, current_total_len, device=device_, dtype=attention_mask.dtype)
        cache_position_gen = torch.tensor([current_pos], device=device_, dtype=torch.long)

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
        recent_last_hidden = new_hidden
        if mrope_last_pos is not None:
            mrope_last_pos = new_mrope_pos

        current_pos += 1

    torch.cuda.synchronize(device)
    decode_end = time.perf_counter()
    decode_latency_s = decode_end - decode_start
    total_latency_s = prefill_latency_s + decode_latency_s

    # 6. 解码文本
    generated_text = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=False)

    # 7. FLOPs 估算
    total_flops, prefill_flops, decode_flops_per_token = _estimate_flops(
        model, prompt_len, num_decode_steps,
        num_latent_events=len(latent_trigger_positions),
        num_total_thought_steps=num_thought_steps,
    )

    return {
        "generated_text": generated_text,
        "prompt_len": prompt_len,
        "num_decode_steps": num_decode_steps,
        "num_latent_triggers": len(latent_trigger_positions),
        "num_thought_steps": num_thought_steps,
        "prefill_latency_s": prefill_latency_s,
        "decode_latency_s": decode_latency_s,
        "total_latency_s": total_latency_s,
        "total_flops": total_flops,
        "prefill_flops": prefill_flops,
        "decode_flops_per_token": decode_flops_per_token,
    }


# ============================================================
# 基座模型推理 (Qwen3-VL-8B-Instruct, 无 latent 架构)
# ============================================================
@torch.no_grad()
def generate_base_model(
    model,
    processor,
    messages,
    max_new_tokens=2048,
    device="cuda",
):
    """纯基座模型 (Qwen3-VL-8B-Instruct) 推理 + latency 测量.

    Returns:
        dict with same keys as generate_and_measure.
    """
    import time

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages) if process_vision_info else (None, None)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    prompt_len = inputs['input_ids'].shape[1]

    # 使用 model.generate 进行推理 (更贴近实际用法)
    # 先 warm up
    _ = model.generate(**inputs, max_new_tokens=1)
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    torch.cuda.synchronize(device)
    end = time.perf_counter()

    total_latency_s = end - start
    num_decode_steps = output_ids.shape[1] - prompt_len
    generated_text = processor.decode(output_ids[0, prompt_len:], skip_special_tokens=False)

    # FLOPs 估算 (无 latent)
    total_flops, prefill_flops, decode_flops_per_token = _estimate_flops(
        model, prompt_len, num_decode_steps,
        num_latent_events=0, num_total_thought_steps=0,
    )

    return {
        "generated_text": generated_text,
        "prompt_len": prompt_len,
        "num_decode_steps": num_decode_steps,
        "num_latent_triggers": 0,
        "num_thought_steps": 0,
        "prefill_latency_s": total_latency_s * 0.5,  # 粗略拆分
        "decode_latency_s": total_latency_s * 0.5,
        "total_latency_s": total_latency_s,
        "total_flops": total_flops,
        "prefill_flops": prefill_flops,
        "decode_flops_per_token": decode_flops_per_token,
    }


# ============================================================
# 可视化
# ============================================================
def plot_efficiency_comparison(summary, output_dir):
    """绘制 FLOPs 和 Latency 三配置对比图 (合并为单图，ACL风格)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
    except ImportError:
        print("[Efficiency] ⚠️ matplotlib 未安装, 跳过绘图")
        return

    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.titlesize'] = 12
    matplotlib.rcParams['axes.labelsize'] = 11

    # === 颜色方案 (与 modality_manifold_analysis 和谐一致) ===
    # 深蓝 (Instruct) → 青色/蓝 (Thinking) → 紫色 (Interlude)
    # 参考配色：蓝色 #2196F3, 紫色 #9C27B0
    # 使用更专业、柔和且区分度高的颜色
    color_inst  = '#78B3CE'   # 柔和蓝 (Instruct)
    color_think = '#4D869C'   # 深青 (Thinking)
    color_ours  = '#7D71A0'   # 柔和紫 (Ours)
    color_bg    = '#FFFFFF'
    text_color  = '#222222'
    grid_color  = '#E0E0E0'

    # 准备数据
    configs = ["Qwen3-VL\nInstruct", "Qwen3-VL\nThinking", "Interlude\n(Ours)"]
    colors  = [color_inst, color_think, color_ours]

    flops_data = [
        summary.get("base_flops_mean", 0) or 0,
        summary.get("thinking_flops_mean", 0) or 0,
        summary.get("with_latent_flops_mean", 0) or 0,
    ]
    flops_std = [
        summary.get("base_flops_std", 0) or 0,
        summary.get("thinking_flops_std", 0) or 0,
        summary.get("with_latent_flops_std", 0) or 0,
    ]

    latency_data = [
        summary.get("base_latency_mean", 0) or 0,
        summary.get("thinking_latency_mean", 0) or 0,
        summary.get("with_latent_latency_mean", 0) or 0,
    ]
    latency_std = [
        summary.get("base_latency_std", 0) or 0,
        summary.get("thinking_latency_std", 0) or 0,
        summary.get("with_latent_latency_std", 0) or 0,
    ]

    # 转换为更易读的单位
    flops_g = [v / 1e9 for v in flops_data]       # GFLOPs
    latency_s = latency_data

    # === 子图布局: 1行2列 ===
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor(color_bg)
    for ax in axes:
        ax.set_facecolor(color_bg)

    bar_width = 0.52
    x = range(len(configs))

    # ---- 左图: FLOPs ----
    ax = axes[0]
    bars = ax.bar(x, flops_g, bar_width, color=colors, edgecolor='none', zorder=3, yerr=[s/1e9 for s in flops_std], error_kw={'ecolor':'#555555','elinewidth':1.2,'capthick':1,'capsize':3})
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9.5, color=text_color)
    ax.set_ylabel(r"FLOPs per Sample ($\times 10^9$)", fontsize=11, color=text_color)
    ax.set_title("Computational Cost", fontsize=12, fontweight='bold', color=text_color, pad=8)
    ax.tick_params(axis='both', which='major', labelcolor=text_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.grid(axis='y', ls='-', lw=0.5, alpha=0.4, color=grid_color, zorder=0)
    # 数值标签
    for i, (bar, val) in enumerate(zip(bars, flops_g)):
        if val > 0:
            label = f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (flops_std[i]/1e9)*1.15,
                    label, ha='center', va='bottom', fontsize=9.5, color=text_color, fontweight='bold')

    # ---- 右图: Latency ----
    ax = axes[1]
    bars = ax.bar(x, latency_s, bar_width, color=colors, edgecolor='none', zorder=3, yerr=latency_std, error_kw={'ecolor':'#555555','elinewidth':1.2,'capthick':1,'capsize':3})
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9.5, color=text_color)
    ax.set_ylabel("Avg. Latency (s/sample)", fontsize=11, color=text_color)
    ax.set_title("Wall-Clock Latency", fontsize=12, fontweight='bold', color=text_color, pad=8)
    ax.tick_params(axis='both', which='major', labelcolor=text_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.grid(axis='y', ls='-', lw=0.5, alpha=0.4, color=grid_color, zorder=0)
    # 数值标签
    for i, (bar, val) in enumerate(zip(bars, latency_s)):
        if val > 0:
            label = f"{val:.2f}s"
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + latency_std[i]*1.15,
                    label, ha='center', va='bottom', fontsize=9.5, color=text_color, fontweight='bold')

    # 整体标题
    fig.suptitle("Efficiency Comparison: Qwen3-VL-8B vs Interlude", fontsize=13, fontweight='bold', color=text_color, y=1.01)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "efficiency_comparison.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Efficiency] 对比图已保存: {output_path}")
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="FLOPs & Latency 效率分析: 模型配置对比")
    parser.add_argument("--model_path", type=str,
                        default="<PATH_TO_QWEN3_VL_8B_INSTRUCT>",
                        help="Qwen3-VL-8B-Instruct 基座模型路径")
    parser.add_argument("--checkpoint", type=str, default=None, required=True,
                        help="Interlude checkpoint 路径 (必须)")
    parser.add_argument("--data_file", type=str, default="./data/erqa/erqa_test.jsonl")
    parser.add_argument("--data_root", type=str, default="./data/erqa")
    parser.add_argument("--output_dir", type=str, default="./outputs/efficiency_analysis")
    parser.add_argument("--max_samples", type=int, default=200, help="0=全量")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--thinking_model_path", type=str,
                        default="<PATH_TO_QWEN3_VL_8B_THINKING>",
                        help="Qwen3-VL-8B-Thinking 模型路径 (不带 latent 的 CoT 基线)")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)
    device = args.device

    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"efficiency_analysis_{ts}.log")

    print("=" * 60)
    print("  FLOPs & Latency 效率分析")
    print("=" * 60)
    print(f"  model_path:         {args.model_path}")
    print(f"  thinking_model_path:{args.thinking_model_path}")
    print(f"  checkpoint:   {args.checkpoint}")
    print(f"  data_file:    {args.data_file}")
    print(f"  data_root:    {args.data_root}")
    print(f"  output_dir:   {args.output_dir}")
    print(f"  max_samples:  {args.max_samples}")
    print(f"  max_new:      {args.max_new_tokens}")
    print(f"  dtype:        {args.dtype}")
    print(f"  device:       {args.device}")
    print("=" * 60)

    # ---- 加载数据 ----
    samples = load_erqa_data(args.data_file, args.data_root, args.max_samples)
    n_total = len(samples)
    print(f"[Efficiency] 加载 {n_total} 条样本")

    if n_total == 0:
        print("[Efficiency] ❌ 无数据, 退出")
        return

    # ============================================================
    # 配置 A: Qwen3-VL-8B-Instruct (纯基座)
    # ============================================================
    print("\n[Efficiency] === 加载配置 A: Qwen3-VL-8B-Instruct (纯基座) ===")
    base_model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    base_model = base_model.to(device).eval()
    print(f"[Efficiency] 基座模型已加载: {sum(p.numel() for p in base_model.parameters())/1e9:.2f}B params")

    # ============================================================
    # 配置 B & C: Interlude (checkpoint)
    # ============================================================
    print("\n[Efficiency] === 加载配置 B/C: Interlude (checkpoint) ===")
    nld_model = NLDModel(model_path=args.model_path, torch_dtype=dtype, attn_implementation="flash_attention_2")
    nld_processor = AutoProcessor.from_pretrained(args.model_path)

    existing_specials = set(nld_processor.tokenizer.additional_special_tokens or [])
    to_add = [t for t in [LATENT_TOKEN, LATENT_END_TOKEN] if t not in existing_specials]
    if to_add:
        nld_processor.tokenizer.add_special_tokens({"additional_special_tokens": to_add})

    _V_old = nld_model.base_model.get_input_embeddings().weight.shape[0]
    _target_vocab = max(len(nld_processor.tokenizer), _V_old)
    if _target_vocab != _V_old:
        nld_model.base_model.resize_token_embeddings(_target_vocab)
    nld_model.set_processor(nld_processor)
    nld_model.load_pretrained(args.checkpoint)
    nld_model = nld_model.to(device).eval()
    print(f"[Efficiency] Interlude 已加载: checkpoint={args.checkpoint}")
    print(f"[Efficiency]  vocab: {_V_old} -> {nld_model.base_model.get_input_embeddings().weight.shape[0]}")
    print(f"[Efficiency]  latent_token_id = {nld_model.latent_token_id}")

    # ============================================================
    # 配置 D: Qwen3-VL-8B-Thinking (带 CoT 的基线)
    # ============================================================
    print("\n[Efficiency] === 加载配置 D: Qwen3-VL-8B-Thinking (CoT 基线) ===")
    thinking_model = AutoModelForVision2Seq.from_pretrained(
        args.thinking_model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    thinking_processor = AutoProcessor.from_pretrained(args.thinking_model_path, trust_remote_code=True)
    thinking_model = thinking_model.to(device).eval()
    print(f"[Efficiency] Thinking 模型已加载: {sum(p.numel() for p in thinking_model.parameters())/1e9:.2f}B params")

    # ============================================================
    # 逐条推理
    # ============================================================
    per_sample_results = []

    for idx, item in enumerate(samples):
        question = item.get("question", "")
        answer_gt = item.get("answer", "")
        qid = item.get("question_id", f"q_{idx}")

        # 构造图片路径
        image_paths = []
        for key in ["image_path", "image"]:
            val = item.get(key, "")
            if val:
                if not os.path.isabs(val):
                    val = os.path.join(args.data_root, val)
                if os.path.exists(val):
                    image_paths.append(val)
        for i in range(10):
            key = f"image_path_{i}"
            val = item.get(key, "")
            if val:
                if not os.path.isabs(val):
                    val = os.path.join(args.data_root, val)
                if os.path.exists(val):
                    image_paths.append(val)

        # 构建 messages: Base 模型需要额外的逐步推理要求以保证公平对比
        content = []
        for img_path in image_paths:
            content.append({"type": "image", "image": f"file://{img_path}"})
        content.append({"type": "text", "text": question})

        # 系统提示: Base 模型使用与 Interlude 一致的结构，要求详细自然语言推理
        BASE_SYSTEM_PROMPT = """You are a visual reasoning assistant. Analyze images carefully and think deeply.

Rules:
1. Describe your reasoning process in detail, grounded in the visual evidence. Be thorough and explicit about how you arrive at each conclusion.
2. When you need to perform high-level visual thinking — such as mentally reconstructing 3D layouts from 2D views, reasoning about spatial relationships, simulating physical dynamics, resolving occlusions, or imagining counterfactual scenes — reason through it step by step in natural language.
3. If the question can be answered directly from the image without deep mental simulation, provide a concise but complete reasoning. If it requires multiple distinct mental operations (e.g. compare two viewpoints, or trace before/after states), describe each operation clearly and thoroughly in natural language.
4. End your response with "Final Answer:" on its own line, followed by your final answer.

Example A (direct observation):
The image shows a red sedan parked beside a green bench under a tree. The sedan has four doors and tinted windows, and its license plate is partially visible but too blurry to read. The bench appears weathered, suggesting it has been outdoors for some time. No mental simulation is needed to describe the visible scene.
Final Answer:
A red four-door sedan is parked next to a green bench beneath a tree.

Example B (single mental simulation):
The scene shows a ball mid-flight toward a glass window. To predict the outcome, I need to mentally simulate the collision dynamics and trace the impact geometry. First, I observe the ball's trajectory: it is approaching the window at an angle from the left, suggesting a diagonal path. The window has four panes, and the ball is heading toward the lower-left pane. The ball's size relative to the pane suggests it has enough mass to shatter the glass upon impact. Next, I consider the physics: when a hard spherical object hits a glass pane at moderate velocity, the glass will likely crack and break at the point of impact, with fragments propagating outward in a cone pattern due to the force distribution. Finally, based on the trajectory angle and estimated velocity, the ball will shatter the lower-left pane on impact, sending glass fragments inward in a cone pattern.
Final Answer:
The ball will break through the lower-left window pane, scattering glass fragments inward.

Example C (two mental simulations):
Two people stand on opposite sides of a chess board. To compare their perspectives, I first reconstruct what the left player sees. From the left player's position, the white pieces are in the foreground and the board tilts away to the right. The player would see the white king at the bottom-right of their field of view and the white queen to its left. The pawns form a protective wall in front of the major pieces. Now I switch to the right player's viewpoint. From the right, the black pieces dominate the foreground with the board tilting in the opposite direction — away to the left. The black king would appear at the bottom-left of this player's view, with the black queen to its right. The coordinate system is inverted: what is "left" for one player is "right" for the other, and "forward" for one is "backward" for the other. The two players see mirror-opposite layouts of the same board.
Final Answer:
Each player sees the board tilted away from them with their own pieces in the foreground, producing mirror-opposite perspectives."""
        base_messages = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        # Interlude 使用训练时的 NLD_SYSTEM_PROMPT
        nld_messages = [
            {"role": "system", "content": NLD_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        print(f"\n  [{idx+1}/{n_total}] {qid}")

        try:
            # ---- Config A: 基座模型 (带推理要求) ----
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            result_base = generate_base_model(
                base_model, processor, base_messages,
                max_new_tokens=args.max_new_tokens, device=device,
            )
            print(f"      Qwen3-VL-8B-Instruct: steps={result_base['num_decode_steps']}, "
                  f"latency={result_base['total_latency_s']:.3f}s, "
                  f"FLOPs={result_base['total_flops']:.2e}")

            # ---- Config C: Ours (正常 latent) ----
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            # reset rope_deltas
            if hasattr(nld_model, '_inner_model'):
                inner = nld_model._inner_model
                if hasattr(inner, 'rope_deltas'):
                    inner.rope_deltas = None
            result_with_latent = generate_and_measure(
                nld_model, nld_processor, nld_messages,
                max_new_tokens=args.max_new_tokens, device=device,
                disable_latent=False,
            )
            print(f"      Interlude (Ours): steps={result_with_latent['num_decode_steps']}, "
                  f"latent×{result_with_latent['num_latent_triggers']}, "
                  f"thoughts={result_with_latent['num_thought_steps']}, "
                  f"latency={result_with_latent['total_latency_s']:.3f}s, "
                  f"FLOPs={result_with_latent['total_flops']:.2e}")

            # ---- Config D: Thinking 模型 ----
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            result_thinking = generate_base_model(
                thinking_model, thinking_processor, base_messages,
                max_new_tokens=args.max_new_tokens, device=device,
            )
            print(f"      Qwen3-VL-8B-Thinking: steps={result_thinking['num_decode_steps']}, "
                  f"latency={result_thinking['total_latency_s']:.3f}s, "
                  f"FLOPs={result_thinking['total_flops']:.2e}")

            per_sample_results.append({
                "question_id": qid,
                "base": result_base,
                "thinking": result_thinking,
                "with_latent": result_with_latent,
            })

        except Exception as e:
            print(f"      ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ============================================================
    # 汇总统计
    # ============================================================
    print("\n" + "=" * 60)
    print("  汇总统计")
    print("=" * 60)

    base_flops = [r["base"]["total_flops"] for r in per_sample_results if r["base"].get("total_flops", 0) > 0]
    base_latency = [r["base"]["total_latency_s"] for r in per_sample_results if r["base"].get("total_latency_s", 0) > 0]
    base_decode = [r["base"]["num_decode_steps"] for r in per_sample_results if r["base"].get("num_decode_steps", 0) > 0]

    with_latent_flops = [r["with_latent"]["total_flops"] for r in per_sample_results if r["with_latent"].get("total_flops", 0) > 0]
    with_latent_latency = [r["with_latent"]["total_latency_s"] for r in per_sample_results if r["with_latent"].get("total_latency_s", 0) > 0]
    with_latent_decode = [r["with_latent"]["num_decode_steps"] for r in per_sample_results if r["with_latent"].get("num_decode_steps", 0) > 0]

    thinking_flops = [r["thinking"]["total_flops"] for r in per_sample_results if r["thinking"].get("total_flops", 0) > 0]
    thinking_latency = [r["thinking"]["total_latency_s"] for r in per_sample_results if r["thinking"].get("total_latency_s", 0) > 0]
    thinking_decode = [r["thinking"]["num_decode_steps"] for r in per_sample_results if r["thinking"].get("num_decode_steps", 0) > 0]

    def _mean_std(data):
        arr = np.array(data)
        return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0

    summary = {
        "n_samples": len(per_sample_results),
        # Base
        "base_flops_mean": _mean_std(base_flops)[0] if base_flops else None,
        "base_flops_std": _mean_std(base_flops)[1] if base_flops else None,
        "base_latency_mean": _mean_std(base_latency)[0] if base_latency else None,
        "base_latency_std": _mean_std(base_latency)[1] if base_latency else None,
        "base_decode_steps_mean": _mean_std(base_decode)[0] if base_decode else None,
        # With-latent
        "with_latent_flops_mean": _mean_std(with_latent_flops)[0] if with_latent_flops else None,
        "with_latent_flops_std": _mean_std(with_latent_flops)[1] if with_latent_flops else None,
        "with_latent_latency_mean": _mean_std(with_latent_latency)[0] if with_latent_latency else None,
        "with_latent_latency_std": _mean_std(with_latent_latency)[1] if with_latent_latency else None,
        "with_latent_decode_steps_mean": _mean_std(with_latent_decode)[0] if with_latent_decode else None,
        # Thinking
        "thinking_flops_mean": _mean_std(thinking_flops)[0] if thinking_flops else None,
        "thinking_flops_std": _mean_std(thinking_flops)[1] if thinking_flops else None,
        "thinking_latency_mean": _mean_std(thinking_latency)[0] if thinking_latency else None,
        "thinking_latency_std": _mean_std(thinking_latency)[1] if thinking_latency else None,
        "thinking_decode_steps_mean": _mean_std(thinking_decode)[0] if thinking_decode else None,
    }

    # 打印汇总
    print(f"\n  样本数: {summary['n_samples']}")
    print(f"\n  [Qwen3-VL-8B-Instruct]")
    if summary["base_flops_mean"]:
        print(f"    FLOPs:   {summary['base_flops_mean']:.3e}  ± {summary['base_flops_std']:.3e}")
        print(f"    Latency: {summary['base_latency_mean']:.3f}s ± {summary['base_latency_std']:.3f}s")
        print(f"    Decode steps: {summary['base_decode_steps_mean']:.1f}")

    print(f"\n  [Interlude (Ours)]")
    if summary["with_latent_flops_mean"]:
        print(f"    FLOPs:   {summary['with_latent_flops_mean']:.3e}  ± {summary['with_latent_flops_std']:.3e}")
        print(f"    Latency: {summary['with_latent_latency_mean']:.3f}s ± {summary['with_latent_latency_std']:.3f}s")
        print(f"    Decode steps: {summary['with_latent_decode_steps_mean']:.1f}")

    print(f"\n  [Qwen3-VL-8B-Thinking]")
    if summary["thinking_flops_mean"]:
        print(f"    FLOPs:   {summary['thinking_flops_mean']:.3e}  ± {summary['thinking_flops_std']:.3e}")
        print(f"    Latency: {summary['thinking_latency_mean']:.3f}s ± {summary['thinking_latency_std']:.3f}s")
        print(f"    Decode steps: {summary['thinking_decode_steps_mean']:.1f}")

    # 对比
    if summary["base_flops_mean"] and summary["with_latent_flops_mean"]:
        print(f"\n  [对比: Interlude (Ours) vs Qwen3-VL-8B-Instruct]")
        print(f"    FLOPs  增加: {(summary['with_latent_flops_mean']/summary['base_flops_mean']-1)*100:.1f}%")
        print(f"    Latency 增加: {(summary['with_latent_latency_mean']/summary['base_latency_mean']-1)*100:.1f}%")
    if summary["thinking_flops_mean"] and summary["with_latent_flops_mean"]:
        print(f"\n  [对比: Interlude (Ours) vs Qwen3-VL-8B-Thinking]")
        print(f"    FLOPs  增加: {(summary['with_latent_flops_mean']/summary['thinking_flops_mean']-1)*100:.1f}%")
        print(f"    Latency 增加: {(summary['with_latent_latency_mean']/summary['thinking_latency_mean']-1)*100:.1f}%")
    if summary["base_flops_mean"] and summary["thinking_flops_mean"]:
        print(f"\n  [对比: Qwen3-VL-8B-Thinking vs Qwen3-VL-8B-Instruct]")
        print(f"    FLOPs  增加: {(summary['thinking_flops_mean']/summary['base_flops_mean']-1)*100:.1f}%")
        print(f"    Latency 增加: {(summary['thinking_latency_mean']/summary['base_latency_mean']-1)*100:.1f}%")

    # 保存结果
    per_sample_path = os.path.join(args.output_dir, "per_sample_results.json")
    with open(per_sample_path, "w", encoding="utf-8") as f:
        json.dump(per_sample_results, f, ensure_ascii=False, indent=2)
    print(f"\n[Efficiency] 逐样本结果: {per_sample_path}")

    summary_path = os.path.join(args.output_dir, "summary_stats.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Efficiency] 汇总统计:    {summary_path}")

    # 绘图
    plot_efficiency_comparison(summary, args.output_dir)

    # 文本报告
    report_path = os.path.join(args.output_dir, "efficiency_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  FLOPs & Latency 效率分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"样本数: {summary['n_samples']}\n\n")
        f.write("[Qwen3-VL-8B-Instruct]\n")
        f.write(f"  FLOPs:   {summary['base_flops_mean']:.3e} ± {summary['base_flops_std']:.3e}\n")
        f.write(f"  Latency: {summary['base_latency_mean']:.3f}s ± {summary['base_latency_std']:.3f}s\n\n")
        f.write("[Qwen3-VL-8B-Thinking]\n")
        f.write(f"  FLOPs:   {summary['thinking_flops_mean']:.3e} ± {summary['thinking_flops_std']:.3e}\n")
        f.write(f"  Latency: {summary['thinking_latency_mean']:.3f}s ± {summary['thinking_latency_std']:.3f}s\n\n")
        f.write("[Interlude (Ours)]\n")
        f.write(f"  FLOPs:   {summary['with_latent_flops_mean']:.3e} ± {summary['with_latent_flops_std']:.3e}\n")
        f.write(f"  Latency: {summary['with_latent_latency_mean']:.3f}s ± {summary['with_latent_latency_std']:.3f}s\n\n")
        if summary["base_flops_mean"] and summary["with_latent_flops_mean"]:
            f.write("[Interlude (Ours) vs Qwen3-VL-8B-Instruct]\n")
            f.write(f"  FLOPs  +{(summary['with_latent_flops_mean']/summary['base_flops_mean']-1)*100:.1f}%\n")
            f.write(f"  Latency +{(summary['with_latent_latency_mean']/summary['base_latency_mean']-1)*100:.1f}%\n\n")
        if summary["thinking_flops_mean"] and summary["with_latent_flops_mean"]:
            f.write("[Interlude (Ours) vs Qwen3-VL-8B-Thinking]\n")
            f.write(f"  FLOPs  +{(summary['with_latent_flops_mean']/summary['thinking_flops_mean']-1)*100:.1f}%\n")
            f.write(f"  Latency +{(summary['with_latent_latency_mean']/summary['thinking_latency_mean']-1)*100:.1f}%\n\n")
        if summary["base_flops_mean"] and summary["thinking_flops_mean"]:
            f.write("[Qwen3-VL-8B-Thinking vs Qwen3-VL-8B-Instruct]\n")
            f.write(f"  FLOPs  +{(summary['thinking_flops_mean']/summary['base_flops_mean']-1)*100:.1f}%\n")
            f.write(f"  Latency +{(summary['thinking_latency_mean']/summary['base_latency_mean']-1)*100:.1f}%\n")
    print(f"[Efficiency] 文本报告:    {report_path}")

    print("\n" + "=" * 60)
    print("  ✅ 效率分析完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
