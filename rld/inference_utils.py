#!/usr/bin/env python3
"""Interlude 默认推理时入口: ``infer_with_latent_fallback``.

设计目标
========
把 "一旦命中 hit_max / 病态重复 / 没有 Final Answer 等失败模式, 立即换一个
'directly answer' 的 prompt 重跑一次" 做成 Interlude **默认的推理时行为**.
所有 inference / eval 脚本只要从原本的::

    generated_ids = model.generate(..., max_new_tokens=N)
    text = processor.tokenizer.decode(generated_ids[0, prompt_len:], ...)

切到::

    out = infer_with_latent_fallback(model, processor, messages, max_new_tokens=N)
    text = out['text']                # 已是后处理过的最终 clean output
    ...

就自动具备 fallback. **fallback 的实现保持极简**: 不在 ``model.generate`` 内部
做检测, 不复用 KV cache, 也不用 forced_decoder_ids -- 第一次跑完后做事后检测,
命中就**用一个新的 messages (附 directly-answer hint + 强制 'Final Answer:'
assistant prefix) 完整重跑一次**, 仅此而已.

为什么不改 ``NLDModel.generate``
----------------------------------
``model_v2.py::NLDModel.generate`` 是项目 token-level 生成原语, 内含 KV cache /
``<|latent|>`` 触发 / NativeLatentThinker prefix 写回 等高耦合逻辑, 改起来
风险大, 收益低 (它做不到"换 prompt"这个语义层面的事). 把决策放到调用层是
最干净的分工.

依赖关系
--------
* PathologyDetector 类型 / DIRECTLY_ANSWER_HINT 等保留在 ``diagnose_latent_ab``
  里 (offline 诊断也要用), 这里**事后检测**只用其中的 ngram_repeat 规则,
  实现一个 ``_post_hoc_pathology_scan`` 即可, 完全无运行期 hook 依赖.
* 不依赖 ``scripts.rerun_wrong_samples`` (那是 offline 工具).
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from rld.data import LATENT_TOKEN, LATENT_END_TOKEN

try:
    from qwen_vl_utils import process_vision_info as _process_vision_info
except ImportError:  # 推理脚本本身可能在没装 qwen_vl_utils 的环境里 import 本模块,
    # 让 import 不失败; 真正调用时若 messages 含 image 会在内部报错.
    _process_vision_info = None


# ============================================================================
# Directly-answer prompt 设计 (与 scripts/rerun_wrong_samples.py 行为一致)
# ============================================================================
# 不在 hint 里写死 'Final Answer:' 字样: 我们靠 assistant_prefix 强制注入到
# assistant turn 的开头, 写两遍反而让模型有 "echo + 重新开段" 的失败模式.
# 新策略 (2026-05): hint 不再规定答案形态 (不再写 'single letter / single word /
# single number'), 改为让模型遵循 query 自身的 answer-format 约定 -- 题面里通常
# 已经写了 'Please answer with the letter of the correct option' / 'answer with
# a number' / 自由文本 等, 让 query 作为唯一权威, 避免与 hint 打架 (e.g. ERQA
# 题面要字母, hint 说 'single number', 模型按 hint 输出 '6' -> evaluator 解析失败).
DIRECTLY_ANSWER_HINT = (
    "\n\nIMPORTANT: Stop reasoning and answer the question above NOW. "
    "Do NOT explain, do NOT describe what you see, do NOT plan, "
    "do NOT use any chain-of-thought or latent thoughts. "
    "Follow the question's own answer-format requirement "
    "(e.g. a letter for multiple-choice, a number for counting, "
    "a short phrase for open-ended) exactly. "
    "Output ONLY the final answer and nothing else."
)

# fallback 时强制注入到 assistant turn 起始位置的字面量, 模型只能从冒号后
# 开始生成, 没空间再写一段新推理.
DEFAULT_ASSISTANT_PREFIX = "\n\nFinal Answer: "


# ============================================================================
# 事后 ngram 重复检测
# ============================================================================
def _post_hoc_ngram_repeat(token_ids: List[int],
                           ngram_size: int,
                           ngram_window: int,
                           ngram_repeat_thresh: int) -> Tuple[bool, str]:
    """事后扫描: 在 token_ids 末尾的 ngram_window 个 token 里, 任一 ngram
    出现次数 >= ngram_repeat_thresh 就视为病态重复.

    复杂度 O(W * n), W=window, n=ngram_size; 默认 W=256, n=8 → 每次扫几千次
    比较, 微秒级开销.
    """
    if (ngram_size <= 0 or ngram_repeat_thresh <= 0
            or len(token_ids) < ngram_size + ngram_repeat_thresh - 1):
        return False, ""
    window = token_ids[-ngram_window:] if ngram_window > 0 else token_ids
    if len(window) < ngram_size:
        return False, ""
    n = ngram_size
    # 把窗口里所有 ngram 哈希一遍, 取最高频
    counter: Counter = Counter()
    for i in range(0, len(window) - n + 1):
        counter[tuple(window[i:i + n])] += 1
    most_common = counter.most_common(1)
    if most_common and most_common[0][1] >= ngram_repeat_thresh:
        return True, (f"ngram_repeat: {n}-gram repeated "
                      f"{most_common[0][1]}x in last {len(window)} tokens")
    return False, ""


# ============================================================================
# 内部工具: 构造 inputs / 解码
# ============================================================================
def _build_inputs(processor,
                  messages: List[Dict[str, Any]],
                  device: torch.device,
                  assistant_prefix: str = ""):
    """把 messages 走 chat template + processor; 可选地在 assistant turn 起头
    强制追加 ``assistant_prefix``.

    Qwen 系列 chat template 在 ``add_generation_prompt=True`` 时末尾就是
    ``<|im_start|>assistant\\n``, 把 ``assistant_prefix`` 直接 string append
    到 text 末尾, tokenize 之后这些字面量就成了 assistant 的"已生成"前缀,
    模型只能从其后继续, 永远不可能再出现一段全新的 reasoning.
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if assistant_prefix:
        text = text + assistant_prefix
    if _process_vision_info is None:
        # 兜底: 没 qwen_vl_utils, messages 里若有 image 会失败
        image_inputs = None
    else:
        image_inputs, _ = _process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, return_tensors="pt",
    ).to(device)
    return inputs




def _decode_clean(processor, generated_ids: torch.Tensor, prompt_len: int) -> Tuple[str, str]:
    """把 ``generated_ids[0, prompt_len:]`` 解码出 "raw" (含 special token,
    用于诊断) 与 "clean" (通常用于答案匹配) 两个版本.

    clean 版本: special token 全部丢掉, 但保留 ``<|latent|> <|/latent|>``
    位置上的 ``" [latent thinking] "`` 痕迹 -- 这与原各脚本里的后处理保持
    一致 (例如 scripts/inference.py / scripts/test_samples_inference.py).
    """
    gen_tokens = generated_ids[0, prompt_len:]
    raw_output = processor.tokenizer.decode(gen_tokens, skip_special_tokens=False)
    # 把 latent 标记替换掉, 后处理与 inference.py 对齐
    clean_output = (
        raw_output
        .replace(LATENT_TOKEN, " [latent thinking] ")
        .replace(LATENT_END_TOKEN, " [/latent thinking] ")
    )
    # 去掉 EOS / 模板尾巴
    clean_output = clean_output.replace("<|im_end|>", "").strip()
    return raw_output, clean_output


def _make_fallback_messages(messages: List[Dict[str, Any]],
                            hint: str = DIRECTLY_ANSWER_HINT) -> List[Dict[str, Any]]:
    """把 directly-answer hint 拼到最后一条 user message 末尾, 不动 system /
    image / 其它历史消息. 浅拷贝 messages 列表, 深拷贝最后一条 user content.
    """
    if not messages:
        return messages
    new_msgs = list(messages)            # 浅 copy 列表
    # 找到最后一条 role=user 的消息
    last_user_idx = None
    for i in range(len(new_msgs) - 1, -1, -1):
        if new_msgs[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        # 没有 user 消息 (理论上不会), 直接附加一条
        new_msgs.append({"role": "user", "content": [{"type": "text", "text": hint.strip()}]})
        return new_msgs
    last_msg = new_msgs[last_user_idx]
    content = last_msg.get("content", "")
    if isinstance(content, str):
        new_msgs[last_user_idx] = {**last_msg, "content": content + hint}
    elif isinstance(content, list):
        # qwen_vl_utils 风格: [{"type":"image",...}, {"type":"text","text":...}, ...]
        # 找到最后一条 text item, 在其文本末尾加 hint; 找不到就 append 一条 text item
        new_content = list(content)
        last_text_idx = None
        for j in range(len(new_content) - 1, -1, -1):
            if new_content[j].get("type") == "text":
                last_text_idx = j
                break
        if last_text_idx is not None:
            old_item = new_content[last_text_idx]
            new_content[last_text_idx] = {**old_item, "text": old_item.get("text", "") + hint}
        else:
            new_content.append({"type": "text", "text": hint.strip()})
        new_msgs[last_user_idx] = {**last_msg, "content": new_content}
    else:
        # 未知 content 结构, 退化处理
        new_msgs[last_user_idx] = {
            **last_msg,
            "content": [{"type": "text", "text": str(content) + hint}],
        }
    return new_msgs


# ============================================================================
# 主入口
# ============================================================================
def infer_with_latent_fallback(
    model,
    processor,
    messages: List[Dict[str, Any]],
    *,
    # ---- 第一次推理参数 (透传给 model.generate) ----
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.9,
    do_sample: Optional[bool] = None,
    return_diagnostics: bool = False,
    # ---- fallback 开关与参数 ----
    enable_fallback: bool = True,
    fallback_assistant_prefix: str = DEFAULT_ASSISTANT_PREFIX,
    fallback_max_new_tokens: int = 32,
    # 触发条件 (任一命中即触发, 全部默认 ON)
    fallback_on_hit_max: bool = True,
    fallback_on_no_final_answer: bool = True,
    fallback_on_ngram_repeat: bool = True,
    pathology_ngram_size: int = 8,
    pathology_ngram_window: int = 256,
    pathology_ngram_repeat_thresh: int = 4,
    # ---- 杂项 ----
    final_answer_marker: str = "Final Answer:",
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Interlude 默认推理函数 (含 directly-answer fallback).

    Parameters
    ----------
    model : NLDModel
        已加载好权重并 to(device).eval() 的 NLDModel 实例.
    processor : AutoProcessor
        与 ``model`` 配套的 processor (tokenizer 已加 ``<|latent|>`` 等
        special token).
    messages : list of dict
        Qwen3-VL 风格的 chat messages, 例如::

            [{"role": "system", "content": [{"type": "text", "text": ...}]},
             {"role": "user",   "content": [{"type": "image", "image": ...},
                                            {"type": "text",  "text": ...}]}]

    max_new_tokens, temperature, top_p, do_sample : 透传 model.generate.
    return_diagnostics : 透传 model.generate; 若 True, 返回值额外包含
                        ``latent_diagnostics_first`` / ``latent_diagnostics_fallback``.
    enable_fallback : True (默认) -> 启用 fallback; False -> 行为退化为 1 次
                      标准 generate, 调用方拿到 ``{'text': ..., 'fallback_triggered': False}``.
    fallback_assistant_prefix : 命中 fallback 时强制注入到 assistant turn 头
                                的字面量, 默认 ``'\\n\\nFinal Answer: '``.
                                注意末尾**不要**加 ``\\n`` -- 模型紧跟着冒号后
                                即可吐出答案.
    fallback_max_new_tokens : fallback 推理的 max_new_tokens, 默认 32
                              (single letter / yes-no / number 答案绰绰有余,
                              且能顺带阻止模型再自作主张写一长段).
    fallback_on_* / pathology_* : 见各 bool / int 名字.
    final_answer_marker : 用于 "no_final_answer" 检测的字符串字面量;
                          下游 judge 通常也是搜索它.

    Returns
    -------
    dict 字段:
        text                 : str, 最终给下游 judge 用的 clean 文本
                              (若触发 fallback, 这里返回的是**fallback 那次**
                              的结果, 且开头会包含
                              ``fallback_assistant_prefix.strip() + ...`` 形式
                              的 'Final Answer: <answer>')
        raw_text             : str, decode(skip_special_tokens=False) 的版本
        first_text           : str, 第一次推理的 clean 文本 (诊断用)
        first_raw_text       : str, 第一次推理的 raw 文本
        fallback_triggered   : bool
        fallback_reason      : Optional[str], 'hit_max' / 'no_final_answer'
                               / 'ngram_repeat' / None
        first_num_new_tokens : int
        fallback_num_new_tokens : int (没触发则 0)
        gen_time_first_s     : float
        gen_time_fallback_s  : float
        gen_time_total_s     : float
        latent_diagnostics_first    : list (return_diagnostics=True 时)
        latent_diagnostics_fallback : list (return_diagnostics=True 时)
    """
    if device is None:
        device = next(model.parameters()).device
    if do_sample is None:
        do_sample = bool(temperature and temperature > 0)

    out: Dict[str, Any] = {
        'text': "",
        'raw_text': "",
        'first_text': "",
        'first_raw_text': "",
        'fallback_triggered': False,
        'fallback_reason': None,
        'first_num_new_tokens': 0,
        'fallback_num_new_tokens': 0,
        'gen_time_first_s': 0.0,
        'gen_time_fallback_s': 0.0,
        'gen_time_total_s': 0.0,
    }

    # ------------------------------------------------------------
    # 1) 第一次推理 (常规 prompt + CoT + latent)
    # ------------------------------------------------------------
    inputs1 = _build_inputs(processor, messages, device, assistant_prefix="")
    prompt_len1 = inputs1['input_ids'].shape[1]

    t0 = time.time()
    with torch.no_grad():
        gen_kwargs = dict(
            pixel_values=inputs1.get('pixel_values'),
            image_grid_thw=inputs1.get('image_grid_thw'),
            input_ids=inputs1['input_ids'],
            attention_mask=inputs1['attention_mask'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        # 兼容: 没图像的纯文本调用方
        if gen_kwargs['pixel_values'] is None:
            gen_kwargs.pop('pixel_values')
            gen_kwargs.pop('image_grid_thw', None)
        if return_diagnostics:
            gen_kwargs['return_diagnostics'] = True
        gen_out1 = model.generate(**gen_kwargs)
    out['gen_time_first_s'] = time.time() - t0

    if return_diagnostics and isinstance(gen_out1, tuple) and len(gen_out1) == 2:
        gen_ids1, latent_diag1 = gen_out1
        out['latent_diagnostics_first'] = latent_diag1
    else:
        gen_ids1 = gen_out1 if not isinstance(gen_out1, tuple) else gen_out1[0]
        if return_diagnostics:
            out['latent_diagnostics_first'] = []

    first_raw, first_clean = _decode_clean(processor, gen_ids1, prompt_len1)
    first_new_token_ids = gen_ids1[0, prompt_len1:].tolist()
    first_num_new = len(first_new_token_ids)
    out['first_text'] = first_clean
    out['first_raw_text'] = first_raw
    out['first_num_new_tokens'] = first_num_new

    # 默认: 第一次推理结果就是最终结果
    out['text'] = first_clean
    out['raw_text'] = first_raw

    if not enable_fallback:
        out['gen_time_total_s'] = out['gen_time_first_s']
        return out

    # ------------------------------------------------------------
    # 2) 病态检测 (事后)
    # ------------------------------------------------------------
    fb_reason: Optional[str] = None

    # (a) hit_max
    if fallback_on_hit_max and first_num_new >= max_new_tokens - 2:
        fb_reason = 'hit_max'

    # (b) no_final_answer
    if fb_reason is None and fallback_on_no_final_answer:
        if final_answer_marker not in first_clean:
            fb_reason = 'no_final_answer'

    # (c) ngram_repeat
    if fb_reason is None and fallback_on_ngram_repeat:
        hit, detail = _post_hoc_ngram_repeat(
            first_new_token_ids,
            ngram_size=pathology_ngram_size,
            ngram_window=pathology_ngram_window,
            ngram_repeat_thresh=pathology_ngram_repeat_thresh,
        )
        if hit:
            fb_reason = 'ngram_repeat'
            if verbose:
                print(f"[infer_with_latent_fallback] ngram detail: {detail}")

    if fb_reason is None:
        # 一切正常, 返回第一次结果
        out['gen_time_total_s'] = out['gen_time_first_s']
        return out

    # ------------------------------------------------------------
    # 3) Fallback: 换 prompt 重 infer 一次
    # ------------------------------------------------------------
    if verbose:
        print(f"[infer_with_latent_fallback] fallback triggered: reason={fb_reason}; "
              f"first_num_new_tokens={first_num_new}/{max_new_tokens}")
    out['fallback_triggered'] = True
    out['fallback_reason'] = fb_reason

    fb_messages = _make_fallback_messages(messages, hint=DIRECTLY_ANSWER_HINT)
    inputs2 = _build_inputs(
        processor, fb_messages, device,
        assistant_prefix=fallback_assistant_prefix,
    )
    # 计算"不带 prefix 的 prompt_len" -- 这样 decode 出来会自然包含
    # 'Final Answer: ...' 字面量, 与下游 judge 的 'Final Answer:' 搜索匹配.
    # 优化: 直接 tokenize prefix 字符串得到它的 token 数, 从 inputs2 的
    # 总长里减去, 避免再走一遍 vision encoder + apply_chat_template.
    if fallback_assistant_prefix:
        # add_special_tokens=False: prefix 是 chat template 内部插入的字面量,
        # 不应再被自动加 BOS / system prompt 等; 取得的 ids 长度才与
        # _build_inputs(... assistant_prefix=prefix) 里多出的那一段对齐.
        prefix_ids = processor.tokenizer(
            fallback_assistant_prefix, add_special_tokens=False,
        )['input_ids']
        prefix_len = len(prefix_ids)
    else:
        prefix_len = 0
    prompt_len2 = inputs2['input_ids'].shape[1] - prefix_len

    t1 = time.time()
    with torch.no_grad():
        gen_kwargs2 = dict(
            pixel_values=inputs2.get('pixel_values'),
            image_grid_thw=inputs2.get('image_grid_thw'),
            input_ids=inputs2['input_ids'],
            attention_mask=inputs2['attention_mask'],
            max_new_tokens=fallback_max_new_tokens,
            # 强烈倾向 deterministic: fallback 就是要给一个干脆的答案,
            # 不需要 sampling 多样性.
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        if gen_kwargs2['pixel_values'] is None:
            gen_kwargs2.pop('pixel_values')
            gen_kwargs2.pop('image_grid_thw', None)
        if return_diagnostics:
            gen_kwargs2['return_diagnostics'] = True
        gen_out2 = model.generate(**gen_kwargs2)
    out['gen_time_fallback_s'] = time.time() - t1

    if return_diagnostics and isinstance(gen_out2, tuple) and len(gen_out2) == 2:
        gen_ids2, latent_diag2 = gen_out2
        out['latent_diagnostics_fallback'] = latent_diag2
    else:
        gen_ids2 = gen_out2 if not isinstance(gen_out2, tuple) else gen_out2[0]
        if return_diagnostics:
            out['latent_diagnostics_fallback'] = []

    fb_raw, fb_clean = _decode_clean(processor, gen_ids2, prompt_len2)
    fb_num_new = gen_ids2.shape[1] - prompt_len2
    out['fallback_num_new_tokens'] = int(fb_num_new)
    out['text'] = fb_clean
    out['raw_text'] = fb_raw
    out['gen_time_total_s'] = out['gen_time_first_s'] + out['gen_time_fallback_s']
    return out


# ============================================================================
# 便捷函数: 给已有脚本一个最薄替换
# ============================================================================
def split_reasoning_and_answer(clean_text: str,
                               final_answer_marker: str = "Final Answer:") -> Tuple[str, str]:
    """与 inference.py / test_samples_inference.py 里的解析逻辑一致.

    用法::

        out = infer_with_latent_fallback(model, processor, messages, ...)
        reasoning, answer = split_reasoning_and_answer(out['text'])
    """
    text = clean_text.strip()
    if final_answer_marker in text:
        idx = text.rfind(final_answer_marker)
        reasoning = text[:idx].strip()
        answer = text[idx + len(final_answer_marker):].strip()
        answer = answer.replace("<|im_end|>", "").strip()
        return reasoning, answer
    return text, ""


# ============================================================================
# 给现有脚本的最小侵入式接入点
# ============================================================================
def generate_with_fallback(
    model,
    processor,
    messages: List[Dict[str, Any]],
    inputs,
    *,
    # 第一次推理参数
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.9,
    do_sample: Optional[bool] = None,
    return_diagnostics: bool = False,
    # fallback 控制
    enable_fallback: bool = True,
    # ---- L1 (拑制 latent 重生成, 同 prompt) ----
    level1_fallback_enabled: bool = True,
    # ---- L2 (directly-answer rewrite, 换 prompt) ----
    fallback_assistant_prefix: str = DEFAULT_ASSISTANT_PREFIX,
    fallback_max_new_tokens: int = 32,
    fallback_on_hit_max: bool = True,
    fallback_on_no_final_answer: bool = True,
    fallback_on_ngram_repeat: bool = True,
    pathology_ngram_size: int = 8,
    pathology_ngram_window: int = 256,
    pathology_ngram_repeat_thresh: int = 4,
    final_answer_marker: str = "Final Answer:",
    device: Optional[torch.device] = None,
    verbose: bool = False,
):
    """**最小侵入式接入点**: 给已有 eval / test 脚本用.

    现有脚本通常已经自己调好了 ``processor.apply_chat_template`` + processor
    并拿到 ``inputs`` 字典 (含 ``input_ids`` / ``pixel_values`` / ``image_grid_thw``
    / ``attention_mask``). 改成调用本函数, 只需要把那块::

        with torch.no_grad():
            gen_out = model.generate(
                pixel_values=inputs['pixel_values'],
                ...,
                max_new_tokens=max_new_tokens,
                ...,
                return_diagnostics=True,
            )
        if isinstance(gen_out, tuple):
            generated_ids, latent_diags = gen_out
        else:
            generated_ids, latent_diags = gen_out, []
        prompt_len = inputs['input_ids'].shape[1]

    替换为::

        from rld.inference_utils import generate_with_fallback
        generated_ids, prompt_len, latent_diags, fb_meta = generate_with_fallback(
            model, processor, messages, inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1, top_p=0.9, do_sample=False,
            return_diagnostics=True,
        )

    后续的 ``processor.tokenizer.decode(generated_ids[0, prompt_len:], ...)``
    / latent 统计 / 写日志逻辑**完全不需要改**: 命中 fallback 时返回的
    ``generated_ids`` / ``prompt_len`` 已经指向 fallback 那次的结果, 解出来
    会自然包含 'Final Answer: <answer>' 文本 (前缀靠 chat-template 注入).

    Returns
    -------
    generated_ids : torch.Tensor [1, T_total]
    prompt_len    : int, 该 generated_ids 对应的 prompt 长度
                    (decode 时用 ``generated_ids[0, prompt_len:]``)
    latent_diags  : list (return_diagnostics=True 时), 取 fallback 路径的;
                    没触发 fallback 时取第一次的; 否则空 list.
    fb_meta       : dict, fallback 元信息
        {
          'triggered': bool,
          'reason': Optional[str],     # 'hit_max'/'no_final_answer'/'ngram_repeat'/None
          'first_num_new_tokens': int,
          'fallback_num_new_tokens': int,
          'gen_time_first_s': float,
          'gen_time_fallback_s': float,
          'latent_diagnostics_first': list (return_diagnostics 时),
        }
    """
    if device is None:
        device = next(model.parameters()).device
    if do_sample is None:
        do_sample = bool(temperature and temperature > 0)

    fb_meta: Dict[str, Any] = {
        'triggered': False,
        'reason': None,
        'first_num_new_tokens': 0,
        'fallback_num_new_tokens': 0,
        'gen_time_first_s': 0.0,
        'gen_time_fallback_s': 0.0,
        # 新增: 两级 fallback 状态 (None / 'L1' / 'L2')
        'fallback_level': None,
        'l1_retry_triggered': False,
        'l1_num_new_tokens': 0,
        'l1_residual_reason': None,
        'gen_time_l1_s': 0.0,
    }

    prompt_len1 = inputs['input_ids'].shape[1]

    # ------------------------------------------------------------
    # 1) 第一次推理
    # ------------------------------------------------------------
    t0 = time.time()
    with torch.no_grad():
        gen_kwargs = dict(
            pixel_values=inputs.get('pixel_values'),
            image_grid_thw=inputs.get('image_grid_thw'),
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        if gen_kwargs['pixel_values'] is None:
            gen_kwargs.pop('pixel_values')
            gen_kwargs.pop('image_grid_thw', None)
        if return_diagnostics:
            gen_kwargs['return_diagnostics'] = True
        gen_out1 = model.generate(**gen_kwargs)
    fb_meta['gen_time_first_s'] = time.time() - t0

    if return_diagnostics and isinstance(gen_out1, tuple) and len(gen_out1) == 2:
        gen_ids1, latent_diag1 = gen_out1
    else:
        gen_ids1 = gen_out1 if not isinstance(gen_out1, tuple) else gen_out1[0]
        latent_diag1 = []
    fb_meta['latent_diagnostics_first'] = latent_diag1

    first_new_token_ids = gen_ids1[0, prompt_len1:].tolist()
    first_num_new = len(first_new_token_ids)
    fb_meta['first_num_new_tokens'] = first_num_new

    # ------------------------------------------------------------
    # 2) 病态检测 (事后)
    # ------------------------------------------------------------
    if not enable_fallback:
        return gen_ids1, prompt_len1, latent_diag1, fb_meta

    # 解码一次 first_clean 仅用于判断 'Final Answer:' 是否在里面
    first_raw, first_clean = _decode_clean(processor, gen_ids1, prompt_len1)

    fb_reason: Optional[str] = None
    if fallback_on_hit_max and first_num_new >= max_new_tokens - 2:
        fb_reason = 'hit_max'
    if fb_reason is None and fallback_on_no_final_answer:
        if final_answer_marker not in first_clean:
            fb_reason = 'no_final_answer'
    if fb_reason is None and fallback_on_ngram_repeat:
        hit, detail = _post_hoc_ngram_repeat(
            first_new_token_ids,
            ngram_size=pathology_ngram_size,
            ngram_window=pathology_ngram_window,
            ngram_repeat_thresh=pathology_ngram_repeat_thresh,
        )
        if hit:
            fb_reason = 'ngram_repeat'
            if verbose:
                print(f"[generate_with_fallback] ngram detail: {detail}")

    if fb_reason is None:
        return gen_ids1, prompt_len1, latent_diag1, fb_meta

    # ------------------------------------------------------------
    # 3) L1 fallback: 同 prompt + 拑制 latent 的重生成 (纯文本 CoT)
    # 丢弃 L0 输出, 在 model.generate 内部走 suppress_latent=True 路径,
    # 同时禁用其内部的二级 fallback (enable_fallback=False), 避免重复 L2.
    # ------------------------------------------------------------
    l1_reason: Optional[str] = fb_reason  # 默认: L1 未启用时, 吃 L0 原因进 L2
    if level1_fallback_enabled:
        if verbose:
            print(f"[generate_with_fallback] L0 pathology={fb_reason}; "
                  f"first_num_new_tokens={first_num_new}/{max_new_tokens}; "
                  f"→ launching L1 (suppress_latent retry, same prompt) ...")
        t_l1 = time.time()
        try:
            with torch.no_grad():
                gen_kwargs_l1 = dict(
                    pixel_values=inputs.get('pixel_values'),
                    image_grid_thw=inputs.get('image_grid_thw'),
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    suppress_latent=True,
                    enable_fallback=False,
                )
                if gen_kwargs_l1['pixel_values'] is None:
                    gen_kwargs_l1.pop('pixel_values')
                    gen_kwargs_l1.pop('image_grid_thw', None)
                if return_diagnostics:
                    gen_kwargs_l1['return_diagnostics'] = True
                gen_out_l1 = model.generate(**gen_kwargs_l1)
            fb_meta['gen_time_l1_s'] = time.time() - t_l1
            if return_diagnostics and isinstance(gen_out_l1, tuple) and len(gen_out_l1) == 2:
                gen_ids_l1, latent_diag_l1 = gen_out_l1
            else:
                gen_ids_l1 = gen_out_l1 if not isinstance(gen_out_l1, tuple) else gen_out_l1[0]
                latent_diag_l1 = []
            l1_new_token_ids = gen_ids_l1[0, prompt_len1:].tolist()
            l1_num_new = len(l1_new_token_ids)
            fb_meta['l1_retry_triggered'] = True
            fb_meta['l1_num_new_tokens'] = l1_num_new
            # 二次病态检测
            _, l1_clean = _decode_clean(processor, gen_ids_l1, prompt_len1)
            l1_pathology: Optional[str] = None
            if fallback_on_hit_max and l1_num_new >= max_new_tokens - 2:
                l1_pathology = 'hit_max'
            if l1_pathology is None and fallback_on_no_final_answer:
                if final_answer_marker not in l1_clean:
                    l1_pathology = 'no_final_answer'
            if l1_pathology is None and fallback_on_ngram_repeat:
                _hit, _detail = _post_hoc_ngram_repeat(
                    l1_new_token_ids,
                    ngram_size=pathology_ngram_size,
                    ngram_window=pathology_ngram_window,
                    ngram_repeat_thresh=pathology_ngram_repeat_thresh,
                )
                if _hit:
                    l1_pathology = 'ngram_repeat'
            fb_meta['l1_residual_reason'] = l1_pathology
            if l1_pathology is None:
                # L1 救回
                if verbose:
                    print(f"[generate_with_fallback] L1 ok, l1_num_new_tokens={l1_num_new}; "
                          f"→ returning L1 result")
                fb_meta['triggered'] = True
                fb_meta['reason'] = f"{fb_reason}_resolved_by_l1"
                fb_meta['fallback_level'] = 'L1'
                fb_meta['fallback_num_new_tokens'] = l1_num_new
                if return_diagnostics:
                    fb_meta['latent_diagnostics_l1'] = latent_diag_l1
                return gen_ids_l1, prompt_len1, latent_diag_l1, fb_meta
            else:
                # L1 仍病态 -> 走 L2; l1_reason 作为 L2 最终 reason 的依据
                if verbose:
                    print(f"[generate_with_fallback] L1 still pathology={l1_pathology} "
                          f"(l1_num_new={l1_num_new}); → falling through to L2")
                l1_reason = l1_pathology
        except Exception as l1_err:
            fb_meta['l1_retry_triggered'] = False
            fb_meta['l1_residual_reason'] = f'l1_failed:{type(l1_err).__name__}'
            if verbose:
                print(f"[generate_with_fallback] L1 retry failed: {l1_err}; "
                      f"→ falling through to L2")
            l1_reason = fb_reason

    # ------------------------------------------------------------
    # 4) L2 Fallback: 换 prompt 重 infer (directly-answer + 'Final Answer: ' prefix)
    # ------------------------------------------------------------
    if verbose:
        print(f"[generate_with_fallback] L2 (directly-answer) triggered: "
              f"L0_reason={fb_reason}, L1_reason={l1_reason}; "
              f"first_num_new_tokens={first_num_new}/{max_new_tokens}")
    fb_meta['triggered'] = True
    fb_meta['reason'] = l1_reason if l1_reason else fb_reason
    fb_meta['fallback_level'] = 'L2'

    fb_messages = _make_fallback_messages(messages, hint=DIRECTLY_ANSWER_HINT)
    inputs2 = _build_inputs(
        processor, fb_messages, device,
        assistant_prefix=fallback_assistant_prefix,
    )
    if fallback_assistant_prefix:
        prefix_ids = processor.tokenizer(
            fallback_assistant_prefix, add_special_tokens=False,
        )['input_ids']
        prefix_len = len(prefix_ids)
    else:
        prefix_len = 0
    prompt_len2 = inputs2['input_ids'].shape[1] - prefix_len

    t1 = time.time()
    with torch.no_grad():
        gen_kwargs2 = dict(
            pixel_values=inputs2.get('pixel_values'),
            image_grid_thw=inputs2.get('image_grid_thw'),
            input_ids=inputs2['input_ids'],
            attention_mask=inputs2['attention_mask'],
            max_new_tokens=fallback_max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        if gen_kwargs2['pixel_values'] is None:
            gen_kwargs2.pop('pixel_values')
            gen_kwargs2.pop('image_grid_thw', None)
        if return_diagnostics:
            gen_kwargs2['return_diagnostics'] = True
        gen_out2 = model.generate(**gen_kwargs2)
    fb_meta['gen_time_fallback_s'] = time.time() - t1

    if return_diagnostics and isinstance(gen_out2, tuple) and len(gen_out2) == 2:
        gen_ids2, latent_diag2 = gen_out2
    else:
        gen_ids2 = gen_out2 if not isinstance(gen_out2, tuple) else gen_out2[0]
        latent_diag2 = []
    fb_meta['fallback_num_new_tokens'] = int(gen_ids2.shape[1] - prompt_len2)
    return gen_ids2, prompt_len2, latent_diag2, fb_meta
