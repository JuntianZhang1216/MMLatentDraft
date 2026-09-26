"""
NativeLatentThinker: 基于 VLM 原生隐空间的自适应多步思考模块 (COCONUT 等价)

核心思想:
  在每个 step boundary 处，取最后一个 token 的 hidden state 作为 "thought token"，
  在隐空间中做多步步进演化 (recurrence)，等价于 COCONUT 的连续思考过程：
  - 历史信息全部通过 KV cache 提供 (thought token 通过 attention 看到所有历史)
  - 每步: thought_token → text_model() 整体调用 (全部 36 层 + norm) → new_hidden
  - 多步: new_hidden 作为下一步的输入 → 再过一遍全部层 (recurrence)
  最终输出: 全部步的 hidden states [B, num_steps, H]，作为 prefix 注入下一段自然语言推理

  与 COCONUT 的等价性:
  - COCONUT: last hidden → 过 VLM 全部层 → 新 hidden (不解码) → 循环
  - NLD: last hidden → text_model() (全部 36 层 + norm, with KV cache) → 新 hidden → 循环
  - 两者都: 把 Transformer 当 RNN cell 用，hidden state 循环反馈

  关键设计:
  - 单个 thought token: 不需要 "last N hidden"，历史信息全在 KV cache 中
  - thought token 只是一个 "查询探针"，通过 attention 访问所有历史
  - 多步演化 = 加深 Transformer 层数 (recurrence = depth)
  - KV 累积: 每一步 thought 的 KV 自动追加到 cache 中，
    后续步骤能通过 attention 看到之前所有 thought + 全部历史 token
  - 统一序列: thought tokens 作为特殊 token 永久存在于推理序列中，
    隐式推理 + 自然语言 CoT 构成完整的、连续的推理过程

  方案 B (梯度可回传, 训推一致):
  - 训练和推理都使用局部 cache (clone detached global cache)
    thought KV 写入局部 cache，不修改 global_cache
    thinker 完成后，thought_output 作为 prefix embedding 注入下一个 segment/token
    下一个 segment 的 text_model() 把 thought prefix + segment tokens 的 KV 一起写入 global_cache
  - 训练时: 梯度路径: CE loss → Segment hidden → inputs_embeds → thought_output → thinker → VLM
  - 推理时: thought_output 作为 prefix 再过一遍 text_model() 写入主 cache
  - 训推行为完全一致: thought tokens 在 global_cache 中的 KV 都是 text_model(thought_output) 产生的

  统一 forward 架构:
  - 训练和推理都使用 text_model() 整体调用 (而非手动逐层 forward)
  - KV cache 由 DynamicCache 自动管理
  - gradient checkpointing 由 text_model 内部正确处理
  - 代码简洁，训推一致

信号流:
  Step boundary 处:
  last_token_hidden [B, 1, H] (最后一个 token 的 hidden state, 已过 norm)
  → + step_embedding (标识当前是第几步隐空间演化)
  → text_model() 整体调用 (全部 36 层 + norm, with KV cache)
    (通过 KV cache attend 到所有历史 token + 之前所有 thought)
  → 输出 hidden [B, 1, H] (已过 norm)
  → KV 写入局部 cache (训练时) 或全局 cache (推理时)
  → 作为下一步的输入 (recurrence) 或退出
  → 退出判断: 退出 token 预测 OR 饱和信号 (推理时)
  → output_norm (额外 norm, 用于 prefix embedding)
  → 收集全部步的 hidden states → thought_output [B, num_steps, H]
  → 全部步的 hidden states 作为下一个 segment 的 prefix embeddings (训练和推理都一样)
  → prefix 过 text_model() 写入 global_cache (训推一致)
  
  Hidden States 监督 (SW-SRS, 参考 Laser arxiv 2601.06803):
  - Stage-Windowed Self-Refined Supervision:
      窗口 W_s = ∪_{s' >= s} K_{s'}  (本 stage + 所有 future stage 的 key tokens 并集)
      Self-Refined soft target: Q_s = softmax(StopGrad(z_s[W_s]) / τ)
      熵门控 hybrid: H(Q_s) > η 时混入 hard target (本 stage K_s 均匀)
      KL(P_target ‖ P_student) 仅在子词表 W_s 上计算
  - Exit Token Loss: 最后一步 hidden → lm_head (frozen) → 应预测 <|/latent|>
  - 不引入视觉侧监督 (cone 坍缩诊断后移除): 视觉对齐由 next-token CE 自然反传提供
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from transformers.cache_utils import DynamicCache

# path B2 (视觉侧 vision loss): role-driven slerp + 双 anchor margin
from .visual_anchor import compute_vision_loss as _compute_vision_loss_fn


def compute_saturation(
    prev_hidden: torch.Tensor,
    curr_hidden: torch.Tensor,
) -> torch.Tensor:
    """
    计算相邻两步 hidden state 的饱和度 (余弦相似度)
    
    饱和度 ≈ 1.0 表示 hidden state 不再变化 (充分条件: 可以退出)
    饱和度 ≈ 0.0 表示 hidden state 仍在剧烈变化
    
    Args:
        prev_hidden: [B, 1, H] 上一步的 hidden state
        curr_hidden: [B, 1, H] 当前步的 hidden state
    
    Returns:
        saturation: [B] 饱和度 ∈ [-1, 1]，通常 ∈ [0.8, 1.0]
    """
    prev = prev_hidden.squeeze(1)  # [B, H]
    curr = curr_hidden.squeeze(1)  # [B, H]
    return F.cosine_similarity(prev, curr, dim=-1)  # [B]


class NativeLatentThinker(nn.Module):
    """
    原生隐空间思考器: COCONUT 等价的隐空间推理 (单 token recurrence, 全层)
    
    核心设计:
    1. 单个 thought token — 最后一个 token 的 hidden state 作为 "查询探针"
    2. 历史信息全部通过 KV cache 提供 — thought token 通过 attention 访问所有历史
    3. 多步演化 = recurrence — 输出 hidden 作为下一步输入，再过一遍全部层
    4. 不过 lm_head 解码 — 纯隐空间步进
    5. 复用 VLM 的权重 — 利用 VLM 已有的推理能力
    
    统一 forward 架构:
    - 使用 text_model() 整体调用替代手动逐层 forward
    - KV cache 由 DynamicCache 自动管理
    - gradient checkpointing 由 text_model 内部正确处理
    - 训推一致: 训练和推理使用完全相同的 forward 路径
    
    退出策略 (简化版, 无 ExitGate):
    - 训练时: 固定执行 target_steps 步 (由数据中 latent_key_tokens 的 stage 数决定)
    - 推理时: 退出 token 预测 OR 饱和信号 (双保险) OR max_think_steps 硬上限
    
    Hidden States 监督:
    - Key Token Loss (Soft Concept Anchoring): hidden → lm_head (frozen) → logit distribution
      与 key tokens 的 embedding 语义邻域构成的 soft target 做 KL 散度对齐
    - Exit Token Loss: 最后一步 hidden → lm_head (frozen) → 应预测 <|/latent|>
    - lm_head 在这两个 loss 中冻结 (detach)，只塑形 hidden states
    
    与 COCONUT 的等价性:
    - COCONUT: last hidden → 过 VLM 全部层 → 新 hidden (不解码) → 循环
    - NLD: last hidden → text_model() (全部 36 层 + norm) → 新 hidden (不解码) → 循环
    - 两者都: 把 Transformer 当 RNN cell 用，hidden state 循环反馈
    - 关键: thought token 只有 1 个，历史信息全在 KV cache 中
    - KV 累积: 每步 thought KV 追加到 cache，后续步可 attend 所有之前的 thought + 历史 token
    
    Args:
        hidden_size: VLM 的隐藏维度 (如 4096)
        max_think_steps: 最大隐空间演化步数 (硬上限, 无论任何信号都强制退出, 如 4~8)
        saturation_exit_threshold: 推理时饱和信号的退出阈值
        latent_end_token_id: <|/latent|> token id，用于退出预测
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        max_think_steps: int = 4,
        saturation_exit_threshold: float = 0.99,
        latent_end_token_id: Optional[int] = None,
        # 以下参数保留接口兼容性，但不再使用
        adaptive_exit: bool = True,
        exit_threshold: float = 0.5,
        ponder_cost_weight: float = 0.01,
        saturation_aux_weight: float = 0.005,
        # ====== 3 层 anti-collapse 修复超参 ======
        # Layer 1 (Soft Anti-Premature-Exit): 中间步软约束, 让 <|/latent|> 的 logit 必须低于其他 token 至少 margin
        exit_margin: float = 2.0,
        exit_margin_weight: float = 0.5,
        # Layer 2 (SW-SRS 增强 + alpha schedule):
        #   - sw_srs_alpha:   hard target 在 SW-SRS 中的混合权重 (默认 0.3, 可在 yaml 调高到 0.7 以拽出塌缩态)
        #   - swsrs_anti_collapse_weight: SW-SRS 内部 anti-collapse loss 的权重 (惩罚 hidden 在 W_s 上的 mass 低于 <|/latent|>)
        sw_srs_alpha: float = 0.3,
        swsrs_anti_collapse_weight: float = 0.5,
        swsrs_anti_collapse_margin: float = 1.0,
        # Layer 3 (Stage-Diversity Regularizer): 跨步 hidden 不能复读
        diversity_threshold: float = 0.95,
        diversity_weight: float = 1.0,
        # ====== 语义侧监督 loss 模式开关 ======
        # 'laser_dwal' (默认, 推荐): 严格对齐 Laser 官方实现 (forward_dwal.py)
        #               student = 全词表 log_softmax,
        #               teacher = stopgrad(同 logit 在 W_s 子集 softmax) → 当作 K 个未来
        #                         token 的 CE 权重 (weighted-CE), 非 KL.
        #               天然不退化, 与 Laser 论文 Sec 3.3 公式一致.
        # 'stage_kw'  : 替代实现 — teacher 用 K_s 上均匀的外部 hard anchor + KL on W_s.
        #               (与 student 不同源, KL 不退化, 但梯度形式与 Laser 不同).
        # 'sw_srs'    : 旧版 self-distill KL on W_s (KL=0 trivially 退化, 仅保留向后兼容).
        loss_mode: str = 'laser_dwal',
        # ====== path B2 视觉侧 vision loss 超参 (零可学参数) ======
        # vision_loss_weight: 本地权重, 0 表示关闭 (向后兼容, 默认 0).
        #                     model_v2 侧还有 total_w 乘上, 供两级调丰.
        # vision_top_k:       视觉 token 检索 top-K 均值 (默认 6).
        # vision_margin:      双 anchor margin loss 的 δ (默认 0.05).
        vision_loss_weight: float = 0.0,
        vision_top_k: int = 6,
        vision_margin: float = 0.05,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_think_steps = max_think_steps
        self.saturation_exit_threshold = saturation_exit_threshold
        self.latent_end_token_id = latent_end_token_id  # <|/latent|> token id，用于退出预测

        # 保留接口兼容性 (不再实际使用)
        self.adaptive_exit = adaptive_exit

        # ====== 3 层 anti-collapse 修复超参 (forward 时使用) ======
        self.exit_margin = float(exit_margin)
        self.exit_margin_weight = float(exit_margin_weight)
        self.sw_srs_alpha = float(sw_srs_alpha)
        self.swsrs_anti_collapse_weight = float(swsrs_anti_collapse_weight)
        self.swsrs_anti_collapse_margin = float(swsrs_anti_collapse_margin)
        self.diversity_threshold = float(diversity_threshold)
        self.diversity_weight = float(diversity_weight)
        # ====== 语义侧监督 loss mode ======
        self.loss_mode = str(loss_mode).lower().strip()
        if self.loss_mode not in ('laser_dwal', 'sw_srs', 'stage_kw'):
            print(f"[NLD-Thinker] ⚠️ unknown loss_mode={loss_mode!r}, fallback to 'laser_dwal'")
            self.loss_mode = 'laser_dwal'
        if self.loss_mode == 'sw_srs':
            print("[NLD-Thinker] ⚠️ loss_mode='sw_srs' (KL self-distill) is deprecated due to "
                  "KL(P, stopgrad(P))=0 degeneracy. Recommend 'laser_dwal' (default) or 'stage_kw'.")

        # ====== path B2 视觉侧 vision loss 配置 ======
        self.vision_loss_weight = float(vision_loss_weight)
        self.vision_top_k = int(vision_top_k)
        self.vision_margin = float(vision_margin)
        
        # ====== 1. 步骤编码 (多步演化: 让模型知道当前是第几步) ======
        # 使用较大的 embedding 表大小，避免 max_think_steps 限制训练时的实际步数
        # 训练时 target_steps 可能超过 max_think_steps，需要足够的 step embedding
        self._step_embed_size = max(max_think_steps, 32)  # 至少 32，覆盖绝大多数场景
        if self._step_embed_size > 1:
            self.step_embedding = nn.Embedding(self._step_embed_size, hidden_size)
            nn.init.normal_(self.step_embedding.weight, std=0.02)
        else:
            self.step_embedding = None
        
        # ====== 2. 输出 Norm ======
        # thought_output 直接作为 prefix embedding 注入 VLM
        # text_model() 返回的 last_hidden_state 已过 final norm，
        # 这里的 output_norm 是额外的 norm，用于将 thought hidden 转换为 prefix embedding
        self.output_norm = nn.RMSNorm(hidden_size)

        # ====== 3. SW-SRS / stage_kw 的 W_t 排除集合 (T2': 贴合 LASER paper) ======
        # LASER (arxiv 2601.06803, Sec 3.3) 显式把 <laser_end> 排除在 W_t 外,
        # 避免 student 把概率塞给特殊 token. 我们的对应物是:
        #   <|latent|> / <|/latent|> / <|pause|> / bos / eos / pad / unk /
        #   <|im_start|> / <|im_end|> / image_token_id 等.
        # 由 model_v2.set_processor() 在 tokenizer 就绪后注入到此 buffer.
        # None / 空 tensor 表示不做排除 (向后兼容老 checkpoint).
        self.register_buffer(
            "_sw_srs_excluded_ids",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )

        # ====== 统计 ======
        self._init_param_count()
    
    def _init_param_count(self):
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._total_params = total
        self._trainable_params = trainable
    
    @staticmethod
    def _clone_cache_detached(cache) -> DynamicCache:
        """Clone KV cache 并 detach 所有 tensor，用于创建局部 cache。
        
        thinker 内部多步演化时使用局部 cache，不修改传入的 global_cache。
        局部 cache 包含 global_cache 的 detached 副本（提供历史 context），
        thinker 的 thought KV 追加到局部 cache 中。
        
        Args:
            cache: 需要 clone 的 DynamicCache (global_cache)
        
        Returns:
            新的 DynamicCache，所有 KV tensor 已 detach (不影响原 cache)
        """
        from transformers.cache_utils import DynamicLayer as _DL
        cloned = DynamicCache()
        for layer in cache.layers:
            new_layer = _DL()
            if hasattr(layer, 'keys') and layer.keys is not None:
                new_layer.keys = layer.keys.detach()
                new_layer.values = layer.values.detach()
                new_layer.is_initialized = True
            cloned.layers.append(new_layer)
        return cloned
    
    def _run_think_step(
        self,
        thought_hidden: torch.Tensor,
        text_model: nn.Module,
        step_idx: int,
        past_key_values,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        执行单步隐空间演化 (COCONUT 等价):
        thought_hidden [B, 1, H] → text_model() 整体调用 → updated [B, 1, H]
        
        使用 text_model() 整体调用替代手动逐层 forward:
        - KV cache 由 DynamicCache 自动管理 (每步 thought KV 自动追加)
        - gradient checkpointing 由 text_model 内部正确处理
        - attention mask 由 text_model 内部自动构建
        
        关键: 通过 past_key_values 看到所有历史 token，
        这等价于 COCONUT 中 hidden state 能看到所有之前的 context。
        thought token 只是一个 "查询探针"，通过 attention 访问所有历史信息。
        
        Args:
            thought_hidden: [B, 1, H] 当前 thought token 的 hidden state
            text_model: VLM 的 language_model (整体调用)
            step_idx: 当前步骤索引 (用于步骤编码)
            past_key_values: 历史 KV cache (DynamicCache, 所有历史 token 的 context)
            attention_mask: attention mask (通常为 None, 让 text_model 自动处理)
            cache_position: cache 中的位置
            position_ids: position ids (4D: [text_pos + mrope_3d])
        
        Returns:
            updated_hidden: [B, 1, H] 更新后的 thought token hidden state (已过 norm)
        """
        device = thought_hidden.device
        hidden = thought_hidden
        
        # 加入步骤编码 (让模型知道当前是第几步演化)
        if self.step_embedding is not None:
            # 如果 step_idx 超出 embedding 表大小，使用最后一个 embedding (安全兜底)
            clamped_idx = min(step_idx, self._step_embed_size - 1)
            step_emb = self.step_embedding(
                torch.tensor(clamped_idx, device=device)
            )  # [H]
            hidden = hidden + step_emb.unsqueeze(0).unsqueeze(0)  # [B, 1, H] + [1, 1, H]
        
        # 使用 text_model() 整体调用 (全部 36 层 + norm)
        # KV cache 自动管理: thought token 的 KV 自动追加到 past_key_values
        # gradient checkpointing: text_model 内部正确处理 (不会导致 KV 重复追加)
        outputs = text_model(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
        )
        
        # last_hidden_state 已过 final norm
        updated_hidden = outputs.last_hidden_state  # [B, 1, H]
        
        return updated_hidden
    
    def forward(
        self,
        last_hidden: torch.Tensor,
        text_model: nn.Module,
        past_key_values,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb_fn=None,
        mrope_position_ids=None,
        base_thought_pos: Optional[int] = None,
        B: Optional[int] = None,
        target_steps: Optional[int] = None,
        lm_head: Optional[nn.Module] = None,
        embed_tokens: Optional[nn.Module] = None,
        stage_key_token_ids: Optional[List[List[List[int]]]] = None,
        # ---- TGVR 档 A (诊断版, 无 loss): stage key tokens 召回视觉特征 ----
        # image_embeds_per_sample: List[Tensor[N_i, H]], 每个样本的视觉 embedding
        #   (已过 vision encoder + projector, 与 lm hidden 同维度).
        #   若为 None 则跳过 TGVR 诊断 (向后兼容, 推理路径).
        image_embeds_per_sample: Optional[List[torch.Tensor]] = None,
        # ---- path B2 视觉侧 vision loss: 每个 stage 的 role (abstract/bridge/unified/concrete) ----
        # 仅 vision_loss_weight > 0 时生效; 为 None 则默认 alpha=0.5 (中点).
        stage_roles: Optional[List[List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        COCONUT 等价的隐空间推理 (单 token recurrence, text_model() 整体调用)
        
        统一 forward 架构:
        - 使用 text_model() 整体调用替代手动逐层 forward
        - KV cache 由 DynamicCache 自动管理
        - 训推一致: 训练和推理使用完全相同的 forward 路径
        
        退出策略 (简化版, 无 ExitGate):
        - 训练时: 固定执行 target_steps 步 (由数据中 latent_key_tokens 的 stage 数决定)
        - 推理时: 退出 token 预测 OR 饱和信号 OR max_think_steps 硬上限
        
    Hidden States 监督 (Soft Concept Anchoring + Exit Token):
        - Key Token Loss (SCA): 每步 hidden 过 lm_head (frozen) 后的 logit distribution
          与 key tokens 的 embedding 语义邻域构成的 soft target 做 KL 散度对齐 (Top-K 近似)
        - Exit Token Loss: 最后一步 hidden 过 lm_head (frozen) 后应预测 <|/latent|>
        - lm_head 在这两个 loss 中冻结 (detach)，只塑形 hidden states
        
        Args:
            last_hidden: [B, 1, H] 最后一个 token 的 hidden state (已过 norm)
            text_model: VLM 的 language_model (整体调用)
            past_key_values: 历史 KV cache (DynamicCache)
            attention_mask: attention mask
            cache_position: 第一步 thought token 在 cache 中的位置
            position_ids: thought token 的 position ids (4D)
            rotary_emb_fn: VLM 的 rotary_emb 函数
            mrope_position_ids: [3, B, 1] 第一步的 mrope position ids
            base_thought_pos: int, 第一步 thought 的起始位置
            B: batch size
            target_steps: 目标迭代步数 (训练时由数据决定, 推理时为 None)
            lm_head: VLM 的 lm_head (用于 key token loss 和退出 token 预测)
            embed_tokens: VLM 的 input embedding 层 (用于 Soft Concept Anchoring 构建 soft target)
            stage_key_token_ids: [B][stage][token_ids] 每个 batch 样本每个 stage 的关键词 token ids
        
        Returns:
            dict 包含:
            - thought_output: [B, num_steps, H] 全部步的隐空间思考结果
            - exit_token_loss: scalar, 退出 token 预测 loss
            - key_token_loss: scalar, Key Token Loss (Soft Concept Anchoring via KL divergence)
            - exit_stats: dict, 退出统计信息
            - num_thought_steps: int, 实际执行的思考步数
        """
        if B is None:
            B = last_hidden.shape[0]
        device = last_hidden.device
        dtype = last_hidden.dtype
        
        # ---- 单个 thought token: 最后一个 token 的 hidden state ----
        thought_hidden = last_hidden  # [B, 1, H]
        
        # ---- 方案 B: 始终使用局部 cache，不修改 global cache ----
        local_cache = self._clone_cache_detached(past_key_values)
        think_cache = local_cache
        
        # ---- 确定实际迭代步数 ----
        # 训练时: 完全由 target_steps 决定 (数据中几个 stage 就训练几步，不受 max_think_steps 截断)
        # 推理时: max_think_steps 作为安全上限 (配合自适应退出)
        if self.training and target_steps is not None and target_steps > 0:
            effective_steps = target_steps  # 训练时不截断，完全由数据决定
        elif target_steps is not None and target_steps > 0:
            effective_steps = target_steps  # 推理时也优先使用指定步数
        else:
            effective_steps = self.max_think_steps  # 推理时无指定步数，使用 max_think_steps
        
        # ---- 当前步的 cache_position 和 position_ids (每步递增) ----
        current_cache_pos = cache_position
        current_position_ids = position_ids
        
        # ---- 辅助函数: 构建某一步的 position_ids (4D: text_pos + mrope_3d) ----
        def _build_step_position_ids(step_idx):
            """为第 step_idx 步构建 position_ids [4, B, 1]"""
            step_pos = base_thought_pos + step_idx
            step_positions = torch.arange(step_pos, step_pos + 1, device=device)
            step_cache_pos = step_positions
            
            # text_pos: [B, 1]
            text_pos = torch.full((B, 1), step_pos, device=device, dtype=torch.long)
            
            # mrope: [3, B, 1]
            if mrope_position_ids is not None:
                step_mrope = mrope_position_ids[:, :, :1] + step_idx  # [3, B, 1]
            else:
                step_mrope = step_positions.unsqueeze(0).unsqueeze(0).expand(3, B, -1)
            
            # 拼接为 [4, B, 1]: [text_pos, mrope_3d]
            pos_ids = torch.cat([text_pos.unsqueeze(0), step_mrope], dim=0)  # [4, B, 1]
            return step_cache_pos, pos_ids
        
        # ---- 多步隐空间演化 (Coconut recurrence, KV 累积) ----
        all_step_outputs = []
        all_saturations = []
        prev_hidden_for_saturation = thought_hidden.detach()
        
        for step_idx in range(effective_steps):
            # 为当前步生成 position_ids 和 cache_position (每步递增位置)
            if step_idx > 0:
                current_cache_pos, current_position_ids = _build_step_position_ids(step_idx)
            
            # 构建 attention_mask
            step_pos = base_thought_pos + step_idx
            step_attn_mask = torch.ones(B, step_pos + 1, device=device, dtype=torch.long)
            
            # 单步演化: thought_hidden → text_model() → new_hidden
            thought_hidden = self._run_think_step(
                thought_hidden, text_model, step_idx,
                past_key_values=think_cache,
                attention_mask=step_attn_mask,
                cache_position=current_cache_pos,
                position_ids=current_position_ids,
            )
            all_step_outputs.append(thought_hidden)
            
            # 计算饱和度
            saturation = compute_saturation(
                prev_hidden_for_saturation, thought_hidden
            )  # [B]
            all_saturations.append(saturation)
            prev_hidden_for_saturation = thought_hidden.detach()
            
            # ---- 推理时: 退出 token 预测 OR 饱和信号 (双保险) ----
            if not self.training and step_idx < effective_steps - 1:
                saturation_exit = (saturation > self.saturation_exit_threshold).all()
                
                # 退出 token 检测: 如果 lm_head 预测出 <|/latent|>，退出
                token_exit = False
                if (lm_head is not None and self.latent_end_token_id is not None):
                    with torch.no_grad():
                        step_logits = lm_head(thought_hidden[:, -1:, :])  # [B, 1, V]
                        predicted_token = step_logits.argmax(dim=-1).squeeze(-1)  # [B]
                        token_exit = (predicted_token == self.latent_end_token_id).all().item()
                
                if saturation_exit or token_exit:
                    break
        
        actual_steps = len(all_step_outputs)
        
        # ---- 输出全部步的 hidden states ----
        normed_outputs = [self.output_norm(out) for out in all_step_outputs]
        thought_output = torch.cat(normed_outputs, dim=1)  # [B, actual_steps, H]
        
        # ---- 统一计算 step_logits (Exit Token Loss + Key Token Loss 共用) ----
        # ⚠️ 关键修复 (FSDP full_shard 兼容性):
        #   不要直接访问 `lm_head.weight.detach()` 做 `F.linear(...)`!
        #   在 FSDP full_shard (use_orig_params=True) 下, 模块的 .weight 属性在
        #   forward() 调用外是分片状态 (含 padding/未初始化内存), 直接访问会拿到
        #   垃圾值. 必须走 `lm_head(...)` 正常调用路径, 让 FSDP 自动触发
        #   AllGather 得到完整权重, 再做矩阵乘法.
        #
        #   上一版 `F.linear(hidden, lm_head.weight.detach())` 会导致:
        #     1) 第一次 forward 的 step_logits 已经含 NaN/极端值
        #     2) NaN 通过 exit_token_loss / sw_srs_loss 回传 backward
        #     3) FSDP reduce-scatter 把 NaN 梯度广播到所有参数
        #     4) 一个 optimizer step 后 vision tower / lm_head / embed_tokens 全部
        #        被污染成 NaN, 从此所有 forward 永久 NaN.
        #
        #   此处不再 detach lm_head.weight: 梯度会自然回传到 lm_head (与 next-token
        #   CE 的梯度共享同一条链路), 等于把 lm_head 交给主 CE + SW-SRS 联合塑形,
        #   这与 "lm_head 冻结" 的原始意图相比是放松的, 但与 FSDP 兼容, 且
        #   全量微调场景下 lm_head 本就是可训练的, 不会引入额外问题.
        exit_token_loss = torch.tensor(0.0, device=device, dtype=dtype)
        key_token_loss = torch.tensor(0.0, device=device, dtype=dtype)
        # ====== 3 层 anti-collapse 新增 loss ======
        exit_margin_loss = torch.tensor(0.0, device=device, dtype=dtype)
        diversity_loss = torch.tensor(0.0, device=device, dtype=dtype)
        swsrs_anti_collapse_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        if (self.training and lm_head is not None and actual_steps > 0):
            all_hiddens = torch.cat(all_step_outputs, dim=1)  # [B, actual_steps, H]
            
            # ✅ 走 lm_head(...) 正常调用, 触发 FSDP AllGather, 得到完整 logits
            step_logits = lm_head(all_hiddens)  # [B, actual_steps, V]
            
            # ---- Exit Token Loss: 最后一步应预测 <|/latent|> ----
            if self.latent_end_token_id is not None:
                exit_targets = torch.full(
                    (B, actual_steps), -100, device=device, dtype=torch.long
                )
                exit_targets[:, -1] = self.latent_end_token_id  # 最后一步 → <|/latent|>
                
                # 升 fp32 计算 CE (数值稳定, 防 bf16 下 log_softmax 溢出导致 NaN)
                exit_token_loss = F.cross_entropy(
                    step_logits.float().view(-1, step_logits.size(-1)),
                    exit_targets.view(-1),
                    ignore_index=-100,
                )

                # ====== Layer 1: Soft Anti-Premature-Exit ======
                # 中间步 (step 0 ~ S-2) 不应过早预测 <|/latent|>
                # 软约束: 让 logit_<|/latent|> < max(other_logits) - margin
                # 这不是绝对禁止 exit, 只是"中间步没确信就别 exit"
                # 当 hidden 真的演化到结束时, 自然 end_logit 会高于其他 token, loss → 0
                if actual_steps > 1 and self.exit_margin_weight > 0:
                    end_id = self.latent_end_token_id
                    mid_logits = step_logits[:, :-1, :].float()  # [B, S-1, V]
                    end_logit = mid_logits[..., end_id]          # [B, S-1]
                    # 排除 end_id 后求 max (用 scatter mask 把 end_id 列置 -inf)
                    mid_logits_masked = mid_logits.clone()
                    mid_logits_masked[..., end_id] = -1e9
                    other_max_logit = mid_logits_masked.max(dim=-1).values  # [B, S-1]
                    # margin loss: 期望 end_logit + margin < other_max
                    # = ReLU(end_logit - other_max + margin)
                    exit_margin_loss = F.relu(
                        end_logit - other_max_logit + self.exit_margin
                    ).mean().to(dtype)

            # 语义侧监督 — 参考 Laser arxiv 2601.06803 (forward_dwal.py 官方实现)
            # 通过 self.loss_mode 切换三种实现:
            #   - 'laser_dwal' (默认): student 全词表 log_softmax + teacher 子集 softmax 当 weighted-CE 权重,
            #                          严格对齐 Laser 官方公式, 不退化.
            #   - 'stage_kw'         : teacher = K_s 上外部均匀 hard anchor + KL on W_s, 不退化.
            #   - 'sw_srs'           : 旧版 KL self-distill on W_s, 数学上 KL=0 退化 (仅向后兼容).
            if stage_key_token_ids is not None:
                if self.loss_mode == 'stage_kw':
                    key_token_loss = self._compute_stage_keyword_distill_loss(
                        step_logits, stage_key_token_ids, B, actual_steps, device, dtype,
                    )
                elif self.loss_mode == 'sw_srs':
                    # Deprecated: 旧版 KL self-distill (退化), 仅保留向后兼容
                    key_token_loss = self._compute_sw_srs_loss(
                        step_logits, stage_key_token_ids, B, actual_steps, device, dtype,
                        alpha=self.sw_srs_alpha,
                        loss_form='kl',
                    )
                else:
                    # 默认 'laser_dwal': Laser 官方 weighted-CE
                    key_token_loss = self._compute_sw_srs_loss(
                        step_logits, stage_key_token_ids, B, actual_steps, device, dtype,
                        alpha=self.sw_srs_alpha,
                        loss_form='weighted_ce',
                    )
                # ====== Layer 2 增强: SW-SRS Anti-Collapse ======
                # 期望 stage keys 的总 logsumexp > <|/latent|> 的 logit
                # 这强制 hidden 在 stage keys 上的 mass 高于 exit token
                # 直接禁止 hidden 塌缩到 exit (而 SW-SRS 自蒸馏在塌缩态下 KL=0 拦不住)
                if self.latent_end_token_id is not None and self.swsrs_anti_collapse_weight > 0:
                    swsrs_anti_collapse_loss = self._compute_swsrs_anti_collapse_loss(
                        step_logits, stage_key_token_ids, B, actual_steps,
                        device, dtype,
                    )

            # ====== Layer 3: Stage-Diversity Regularizer ======
            # 期望任意两步 hidden 的 cos 相似度 < diversity_threshold
            # 直接对应"hidden 必须演化, 不能复读"的物理直觉
            # 在塌缩态时 cos_ij ≈ 1.0, diversity_loss 会很大, 提供真实反塌缩梯度
            if actual_steps >= 2 and self.diversity_weight > 0:
                _diversity_pairs = []
                for _i in range(actual_steps):
                    h_i = all_step_outputs[_i].squeeze(1)  # [B, H]
                    for _j in range(_i + 1, actual_steps):
                        h_j = all_step_outputs[_j].squeeze(1)  # [B, H]
                        cos_ij = F.cosine_similarity(
                            h_i.float(), h_j.float(), dim=-1
                        )  # [B]
                        # 软约束: cos > threshold 时罚
                        _diversity_pairs.append(
                            F.relu(cos_ij - self.diversity_threshold)
                        )
                if _diversity_pairs:
                    diversity_loss = torch.stack(_diversity_pairs, dim=-1).mean().to(dtype)
        
        # ---- 退出统计 ----
        with torch.no_grad():
            exit_reason = "training" if self.training else "max_steps"
            if not self.training and actual_steps < effective_steps:
                # 检查退出原因
                _token_predicted = False
                if (lm_head is not None and self.latent_end_token_id is not None):
                    _last_logits = lm_head(all_step_outputs[-1])
                    _pred = _last_logits.argmax(dim=-1).squeeze(-1)
                    _token_predicted = (_pred == self.latent_end_token_id).all().item()
                
                if _token_predicted:
                    exit_reason = "exit_token"
                elif all_saturations and (all_saturations[-1] > self.saturation_exit_threshold).all():
                    exit_reason = "saturation"
                else:
                    exit_reason = "mixed_signal"
        
        exit_stats = {
            'mean_steps': float(actual_steps),
            'saturations': [sat.detach() for sat in all_saturations],
            'actual_steps': actual_steps,
            'exit_reason': exit_reason,
        }
        
        # ---- Hidden 几何统计 (anti-collapse 监控) ----
        # 核心指标:
        #   - h_norm_mean / h_batch_cos_mean / h_effective_rank: 基于最后一步 hidden
        #   - h_first_last_cos: step_0 vs step_last 宏观演化幅度
        #   - h_adj_cos_mean/min/max: 相邻步 cos 序列 (细粒度, 定位"第几步原地踏步")
        hidden_stats = {}
        with torch.no_grad():
            if actual_steps > 0:
                # 取最后一步 hidden: [B, 1, H] → [B, H]
                _last_h = all_step_outputs[-1].squeeze(1).float()  # [B, H]
                # 1) L2 范数 (监测数值爆炸/消失)
                hidden_stats['h_norm_mean'] = _last_h.norm(dim=-1).mean().item()
                # 2) Batch 内两两 cos (跨样本同构坍塌) + Effective rank (维度坍塌) — 仅当 B >= 2
                if B >= 2:
                    _h_normed = F.normalize(_last_h, p=2, dim=-1)  # [B, H]
                    _cos_mat = _h_normed @ _h_normed.t()  # [B, B]
                    _triu_mask = torch.triu(torch.ones(B, B, device=device), diagonal=1).bool()
                    hidden_stats['h_batch_cos_mean'] = _cos_mat[_triu_mask].mean().item()
                    try:
                        _s = torch.linalg.svdvals(_last_h)  # min(B, H) 个奇异值
                        _p = _s.pow(2) / _s.pow(2).sum().clamp(min=1e-12)
                        _p_c = _p.clamp(min=1e-12)
                        hidden_stats['h_effective_rank'] = (-(_p_c * _p_c.log()).sum()).exp().item()
                    except Exception:
                        hidden_stats['h_effective_rank'] = -1.0
                # 3) 跨步演化幅度 (step_0 vs step_last) — 验证迭代是否有效, 而非原地踏步
                if actual_steps >= 2:
                    _first_h = all_step_outputs[0].squeeze(1).float()
                    hidden_stats['h_first_last_cos'] = F.cosine_similarity(
                        _first_h, _last_h, dim=-1
                    ).mean().item()

                    # 4) 相邻步 cos 序列 (细粒度 anti-collapse 信号)
                    # 对每对 (step_i, step_{i+1}) 取 cos, 得到长度 S-1 的序列
                    # 4) 相邻步 cos 序列 (细粒度 anti-collapse 信号)
                    # adj_cos_mean: 整体步间相似度 (→ 1 → 全局坍塌)
                    # adj_cos_min:  最活跃的一跳
                    # adj_cos_max:  最死板的一跳 (→ 1 说明存在"完全复读"的中间步)
                    _adj_cos_list = []
                    for _s_i in range(actual_steps - 1):
                        _h_a = all_step_outputs[_s_i].squeeze(1).float()
                        _h_b = all_step_outputs[_s_i + 1].squeeze(1).float()
                        _adj = F.cosine_similarity(_h_a, _h_b, dim=-1).mean().item()
                        _adj_cos_list.append(_adj)
                    if _adj_cos_list:
                        hidden_stats['h_adj_cos_mean'] = float(sum(_adj_cos_list) / len(_adj_cos_list))
                        hidden_stats['h_adj_cos_min'] = float(min(_adj_cos_list))
                        hidden_stats['h_adj_cos_max'] = float(max(_adj_cos_list))

            # 5) Stage Alignment Matrix (SAM) — 真正监控 hidden 推理语义
            #    只在训练 (有 step_logits 和 stage_key_token_ids) 时计算.
            #    构造 attn[i, j] = step i 的 hidden logit 在 stage j 关键词上的 softmax 概率
            #    理想行为是 attn 接近对角线: step 0 偏好 stage 0 keys, step 1 偏好 stage 1 keys, ...
            if (
                self.training
                and lm_head is not None
                and actual_steps >= 2
                and stage_key_token_ids is not None
            ):
                _diag_scores = []      # 每个样本的 mean(attn[i, i])
                _mono_scores = []      # argmax 沿对角线递增的比例
                _shift_kls = []        # KL(attn[0, :] || attn[-1, :])  分布位移
                # step_logits 由前面已计算: [B, actual_steps, V]
                for _b in range(B):
                    _b_stages = (
                        stage_key_token_ids[_b] if _b < len(stage_key_token_ids) else []
                    )
                    if not _b_stages:
                        continue
                    _S = min(len(_b_stages), actual_steps)
                    if _S < 2:
                        continue
                    # 为每个 stage 计算其 token id list (去重)
                    _stage_ids = []
                    _valid_stages = []
                    for _s in range(_S):
                        _ids = _b_stages[_s]
                        if _ids:
                            _stage_ids.append(
                                torch.tensor(_ids, device=device, dtype=torch.long).unique()
                            )
                            _valid_stages.append(_s)
                    _Sv = len(_valid_stages)
                    if _Sv < 2:
                        continue
                    # attn matrix: [actual_steps_used, _Sv]
                    # 每个元素 = step_i 的 logits 在 stage_j 关键词上的 logsumexp 后做跨 stage 的 softmax
                    _atts_per_step = []
                    for _i in range(_Sv):
                        _step_logit = step_logits[_b, _i, :].float()  # [V]
                        # 每个 stage 的 "亲和度" = stage keys 上的 mean logit (温度 1)
                        _stage_affinity = torch.stack([
                            _step_logit[_sids].mean() for _sids in _stage_ids
                        ])  # [_Sv]
                        _attn = torch.softmax(_stage_affinity, dim=0)  # [_Sv]
                        _atts_per_step.append(_attn)
                    _attn_mat = torch.stack(_atts_per_step)  # [_Sv, _Sv]
                    # (a) diag_score: 对角线均值 (期望 > 1/_Sv = 随机)
                    _diag_scores.append(
                        torch.diagonal(_attn_mat).mean().item()
                    )
                    # (b) monotonic_ratio: argmax 是否随 step 递增
                    _argmax_per_step = _attn_mat.argmax(dim=-1).cpu().tolist()
                    _mono_count = sum(
                        1 for _i in range(1, len(_argmax_per_step))
                        if _argmax_per_step[_i] >= _argmax_per_step[_i - 1]
                    )
                    _mono_scores.append(_mono_count / max(len(_argmax_per_step) - 1, 1))
                    # (c) shift_kl: 第一步 vs 最后一步 attn 分布的 KL
                    _p0 = _attn_mat[0].clamp(min=1e-8)
                    _pL = _attn_mat[-1].clamp(min=1e-8)
                    _shift = (_p0 * (_p0.log() - _pL.log())).sum().item()
                    _shift_kls.append(_shift)
                if _diag_scores:
                    hidden_stats['h_stage_diag_score'] = float(
                        sum(_diag_scores) / len(_diag_scores)
                    )
                if _mono_scores:
                    hidden_stats['h_stage_monotonic'] = float(
                        sum(_mono_scores) / len(_mono_scores)
                    )
                if _shift_kls:
                    hidden_stats['h_stage_shift_kl'] = float(
                        sum(_shift_kls) / len(_shift_kls)
                    )

            # 6) Saturation 演化 (推理期 early-exit 病态预警)
            #    all_saturations 已采集: list of [B] tensor (长度 = actual_steps)
            #    saturations[0] = step 0 → step 1 的 cos 相似度 (饱和度)
            if all_saturations and len(all_saturations) >= 1:
                _sat_step1 = all_saturations[0].float().mean().item()
                hidden_stats['h_sat_step1'] = _sat_step1
                hidden_stats['h_sat_step_last'] = (
                    all_saturations[-1].float().mean().item()
                )
                # early_exit_ratio: 第 1 步 (step 0→1) 就达到 saturation_exit_threshold 的比例
                # 这是"固定第 2 步 exit"病态的直接证据
                _sat_thresh = self.saturation_exit_threshold
                _early = (all_saturations[0] > _sat_thresh).float().mean().item()
                hidden_stats['h_sat_early_exit_ratio'] = _early

            # 7) num_thought_steps (本次 latent 实际跑了几步)
            #    训练期 = target_steps (数据决定); 推理期 = 自适应退出后的实际步数
            #    用于诊断"固定第 N 步 exit": 若推理期均值远小于数据 num_stages 均值,
            #    说明模型已陷入 early-exit 死结
            if actual_steps > 0:
                hidden_stats['num_thought_steps'] = int(actual_steps)

            # 8) TGVR 档 A (Text-Guided Visual Recall, 诊断版, 无 loss)
            #    用 stage_key_token_ids[s] 作 query, 召回该样本视觉 token 的 v_pos_s,
            #    再算 thought hidden 与 v_pos_s 的对齐度. 仅写入 hidden_stats, 不参 loss.
            #    必备依赖: stage_key_token_ids / image_embeds_per_sample / embed_tokens / B / actual_steps
            if (
                stage_key_token_ids is not None
                and image_embeds_per_sample is not None
                and embed_tokens is not None
                and B is not None
                and actual_steps > 0
            ):
                try:
                    _tgvr_stats = self._compute_tgvr_diagnostics(
                        all_step_outputs=all_step_outputs,
                        stage_key_token_ids=stage_key_token_ids,
                        image_embeds_per_sample=image_embeds_per_sample,
                        embed_tokens=embed_tokens,
                        B=B,
                        actual_steps=actual_steps,
                        device=device,
                    )
                    hidden_stats.update(_tgvr_stats)
                except Exception as _tgvr_err:
                    # 诊断不能影响训练; 失败时跳过, 写一个标记便于排查
                    hidden_stats['tgvr_error'] = 1.0
                    if self.training and getattr(self, '_tgvr_warned', False) is False:
                        print(f"[NLD-Thinker] ⚠️ TGVR diagnostics failed: {_tgvr_err}")
                        self._tgvr_warned = True
        # ---- SW-SRS 内部统计 (Q 分布熵 / 门控触发率 / top-k 命中率) ----
        sw_srs_stats = getattr(self, '_last_sw_srs_stats', {}) or {}

        # ---- path B2 视觉侧 vision_loss (默认权重 0, 向后兼容) ----
        # 仅在 vision_loss_weight > 0 且必要依赖齐备时计算; 其余场景返回 0 张量.
        vision_loss = torch.tensor(0.0, device=device, dtype=dtype)
        vision_stats: Dict[str, float] = {}
        if (
            self.training
            and self.vision_loss_weight > 0.0
            and stage_key_token_ids is not None
            and image_embeds_per_sample is not None
            and embed_tokens is not None
            and B is not None
            and actual_steps > 0
        ):
            try:
                _excluded = getattr(self, '_sw_srs_excluded_ids', None)
                _v_loss, vision_stats = _compute_vision_loss_fn(
                    all_step_outputs=all_step_outputs,
                    stage_key_token_ids=stage_key_token_ids,
                    stage_roles=stage_roles,
                    image_embeds_per_sample=image_embeds_per_sample,
                    embed_tokens=embed_tokens,
                    B=B,
                    actual_steps=actual_steps,
                    device=device,
                    top_k=self.vision_top_k,
                    margin=self.vision_margin,
                    excluded_ids=_excluded,
                )
                # 严格安全: 检查 NaN/Inf 避免污染总 loss
                if torch.is_tensor(_v_loss) and torch.isfinite(_v_loss):
                    vision_loss = _v_loss.to(dtype=dtype)
            except Exception as _vl_err:
                vision_stats = {'vis_error': 1.0}
                if getattr(self, '_vis_warned', False) is False:
                    print(f"[NLD-Thinker] ⚠️ vision_loss failed: {_vl_err}")
                    self._vis_warned = True

        return {
            'thought_output': thought_output,  # [B, actual_steps, H]
            'exit_token_loss': exit_token_loss,
            'key_token_loss': key_token_loss,
            # ====== 3 层 anti-collapse 新增 loss ======
            'exit_margin_loss': exit_margin_loss,             # Layer 1: 中间步软约束
            'swsrs_anti_collapse_loss': swsrs_anti_collapse_loss,  # Layer 2 增强
            'diversity_loss': diversity_loss,                 # Layer 3: 跨步 hidden 多样性
            # ====== path B2 视觉侧 ======
            'vision_loss': vision_loss,
            'vision_stats': vision_stats,
            'exit_stats': exit_stats,
            'hidden_stats': hidden_stats,
            'sw_srs_stats': sw_srs_stats,
            'num_thought_steps': actual_steps,
        }
    
    @torch.no_grad()
    def _compute_tgvr_diagnostics(
        self,
        all_step_outputs: List[torch.Tensor],
        stage_key_token_ids: List[List[List[int]]],
        image_embeds_per_sample: List[torch.Tensor],
        embed_tokens: nn.Module,
        B: int,
        actual_steps: int,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        TGVR 档 A: Text-Guided Visual Recall — 诊断版 (无 loss, 不进梯度图).

        目的:
          回答"latent hidden 在迭代过程中是否拉近了与本 stage ' 关键视觉特征 ' 的距离".
          这里"关键视觉特征" = 用 stage key tokens 作 query, 在该样本的视觉 token 序列上
          做 attention 召回, 得到 v_pos_s ∈ R^H (相比对全图视觉特征求平均, 更聚焦在
          stage 真正涉及的 patch 上).

        指标 (mean over (sample × stage)):
          - tgvr_cos_h_v        : cos(thought_hidden_s, v_pos_s) 平均, [-1, 1]
                                  → 1 表示 hidden 已经在视觉对齐方向上, → 0/负值 = 没对齐
          - tgvr_cos_h_v_first  : 仅 stage 0 (早期) 平均, 用作"启动点"参考
          - tgvr_cos_h_v_last   : 仅 stage S-1 (后期) 平均, 用作"收敛点"参考
                                  收敛信号: last > first (越靠后越对齐)
          - tgvr_v_pos_norm     : ||v_pos_s||_2 平均, 反映召回向量集中度
          - tgvr_attn_entropy   : softmax(attn_weights) 的熵 / log(N_i), 归一化到 [0,1]
                                  → 0 = 召回非常聚焦, → 1 = 均匀近似平均池化
          - tgvr_topk_recall    : top-1 召回 patch 上的 attention 权重平均
                                  → 高表示存在显著主导 patch, 低则视觉中性

        Args:
            all_step_outputs:  list of [B, 1, H], 长度 = actual_steps
            stage_key_token_ids: [B][stage][token_ids]  (model_v2 已切到当前 boundary)
            image_embeds_per_sample: list of [N_i, H], 长度 = B
            embed_tokens: VLM input embedding 层 (用于把 token id -> embedding)
            B, actual_steps, device: 上下文

        Returns:
            dict with floats; 失败/无可用 stage 时返回空 dict (调用方决定不写入).
        """
        # 维度安全检查
        if not stage_key_token_ids or not image_embeds_per_sample:
            return {}
        if len(image_embeds_per_sample) < B:
            return {}

        cos_list_all: List[float] = []
        cos_first_list: List[float] = []
        cos_last_list: List[float] = []
        v_pos_norm_list: List[float] = []
        attn_entropy_list: List[float] = []
        topk_w_list: List[float] = []
        eps = 1e-6

        for b_idx in range(B):
            stages = (
                stage_key_token_ids[b_idx]
                if b_idx < len(stage_key_token_ids) else []
            )
            if not stages:
                continue
            v_b = image_embeds_per_sample[b_idx]  # [N_i, H]
            if v_b is None or v_b.numel() == 0 or v_b.dim() != 2:
                continue
            N_i = v_b.shape[0]
            if N_i < 1:
                continue

            S_data = len(stages)
            # 与 actual_steps 取交集 (训练期 actual_steps == S_data 但仍取 min 防御)
            S_use = min(S_data, actual_steps)
            if S_use < 1:
                continue

            # 视觉特征统一到 float (避免 bf16 数值不稳定)
            v_f = v_b.detach().to(device=device, dtype=torch.float32)

            for s in range(S_use):
                tok_ids = stages[s] or []
                if not tok_ids:
                    continue
                # 排除 specials (与 SW-SRS 一致)
                excluded = getattr(self, '_sw_srs_excluded_ids', None)
                if excluded is not None and excluded.numel() > 0:
                    excluded_set = set(int(x) for x in excluded.tolist())
                    tok_ids = [int(t) for t in tok_ids
                               if int(t) not in excluded_set]
                if not tok_ids:
                    continue
                tok_ids_t = torch.tensor(
                    tok_ids, device=device, dtype=torch.long
                ).unique()
                if tok_ids_t.numel() == 0:
                    continue

                # ---- 构造 query: stage key tokens 的 mean embedding ----
                # embed_tokens 输出 [K, H], 然后 mean → [H]
                tok_embs = embed_tokens(tok_ids_t).detach().to(
                    dtype=torch.float32
                )
                q_s = tok_embs.mean(dim=0)  # [H]
                q_norm = q_s.norm().clamp(min=eps)
                q_unit = q_s / q_norm  # [H]

                # ---- v_pos_s: 用 q_s 对 v_f 做 dot-product attention ----
                # logits: [N_i] = v_f @ q_s / sqrt(H)
                H = v_f.shape[-1]
                attn_logits = (v_f @ q_s) / math.sqrt(max(H, 1))
                attn_weights = F.softmax(attn_logits, dim=0)  # [N_i]
                v_pos_s = (attn_weights.unsqueeze(-1) * v_f).sum(dim=0)  # [H]

                # ---- thought hidden at stage s ----
                # all_step_outputs[s]: [B, 1, H]; 取本样本的 [H]
                if s >= len(all_step_outputs):
                    continue
                h_s = all_step_outputs[s][b_idx].detach().squeeze(0).to(
                    dtype=torch.float32
                )  # [H]

                # ---- cos(h_s, v_pos_s) ----
                cos_hv = F.cosine_similarity(
                    h_s.unsqueeze(0), v_pos_s.unsqueeze(0), dim=-1
                ).item()
                cos_list_all.append(cos_hv)
                if s == 0:
                    cos_first_list.append(cos_hv)
                if s == S_use - 1:
                    cos_last_list.append(cos_hv)

                # ---- v_pos 范数 / attn 熵 / top-1 权重 ----
                v_pos_norm_list.append(v_pos_s.norm().item())
                p_safe = attn_weights.clamp(min=eps)
                ent = -(p_safe * p_safe.log()).sum().item()
                ent_norm = ent / max(math.log(N_i), 1e-6) if N_i > 1 else 0.0
                attn_entropy_list.append(ent_norm)
                topk_w_list.append(attn_weights.max().item())

        if not cos_list_all:
            return {}

        out: Dict[str, float] = {
            'tgvr_cos_h_v_mean': float(sum(cos_list_all) / len(cos_list_all)),
            'tgvr_v_pos_norm_mean': float(sum(v_pos_norm_list) / len(v_pos_norm_list)),
            'tgvr_attn_entropy_mean': float(sum(attn_entropy_list) / len(attn_entropy_list)),
            'tgvr_topk_recall_mean': float(sum(topk_w_list) / len(topk_w_list)),
        }
        if cos_first_list:
            out['tgvr_cos_h_v_first'] = float(sum(cos_first_list) / len(cos_first_list))
        if cos_last_list:
            out['tgvr_cos_h_v_last'] = float(sum(cos_last_list) / len(cos_last_list))
        return out

    def _compute_sw_srs_loss(
        self,
        step_logits: torch.Tensor,
        stage_key_token_ids: List[List[List[int]]],
        B: int,
        actual_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        tau: float = 1.0,
        eta: float = 0.6,
        alpha: float = 0.3,
        loss_form: str = 'weighted_ce',
    ) -> torch.Tensor:
        """
        Stage-Windowed Supervision (语义侧监督), 借鉴 Laser (arxiv 2601.06803, Sec 3.3)
        的 Dynamic Windowed Alignment Learning, 适配 Interlude 的 stage-organized
        key token 数据结构。

        统一前置流程 (两种 loss_form 共享):
          1. 动态语义窗口: W_s = ∪_{s' >= s} K_{s'}  (本 stage + 所有 future stage
             key tokens 去重合集). 早期 W_s 大 (global superposition, Forest),
             晚期 W_s 小 (local precision, Trees).
          2. Teacher 权重 q_s ∈ R^|W_s|:
               q_s = softmax(StopGrad(z_s[W_s]) / τ),
             entropy-regularized hard-mix:
               若归一化熵 H(q_s) > η, 则 q_s ← α·y_hard + (1-α)·q_s,
               其中 y_hard 在本 stage K_s 上均匀.

        loss_form 控制最后的 loss 形式:
          - 'weighted_ce' (默认, 与 Laser 官方 forward_dwal.py 严格一致):
                student 在 *全词表 V* 上 log_softmax: log p_θ ∈ R^V,
                取 W_s 子集 log p_θ[W_s] ∈ R^|W_s|, 单 token CE = -log p_θ[k],
                loss = Σ_k q_s(k) · (-log p_θ[k]).        ← weighted next-token CE
                等价于 "用 q_s 给 W_s 中每个 token 的 single-token CE 加权".
                由于 student 全词表归一化、teacher 仅作子集权重, 不存在
                KL(P, stopgrad(P))=0 的退化 → 永远有非平凡梯度。

          - 'kl' (deprecated, 仅向后兼容旧 'sw_srs' 实验):
                student 与 teacher 都在 W_s 子集上 softmax → 数值相等 →
                KL(P_target ‖ P_student) ≡ 0 (∀ hidden), 梯度 ≡ 0.
                数学上是 trivial 退化, 不能学到任何东西. 仅保留以复现旧 ckpt.

        Args:
            step_logits: [B, actual_steps, V] student logits (含梯度链路)
            stage_key_token_ids: [B][stage][token_ids]
            B, actual_steps, device, dtype: 同其它 loss
            tau: softmax 温度 (Laser 推荐 1.0)
            eta: 归一化熵阈值 (Laser 推荐 0.6)
            alpha: hard-target 混合权重 (Laser 未给值, 取 0.3 作保守起点)
            loss_form: 'weighted_ce' (默认, Laser 官方) | 'kl' (deprecated)

        Returns:
            scalar loss, 始终有 grad_fn

        Side-effect:
            self._last_sw_srs_stats 会被更新, 含 anti-collapse 监控指标:
              - q_entropy_mean / q_entropy_first_stage / q_entropy_last_stage
              - topk_hit_ratio: student argmax 落在本 stage now_ids 的比例
        """
        # 防御: loss_form 校验
        if loss_form not in ('weighted_ce', 'kl'):
            print(f"[NLD-Thinker] ⚠️ unknown loss_form={loss_form!r}, "
                  f"fallback to 'weighted_ce' (Laser official)")
            loss_form = 'weighted_ce'
        losses = []
        eps = 1e-8
        
        # ---- Anti-collapse 统计累积器 (no_grad) ----
        _q_entropies: List[float] = []
        # 与 _q_entropies 等长的并行元数据, 用于将熵分到 "early stage" / "late stage" 两个桶
        _q_entropies_step_idx: List[int] = []     # 该 entropy 对应的 step_idx (从 0 开始)
        _q_entropies_total_steps: List[int] = []  # 该样本的总 stage 数 S_data
        _topk_hits: List[float] = []       # 1.0 if student_argmax in now_ids else 0.0
        
        for b_idx in range(B):
            b_stages = stage_key_token_ids[b_idx] if b_idx < len(stage_key_token_ids) else []
            if not b_stages:
                continue
            S_data = len(b_stages)  # 数据中的 stage 数 (== 本样本期望步数)
            
            for step_idx in range(actual_steps):
                if step_idx >= S_data:
                    # 多余的 step (不应出现, 因 target_steps 由数据 stage 数决定), 跳过
                    continue
                
                # ---- 构造 W_s = 本 stage + 所有 future stage 的 key tokens 合集 ----
                window_ids: List[int] = []
                for s_future in range(step_idx, S_data):
                    window_ids.extend(b_stages[s_future])
                if len(window_ids) == 0:
                    continue
                window_ids_t = torch.tensor(window_ids, device=device, dtype=torch.long).unique()
                
                # 本 stage keys (用于 hard target)
                now_ids = b_stages[step_idx]
                if len(now_ids) == 0:
                    continue
                now_ids_t = torch.tensor(now_ids, device=device, dtype=torch.long).unique()
                # 确保 now_ids 都在 window 内 (应恒成立)
                window_ids_t = torch.cat([window_ids_t, now_ids_t]).unique()

                # ---- T2': 排除 specials (贴合 LASER paper Sec 3.3 排除 <laser_end>) ----
                # 防止 student 把概率塞给 <|latent|>/<|/latent|>/<|pause|>/bos/eos/pad/unk/
                # <|im_start|>/<|im_end|>/image_token_id 等.
                # _sw_srs_excluded_ids 由 model_v2.set_processor 注入; 空 tensor 表示不排除.
                excluded = getattr(self, '_sw_srs_excluded_ids', None)
                if excluded is not None and excluded.numel() > 0:
                    excluded_dev = excluded.to(window_ids_t.device)
                    keep_mask = ~torch.isin(window_ids_t, excluded_dev)
                    window_ids_t = window_ids_t[keep_mask]
                    # now_ids_t 也同步过滤, 保证 hard target 的 mass 不落到 specials
                    now_keep_mask = ~torch.isin(now_ids_t, excluded_dev)
                    now_ids_t = now_ids_t[now_keep_mask]
                    if now_ids_t.numel() == 0:
                        # 本 stage keys 全是 specials (异常数据), 跳过
                        continue

                K_size = window_ids_t.numel()
                if K_size < 2:
                    # 窗口太小, softmax 退化为确定性, 跳过以避免 log(1)=0 除零
                    continue
                
                # ---- Teacher 权重 q_s 在 W_s 子集上 (两种 loss_form 共享) ----
                # 数值稳定: 升到 fp32 + 守卫 nan/inf (bf16 logits 可能溢出到 inf)
                full_logits_fp32 = step_logits[b_idx, step_idx].float()  # [V]
                if torch.isnan(full_logits_fp32).any() or torch.isinf(full_logits_fp32).any():
                    continue
                # 对 W_s 子集 logit 做 clamp (Laser 官方做法, bf16 safety)
                sub_logits_for_teacher = full_logits_fp32[window_ids_t].clamp(-1e4, 1e4)

                with torch.no_grad():
                    teacher_sub_logits = sub_logits_for_teacher.detach()
                    q_soft = F.softmax(teacher_sub_logits / tau, dim=0)  # [K_size]
                    # 归一化熵 H(Q)/log|W|
                    q_clamped = q_soft.clamp(min=eps)
                    log_q = q_clamped.log()
                    H_norm = -(q_clamped * log_q).sum() / math.log(K_size)  # scalar in [0, 1]

                    # hard target: 本 stage now_ids 上均匀 (在 window 子集内的位置)
                    now_pos_in_window = torch.searchsorted(window_ids_t, now_ids_t)
                    y_hard = torch.zeros(K_size, device=device, dtype=q_soft.dtype)
                    y_hard[now_pos_in_window] = 1.0 / now_ids_t.numel()

                    # 熵门控混合
                    if H_norm.item() > eta:
                        p_target = alpha * y_hard + (1.0 - alpha) * q_soft
                    else:
                        p_target = q_soft
                    # 再次归一化 (数值安全)
                    p_target = p_target / p_target.sum().clamp(min=eps)

                # ---- 按 loss_form 选择 student 端构造与 loss 形式 ----
                if loss_form == 'weighted_ce':
                    # Laser 官方 forward_dwal.py: weighted next-token CE
                    # student: 全词表 log_softmax (含梯度) → 在 W_s 子集 gather
                    # loss = Σ_k q_s(k) · (-log p_θ(k));  q_s 为 stopgrad 权重
                    log_p_student_full = F.log_softmax(full_logits_fp32, dim=-1)  # [V]
                    window_log_probs = log_p_student_full[window_ids_t]           # [K_size]
                    # 单 token CE = -log p (注意: 这里的 softmax 在全词表 V 上归一化,
                    # 与 teacher 子集 softmax 不同源 → 永远有非平凡梯度)
                    ce_per_token = -window_log_probs                              # [K_size]
                    loss_b_step = (p_target * ce_per_token).sum()                 # scalar
                    losses.append(loss_b_step)

                    # 留作后续 anti-collapse 统计 (用 student 在 W_s 子集 argmax 判断 topk_hit)
                    with torch.no_grad():
                        # 在子集上的相对偏好 (仅作监控, 不参与 loss)
                        log_p_student_sub = window_log_probs.detach()
                else:
                    # 'kl' (deprecated): 旧版 KL self-distill, 数学上恒为 0 → 仅向后兼容
                    student_sub_logits = full_logits_fp32[window_ids_t]  # [K_size]
                    log_p_student_sub = F.log_softmax(student_sub_logits / tau, dim=0)
                    kl = F.kl_div(log_p_student_sub, p_target, reduction='sum')
                    kl = kl * (tau ** 2)  # Distillation 标准温度缩放
                    losses.append(kl)

                # ---- Anti-collapse 统计 (no_grad) ----
                with torch.no_grad():
                    _q_entropies.append(H_norm.item())
                    _q_entropies_step_idx.append(step_idx)
                    _q_entropies_total_steps.append(S_data)
                    # Top-k hit: student argmax 是否落在本 stage now_ids (window 子集内的索引)
                    _student_argmax_idx = log_p_student_sub.argmax().item()
                    _now_pos_set = set(now_pos_in_window.tolist())
                    _topk_hits.append(1.0 if _student_argmax_idx in _now_pos_set else 0.0)
        
        if losses:
            sw_srs_loss = torch.stack(losses).mean()
        else:
            # 无有效样本, 返回 0 但保持梯度链路
            sw_srs_loss = (step_logits.sum() * 0.0).to(dtype)
        
        # ---- 导出 anti-collapse 统计 (供 model_v2.py 读取写入 TensorBoard) ----
        # q_entropy_mean:        全局均值 (所有 stage)
        # q_entropy_first_stage: 仅 step_idx == 0 (最早 stage), 期望偏大 (global superposition)
        # q_entropy_last_stage:  仅 step_idx == S_data - 1 (最晚 stage), 期望偏小 (local precision)
        # 两者差距越大, 说明 SW-SRS 的 "先宽后窄" 设计越生效
        _early_q = [
            q for q, si in zip(_q_entropies, _q_entropies_step_idx) if si == 0
        ]
        _late_q = [
            q for q, si, ts in zip(_q_entropies, _q_entropies_step_idx, _q_entropies_total_steps)
            if si == ts - 1
        ]
        self._last_sw_srs_stats = {
            'q_entropy_mean': sum(_q_entropies) / max(len(_q_entropies), 1) if _q_entropies else 0.0,
            'q_entropy_first_stage': (sum(_early_q) / len(_early_q)) if _early_q else 0.0,
            'q_entropy_last_stage': (sum(_late_q) / len(_late_q)) if _late_q else 0.0,
            'topk_hit_ratio': sum(_topk_hits) / max(len(_topk_hits), 1) if _topk_hits else 0.0,
        }
        
        return sw_srs_loss

    def _compute_stage_keyword_distill_loss(
        self,
        step_logits: torch.Tensor,
        stage_key_token_ids: List[List[List[int]]],
        B: int,
        actual_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        tau: float = 1.0,
    ) -> torch.Tensor:
        """
        Stage Keyword Distillation Loss (Phase 1 推荐, 替代 self-distill 的 SW-SRS)

        设计动机:
          原 SW-SRS 用 teacher = stopgrad(student_logits) 做 KL 自蒸馏,
          数学上 KL(P, stopgrad(P)) = 0 对任意 hidden 都成立 → 该 loss
          根本不约束 hidden 学到任何东西, 退化为 "无监督".

          本 loss 把 teacher 替换为 **外部 anchor: 第 s stage 关键词集合 K_s 上的均匀分布**:
             P_target(k) = 1/|K_s| if k in K_s else 0  (定义在 window W_s 上)
          这样:
            1. teacher 与 student 不再同源, KL=0 不可能 trivially 满足;
            2. 不同 stage 有不同 K_s → hidden 复读会让某些 stage loss 偏大;
            3. "forest-before-trees" 由 W_s 由大变小天然表达
               (早期 W_s 包含 future stage keys, 概率叠加; 晚期窗口收紧, 精确化).
          严格地, 这是 Laser DWAL 公式 (arxiv 2601.06803, Sec 3.3) 的
          forward-KL 直接实现, 不绕道 self-distill.

        Args:
            step_logits:           [B, actual_steps, V]
            stage_key_token_ids:   [B][stage][token_ids]
            B, actual_steps, device, dtype: 同 _compute_sw_srs_loss
            tau:                   温度 (与 SW-SRS 保持一致, 默认 1.0)

        Returns:
            scalar loss, 始终有 grad_fn

        Side-effect:
            self._last_sw_srs_stats 仍然按 SW-SRS 同样 schema 写出, 便于打印复用.
              - q_entropy_mean / first / last: 这里复用为 P_target 的归一化熵
                (uniform on K_s 时熵 = log|K_s|/log|W_s|, 反映窗口粒度)
              - topk_hit_ratio: student argmax 是否落在 K_s 内 (核心收敛信号)
        """
        losses = []
        eps = 1e-8

        _q_entropies: List[float] = []
        _q_entropies_step_idx: List[int] = []
        _q_entropies_total_steps: List[int] = []
        _topk_hits: List[float] = []

        for b_idx in range(B):
            b_stages = stage_key_token_ids[b_idx] if b_idx < len(stage_key_token_ids) else []
            if not b_stages:
                continue
            S_data = len(b_stages)

            for step_idx in range(actual_steps):
                if step_idx >= S_data:
                    continue

                # ---- 构造 W_s = 本 stage + 所有 future stage 的 key tokens 合集 ----
                window_ids: List[int] = []
                for s_future in range(step_idx, S_data):
                    window_ids.extend(b_stages[s_future])
                if len(window_ids) == 0:
                    continue
                window_ids_t = torch.tensor(window_ids, device=device, dtype=torch.long).unique()

                # 本 stage keys (K_s)
                now_ids = b_stages[step_idx]
                if len(now_ids) == 0:
                    continue
                now_ids_t = torch.tensor(now_ids, device=device, dtype=torch.long).unique()
                # 确保 now_ids 都在 window 内 (应恒成立)
                window_ids_t = torch.cat([window_ids_t, now_ids_t]).unique()

                # ---- T2': 排除 specials (贴合 LASER paper Sec 3.3) ----
                excluded = getattr(self, '_sw_srs_excluded_ids', None)
                if excluded is not None and excluded.numel() > 0:
                    excluded_dev = excluded.to(window_ids_t.device)
                    keep_mask = ~torch.isin(window_ids_t, excluded_dev)
                    window_ids_t = window_ids_t[keep_mask]
                    now_keep_mask = ~torch.isin(now_ids_t, excluded_dev)
                    now_ids_t = now_ids_t[now_keep_mask]
                    if now_ids_t.numel() == 0:
                        continue

                K_size = window_ids_t.numel()
                if K_size < 2:
                    continue

                # ---- Student: log_softmax(logits[W_s] / tau), 保留梯度 ----
                student_sub_logits = step_logits[b_idx, step_idx, window_ids_t].float()
                if torch.isnan(student_sub_logits).any() or torch.isinf(student_sub_logits).any():
                    continue
                log_p_student = F.log_softmax(student_sub_logits / tau, dim=0)

                # ---- Teacher: K_s 上均匀分布 (外部 anchor, 与 student 同源解耦) ----
                with torch.no_grad():
                    now_pos_in_window = torch.searchsorted(window_ids_t, now_ids_t)
                    p_target = torch.zeros(K_size, device=device, dtype=log_p_student.dtype)
                    p_target[now_pos_in_window] = 1.0 / now_ids_t.numel()
                    # 数值稳定: 给非 K_s 的位置一个极小 floor, 避免 KL inf
                    # (target=0 但 student>0 时 KL 仍有限, 故 floor 可选; 此处保守加上)
                    p_target = p_target + eps
                    p_target = p_target / p_target.sum().clamp(min=eps)

                # ---- KL(p_target ‖ p_student) ----
                kl = F.kl_div(log_p_student, p_target, reduction='sum')
                kl = kl * (tau ** 2)
                losses.append(kl)

                # ---- 监控指标 (与 SW-SRS 保持同 schema) ----
                with torch.no_grad():
                    # P_target 是 uniform on K_s, 其归一化熵 = log|K_s|/log|W_s|
                    # 早期 stage W_s 大 → 比值小; 晚期 W_s≈K_s → 比值接近 1
                    H_norm = math.log(max(now_ids_t.numel(), 1)) / math.log(K_size)
                    _q_entropies.append(float(H_norm))
                    _q_entropies_step_idx.append(step_idx)
                    _q_entropies_total_steps.append(S_data)
                    _student_argmax_idx = log_p_student.argmax().item()
                    _now_pos_set = set(now_pos_in_window.tolist())
                    _topk_hits.append(1.0 if _student_argmax_idx in _now_pos_set else 0.0)

        if losses:
            stage_kw_loss = torch.stack(losses).mean()
        else:
            stage_kw_loss = (step_logits.sum() * 0.0).to(dtype)

        # ---- 同样导出 _last_sw_srs_stats 供 model_v2 复用监控 ----
        _early_q = [
            q for q, si in zip(_q_entropies, _q_entropies_step_idx) if si == 0
        ]
        _late_q = [
            q for q, si, ts in zip(_q_entropies, _q_entropies_step_idx, _q_entropies_total_steps)
            if si == ts - 1
        ]
        self._last_sw_srs_stats = {
            'q_entropy_mean': sum(_q_entropies) / max(len(_q_entropies), 1) if _q_entropies else 0.0,
            'q_entropy_first_stage': (sum(_early_q) / len(_early_q)) if _early_q else 0.0,
            'q_entropy_last_stage': (sum(_late_q) / len(_late_q)) if _late_q else 0.0,
            'topk_hit_ratio': sum(_topk_hits) / max(len(_topk_hits), 1) if _topk_hits else 0.0,
        }

        return stage_kw_loss

    def _compute_swsrs_anti_collapse_loss(
        self,
        step_logits: torch.Tensor,
        stage_key_token_ids: List[List[List[int]]],
        B: int,
        actual_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Layer 2 增强: SW-SRS Anti-Collapse Loss

        核心: 强制让 stage keys 的 logsumexp(W_s) > <|/latent|> 的 logit
        这是直接拦截 "hidden 塌缩到 exit" 病态的关键监督.

        SW-SRS 自蒸馏 KL(detach, self) 在 hidden 全部塌缩到 exit 时也满足 (KL=0),
        无法拦住塌缩. 本 loss 通过比较 W_s 子词表的 logsumexp 与 <|/latent|> 的 logit,
        当 hidden 上 stage keys 总 mass 低于 <|/latent|> 时给出反向梯度.

        Args:
            step_logits: [B, actual_steps, V] 已算好的 student logits (有梯度链路)
            stage_key_token_ids: [B][stage][token_ids]
            B, actual_steps, device, dtype: 同 _compute_sw_srs_loss

        Returns:
            scalar loss, 始终有 grad_fn
        """
        if self.latent_end_token_id is None:
            return (step_logits.sum() * 0.0).to(dtype)

        end_id = self.latent_end_token_id
        margin = self.swsrs_anti_collapse_margin
        # ====== 数值稳定上界 (防止极端 batch 单步爆炸) ======
        # 旧版无界 ReLU 在某些极端 batch 上 (lm_head 对 <|/latent|> 偏置过大,
        # end_logit 高达 +60, window_logsumexp 仅 +5) 会让 loss 高达 56,
        # bf16 + Adam 累积 → grad NaN → 整个模型权重污染 (已在 step ~512 实证).
        # 用 clamp(max) 把单 (b, step) 项 loss 限制在合理量级,
        # 训练信号仍存在 (>0 时仍提供反向梯度) 但单步 total loss 不会失控.
        # 2026-05-18 更新: 10.0 → 6.0
        #   旧 cap=10.0 时, 多项 mean 仍可达 23~48, 经常被外层 _ac_loss_cap=20
        #   守卫吃掉, anti-collapse 信号实际未进 total_loss. 降到 6.0 让 loss
        #   量级 ≤ ~6 * mean_factor, 大概率落在守卫阈值之下.
        per_term_cap = 6.0
        terms = []
        # ---- 诊断: 收集 end_logit / window_logsumexp 数值, 用于排查塌缩 ----
        diag_end_logits: List[float] = []
        diag_window_lse: List[float] = []
        diag_hit_cap = 0
        diag_total = 0
        for b_idx in range(B):
            b_stages = stage_key_token_ids[b_idx] if b_idx < len(stage_key_token_ids) else []
            if not b_stages:
                continue
            S_data = len(b_stages)
            for step_idx in range(actual_steps):
                if step_idx >= S_data:
                    continue
                # W_s = 本 stage + 所有 future stage 的 key tokens 合集 (与 SW-SRS 同)
                window_ids: List[int] = []
                for s_future in range(step_idx, S_data):
                    window_ids.extend(b_stages[s_future])
                if not window_ids:
                    continue
                window_ids_t = torch.tensor(
                    window_ids, device=device, dtype=torch.long
                ).unique()
                if window_ids_t.numel() < 1:
                    continue
                step_logit = step_logits[b_idx, step_idx, :].float()  # [V]
                if torch.isnan(step_logit).any() or torch.isinf(step_logit).any():
                    continue
                # logsumexp 在 stage keys 子词表上 (有梯度)
                window_logsumexp = step_logit[window_ids_t].logsumexp(dim=0)
                end_logit = step_logit[end_id]
                # ⛑ 数值守卫: 若 end_logit / logsumexp 已经异常大, 跳过此项
                #   (bf16 max ≈ 3.4e38, 但乘 grad 后可能 inf; 给保守阈值 1e4)
                if (
                    torch.isnan(end_logit) or torch.isinf(end_logit)
                    or torch.isnan(window_logsumexp) or torch.isinf(window_logsumexp)
                    or end_logit.abs() > 1e4
                    or window_logsumexp.abs() > 1e4
                ):
                    continue
                # 期望: window_logsumexp > end_logit + margin
                # → loss = ReLU(end_logit - window_logsumexp + margin)
                # 用 clamp(max=per_term_cap) 限制单项最大值, 防止极端 batch 爆炸
                raw_term = F.relu(end_logit - window_logsumexp + margin)
                terms.append(raw_term.clamp(max=per_term_cap))
                # ---- 诊断: 记录数值 (no_grad) ----
                with torch.no_grad():
                    diag_end_logits.append(float(end_logit.detach().item()))
                    diag_window_lse.append(float(window_logsumexp.detach().item()))
                    if raw_term.detach().item() >= per_term_cap - 1e-3:
                        diag_hit_cap += 1
                    diag_total += 1

        # ---- 诊断: 把统计写入 module attr, 由 model_v2 决定何时打印 ----
        if diag_total > 0:
            self._last_swsrs_anti_collapse_diag = {
                'end_logit_mean':    sum(diag_end_logits) / max(1, len(diag_end_logits)),
                'end_logit_max':     max(diag_end_logits) if diag_end_logits else 0.0,
                'window_lse_mean':   sum(diag_window_lse) / max(1, len(diag_window_lse)),
                'window_lse_max':    max(diag_window_lse) if diag_window_lse else 0.0,
                'gap_mean':          (sum(diag_end_logits) - sum(diag_window_lse)) / max(1, diag_total),
                'hit_cap_ratio':     diag_hit_cap / max(1, diag_total),
                'n_terms':           diag_total,
                'per_term_cap':      per_term_cap,
                'margin':            margin,
            }
        else:
            self._last_swsrs_anti_collapse_diag = None

        if terms:
            return torch.stack(terms).mean().to(dtype)
        else:
            return (step_logits.sum() * 0.0).to(dtype)

    def print_summary(self):
        """打印模块摘要"""
        print(f"\n{'='*60}")
        print(f"NativeLatentThinker 模块摘要 (Coconut 等价, 单 token recurrence)")
        print(f"{'='*60}")
        print(f"  hidden_size: {self.hidden_size}")
        print(f"  thought_tokens: 1 (单个查询探针, 历史信息全在 KV cache 中)")
        print(f"  forward 方式: text_model() 整体调用 (全部层 + norm)")
        print(f"  max_think_steps: {self.max_think_steps} (推理时安全上限, 训练时由数据 target_steps 决定)")
        print(f"  step_embed_size: {self._step_embed_size} (step embedding 表大小)")
        print(f"  saturation_exit_threshold: {self.saturation_exit_threshold}")
        print(f"  推理退出策略: 退出 token 预测 OR 饱和信号 (双保险)")
        print(f"  训练监督: Exit Token Loss + SW-SRS (Stage-Windowed Self-Refined Supervision)")
        print(f"  设计: last_hidden [B,1,H] → text_model() (with KV cache) × steps → prefix")
        print(f"  COCONUT 等价: hidden state 循环反馈 (Transformer as RNN cell, 全层)")
        print(f"  KV 累积: 每步 thought KV 自动追加到 cache, 后续步可 attend 所有历史 + 之前 thought")
        print(f"  总参数: {self._total_params:,} ({self._total_params/1e6:.2f}M)")
        print(f"  可训练参数: {self._trainable_params:,} ({self._trainable_params/1e6:.2f}M)")
        print(f"{'='*60}\n")


class ThoughtPrefixInjector(nn.Module):
    """
    将 NativeLatentThinker 的输出转换为 prefix，注入到 VLM 的 attention 层
    
    方式 A (Embedding Prefix): 直接拼接到下一个 chunk 的 input embeddings 前面
      → 最简单，完全兼容 flash_attention_2
      → thought_output [B, 1, H] 直接作为 prefix embedding
    """
    
    def __init__(self, mode: str = "embedding"):
        super().__init__()
        self.mode = mode
    
    def forward(
        self,
        thought_output: torch.Tensor,
        chunk_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        将思考结果注入到下一个 chunk 的输入中
        
        Args:
            thought_output: [B, N_t, H] 隐空间思考结果 (N_t 个 thought token，
                           即隐式推理进行了几步就有几个)
            chunk_embeds: [B, chunk_len, H] 下一个 chunk 的 token embeddings
        
        Returns:
            injected_embeds: [B, N_t + chunk_len, H] 注入后的 embeddings
        """
        if self.mode == "embedding":
            return torch.cat([thought_output, chunk_embeds], dim=1)
        else:
            raise ValueError(f"不支持的注入模式: {self.mode}")
