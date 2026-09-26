"""
NLD Trainer: 自定义 Trainer 支持 Interlude 训练

核心特性:
1. 差异化学习率: NativeLatentThinker lr vs VLM 基座 lr
2. 监控 NLD 特有指标: thought_count, mean_think_steps, ponder_cost 等
3. 兼容 FSDP / DDP 分布式训练
"""

import os
import logging
import time
import torch
from transformers import Trainer
from typing import Optional, Dict, List
from collections import defaultdict

logging.getLogger("transformers.trainer").setLevel(logging.WARNING)


class NLDTrainer(Trainer):
    """
    NLD Trainer
    
    扩展 HuggingFace Trainer:
    - 自定义 compute_loss: 调用 NLDModel.forward
    - 差异化学习率: NativeLatentThinker vs VLM 基座
    - 监控 NLD 特有指标
    - 兼容 FSDP / DDP 分布式训练
    """

    def __init__(
        self,
        *args,
        thinker_lr: float = 1e-4,
        vlm_lr: float = None,
        monitor_every_n_steps: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.thinker_lr = thinker_lr
        self.vlm_lr = vlm_lr  # VLM 基座学习率 (全量微调)
        self.monitor_every_n_steps = monitor_every_n_steps
        self._step_count = 0
        self._fwd_count = 0
        self._train_start_time = None
        self._sample_stats_accum = defaultdict(list)
        self._recent_losses = []
        self._loss_window_size = 50

    def _is_main_process(self):
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
        local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', -1)))
        return local_rank in (-1, 0)

    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        """覆盖 Trainer.log(): 只写 TensorBoard"""
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if start_time is not None:
            logs["step_time"] = round(start_time, 4)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        
        for callback in self.callback_handler.callbacks:
            cb_name = type(callback).__name__
            if cb_name in ("PrinterCallback", "ProgressCallback", "DefaultFlowCallback"):
                continue
            if hasattr(callback, 'on_log'):
                callback.on_log(self.args, self.state, self.control, logs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """自定义损失计算"""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs.get("labels")
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        step_boundaries = inputs.get("step_boundaries")
        prompt_lens = inputs.get("prompt_lens")
        loss_weight_mask = inputs.get("loss_weight_mask")
        latent_think_steps = inputs.get("latent_think_steps")
        stage_concept_token_ids = inputs.get("stage_concept_token_ids")
        stage_concept_roles = inputs.get("stage_concept_roles")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            step_boundaries=step_boundaries,
            prompt_lens=prompt_lens,
            loss_weight_mask=loss_weight_mask,
            latent_think_steps=latent_think_steps,
            stage_concept_token_ids=stage_concept_token_ids,
            stage_concept_roles=stage_concept_roles,
        )

        loss = outputs['loss']

        # 首次 forward 验证计算图
        if not hasattr(self, '_grad_path_verified'):
            self._grad_path_verified = True
            if self._is_main_process():
                _has_grad_fn = loss.grad_fn is not None
                print(f"\n  🔬 计算图验证 (首次 forward):")
                print(f"    loss.grad_fn: {'✅ 存在' if _has_grad_fn else '❌ None'}")
                if _has_grad_fn:
                    print(f"    loss.grad_fn 类型: {type(loss.grad_fn).__name__}")
                print()

        self._fwd_count += 1
        self._pending_draft_metrics = outputs.get('draft_metrics', {})
        self._pending_main_loss = outputs.get('main_loss', torch.tensor(0.0))
        self._pending_total_loss = loss

        # 累积样本级统计
        sample_meta = inputs.get('sample_meta', {})
        if sample_meta:
            for key, values in sample_meta.items():
                if isinstance(values, list):
                    self._sample_stats_accum[key].extend(values)
                else:
                    self._sample_stats_accum[key].append(values)

        if labels is not None:
            _valid_tokens = (labels != -100).sum().item()
            _total_tokens = labels.numel()
            self._sample_stats_accum['valid_tokens'].append(_valid_tokens)
            self._sample_stats_accum['total_tokens'].append(_total_tokens)

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """重写 training_step"""
        _t_start = time.time()
        _rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))

        if self._train_start_time is None:
            self._train_start_time = _t_start

        # 调用父类 training_step
        loss = super().training_step(model, inputs, num_items_in_batch)

        # 更新 step 计数 (在梯度累积完成后)
        if (self._fwd_count % self.args.gradient_accumulation_steps == 0):
            self._step_count += 1

            # 记录 loss 到滑动窗口
            _total_loss_val = self._pending_total_loss.item() if isinstance(self._pending_total_loss, torch.Tensor) else self._pending_total_loss
            self._recent_losses.append(_total_loss_val)
            if len(self._recent_losses) > self._loss_window_size:
                self._recent_losses = self._recent_losses[-self._loss_window_size:]

            # 定期打印监控信息
            if self._is_main_process() and self._step_count % self.monitor_every_n_steps == 0:
                self._print_monitor_info()

            # 写入 TensorBoard
            if self._is_main_process():
                log_dict = {}
                _main_loss = self._pending_main_loss.item() if isinstance(self._pending_main_loss, torch.Tensor) else self._pending_main_loss
                log_dict['loss/total'] = _total_loss_val
                log_dict['loss/main_ce'] = _main_loss

                # NLD 特有指标
                draft_metrics = getattr(self, '_pending_draft_metrics', {})
                for k, v in draft_metrics.items():
                    if isinstance(v, (int, float)):
                        log_dict[k] = v

                self.log(log_dict)

            # 清空累积器
            self._sample_stats_accum.clear()

        return loss

    def _print_monitor_info(self):
        """打印监控信息"""
        _elapsed = time.time() - self._train_start_time if self._train_start_time else 0
        _hours = _elapsed / 3600
        _steps_per_hour = self._step_count / max(_hours, 0.001)

        # 估算 ETA
        total_steps = self.state.max_steps if self.state.max_steps > 0 else 0
        remaining_steps = total_steps - self._step_count
        eta_hours = remaining_steps / max(_steps_per_hour, 1)

        _total_loss = self._pending_total_loss.item() if isinstance(self._pending_total_loss, torch.Tensor) else 0
        _main_loss = self._pending_main_loss.item() if isinstance(self._pending_main_loss, torch.Tensor) else 0

        # 滑动窗口 loss 统计
        _avg_loss = sum(self._recent_losses) / max(len(self._recent_losses), 1)

        # ---- Loss Spike 自动告警 (优先级最高, 训练崩溃前的早期信号) ----
        # 双判据 (任一触发即告警):
        #   (a) z-score:  current > μ + 3σ  (统计离群)
        #   (b) 倍数:     current > 3 × μ   (绝对量级)
        # 需窗口 ≥ 10 才启用 (前期样本太少会误报); μ/σ 由 _recent_losses 估算.
        if len(self._recent_losses) >= 10:
            import math as _math
            _mu = _avg_loss
            _var = sum((x - _mu) ** 2 for x in self._recent_losses) / len(self._recent_losses)
            _sigma = _math.sqrt(max(_var, 1e-12))
            _z_thr = _mu + 3.0 * _sigma
            _mult_thr = 3.0 * _mu if _mu > 0 else float('inf')
            if _total_loss > _z_thr or _total_loss > _mult_thr:
                _z = (_total_loss - _mu) / max(_sigma, 1e-6)
                print(f"\n  🚨 LOSS SPIKE DETECTED at step {self._step_count}: "
                      f"current={_total_loss:.4f}, "
                      f"window_mean(μ)={_mu:.4f}, window_std(σ)={_sigma:.4f}, "
                      f"z={_z:+.2f}σ, ratio={_total_loss / max(_mu, 1e-6):.1f}×μ",
                      flush=True)

        draft_metrics = getattr(self, '_pending_draft_metrics', {})
        _thought_count = draft_metrics.get('nld/thought_count', 0)

        # ---- Line 1: Step / ETA ----
        print(f"\n📊 [Step {self._step_count}/{total_steps}] "
              f"({_hours:.1f}h, {_steps_per_hour:.0f} steps/h, ETA {eta_hours:.1f}h)")

        # ---- Line 2: Loss (total / main_ce / exit / latent_kw) ----
        # 注: nld/sw_srs_loss 这个 tensorboard key 是历史名称【通用容器】,
        #   实际装的是【当前 loss_mode 对应的 latent monitor loss】:
        #     - loss_mode='laser_dwal' (默认) → Laser 官方 weighted-CE on W_s
        #     - loss_mode='stage_kw'              → stage keyword distillation (KL on W_s, K_s anchor)
        #     - loss_mode='sw_srs'   (deprecated) → 旧 self-distill SW-SRS loss
        #   为了阅读清晰, 这里统一打印为 'latent_kw' (latent keyword 监督项).
        _exit_token_loss = draft_metrics.get('nld/exit_token_loss', None)
        _sw_srs_loss = draft_metrics.get('nld/sw_srs_loss', None)
        _loss_parts = [f"total={_total_loss:.4f}", f"main_ce={_main_loss:.4f}",
                       f"avg({len(self._recent_losses)})={_avg_loss:.4f}"]
        if _exit_token_loss is not None:
            _loss_parts.append(f"exit={_exit_token_loss:.4f}")
        if _sw_srs_loss is not None:
            _loss_parts.append(f"latent_kw={_sw_srs_loss:.4f}")
        print(f"  Loss: {', '.join(_loss_parts)}")

        # ---- Line 2.5: Anti-Collapse Loss (3 层修复) ----
        # margin:    Layer 1, 中间步软约束 (>0 说明中间步有人在偷预测 <|/latent|>)
        # anti_col:  Layer 2, SW-SRS Anti-Collapse (>0 说明 hidden 在 stage keys 上 mass < exit logit)
        # diversity: Layer 3, 跨步 hidden 多样性 (>0 说明存在 cos > threshold 的"复读"步对)
        _exit_margin_loss = draft_metrics.get('nld/exit_margin_loss', None)
        _swsrs_anti_loss = draft_metrics.get('nld/swsrs_anti_collapse_loss', None)
        _diversity_loss_v = draft_metrics.get('nld/diversity_loss', None)
        _ac_parts = []
        if _exit_margin_loss is not None:
            _tag = " ⚠️premature" if _exit_margin_loss > 0.5 else ""
            _ac_parts.append(f"margin={_exit_margin_loss:.4f}{_tag}")
        if _swsrs_anti_loss is not None:
            _tag = " ⚠️collapse-to-exit" if _swsrs_anti_loss > 0.5 else ""
            _ac_parts.append(f"anti_col={_swsrs_anti_loss:.4f}{_tag}")
        if _diversity_loss_v is not None:
            _tag = " ⚠️duplicate-steps" if _diversity_loss_v > 0.05 else ""
            _ac_parts.append(f"diversity={_diversity_loss_v:.4f}{_tag}")
        if _ac_parts:
            print(f"  AntiC:  {', '.join(_ac_parts)}")

        # ---- Line 2.6: path B2 Vision Loss + 几何诊断 (vis_loss / cos(h,r/v/t) / alpha) ----
        # vis_loss: 双 anchor margin 损失 (>0 说明 hidden 还没落进过渡邻域)
        # cos_h_r:  hidden 与 slerp 参考目标 r 的相似度 (期望 > cos_h_v 与 cos_h_t)
        # cos_h_v:  hidden 与视觉重心 v_bar 的相似度 (越接近 1 越偏视觉锥)
        # cos_h_t:  hidden 与文本重心 t_bar 的相似度 (越接近 1 越偏语言锥)
        # alpha:    role-driven 插值系数均值 (abstract=0.75, bridge/unified=0.5, concrete=0.25)
        _vis_loss_v = draft_metrics.get('nld/vision_loss', None)
        _vis_cos_hr = draft_metrics.get('nld/vis_cos_h_r_mean', None)
        _vis_cos_hv = draft_metrics.get('nld/vis_cos_h_v_mean', None)
        _vis_cos_ht = draft_metrics.get('nld/vis_cos_h_t_mean', None)
        _vis_alpha = draft_metrics.get('nld/vis_alpha_mean', None)
        _vis_count = draft_metrics.get('nld/vis_count', None)
        _vis_parts = []
        if _vis_loss_v is not None:
            _vis_parts.append(f"vis_loss={_vis_loss_v:.4f}")
        if _vis_cos_hr is not None and _vis_cos_hv is not None and _vis_cos_ht is not None:
            # 期望: cos_hr 是三者中最大 (hidden 落在过渡邻域)
            _tag = "" if (_vis_cos_hr >= _vis_cos_hv and _vis_cos_hr >= _vis_cos_ht) else " ⚠️not-in-bridge"
            _vis_parts.append(f"cos(h,r/v/t)={_vis_cos_hr:.3f}/{_vis_cos_hv:.3f}/{_vis_cos_ht:.3f}{_tag}")
        if _vis_alpha is not None:
            _vis_parts.append(f"α={_vis_alpha:.2f}")
        if _vis_count is not None and _vis_count > 0:
            _vis_parts.append(f"n={int(_vis_count)}")
        if _vis_parts:
            print(f"  Vision: {', '.join(_vis_parts)}")

        # ---- Line 3: Hidden 几何 (核心 anti-collapse, 4 项 + 步间演化 3 项) ----
        _h_batch_cos = draft_metrics.get('collapse/h_batch_cos_mean', None)
        _h_eff_rank = draft_metrics.get('collapse/h_effective_rank', None)
        _h_drift = draft_metrics.get('collapse/h_first_last_cos', None)
        _h_norm = draft_metrics.get('collapse/h_norm_mean', None)
        _geo_parts = []
        if _h_batch_cos is not None:
            _tag = " ⚠️" if _h_batch_cos > 0.9 else ""
            _geo_parts.append(f"batch_cos={_h_batch_cos:.3f}{_tag}")
        if _h_eff_rank is not None:
            _tag = " ⚠️" if _h_eff_rank < 2.0 else ""
            _geo_parts.append(f"eff_rank={_h_eff_rank:.2f}{_tag}")
        if _h_drift is not None:
            _tag = " ⚠️" if _h_drift > 0.99 else ""
            _geo_parts.append(f"first_last_cos={_h_drift:.3f}{_tag}")
        if _h_norm is not None:
            _geo_parts.append(f"h_norm={_h_norm:.2f}")
        if _geo_parts:
            print(f"  Hidden: {', '.join(_geo_parts)}")

        # ---- Line 3.5: 步间演化 (相邻步 cos 序列, 细粒度坍塌信号) ----
        # adj_cos_mean → 1: 全局坍塌 (每步都几乎不变)
        # adj_cos_max → 1: 至少有一步完全复读 (定位坍塌发生在第几步)
        # adj_cos_min → 1: 连最活跃的一跳也不变 (严重坍塌)
        _adj_mean = draft_metrics.get('collapse/h_adj_cos_mean', None)
        _adj_min = draft_metrics.get('collapse/h_adj_cos_min', None)
        _adj_max = draft_metrics.get('collapse/h_adj_cos_max', None)
        _step_parts = []
        if _adj_mean is not None:
            _tag = " ⚠️" if _adj_mean > 0.99 else ""
            _step_parts.append(f"adj_mean={_adj_mean:.3f}{_tag}")
        if _adj_min is not None:
            _tag = " ⚠️" if _adj_min > 0.99 else ""
            _step_parts.append(f"adj_min={_adj_min:.3f}{_tag}")
        if _adj_max is not None:
            _tag = " ⚠️" if _adj_max > 0.999 else ""
            _step_parts.append(f"adj_max={_adj_max:.3f}{_tag}")
        if _step_parts:
            print(f"  Steps:  {', '.join(_step_parts)}")

        # ---- Line 4: Latent K_s 对齐质量 (历史名称 "SW-SRS:", 现指代表 latent 对 K_s 的对齐信号) ----
        # 重要: 这些指标的计算仅依赖 hidden × stage_keys, 不依赖具体 loss_mode!
        # 意义随 loss_mode 变化:
        #
        #   loss_mode='laser_dwal' (默认, Laser 官方):
        #     q_ent           → teacher 权重在 W_s 上的归一化熵 (越小越 focused)
        #     q_ent[early→late] → "先宽后窄" 体现 —— 期望 late < early
        #     topk_hit        → student argmax 是否落在本 stage K_s 内
        #
        #   loss_mode='sw_srs' (deprecated):
        #     q_ent           → student logits 在 W_s 上的 softmax 熵 (越小越 focused)
        #     q_ent[early→late] → 期望 late 少于 early
        #     topk_hit        → student argmax 是否落在本 stage K_s 内
        #
        #   loss_mode='stage_kw':
        #     q_ent           → 【P_target 的归一化熵 = log|K_s|/log|W_s|】——仅反映
        #                      窗口粒度 (数据驱动, 并非模型信号), 只在体现 "窗口由大变小"
        #     q_ent[early→late] → 同上, late > early 属正常 (W_s 变小 → 均匀分布熵变大)
        #                      【这与旧 SW-SRS 下 "期望变小" 的含义相反!】—— 不是坏事
        #     topk_hit        → 这才是 stage_kw 下【唯一有意义的模型信号】:
        #                      hidden top-1 是否落在 K_s 内, 越高越好 (足以代表对齐)
        _q_ent = draft_metrics.get('collapse/sw_srs_q_entropy_mean', None)
        _q_ent_early = draft_metrics.get('collapse/sw_srs_q_entropy_first_stage', None)
        _q_ent_late = draft_metrics.get('collapse/sw_srs_q_entropy_last_stage', None)
        _topk_hit = draft_metrics.get('collapse/sw_srs_topk_hit_ratio', None)
        # 读取当前 loss_mode (决定警告逻辑)
        _loss_mode = os.environ.get('NLD_LOSS_MODE', 'laser_dwal').lower().strip()
        _srs_parts = []
        if _q_ent is not None:
            _tag = ""
            # 在 laser_dwal / sw_srs 两种模式下 q_ent 都是模型信号, 可以报 peak/flat;
            # stage_kw 下 q_ent 是数据驱动, 不该报警.
            if _loss_mode in ('laser_dwal', 'sw_srs'):
                if _q_ent < 0.1:
                    _tag = " ⚠️peak"
                elif _q_ent > 0.95:
                    _tag = " ⚠️flat"
            _srs_parts.append(f"q_ent={_q_ent:.3f}{_tag}")
        # early/late 分桶:
        # - laser_dwal / sw_srs: 期望 late 显著 < early ("先宽后窄" 生效), 反之警告
        # - stage_kw         : late > early 是【数据几何】的必然结果 (W_s 由大变小),
        #                      不该警告; 只记录指标以供诊断
        #
        # 重要前提 (no-narrowing 误报修复):
        #   "先宽后窄" 是 Laser 原始数据契约下的现象, 即 K_s 嵌套收缩 (后一步是前一步的子集).
        #   但当前 v6 数据契约下, 各 stage 的 K_s 互相独立 (不嵌套), 物理上没有 narrowing 基础.
        #   独立 K_s 契约下 early/late 熵【不应】持续递减, Δ ≈ 0 是正常状态.
        #   因此引入两层 opt-out:
        #     1) NLD_DWAL_NARROWING_CHECK=0   → 完全关闭 narrowing 判定 (独立 K_s 契约)
        #     2) NLD_DWAL_NARROWING_MIN_K=N   → batch 内 stage 数(用 _ns_mean 近似) < N 时跳过
        #        (默认 3: 至少 3 stage 才有 narrowing 物理意义; K=2 数据天然跳过)
        #     3) NLD_DWAL_NARROWING_DELTA_THRESHOLD=X → 仅在 K≥N 时, 用 X 作为 Δ 下限 (默认 0.01)
        if _q_ent_early is not None and _q_ent_late is not None:
            _delta = _q_ent_early - _q_ent_late
            _tag = ""
            _narrowing_check = os.environ.get('NLD_DWAL_NARROWING_CHECK', '1').strip() not in ('0', 'false', 'False', '')
            _narrowing_min_k = float(os.environ.get('NLD_DWAL_NARROWING_MIN_K', '3'))
            _narrowing_delta_thr = float(os.environ.get('NLD_DWAL_NARROWING_DELTA_THRESHOLD', '0.01'))
            # K 近似: 直接从 draft_metrics 读 num_thought_steps_mean (本 batch 平均 stage 数).
            # 注: 不能用局部变量 _ns_mean —— 它在后续 Sat 段才赋值, 这里读取会 NameError.
            # 不可用时回退 999 (即默认通过 K 检查, 仅靠 _delta 判定).
            _ns_mean_for_check = draft_metrics.get('nld/num_thought_steps_mean', None)
            _approx_k = float(_ns_mean_for_check) if _ns_mean_for_check is not None else 999.0
            if (_narrowing_check
                    and _loss_mode in ('laser_dwal', 'sw_srs')
                    and _approx_k >= _narrowing_min_k
                    and _delta < _narrowing_delta_thr):
                _tag = " ⚠️no-narrowing"
            _srs_parts.append(f"q_ent[early→late]={_q_ent_early:.2f}→{_q_ent_late:.2f} (Δ={_delta:+.3f}){_tag}")
        if _topk_hit is not None:
            # topk_hit 是所有模式下有价值的收敛信号
            _hit_thr = 0.20 if _loss_mode in ('laser_dwal', 'stage_kw') else 0.05
            _tag = " ⚠️miss-K_s" if _topk_hit < _hit_thr else ""
            _srs_parts.append(f"topk_hit={_topk_hit:.3f}{_tag}")
        if _srs_parts:
            # 行首标签随 loss_mode 调整, 消除误导
            if _loss_mode == 'laser_dwal':
                _label = "Laser-DWAL"
            elif _loss_mode == 'stage_kw':
                _label = "Latent-K_s"
            else:
                _label = "SW-SRS"
            print(f"  {_label}: {', '.join(_srs_parts)}")

        # ---- Line 4.5: Stage Alignment Matrix (SAM, 真正监控 hidden 推理语义) ----
        # diag_score: 每步对应自己 stage 的 attention 概率均值 (期望 > 1/S)
        # monotonic:  argmax 沿 stage 对角线递增的比例 (期望 > 0.6)
        # shift_kl:   step 0 vs step_last 的 stage 分布 KL (期望 > 1.0, 证明语义在演化)
        _diag = draft_metrics.get('collapse/h_stage_diag_score', None)
        _mono = draft_metrics.get('collapse/h_stage_monotonic', None)
        _shift = draft_metrics.get('collapse/h_stage_shift_kl', None)
        _sam_parts = []
        if _diag is not None:
            _tag = " ⚠️random" if _diag < 0.25 else ""
            _sam_parts.append(f"diag={_diag:.3f}{_tag}")
        if _mono is not None:
            _tag = " ⚠️disorder" if _mono < 0.4 else ""
            _sam_parts.append(f"mono={_mono:.3f}{_tag}")
        if _shift is not None:
            # 'static' 阈值: shift_kl 反映 step 0 vs step_last 在 stage 维度上的 attn 分布 KL.
            # 期望 hidden 在多步演化中, 关注的 stage 分布形状显著变化.
            #   - K (per-sample stage 数) 大时(≥4), 阶段间差异充分, 阈值取 0.1 合理
            #   - K 小时(=2, 如 v6 b1+b2+b3 数据), 仅 2 个 stage, 分布天然差异有限,
            #     阈值放宽到 0.03 以避免误报.
            # 通过 NLD_SHIFT_KL_STATIC_THRESHOLD 环境变量覆盖 (默认 0.03, 适配 K=2).
            _shift_thr = float(os.environ.get('NLD_SHIFT_KL_STATIC_THRESHOLD', '0.03'))
            _tag = " ⚠️static" if _shift < _shift_thr else ""
            _sam_parts.append(f"shift_kl={_shift:.3f}{_tag}")
        if _sam_parts:
            print(f"  Stage:  {', '.join(_sam_parts)}")

        # ---- Line 4.6: Saturation 演化 + actual_steps (early-exit 病态预警) ----
        # sat_step1: 第一步就达到的饱和度 (越接近 1 越说明模型"想"立即 exit)
        # early_exit_ratio: 第 2 步就超过 saturation_exit_threshold 的比例
        # n_steps:   actual_steps 分布 (训练期 = num_stages, 推理期是关键诊断信号)
        _sat1 = draft_metrics.get('collapse/h_sat_step1', None)
        _sat_last = draft_metrics.get('collapse/h_sat_step_last', None)
        _early_exit = draft_metrics.get('collapse/h_sat_early_exit_ratio', None)
        _ns_mean = draft_metrics.get('nld/num_thought_steps_mean', None)
        _ns_min = draft_metrics.get('nld/num_thought_steps_min', None)
        _ns_max = draft_metrics.get('nld/num_thought_steps_max', None)
        _sat_parts = []
        if _sat1 is not None:
            _tag = " ⚠️fast-saturate" if _sat1 > 0.99 else ""
            _sat_parts.append(f"sat[1]={_sat1:.3f}{_tag}")
        if _sat_last is not None:
            _sat_parts.append(f"sat[L]={_sat_last:.3f}")
        if _early_exit is not None:
            _tag = " ⚠️stuck-at-2" if _early_exit > 0.7 else ""
            _sat_parts.append(f"early_exit={_early_exit:.2%}{_tag}")
        if _ns_mean is not None:
            _ns_str = f"n_steps={_ns_mean:.1f}"
            if _ns_min is not None and _ns_max is not None:
                _ns_str += f" (min={int(_ns_min)}, max={int(_ns_max)})"
            # 'under-stepping' 阈值: 训练期 actual_steps 由数据 num_stages 决定,
            # 实测均值显著低于数据预期 → 模型在某些样本上提前退出 (病态).
            # 不同数据集的 num_stages 分布不同, 阈值需匹配:
            #   - v6_5f (k_latent ≈ 4.5): 阈值 3.0
            #   - v6 b1+b2+b3 (K=2):      阈值 1.5  ← 当前数据
            # 通过 NLD_UNDER_STEPPING_THRESHOLD 环境变量覆盖 (默认 1.5, 适配 K=2).
            _ns_thr = float(os.environ.get('NLD_UNDER_STEPPING_THRESHOLD', '1.5'))
            if _ns_mean < _ns_thr:
                _ns_str += " ⚠️under-stepping"
            _sat_parts.append(_ns_str)
        if _sat_parts:
            print(f"  Sat:    {', '.join(_sat_parts)}")

        # ---- Line 5: Per-region CE & top-1 answer acc ----
        _think_ce = draft_metrics.get('loss/think_ce', None)
        _answer_ce = draft_metrics.get('loss/answer_ce', None)
        _top1_answer = draft_metrics.get('acc/top1_answer', None)
        _acc_parts = []
        if _think_ce is not None:
            _acc_parts.append(f"think_ce={_think_ce:.4f}")
        if _answer_ce is not None:
            _acc_parts.append(f"answer_ce={_answer_ce:.4f}")
        if _top1_answer is not None:
            _acc_parts.append(f"top1_ans={_top1_answer:.4f}")
        if _acc_parts:
            print(f"  CE/Acc: {', '.join(_acc_parts)} | thoughts={_thought_count:.0f}")

        print(flush=True)

    def create_optimizer(self):
        """创建差异化学习率的优化器 (全量微调, FSDP/DDP 兼容)"""
        model = self.model
        
        # 分组参数
        thinker_params = []
        vlm_params_decay = []
        vlm_params_no_decay = []
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "RMSNorm"]
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'latent_thinker' in name or 'thought_injector' in name or 'visual_probe' in name:
                thinker_params.append(param)
            else:
                if any(nd in name for nd in no_decay):
                    vlm_params_no_decay.append(param)
                else:
                    vlm_params_decay.append(param)
        
        _vlm_lr = self.vlm_lr if self.vlm_lr is not None else self.args.learning_rate
        
        # 构建参数组
        optimizer_grouped_parameters = []
        
        if thinker_params:
            optimizer_grouped_parameters.append({
                'params': thinker_params,
                'lr': self.thinker_lr,
                'weight_decay': self.args.weight_decay,
            })
        
        if vlm_params_decay:
            optimizer_grouped_parameters.append({
                'params': vlm_params_decay,
                'lr': _vlm_lr,
                'weight_decay': self.args.weight_decay,
            })
        
        if vlm_params_no_decay:
            optimizer_grouped_parameters.append({
                'params': vlm_params_no_decay,
                'lr': _vlm_lr,
                'weight_decay': 0.0,
            })
        
        if self._is_main_process():
            print(f"\n[NLD Optimizer] 参数组 (全量微调, FSDP/DDP):")
            print(f"  NativeLatentThinker: {len(thinker_params)} params, lr={self.thinker_lr}")
            print(f"  VLM (decay): {len(vlm_params_decay)} params, lr={_vlm_lr}")
            print(f"  VLM (no_decay): {len(vlm_params_no_decay)} params, lr={_vlm_lr}")
            print()
        
        optimizer_cls = torch.optim.AdamW
        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        return self.optimizer
