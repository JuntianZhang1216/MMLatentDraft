"""
Interlude —— 基于 VLM 原生隐空间的自适应多步推理解码

核心模块:
- model_v2: NLD 模型 (VLM 原生隐空间思考 + 全量微调)
- latent_thinker: NativeLatentThinker 模块 (Coconut 式隐空间推理)
- data: 数据集与 Collator
- trainer_nld: NLD 专用 Trainer (DeepSpeed ZeRO 兼容)
"""

from .latent_thinker import NativeLatentThinker, ThoughtPrefixInjector
from .model_v2 import NLDModel
from .data import NLDDataset, NLDCollator, LATENT_TOKEN
from .trainer_nld import NLDTrainer

__version__ = "0.4.0"
