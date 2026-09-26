#!/bin/bash
# ============================================================
# NLD Phase 1 训练启动脚本
# 
# Interlude: 基于 VLM 原生隐空间的自适应多步推理解码
#   - FSDP full_shard + auto_wrap (替代 DeepSpeed)
#   - 全量微调 VLM 8.3B + NativeLatentThinker ~2M
#   - 差异化学习率: thinker 1e-4, VLM 2e-5
#
# 用法:
#   bash scripts/start_training_stage1.sh
#   bash scripts/start_training_stage1.sh --resume /path/to/checkpoint
# ============================================================

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${PROJECT_DIR}/configs/nld_train.yaml"
NUM_GPUS=8
MASTER_PORT=${MASTER_PORT:-29500}

# 解析命令行参数
RESUME_ARG=""
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_ARG="--resume_from_checkpoint $2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

echo "============================================================"
echo "🚀 NLD Phase 1 训练 (Interlude)"
echo "============================================================"
echo "  配置文件:     ${CONFIG}"
echo "  GPU 数量:     ${NUM_GPUS}"
echo "  Master Port:  ${MASTER_PORT}"
echo "  分布式后端:   FSDP (full_shard auto_wrap)"
if [ -n "${RESUME_ARG}" ]; then
    echo "  恢复训练:     ${RESUME_ARG}"
fi
echo "============================================================"

# 检查配置文件
if [ ! -f "${CONFIG}" ]; then
    echo "❌ 配置文件不存在: ${CONFIG}"
    exit 1
fi

# Train data path (from YAML); existence is verified by the trainer itself.
TRAIN_JSON=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['data']['train_json'])")
echo ""
echo "  Train data:   ${TRAIN_JSON}"

echo ""
echo "============================================================"
echo "🚀 开始训练..."
echo "============================================================"

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    "${SCRIPT_DIR}/train_nld.py" \
    --config "${CONFIG}" \
    ${RESUME_ARG} \
    ${EXTRA_ARGS}

echo ""
echo "============================================================"
echo "✅ NLD Phase 1 训练完成！"
echo "============================================================"
