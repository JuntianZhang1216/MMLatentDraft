#!/bin/bash
# ============================================================
# NLD 推理启动脚本 (单图问答 / 批量)
#
# 用法:
#   # 单图问答
#   bash scripts/start_inference.sh --model_path /path/to/model --checkpoint /path/to/checkpoint --image /path/to/image.jpg --question "What is shown?"
#
#   # 批量
#   bash scripts/start_inference.sh --model_path /path/to/model --checkpoint /path/to/checkpoint --batch_file /path/to/queries.json --output_file results.json
# ============================================================

set -e

echo "🔍 NLD 推理"
echo ""

# ======================== 默认参数 ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="<PATH_TO_QWEN3_VL_8B_INSTRUCT>"
NLD_CHECKPOINT="./outputs/nld_train/model"
DEVICE="cuda"

# 透传给 scripts/inference.py 的额外参数
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)        MODEL_PATH="$2";       shift 2 ;;
        --nld_checkpoint|--checkpoint) NLD_CHECKPOINT="$2"; shift 2 ;;
        --device)            DEVICE="$2";           shift 2 ;;
        *)                   EXTRA_ARGS+=("$1");    shift ;;
    esac
done

# ======================== 检测 Python ========================
if command -v python3 &> /dev/null; then PYTHON=python3; else PYTHON=python; fi

# ======================== 运行前检查 ========================
if [ ! -d "${MODEL_PATH}" ]; then
    echo "❌ 基座模型路径不存在: ${MODEL_PATH}"
    exit 1
fi

if [ ! -d "${NLD_CHECKPOINT}" ]; then
    echo "❌ NLD 权重路径不存在: ${NLD_CHECKPOINT}"
    exit 1
fi

echo "============================================================"
echo "  基座模型:     ${MODEL_PATH}"
echo "  NLD 权重:     ${NLD_CHECKPOINT}"
echo "  设备:         ${DEVICE}"
echo "  额外参数:     ${EXTRA_ARGS[*]}"
echo "============================================================"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PYTHON "${SCRIPT_DIR}/inference.py" \
    --model_path "${MODEL_PATH}" \
    --nld_checkpoint "${NLD_CHECKPOINT}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "✅ 推理完成"
