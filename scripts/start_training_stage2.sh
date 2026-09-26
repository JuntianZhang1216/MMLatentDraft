#!/bin/bash
# ============================================================
#  NLD Stage-2 training launcher (generic)
#  ----------------------------------------------------------
#  - Loads the generic Stage-2 config under configs/.
#  - 8-GPU torchrun + FSDP (full_shard auto_wrap).
#  - Replace the <PATH_TO_*> placeholders inside the YAML
#    (model_path, train_json, resume_from_model_only) before
#    running, OR pass --config /path/to/your.yaml.
#
#  Usage:
#    bash scripts/start_training_stage2.sh
#    bash scripts/start_training_stage2.sh --gpus 4
#    bash scripts/start_training_stage2.sh --config /path/to/your.yaml
# ============================================================

set -e

# ======================== Step 1: Env ========================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TOKENIZERS_PARALLELISM=false

# ======================== Step 2: Paths ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/nld_train.yaml"
NUM_GPUS=8
MASTER_PORT=${MASTER_PORT:-29505}

# ======================== Step 3: Args ========================
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ======================== Step 4: Banner ========================
echo "============================================================"
echo "  NLD Stage-2 training"
echo "============================================================"
echo "  Config:       ${CONFIG}"
echo "  GPUs:         ${NUM_GPUS}"
echo "  Master port:  ${MASTER_PORT}"
echo "  Backend:      FSDP (full_shard auto_wrap)"
echo "============================================================"

if [ ! -f "${CONFIG}" ]; then
    echo "Config not found: ${CONFIG}"
    exit 1
fi

# Output dir
OUTPUT_DIR=$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["training"]["output_dir"])' "${CONFIG}")
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# ======================== Step 5: Launch ========================
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    "${SCRIPT_DIR}/train_nld.py" \
    --config "${CONFIG}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "============================================================"
echo "Training finished. Output: ${OUTPUT_DIR}"
echo "============================================================"
