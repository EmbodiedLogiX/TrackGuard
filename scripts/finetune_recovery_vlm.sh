#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

MODE="${MODE:-single}"
MODEL_PRESET="${MODEL_PRESET:-qwen25_3b}"
DATASET_DIR="${DATASET_DIR:-data/recovery_mcq_dataset}"
LF_DATA_DIR="data"

case "${MODE}" in
  single)
    CONFIG="configs/recovery_vlm_lora.yaml"
    OUTPUT_DIR="runs/recovery_vlm_lora"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    unset FORCE_TORCHRUN 2>/dev/null || true
    ;;
  qlora)
    CONFIG="configs/recovery_vlm_qlora.yaml"
    OUTPUT_DIR="runs/recovery_vlm_qlora"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    unset FORCE_TORCHRUN 2>/dev/null || true
    ;;
  dual)
    CONFIG="configs/recovery_vlm_lora.yaml"
    OUTPUT_DIR="runs/recovery_vlm_lora"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    export FORCE_TORCHRUN=1
    ;;
  *)
    echo "unknown MODE=${MODE}; choose: single | qlora | dual"
    exit 1
    ;;
esac

export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

IFS=',' read -r TEMPLATE MODEL_PATH < <(python "${SCRIPT_DIR}/detect_template.py" --recommend --preset "${MODEL_PRESET}")

echo "=== environment check ==="
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"visible GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}, {p.total_memory/1024**3:.1f} GB")
PY

echo "MODE=${MODE}"
echo "MODEL_PRESET=${MODEL_PRESET}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TEMPLATE=${TEMPLATE}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "LF_DATA_DIR=${LF_DATA_DIR}"
echo "CONFIG=${CONFIG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

echo ""
echo "=== Step 1: register dataset into data/dataset_info.json ==="
python "${SCRIPT_DIR}/register_recovery_dataset.py" --dataset_dir "${DATASET_DIR}"

if [[ ! -f "${LF_DATA_DIR}/dataset_info.json" ]]; then
  echo "ERROR: ${LF_DATA_DIR}/dataset_info.json was not generated"
  exit 1
fi

echo ""
echo "=== Step 2: launch fine-tuning ==="
llamafactory-cli train "${CONFIG}" \
    model_name_or_path="${MODEL_PATH}" \
    output_dir="${OUTPUT_DIR}" \
    template="${TEMPLATE}" \
    dataset_dir="${LF_DATA_DIR}"

echo ""
echo "=== done ==="
echo "LoRA weights: ${OUTPUT_DIR}"
echo ""
echo "Evaluate the fine-tuned model:"
echo "  python scripts/eval_recovery_vlm.py \\"
echo "    --base_model ${MODEL_PATH} \\"
echo "    --lora_path ${OUTPUT_DIR} \\"
echo "    --data_json ${DATASET_DIR}/llama_factory/test.json"
echo ""
echo "Evaluate base models (before fine-tuning):"
echo "  python scripts/eval_recovery_vlm_baseline.py \\"
echo "    --data_json ${DATASET_DIR}/llama_factory/test.json"
