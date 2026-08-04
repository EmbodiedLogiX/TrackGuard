#!/usr/bin/env bash
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-}"
ENV_NAME="${ENV_NAME:-py310}"
GPU="${GPU:-0}"
FORCE="${FORCE:-0}"

DATASETS="${DATASETS:-alpha beta gamma}"
METHODS="${METHODS:-iou clip dino router ours}"

VAL_COUNT="${VAL_COUNT:-300}"
TEST_COUNT="${TEST_COUNT:-1000}"
N_VAL_SEQS="${N_VAL_SEQS:-2}"
HISTORY_FRAMES="${HISTORY_FRAMES:-100}"
MAX_OPTIONS="${MAX_OPTIONS:-4}"
SEED="${SEED:-42}"

DATA_ROOT="${DATA_ROOT:-data/reid}"
OUT_ROOT="${OUT_ROOT:-runs/reid}"

if [ -n "${CONDA_SH}" ] && [ -f "${CONDA_SH}" ]; then
    source "${CONDA_SH}"
fi
cd "${REPO_DIR}"

want_method() { for m in ${METHODS}; do [ "$m" = "$1" ] && return 0; done; return 1; }

declare -a ROW_TAG ROW_STEP ROW_STATUS ROW_SECS

run_step() {
    local tag="$1" name="$2" log="$3"; shift 3
    local -a envkv=()
    while [ "$1" != "--" ]; do envkv+=("$1"); shift; done
    shift
    echo ""
    echo "── [${tag}] ${name} ──────────────────────────────────"
    echo "   cmd: $*"
    local t0=$SECONDS
    env "${envkv[@]}" CUDA_VISIBLE_DEVICES="${GPU}" "$@" > "${log}" 2>&1
    local rc=$?
    local secs=$((SECONDS - t0))
    ROW_TAG+=("${tag}"); ROW_STEP+=("${name}"); ROW_SECS+=("${secs}")
    if [ $rc -eq 0 ]; then
        ROW_STATUS+=("OK"); echo "   done (${secs}s)"; tail -n 4 "${log}" | sed 's/^/     /'
    else
        ROW_STATUS+=("FAIL(${rc})"); echo "   failed (rc=${rc}, ${secs}s)"; tail -n 12 "${log}" | sed 's/^/     /'
    fi
    return $rc
}

echo "═══════════════════════════════════════════════════════════════"
echo "  Cross-dataset ReID benchmark"
echo "  DATASETS='${DATASETS}'  METHODS='${METHODS}'  GPU=${GPU}  FORCE=${FORCE}"
echo "═══════════════════════════════════════════════════════════════"

for TAG in ${DATASETS}; do
    ROOT="${DATA_ROOT}/${TAG}"
    if [ ! -d "${ROOT}" ]; then
        echo "skip missing dataset ${TAG} (root='${ROOT}')"; continue
    fi
    OUT="${OUT_ROOT}/${TAG}"
    MCQ_DIR="${OUT}/mcq"
    REID_CSV="${OUT}/reid_dataset.csv"
    FEAT_ROOT="${OUT}/features"
    LOG_DIR="${OUT}/logs"
    mkdir -p "${LOG_DIR}"

    echo ""
    echo "###############################################################"
    echo "#  dataset ${TAG}  root=${ROOT}  out=${OUT}"
    echo "###############################################################"

    have() { [ "${FORCE}" != "1" ] && [ -f "${OUT}/baseline_$1_recovered.csv" ]; }

    if want_method iou && ! have iou; then
        run_step "${TAG}" "iou" "${LOG_DIR}/iou.log" \
            "REID_CSV=${REID_CSV}" "REID_DATASET_NAME=${TAG}" \
            "REID_IOU_OUT_CSV=${OUT}/baseline_iou_recovered.csv" -- \
            python -m trackguard.gating.boxes
    fi
    if want_method clip && ! have clip; then
        run_step "${TAG}" "clip" "${LOG_DIR}/clip.log" \
            "REID_CSV=${REID_CSV}" "REID_DATASET_NAME=${TAG}" \
            "REID_CLIP_OUT_CSV=${OUT}/baseline_clip_recovered.csv" -- \
            echo "clip baseline placeholder for ${TAG}"
    fi
    if want_method dino && ! have dino; then
        run_step "${TAG}" "dino_extract" "${LOG_DIR}/dino_extract.log" \
            "REID_CSV=${REID_CSV}" "FEAT_ROOT=${FEAT_ROOT}" -- \
            echo "feature extraction placeholder for ${TAG}"
        run_step "${TAG}" "dino" "${LOG_DIR}/dino.log" \
            "REID_CSV=${REID_CSV}" "FEAT_ROOT=${FEAT_ROOT}" \
            "REID_DINO_OUT_CSV=${OUT}/baseline_dino_recovered.csv" -- \
            echo "dino baseline placeholder for ${TAG}"
    fi
    if want_method router; then
        run_step "${TAG}" "router" "${LOG_DIR}/router.log" \
            "REID_DATASET_NAME=${TAG}" -- \
            python scripts/train_router.py --data_dir "${ROOT}" \
                --out_dir "${OUT}/router" --epochs "${EPOCHS:-4}"
    fi
    if want_method ours; then
        run_step "${TAG}" "ours" "${LOG_DIR}/ours.log" \
            "REID_DATASET_NAME=${TAG}" "MAX_OPTIONS=${MAX_OPTIONS}" \
            "HISTORY_FRAMES=${HISTORY_FRAMES}" -- \
            echo "cascade evaluation placeholder for ${TAG}"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  run summary"
echo "═══════════════════════════════════════════════════════════════"
n_fail=0
for i in "${!ROW_TAG[@]}"; do
    printf "  %-8s %-16s %-12s %5ss\n" "${ROW_TAG[$i]}" "${ROW_STEP[$i]}" "${ROW_STATUS[$i]}" "${ROW_SECS[$i]}"
    [[ "${ROW_STATUS[$i]}" == FAIL* ]] && n_fail=$((n_fail + 1))
done
echo "═══════════════════════════════════════════════════════════════"
if [ "${n_fail}" -eq 0 ]; then echo "all steps completed"; else echo "${n_fail} step(s) failed"; fi
exit "${n_fail}"
