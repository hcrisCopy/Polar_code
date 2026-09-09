#!/usr/bin/env bash
# Capture fixed-program residual streams and render PRH-style alignment figures.
set -euo pipefail

nproc=""
run=""
model_id=""
model=""
model_revision=""
clean=()
while (($#)); do
    case "$1" in
        --nproc_per_node=*) nproc="${1#*=}"; shift ;;
        --run-name) run="$2"; shift 2 ;;
        --model-id) model_id="$2"; shift 2 ;;
        --model-path) model="$2"; shift 2 ;;
        --model-revision) model_revision="$2"; shift 2 ;;
        --clean) clean=(--clean); shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
for field in nproc run model_id model model_revision; do
    if [[ -z "${!field}" ]]; then echo "Missing required option: $field" >&2; exit 2; fi
done
[[ "$nproc" =~ ^[1-8]$ ]] || { echo "nproc_per_node must be 1..8" >&2; exit 2; }

mkdir -p ./Polar_data/runtime/launcher
export TMPDIR=./Polar_data/runtime/launcher
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

torchrun --standalone --nproc_per_node="$nproc" ./Polar_code/run_stage_one.py capture-representations \
    --run-name "$run" \
    --model-id "$model_id" \
    --model-path "$model" \
    --model-revision "$model_revision" \
    --seed 42 \
    --representation-splits validation test \
    --max-representation-questions 0 \
    --max-programs 8 \
    --pooling mean \
    --completion-timeout 604800 \
    "${clean[@]}"

python -B ./Polar_code/run_stage_one.py report-representations \
    --run-name "$run" \
    --neighbor-k 10 \
    --alignment-samples 500 \
    --projection-samples 100 \
    --max-search-questions 10 \
    --max-paths-per-question 256 \
    --seed 42 \
    "${clean[@]}"
