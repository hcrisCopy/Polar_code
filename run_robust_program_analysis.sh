#!/usr/bin/env bash
# Mine train-only hypotheses, evaluate frozen programs, and render held-out figures.
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
[[ "$run" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || exit 2

[[ ! -L ./Polar_data && ! -L ./Polar_data/runtime && ! -L ./Polar_data/runtime/launcher ]] || exit 2
mkdir -p ./Polar_data/runtime/launcher
export TMPDIR=./Polar_data/runtime/launcher
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

# These are project analysis defaults and are not reported as author settings.
python -B ./Polar_code/run_stage_one.py mine-programs \
    --run-name "$run" \
    --max-candidates 96 \
    --min-train-support 2 \
    --max-consensus-edits 4 \
    --top-layers-per-action 28 \
    --smoothing 0.01 \
    "${clean[@]}"

torchrun --standalone --nproc_per_node="$nproc" ./Polar_code/run_stage_one.py evaluate-programs \
    --run-name "$run" \
    --model-id "$model_id" \
    --model-path "$model" \
    --model-revision "$model_revision" \
    --seed 42 \
    --max-new-tokens 50 \
    --temperature 0 \
    --evaluation-splits validation test \
    --max-eval-candidates 96 \
    --completion-timeout 604800 \
    "${clean[@]}"

python -B ./Polar_code/run_stage_one.py report-programs \
    --run-name "$run" \
    --bootstrap-samples 10000 \
    --bootstrap-seed 42 \
    "${clean[@]}"
