#!/usr/bin/env bash
# Run from the project root. This script installs/downloads nothing.
set -euo pipefail

nproc=""
run=""
data=""
model=""
model_id=""
model_revision=""
source_revision=""
limit=""
diffs=""
train_predictor=""
predictor_config=""
clean=()
while (($#)); do
    case "$1" in
        --nproc_per_node=*) nproc="${1#*=}"; shift ;;
        --run-name) run="$2"; shift 2 ;;
        --data-path) data="$2"; shift 2 ;;
        --model-path) model="$2"; shift 2 ;;
        --model-id) model_id="$2"; shift 2 ;;
        --model-revision) model_revision="$2"; shift 2 ;;
        --source-revision) source_revision="$2"; shift 2 ;;
        --max-questions-per-diff) limit="$2"; shift 2 ;;
        --difficulties) diffs="$2"; shift 2 ;;
        --train-predictor) train_predictor="$2"; shift 2 ;;
        --predictor-config) predictor_config="$2"; shift 2 ;;
        --clean) clean=(--clean); shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
for field in nproc run data model model_id model_revision source_revision limit diffs train_predictor predictor_config; do
    if [[ -z "${!field}" ]]; then echo "Missing required option: $field" >&2; exit 2; fi
done
[[ "$nproc" =~ ^[1-8]$ ]] || { echo "nproc_per_node must be 1..8" >&2; exit 2; }
[[ "$run" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || exit 2
[[ "$train_predictor" == true || "$train_predictor" == false ]] || exit 2
read -r -a difficulty_args <<< "$diffs"

# Launcher-created temporary directories are also explicitly contained.
[[ ! -L ./Polar_data && ! -L ./Polar_data/runtime && ! -L ./Polar_data/runtime/launcher ]] || exit 2
mkdir -p ./Polar_data/runtime/launcher
export TMPDIR=./Polar_data/runtime/launcher
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

if ((${#clean[@]})); then
    # Explicit stage-local cleanup before rank 0's automatic merge.
    python -B ./Polar_code/run_stage_one.py merge --run-name "$run" --clean --clean-only
    python -B ./Polar_code/run_stage_one.py validate --run-name "$run" --clean --clean-only
fi

python -B ./Polar_code/run_stage_one.py environment --run-name "$run" --model-path "$model" --data-path "$data" "${clean[@]}"
python -B ./Polar_code/run_stage_one.py prepare --run-name "$run" --data-path "$data" --data-source hkust-nlp/dart-math-pool-math --source-revision "$source_revision" --difficulties "${difficulty_args[@]}" --seed 42 --split-policy proportional --train-fraction 0.625 --validation-fraction 0.125 --max-questions-per-diff "$limit" "${clean[@]}"
# Every scientific/search option is visible here; these are project defaults.
torchrun --standalone --nproc_per_node="$nproc" ./Polar_code/run_stage_one.py search --run-name "$run" --model-id "$model_id" --model-path "$model" --model-revision "$model_revision" --seed 42 --simulations 1024 --exploration 1.4142135623730951 --length-penalty 0.1 --max-block 4 --max-repeats 4 --max-length-factor 1.15 --max-new-tokens 50 --temperature 0 --completion-timeout 604800 "${clean[@]}"
python -B ./Polar_code/run_stage_one.py merge --run-name "$run" "${clean[@]}"
python -B ./Polar_code/run_stage_one.py validate --run-name "$run" "${clean[@]}"
if [[ "$train_predictor" == true ]]; then
    python -B ./Polar_code/train_stage_one_predictor.py --run-name "$run" --predictor-config "$predictor_config" "${clean[@]}"
fi
