#!/usr/bin/env bash
# Evaluate exactly 12 frozen MCTS programs on 100 held-out questions per difficulty,
# then capture and analyse their residual streams.
set -euo pipefail

nproc=""
devices=""
source_run=""
run=""
data_path=""
source_revision=""
model_id=""
model_path=""
model_revision=""
clean=()

while (($#)); do
    case "$1" in
        --nproc_per_node=*) nproc="${1#*=}"; shift ;;
        --cuda-visible-devices=*) devices="${1#*=}"; shift ;;
        --source-run-name) source_run="$2"; shift 2 ;;
        --run-name) run="$2"; shift 2 ;;
        --data-path) data_path="$2"; shift 2 ;;
        --source-revision) source_revision="$2"; shift 2 ;;
        --model-id) model_id="$2"; shift 2 ;;
        --model-path) model_path="$2"; shift 2 ;;
        --model-revision) model_revision="$2"; shift 2 ;;
        --clean) clean=(--clean); shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

for field in nproc devices source_run run data_path source_revision model_id model_path model_revision; do
    if [[ -z "${!field}" ]]; then
        echo "Missing required option: $field" >&2
        exit 2
    fi
done
[[ "$nproc" =~ ^[1-8]$ ]] || { echo "nproc_per_node must be 1..8" >&2; exit 2; }
[[ "$source_run" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid source run name" >&2; exit 2; }
[[ "$run" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid run name" >&2; exit 2; }
[[ "$source_run" != "$run" ]] || { echo "Source and evaluation run names must differ" >&2; exit 2; }

IFS=',' read -r -a visible_devices <<< "$devices"
[[ "${#visible_devices[@]}" -eq "$nproc" ]] || {
    echo "Visible GPU count must equal nproc_per_node" >&2
    exit 2
}
export CUDA_VISIBLE_DEVICES="$devices"

mkdir -p ./Polar_data/runtime/launcher
export TMPDIR=./Polar_data/runtime/launcher
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

candidate_file="./Polar_data/runs/${source_run}/program_mining/candidates.json"
python -B -c 'import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    payload = json.load(stream)
count = len(payload.get("candidates", []))
if count != 12:
    raise SystemExit(f"Expected exactly 12 frozen candidates, found {count}: {path}")
print(f"Using 12 frozen candidates from {path}")' "$candidate_file"

echo "[1/5] Prepare 50 validation + 50 test questions for each DM difficulty"
python -B ./Polar_code/run_stage_one.py prepare \
    --run-name "$run" \
    --data-path "$data_path" \
    --data-source hkust-nlp/dart-math-pool-math \
    --source-revision "$source_revision" \
    --difficulties 1 2 3 4 5 \
    --seed 20260909 \
    --split-policy proportional \
    --train-fraction 0.5 \
    --validation-fraction 0.25 \
    --max-questions-per-diff 200 \
    --exclude-run-name "$source_run" \
    "${clean[@]}"

manifest_file="./Polar_data/runs/${run}/prepared/manifest.json"
python -B -c 'import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    payload = json.load(stream)
for difficulty in range(1, 6):
    counts = payload["split_counts"].get(str(difficulty), {})
    if counts.get("validation") != 50 or counts.get("test") != 50:
        raise SystemExit(
            f"DM-{difficulty} expected 50 validation + 50 test, found {counts}: {path}")
print("Prepared exactly 100 held-out questions for each of DM-1 through DM-5")' "$manifest_file"

echo "[2/5] Execute baseline and all 12 frozen programs on 500 held-out questions"
torchrun --standalone --nproc_per_node="$nproc" ./Polar_code/run_stage_one.py evaluate-programs \
    --run-name "$run" \
    --candidate-run-name "$source_run" \
    --model-id "$model_id" \
    --model-path "$model_path" \
    --model-revision "$model_revision" \
    --seed 42 \
    --max-new-tokens 50 \
    --temperature 0 \
    --evaluation-splits validation test \
    --max-eval-candidates 12 \
    --completion-timeout 604800 \
    "${clean[@]}"

echo "[3/5] Aggregate accuracy and skip-loop transfer results"
python -B ./Polar_code/run_stage_one.py report-programs \
    --run-name "$run" \
    --bootstrap-samples 1000 \
    --bootstrap-seed 42 \
    "${clean[@]}"

echo "[4/5] Capture baseline plus all 12 programs' residual streams"
torchrun --standalone --nproc_per_node="$nproc" ./Polar_code/run_stage_one.py capture-representations \
    --run-name "$run" \
    --model-id "$model_id" \
    --model-path "$model_path" \
    --model-revision "$model_revision" \
    --seed 42 \
    --representation-splits validation test \
    --max-representation-questions 0 \
    --max-programs 13 \
    --pooling mean \
    --completion-timeout 604800 \
    "${clean[@]}"

echo "[5/5] Build residual alignment and MCTS-path figures"
python -B ./Polar_code/run_stage_one.py report-representations \
    --run-name "$run" \
    --neighbor-k 10 \
    --alignment-samples 500 \
    --projection-samples 100 \
    --max-search-questions 10 \
    --max-paths-per-question 256 \
    --seed 42 \
    "${clean[@]}"

echo "Complete: ./Polar_data/runs/${run}/program_report and representation_report"
