#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-qwen3_path_showcase}"
CLEAN_FLAG="${2:-}"

python -B ./Polar_code/run_path_showcase.py prepare \
  --run-name "${RUN_NAME}" \
  --data-path ./Polar_data/raw/dart-math-pool-math \
  --source-revision local-files \
  --difficulties 1 2 3 4 5 \
  --seed 42 \
  ${CLEAN_FLAG}

python -B ./Polar_code/run_path_showcase.py search \
  --run-name "${RUN_NAME}" \
  --model-id Qwen/Qwen3-8B \
  --model-path ./Polar_data/models/Qwen/Qwen3-8B \
  --model-revision local-snapshot \
  --device 0 \
  --seed 42 \
  --candidate-limit 1024 \
  --batch-size 32 \
  --target-per-label 20 \
  --max-block 4 \
  --max-length-factor 1.15 \
  --max-new-tokens 50 \
  --temperature 0

python -B ./Polar_code/run_path_showcase.py report \
  --run-name "${RUN_NAME}" \
  --paths-per-label 10
