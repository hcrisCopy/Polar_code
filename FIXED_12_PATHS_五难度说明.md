# 12 条固定路径 × 五种难度

本流程读取 `mcts_smoke` 已冻结的 12 条候选路径，不再运行 MCTS。每个难度确定性准备 200 道题，其中 50 道 validation 和 50 道 test 用于路径评测，因此每个难度实际评测 100 道、总计 500 道。`mcts_smoke` 中出现过的全部题目会先排除。

第一步对每道题真实执行完整 baseline 和 12 条路径，共 6,500 次答案生成，并用仓库原有 DART-Math 判分。第二步对相同 500 道题执行 6,500 次无生成 prefill，捕获 post-block residual，再生成 mNN、CKA、PCA 和路径图。因此这是小规模正式评测，不是几分钟即可完成的 10 题连通性检查。

在同时包含 `./Polar_code` 和 `./Polar_data` 的目录执行。当前窗口固定使用物理 GPU 1：

```bash
bash ./Polar_code/run_12_paths_five_difficulties.sh --nproc_per_node=1 --cuda-visible-devices=1 --source-run-name mcts_smoke --run-name paths12_dm100 --data-path ./Polar_data/raw/dart-math-pool-math --source-revision local-files --model-id meta-llama/Llama-3.2-3B-Instruct --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --model-revision modelscope-master --clean
```

首次运行使用 `--clean`。中断后原命令重跑时删除末尾 `--clean`，程序会验证并跳过已经完成的逐题结果。脚本启动时会确认来源候选数恰好为 12；若不是 12 会直接停止。

第一步输出：

- `./Polar_data/runs/paths12_dm100/prepared/`：抽样清单及五个难度的 `100/50/50` train/validation/test 计数。
- `./Polar_data/runs/paths12_dm100/universal_eval/`：逐题 baseline 和 12 条路径的真实判分。
- `./Polar_data/runs/paths12_dm100/program_report/`：validation 选择、test 报告、各难度 accuracy/gain 和 skip-loop 图。其中 `all_candidate_program_layers.svg`（另有 PDF）把全部 12 条候选路径画在同一张图中：每行一条路径，每列一个原始层，S/K/L 分别表示 skip/keep/loop，右侧是 validation/test 相对完整层基线的 accuracy gain。

第二步输出：

- `./Polar_data/runs/paths12_dm100/representations/`：逐题压缩 residual NPZ。
- `./Polar_data/runs/paths12_dm100/representation_report/`：layer-to-layer mNN、分难度 mNN/CKA、PCA、accuracy-alignment 图和 CSV/JSON。

脚本中的抽样种子 `20260909`、bootstrap 1,000 次、mNN `k=10`、alignment 500 题和 projection 100 题均为本项目可调整值，不是论文作者公开配置。答案生成仍使用来源 MCTS 的 seed 42、50 tokens 和 temperature 0，确保路径评测条件一致。
