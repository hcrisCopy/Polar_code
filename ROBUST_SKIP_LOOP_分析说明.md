# 跨任务通用 Skip-Loop 结构分析

MCTS 评估顺序和 residual stream 的 mNN/CKA/PCA 可视化见
[`REPRESENTATION_可视化说明.md`](./REPRESENTATION_可视化说明.md)。

所有命令都在同时包含 `./Polar_code` 和 `./Polar_data` 的项目根目录执行。本流程不会改动 predictor；它回答的是“同一条固定层路径能否跨 DART-Math 难度迁移”。这里把 DM-1 至 DM-5 作为五个任务组。

先完成五个难度的 MCTS 搜索、合并和验证。10 题 smoke run 只能检查程序能否跑通：它只有 6 道 train、1 道 validation、3 道 test，且只有 DM-1，不能支持“跨任务鲁棒”的结论。

## 方法和数据边界

1. 只读取 train 的 MCTS valid/invalid 路径，按题归一化路径权重，统计每层 skip/keep/loop 的描述性倾向。
2. 从 train 冻结三类候选：排名靠前的单层 skip/loop、按难度形成的少量共识组合，以及跨题重复出现的完整成功路径。
3. 在 validation 和 test 上对每个候选真实执行基础 LLM，复用官方答案抽取和数学等价判断。候选仅按 validation 的“最差难度增益、宏平均增益、准确率、路径长度”依次选择。
4. test 只报告一次，绝不参与候选生成或选择。若 validation 最优候选在 test 上没有正增益，就应报告没有找到鲁棒通用结构。

MCTS 路径由自适应搜索产生，所以 train 层频率不能单独证明某层有效。单层 fixed-path 的 held-out 结果提供更直接的迁移证据。一次划分仍不足以证明模型无关的规律；需要重复数据种子和不同基础模型。

下列候选数、支持数、平滑量和 bootstrap 次数均为**本项目默认值，可调整**，不是论文公开的作者配置。固定路径评测量约为 `候选数 × (validation 题数 + test 题数)` 次生成；Llama-3.2-3B-Instruct 正式配置覆盖 28 层的全部 56 个单层 skip/loop，并给共识组合和完整路径保留候选，总上限 96，不会在运行时偷偷缩减。其他深度的模型应把 `--top-layers-per-action` 设为其完整层数，并相应增大两个 candidate 上限。

## 1. 静态环境检查

远程环境缺少依赖时使用 requirements 安装；这里是 `-r`，不是 `-e`：

```bash
python -m pip install -r ./Polar_code/stage_one/requirements.txt
```

```bash
python -B ./Polar_code/check_stage_one_static.py --run-name robust_static --clean
```

输出：`./Polar_data/runs/robust_static/environment/static_report.json`。只检查 AST、代码和参数关系，不加载模型。

## 2. 小规模流程测试

已有 `mcts_smoke` 的 10 题结果时，先生成最多 12 个候选：

```bash
python -B ./Polar_code/run_stage_one.py mine-programs --run-name mcts_smoke --max-candidates 12 --min-train-support 1 --max-consensus-edits 2 --top-layers-per-action 3 --smoothing 0.01 --clean
```

输出：`./Polar_data/runs/mcts_smoke/program_mining/`。

单卡真实执行这些固定路径：

```bash
mkdir -p ./Polar_data/runtime/launcher
TMPDIR=./Polar_data/runtime/launcher PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 torchrun --standalone --nproc_per_node=1 ./Polar_code/run_stage_one.py evaluate-programs --run-name mcts_smoke --model-id meta-llama/Llama-3.2-3B-Instruct --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --model-revision local-snapshot --seed 42 --max-new-tokens 50 --temperature 0 --evaluation-splits validation test --max-eval-candidates 12 --completion-timeout 604800 --clean
```

输出：`./Polar_data/runs/mcts_smoke/universal_eval/`。模型 ID、seed 和生成参数必须与原 MCTS search 一致；模型目录与 revision 标签可以变化，但应由用户保证仍是同一份权重。

生成 smoke 报告：

```bash
python -B ./Polar_code/run_stage_one.py report-programs --run-name mcts_smoke --bootstrap-samples 10000 --bootstrap-seed 42 --clean
```

输出：`./Polar_data/runs/mcts_smoke/program_report/`。这些小样本图只用于检查流程。

## 3. 正式候选挖掘

```bash
python -B ./Polar_code/run_stage_one.py mine-programs --run-name mcts_formal --max-candidates 96 --min-train-support 2 --max-consensus-edits 4 --top-layers-per-action 28 --smoothing 0.01 --clean
```

输出：`./Polar_data/runs/mcts_formal/program_mining/`，包括：

- `layer_statistics.json/csv`：各难度及全体 train 的层操作倾向；
- `robust_layer_hypotheses.json/csv`：train-only 单层候选排序；
- `candidates.json/csv`：已冻结且带校验和的待评测路径；
- `layer_action_heatmaps.svg/pdf`：train valid 路径的 skip/loop 分布。

## 4. 单卡正式评测

```bash
mkdir -p ./Polar_data/runtime/launcher
TMPDIR=./Polar_data/runtime/launcher PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 torchrun --standalone --nproc_per_node=1 ./Polar_code/run_stage_one.py evaluate-programs --run-name mcts_formal --model-id meta-llama/Llama-3.2-3B-Instruct --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --model-revision local-snapshot --seed 42 --max-new-tokens 50 --temperature 0 --evaluation-splits validation test --max-eval-candidates 96 --completion-timeout 604800 --clean
```

输出：`./Polar_data/runs/mcts_formal/universal_eval/rank_00000/`。每题完成后立即原子持久化，原命令重跑会校验并跳过完整题目。

## 5. 单机八卡正式评测

以下命令与单卡只改变进程数：

```bash
mkdir -p ./Polar_data/runtime/launcher
TMPDIR=./Polar_data/runtime/launcher PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 torchrun --standalone --nproc_per_node=8 ./Polar_code/run_stage_one.py evaluate-programs --run-name mcts_formal --model-id meta-llama/Llama-3.2-3B-Instruct --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --model-revision local-snapshot --seed 42 --max-new-tokens 50 --temperature 0 --evaluation-splits validation test --max-eval-candidates 96 --completion-timeout 604800 --clean
```

输出：`./Polar_data/runs/mcts_formal/universal_eval/rank_00000/` 至 `rank_00007/`。按 `global_index % world_size` 互斥分片，一进程一卡，不使用 DDP。

## 6. 生成报告和可视化

```bash
python -B ./Polar_code/run_stage_one.py report-programs --run-name mcts_formal --bootstrap-samples 10000 --bootstrap-seed 42 --clean
```

输出：`./Polar_data/runs/mcts_formal/program_report/`。重点查看：

- `summary.md`：validation 选出的结构、skip/loop 层和 test 增益；
- `report.json`、`candidate_metrics.csv`：全部候选的分难度结果；
- `single_layer_transfer.json/csv` 和 `single_layer_transfer.svg/pdf`：哪些单层 skip/loop 在 validation 最差难度及 test 上更稳；
- `selected_program_accuracy.svg/pdf`：各难度 baseline 与通用路径准确率；
- `selected_programs.csv` 和 `per_difficulty_program_layers.svg/pdf`：每个难度在 validation 上选出的最优路径与通用路径的结构对照；
- `accuracy_depth_tradeoff.svg/pdf`：路径长度与 validation 增益；
- `selected_program_layers.svg/pdf`：最终固定路径逐层 S/K/L 结构；
- `FIGURE_CAPTIONS.md`：图注和数据边界。

若要清理，只有显式加入 `--clean` 才会删除本阶段旧产物，且范围严格限制在对应的 `./Polar_data/runs/<run-name>/` 子目录。候选集、评测参数或 world size 变化时，恢复校验会拒绝混用旧分片。

## 7. 与 predictor 的关系

官方 predictor 学的是“每个 query 选择不同路径”；本流程寻找的是“所有 query 共用一条路径”。它们回答不同问题。分析结果不写回 `merged_mcts_samples.json`，也不改变 predictor 的训练标签、损失或推理逻辑。

## 一键全部运行

以下命令从已经完成并验证的 `mcts_formal` MCTS 结果开始；把 `--nproc_per_node=8` 改为 `1` 即可单卡运行，其他参数不变：

```bash
bash ./Polar_code/run_robust_program_analysis.sh --nproc_per_node=8 --run-name mcts_formal --model-id meta-llama/Llama-3.2-3B-Instruct --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --model-revision local-snapshot --clean
```

输出依次写入 `./Polar_data/runs/mcts_formal/{program_mining,universal_eval,program_report}/`。
