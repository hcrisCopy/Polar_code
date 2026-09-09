# 前 10 条正确/错误路径展示

本流程只完成老师标绿的第一个汇报点：Qwen3-8B-Instruct 关闭思考模式，DART-Math 五个难度各确定性抽一题；每题最多准备 1,024 条候选路径，先评估 32 条，此后每次增加 32 条，得到至少 20 条正确和 20 条错误路径后立即停止。复杂度暂按路径长度衡量，每题输出“最简单 10 条”和“最复杂 10 条”两张图，两组路径严格不重合。

这里沿用模型发布名 `Qwen/Qwen3-8B`；它通过 chat template 的 `enable_thinking=False` 作为 instruct 非推理模式运行。

这不是全量实验，也不声称复现论文 MCTS。候选顺序混合单次局部 skip/loop 与 2–6 次编辑的路径，比当前仓库中从根节点巨大动作集随机弹出候选更适合快速凑齐正负样本。路径块长不超过 4，最长执行深度为原模型的 115%。

所有命令在同时包含 `./Polar_code` 和 `./Polar_data` 的远程服务器目录执行，先进入已有环境：

```bash
conda activate polar
```

## 1. 抽取五道题

```bash
python -B ./Polar_code/run_path_showcase.py prepare \
  --run-name qwen3_path_showcase \
  --data-path ./Polar_data/raw/dart-math-pool-math \
  --source-revision local-files \
  --difficulties 1 2 3 4 5 \
  --seed 42 \
  --clean
```

输出：`./Polar_data/runs/qwen3_path_showcase/path_showcase/questions.json`。

## 2. 单卡分批搜索

```bash
CUDA_VISIBLE_DEVICES=0 python -B ./Polar_code/run_path_showcase.py search \
  --run-name qwen3_path_showcase \
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
```

输出：`./Polar_data/runs/qwen3_path_showcase/path_showcase/search_state.json`。每生成一条路径就原子保存，中断后原命令重跑即可继续；不要加 `--clean`。每 32 条集中调用一次数学判分，避免每条路径重复创建判分进程。

## 3. 生成图片

```bash
python -B ./Polar_code/run_path_showcase.py report \
  --run-name qwen3_path_showcase \
  --paths-per-label 10
```

输出：`./Polar_data/runs/qwen3_path_showcase/path_showcase/figures/` 中每个难度两张 PNG，以及同目录下的 `summary.md`、`report.json`、`report.csv` 和 `path_selections.json`。最后一个文件保留首批发现、最简单及最复杂集合的完整路径、模型原始输出和抽取答案。

图中灰色表示跳过，蓝色表示执行一次，橙色表示 loop；绿色行名为正确路径，红色行名为错误路径。纵轴同时标出候选编号、路径长度和发现顺序。

## 一键运行

首次运行：

```bash
CUDA_VISIBLE_DEVICES=0 bash ./Polar_code/run_path_showcase.sh qwen3_path_showcase --clean
```

中断恢复：

```bash
CUDA_VISIBLE_DEVICES=0 bash ./Polar_code/run_path_showcase.sh qwen3_path_showcase
```

注意：为了保证两张图不重复，必须先获得每类至少 20 条路径。因此报告阶段会检查 `target-per-label >= 2 × paths-per-label`，不满足时直接停止并说明原因。
