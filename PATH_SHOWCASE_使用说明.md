# 前 10 条正确/错误路径展示

本流程只完成老师标绿的第一个汇报点：Qwen3-8B-Instruct 关闭思考模式，DART-Math 五个难度各确定性抽一题；每题使用 `stage_one.search_tree.search` 运行最多 1,024 次 MCTS simulation。每新评估 32 条唯一路径检查一次，得到至少 20 条正确和 20 条错误路径后立即停止。复杂度暂按路径长度衡量，每题输出“最简单 10 条”和“最复杂 10 条”两张图，两组路径严格不重合。

这里沿用模型发布名 `Qwen/Qwen3-8B`；它通过 chat template 的 `enable_thinking=False` 作为 instruct 非推理模式运行。

这不是全量实验。搜索直接复用当前仓库 `stage_one` 的 MCTS：从完整层路径出发，以 UCB 选择节点，通过连续块 skip/repeat 扩展并回传二值正确性 reward。论文没有公开全部搜索超参数，因此 UCB 系数、长度惩罚和 1,024 次上限仍是本项目配置。路径块长和重复次数不超过 4，最长执行深度为原模型的 115%。

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
  --simulations 1024 \
  --check-interval 32 \
  --max-question-seconds 600 \
  --target-per-label 20 \
  --exploration 1.4142135623730951 \
  --length-penalty 0.1 \
  --max-block 4 \
  --max-repeats 4 \
  --max-length-factor 1.15 \
  --max-new-tokens 50 \
  --temperature 0
```

输出：`./Polar_data/runs/qwen3_path_showcase/path_showcase/search_state.json`。每评估一条路径就原子保存。中断后原命令会用同一随机种子快速重放 MCTS 树；已见路径直接读取正确性缓存，不重复运行模型，然后从断点之后继续。MCTS 的下一步依赖上一条路径的 reward，因此路径必须依次判分；代码复用同一个判分器并设置 60 秒单次判分上限。每题累计模型生成与判分最多 600 秒，超时会保存 `time_limit_reached`，防止为了追满 1,024 次而卡数小时。

## 3. 生成图片

```bash
python -B ./Polar_code/run_path_showcase.py report \
  --run-name qwen3_path_showcase \
  --paths-per-label 10
```

输出：`./Polar_data/runs/qwen3_path_showcase/path_showcase/figures/` 中每个难度两张 PNG，以及同目录下的 `summary.md`、`report.json`、`report.csv` 和 `path_selections.json`。最后一个文件保留首批发现、最简单及最复杂集合的完整路径、模型原始输出和抽取答案。

图中灰色表示跳过，蓝色表示执行一次，橙色表示 loop；绿色行名为正确路径，红色行名为错误路径。纵轴同时标出候选编号、路径长度和发现顺序。

搜索尚未全部结束时，可以在另一个终端只生成已经完成难度的图片。例如 DM-1 已完成而 DM-2 正在搜索时执行：

```bash
python -B ./Polar_code/run_path_showcase.py report \
  --run-name qwen3_path_showcase \
  --paths-per-label 10 \
  --difficulties 1
```

这只读取一次原子保存的状态快照，不加载模型，也不会打断 MCTS。局部报告基于当前已有结果：每类不足 10 条时有多少画多少，不足 20 条时允许“最简单”和“最复杂”集合重合，并在报告中给出重合数。图片仍写入 `figures/dm1_*.png`；局部汇总使用 `summary_dm1.md`、`report_dm1.json`、`report_dm1.csv` 和 `path_selections_dm1.json`，不会覆盖最终五难度汇总。

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
