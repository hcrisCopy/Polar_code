# 正确路径与最相似错误路径配对展示

本流程只完成老师标绿的第一个汇报点：Qwen3-8B-Instruct 关闭思考模式，DART-Math 五个难度各确定性抽一题；每题使用 `stage_one.search_tree.search` 运行最多 1,024 次 MCTS simulation。每新评估 32 条唯一路径检查一次，得到至少 20 条正确和 20 条错误路径后立即停止。复杂度暂按正确路径长度衡量，每题输出两张图：最短 10 条正确路径及其最相似错误路径、最长 10 条正确路径及其最相似错误路径。20 条正确路径和 20 条错误路径均不重复。

这里沿用模型发布名 `Qwen/Qwen3-8B`；它通过 chat template 的 `enable_thinking=False` 作为 instruct 非推理模式运行。

这不是全量实验。搜索直接复用当前仓库 `stage_one` 的 MCTS：从完整层路径出发，以 UCB 选择节点，在全部原始层上通过连续块 skip/loop 扩展并回传二值正确性 reward。候选必须能被原项目路径解析器表示：每段最多 4 个连续原始层，loop 段固定整体执行两遍（`×2`），不会生成 `×3/×4`。论文没有公开全部搜索超参数，因此 UCB 系数、长度惩罚和 1,024 次上限仍是本项目配置；最长执行深度为原模型的 115%。

错误路径匹配使用全局一对一最优分配，不逐条贪心，也不重复使用错误路径。相似度依次比较：逐层 S/K/L 操作差异（Skip 与 Loop 的互换代价为 2，其他不同操作为 1）、实际执行层序列的 Levenshtein 编辑距离、路径长度差。

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
  --max-repeats 1 \
  --paths-per-figure 10 \
  --max-length-factor 1.15 \
  --max-new-tokens 50 \
  --temperature 0
```

输出：`./Polar_data/runs/qwen3_path_showcase/path_showcase/search_state.json`。每评估一条路径就原子保存。中断后原命令会用同一随机种子快速重放 MCTS 树；已见路径直接读取正确性缓存，不重复运行模型，然后从断点之后继续。MCTS 的下一步依赖上一条路径的 reward，因此路径必须依次判分；代码复用同一个判分器并设置 60 秒单次判分上限。每题累计模型生成与判分最多 600 秒，超时会保存 `time_limit_reached`，防止为了追满 1,024 次而卡数小时。

每完成一个难度（达到配额、搜索次数上限或时间上限），程序立即生成该难度的两张配对图和 `summary_dmN.md`，然后继续下一个难度。`visualization_status_dmN.json` 记录该难度是否得到完整 10 对、部分配对或路径不足；可视化失败会明确记录，但不会丢失搜索结果或阻断后续难度。

## 3. 生成图片

搜索阶段已经在每个难度完成后自动生成对应图片。只有五个难度都达到 20 条正确和 20 条错误配额、需要重新汇总最终报告时，才单独执行：

```bash
python -B ./Polar_code/run_path_showcase.py report \
  --run-name qwen3_path_showcase \
  --paths-per-label 10
```

输出：`./Polar_data/runs/qwen3_path_showcase/path_showcase/figures/` 中每个难度两张图，每张同时提供 PNG、SVG 和 PDF，以及同目录下的 `summary.md`、`report.json`、`report.csv` 和 `path_selections.json`。最后一个文件保存每个正确—错误对的完整路径、模型原始输出、抽取答案和三项距离。

图的视觉语义沿用原项目 `search_space.png`：蓝色输入、橙色输出、白色实线层表示执行一次、灰色虚线层表示 skip。一个浅绿色外框覆盖完整连续 loop 段，框内回环箭头和 `×2` 表示该段按原顺序整体执行两遍。每个 Pair 的正确路径与匹配错误路径上下相邻，通过左侧括号、`P01-C/P01-W`、绿色/红色三重编码；错误行同时标出 `Δop`、`edit` 和 `Δlen`。

搜索尚未全部结束时，可以在另一个终端只生成已经完成难度的图片。例如 DM-1 已完成而 DM-2 正在搜索时执行：

```bash
python -B ./Polar_code/run_path_showcase.py report \
  --run-name qwen3_path_showcase \
  --paths-per-label 10 \
  --difficulties 1
```

这只读取一次原子保存的状态快照，不加载模型，也不会打断 MCTS。局部报告基于当前已有结果：不足 20 条正确或错误路径时，两张图按能够组成的不重合配对数等量缩减，不复用路径。图仍写入 `figures/dm1_*`，每张都有 PNG、SVG 和 PDF；局部汇总使用 `summary_dm1.md`、`report_dm1.json`、`report_dm1.csv` 和 `path_selections_dm1.json`，不会覆盖最终五难度汇总。

旧版结果仍可用上述局部报告查看，但会排除不能由官方 `×2` 连续段语法表示的路径。正式重跑必须清理旧 run，因为搜索配置和报告结构已经改变。

## 4. 对 DM-1 图中相同路径开启 Thinking 复评

这一步不重新搜索，也不改变路径。它读取 `path_selections_dm1.json` 中两张图实际展示的 40 条不重复路径，使用同一模型、同一道题和同一判分器，把 Qwen3 chat template 改为 `enable_thinking=True` 后逐条重新生成。Thinking 提示词要求模型先完成推理，再在推理之后只输出 boxed 最终答案，避免原 non-thinking 提示词中的“只输出答案”抑制思考；原搜索提示词保持不变。

Thinking 复评使用 `max-new-tokens=2048` 作为安全上限，让模型通常依靠 EOS 自然结束，避免 50-token 上限干扰其真实行为。若异常轨迹仍达到上限且没有输出 boxed answer，结果记为 `truncated`，不误判为错误。

```bash
CUDA_VISIBLE_DEVICES=0 python -B ./Polar_code/run_path_showcase.py think-eval \
  --run-name qwen3_path_showcase \
  --difficulty 1 \
  --model-id Qwen/Qwen3-8B \
  --model-path ./Polar_data/models/Qwen/Qwen3-8B \
  --model-revision local-snapshot \
  --device 0 \
  --seed 42 \
  --max-new-tokens 2048 \
  --temperature 0 \
  --max-total-seconds 1200 \
  --clean
```

输出：`./Polar_data/runs/qwen3_path_showcase/path_showcase/thinking_eval/dm1/`。`summary.md` 和 `comparison.csv` 给出每条路径从 non-thinking 到 thinking 的正确性变化；`state.json` 保留原始 completion、按 Qwen3 `</think>` token 拆出的 thinking 文本与最终答案、各部分 token 数、边界状态和截断状态；`figures/` 生成与原两张配对图相同布局的 PNG、SVG、PDF，并在每行标注 `Think=C/W/T/?`。

每完成一条路径都会原子保存。意外中断后去掉 `--clean` 重跑即可续跑；再次使用 `--clean` 只清理 DM-1 的 thinking 复评产物，不删除 MCTS、原图或其他难度结果。

## 一键运行

首次运行：

```bash
CUDA_VISIBLE_DEVICES=0 bash ./Polar_code/run_path_showcase.sh qwen3_path_showcase --clean
```

中断恢复：

```bash
CUDA_VISIBLE_DEVICES=0 bash ./Polar_code/run_path_showcase.sh qwen3_path_showcase
```

注意：为了保证两张图各有 10 个一对一配对且跨图不重复，必须先获得至少 20 条正确和 20 条错误路径。因此搜索阶段会检查 `target-per-label >= 2 × paths-per-figure`，最终报告也会再次验证。
