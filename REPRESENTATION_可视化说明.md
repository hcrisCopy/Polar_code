# MCTS 路径与残差流表征可视化

本分析包含两部分：读取已有 MCTS `evaluations` 绘制实际评估顺序；对 baseline 和 validation 选出的固定程序做一次无生成 prefill，保存每个执行槽位的 post-block residual stream。

参考 Huh 等人在 ICML 2024 论文 [*The Platonic Representation Hypothesis*](https://proceedings.mlr.press/v235/huh24a.html) 中使用的 mutual k-nearest-neighbor（mNN）方法，对同一批题在不同程序下形成的邻域结构做比较。mNN 实现参照作者公开的 [official metrics code](https://github.com/minyoungg/platonic-rep/blob/main/metrics.py)，显式排除样本自身。命令使用 valid-token mean pooling，与作者对 language features 的公开示例一致；也可显式改成 `--pooling last-token` 做因果语言模型末 token 的补充分析。这里研究的是同一模型的不同 layer programs，不能写成跨模型或跨模态的“柏拉图表征收敛”。linear CKA 和 PCA 只作为补充。

运行前需要已经完成：`evaluate-programs` 和 `report-programs`。全部结果仍写入 `./Polar_data`。

先执行 `mkdir -p ./Polar_data/runtime/launcher`，供 torchrun 保存临时文件。

## 1. 4题 smoke 捕获

```bash
TMPDIR=./Polar_data/runtime/launcher PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 torchrun --standalone --nproc_per_node=1 ./Polar_code/run_stage_one.py capture-representations --run-name mcts_smoke --model-id meta-llama/Llama-3.2-3B-Instruct --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --model-revision modelscope-master --seed 42 --representation-splits validation test --max-representation-questions 4 --max-programs 6 --pooling mean --completion-timeout 604800 --clean
```

输出：`./Polar_data/runs/mcts_smoke/representations/`。每题一个压缩 NPZ，逐题原子保存并支持单卡/八卡互斥分片。该阶段不生成答案，只执行 prompt prefill；仍需加载完整基础模型。

## 2. 4题 smoke 绘图

```bash
python -B ./Polar_code/run_stage_one.py report-representations --run-name mcts_smoke --neighbor-k 1 --alignment-samples 4 --projection-samples 4 --max-search-questions 3 --max-paths-per-question 256 --seed 42 --clean
```

输出：`./Polar_data/runs/mcts_smoke/representation_report/`。4题的 mNN 和 PCA 只检查流程，没有统计意义。

## 3. 正式单卡捕获

```bash
TMPDIR=./Polar_data/runtime/launcher PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 torchrun --standalone --nproc_per_node=1 ./Polar_code/run_stage_one.py capture-representations --run-name mcts_formal --model-id meta-llama/Llama-3.2-3B-Instruct --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --model-revision modelscope-master --seed 42 --representation-splits validation test --max-representation-questions 0 --max-programs 8 --pooling mean --completion-timeout 604800 --clean
```

输出：`./Polar_data/runs/mcts_formal/representations/`。

## 4. 正式八卡捕获

```bash
TMPDIR=./Polar_data/runtime/launcher PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 torchrun --standalone --nproc_per_node=8 ./Polar_code/run_stage_one.py capture-representations --run-name mcts_formal --model-id meta-llama/Llama-3.2-3B-Instruct --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --model-revision modelscope-master --seed 42 --representation-splits validation test --max-representation-questions 0 --max-programs 8 --pooling mean --completion-timeout 604800 --clean
```

输出同样位于 `./Polar_data/runs/mcts_formal/representations/`，每个 rank 写独立子目录。

单卡和八卡只改变问题分片。`--max-representation-questions 0` 表示使用全部 validation/test；`--max-programs 8` 包含 baseline、通用候选、各难度候选和按 validation 排名补齐的候选。

捕获前可按 `4 × hidden_size × 所有程序路径长度之和 × 题目数` 估算未压缩 residual 字节数，NPZ 实际大小取决于可压缩性。`--max-representation-questions` 与 `--max-programs` 是显式资源参数；正式命令没有暗中缩小模型或改变层路径。

## 5. 正式报告

```bash
python -B ./Polar_code/run_stage_one.py report-representations --run-name mcts_formal --neighbor-k 10 --alignment-samples 500 --projection-samples 100 --max-search-questions 10 --max-paths-per-question 256 --seed 42 --clean
```

输出：`./Polar_data/runs/mcts_formal/representation_report/`。

这些均为本项目可调默认值，不是论文作者配置。`alignment-samples` 限制 mNN/CKA 的显式计算规模；原始 residual 仍按捕获命令保存。

重点输出：

- `search_trajectory_<sample_id>.svg/pdf`：MCTS 实际评估顺序、路径长度和结构 PCA。旧结果没有父节点字段，因此连线表示时间顺序，不表示树边。
- `search_path_matrix_<sample_id>.svg/pdf`：每条已评估路径在各原始层上的执行次数；0 是 skip，1 是正常执行，2 及以上是 loop/重复执行，右侧窄列为该路径的 0/1 reward。
- `residual_alignment_heatmap.svg/pdf`：各程序每个执行槽位相对 baseline 最终 residual 几何的 mNN。
- `layer_alignment_<program_id>.svg/pdf`：某个程序的所有执行槽位与 baseline 所有槽位之间的完整 mNN 矩阵，可观察 loop 后的 residual 在 baseline 深度轴上更接近哪里。
- `final_geometry_alignment.svg/pdf`：最终 residual 的 mNN 和 linear CKA。
- `alignment_accuracy_tradeoff.svg/pdf`：最终 residual 的 mNN 与 held-out accuracy gain 的关系；它是相关性图，不作为正确性的因果解释。
- `final_alignment_by_group.svg/pdf`：按 DM-1 至 DM-5 以及 validation/test 分组的最终 residual mNN；样本数不大于 `k` 的组不绘制。
- `residual_norm_trajectories.svg/pdf`：逐执行槽位平均 residual L2 norm。
- `final_residual_pca.svg/pdf`：共同 PCA 投影，仅作探索性观察。
- `program_geometry.csv`：全部 held-out 题上的准确率、mNN 和 CKA。
- `program_geometry_by_group.csv`：按难度与 split 展开的准确率、mNN 和 CKA。
- `report.json`：逐槽位指标、分组样本量以及无法计算 mNN 的小样本组。

mNN 比较的是同一批题的最近邻集合重合率：1 表示邻域完全一致，0 表示没有重合。它衡量表征几何是否相似，不证明该 residual 对答案具有因果作用。

保存的是同一 prompt 的 prefill residual，正确率来自此前 `evaluate-programs` 的真实答案生成。这样保证不同程序之间样本与 token 输入可比，但图中不包含生成答案 token 的动态过程。validation 用于候选选择；判断通用结构是否泛化时以未参与选择的 test 指标为主。中断后重跑捕获命令时去掉 `--clean`，即可验证并跳过已经完整保存的题目。

## 一键正式运行

将进程数改为 1 即可保持其他参数不变地单卡运行：

```bash
bash ./Polar_code/run_representation_analysis.sh --nproc_per_node=8 --run-name mcts_formal --model-id meta-llama/Llama-3.2-3B-Instruct --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --model-revision modelscope-master --clean
```
