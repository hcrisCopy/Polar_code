主流程包括：

1. 从本地 DART-Math parquet 文件读取并按题去重；
2. 使用完整层路径执行 baseline；
3. 对 train/validation 题逐题运行 MCTS；
4. 对每条候选路径真实执行基础 LLM；
5. 使用现有 `dart_math` 逻辑抽取答案并判断数学等价性；
6. 保存每题的有效路径、无效路径、baseline 分数和搜索统计；
7. 合并并验证 predictor 可读取的 `merged_mcts_samples.json`；
8. 使用现有 `polar/data.py`、`polar/model.py` 和 `polar/train.py` 训练 predictor。

以下内容不能从本仓库直接推出：

- MCTS 找到的是全局最优路径；
- 每条搜索到的正确路径都能迁移到其他题目或模型；
- 本项目默认超参数等同于作者实验配置；
- 小规模 smoke test 能复现论文准确率；
- residual 相似性能够证明 skip/loop 导致正确率变化；
- 同一模型不同执行路径已经收敛到“柏拉图表征”。

## 目录约定

所有命令在同时包含 `./Polar_code` 和 `./Polar_data` 的项目根目录执行：

```text
./
├── Polar_code/                 # 本仓库
└── Polar_data/                 # 模型、数据、缓存和全部运行产物
    ├── models/
    ├── raw/
    ├── cache/
    ├── runtime/
    └── runs/
```

本复现流程的所有产物必须位于 `./Polar_data`。不要把正式运行产物写入 `./Polar_code/outputs`。

主要代码结构：

```text
Polar_code/
├── polar/                      # 作者公开的 predictor 实现
├── llm_depth_router/           # 作者公开的动态层执行实现
├── dart_math/                  # 作者公开的数学答案判定实现
├── stage_one/                  # 本仓库补充的 MCTS Stage One
├── run_stage_one.py            # Stage One 分阶段入口
├── run_stage_one_pipeline.sh   # Stage One 串联脚本
├── train_stage_one_predictor.py
└── check_stage_one_static.py
```

## 用户需要准备的内容

本仓库不会自动下载模型、数据集、checkpoint 或依赖。以 Llama-3.2-3B-Instruct 复现为例，需要准备：

```text
./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct/
./Polar_data/raw/dart-math-pool-math/data/
./Polar_data/cache/huggingface/hub/
```

对应资源：

- 基础模型：`meta-llama/Llama-3.2-3B-Instruct`；
- 数据集：`hkust-nlp/dart-math-pool-math` 的五个 parquet 分片；
- predictor embedding model：`Qwen/Qwen3-Embedding-0.6B`；
- Python 依赖：`./Polar_code/stage_one/requirements.txt`。

模型和数据各自受其原始许可证及访问条件约束。

## 安装依赖

在 Linux 环境中运行：

```bash
python -m pip install -r ./Polar_code/stage_one/requirements.txt
```

## 先做静态检查

该检查不加载模型，也不运行推理：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B ./Polar_code/check_stage_one_static.py \
  --run-name static_review \
  --clean
```

输出位于 `./Polar_data/runs/static_review/environment/static_report.json`。

## 小规模流程检查

下面的命令仅检查数据、MCTS、合并、验证和恢复机制能否跑通，不用于报告论文结果：

```bash
CUDA_VISIBLE_DEVICES=1 bash ./Polar_code/run_stage_one_pipeline.sh \
  --nproc_per_node=1 \
  --run-name mcts_smoke \
  --data-path ./Polar_data/raw/dart-math-pool-math \
  --source-revision local-files \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct \
  --model-revision local-snapshot \
  --max-questions-per-diff 8 \
  --difficulties "1" \
  --train-predictor false \
  --predictor-config ./Polar_code/stage_one/predictor_config.json \
  --clean
```

输出位于 `./Polar_data/runs/mcts_smoke/`。首次运行可使用 `--clean`；中断恢复时删除 `--clean`，程序会验证并跳过已经完成的逐题结果。

## 正式运行

正式的单卡、八卡、独立合并、数据验证和 predictor 训练命令见：

- [Stage One 使用说明](./STAGE_ONE_使用说明.md)
- [跨难度固定路径分析说明](./ROBUST_SKIP_LOOP_分析说明.md)
- [12 条固定路径五难度实验](./FIXED_12_PATHS_五难度说明.md)
- [Residual 表征可视化说明](./REPRESENTATION_可视化说明.md)

第一阶段采用一进程一卡的数据并行方式，不使用 DDP，也不进行梯度同步。每个 rank 处理确定且互斥的题目分片，并独立保存逐题结果。

## MCTS 监督与 Predictor 的关系

MCTS 对每道题搜索多个 layer execution programs。只有真实执行基础 LLM 后得到正确答案的路径才进入：

```text
final_valid_transitions
```

`polar/data.py` 会将同一道题的多条有效路径分别展开为训练样本。Predictor 学习的是问题条件下的 segmentation 和 `skip/keep/repeat` 操作概率，而不是一条经过证明的全局最优路径。

推理时 beam search 可以组合出训练集中没有出现过的新路径。结构合法不代表答案正确，因此未知路径仍需要执行基础 LLM 才能得到真实 reward。

## `merged_mcts_samples.json`

每个难度最终生成一个文件：

```text
./Polar_data/runs/<run-name>/merged/<model-id>/dart-math-diff-<1..5>/merged_mcts_samples.json
```

核心格式：

```json
{
  "samples": [
    {
      "sample_id": "...",
      "question": "Solve ...",
      "gt_ans": "\\boxed{42}",
      "initial_score": 1,
      "final_valid_transitions": [
        [0, 1, 2, 4, 5, 6],
        [0, 1, 2, 2, 3, 4, 5]
      ],
      "final_invalid_transitions": [
        [0, 1, 3, 4, 5]
      ]
    }
  ]
}
```

- 跳过某层：路径中不出现该层索引；
- 重复某层或连续层段：相应索引重复出现；
- `initial_score`：完整层 baseline 的二值正确性；
- `final_valid_transitions`：实际执行且判定正确的路径；
- `final_invalid_transitions`：实际执行且判定错误的路径。

标准 predictor 训练读取有效路径。无效路径主要用于数据审计和评测缓存，并没有直接作为标准 Predictor 损失中的负路径样本。