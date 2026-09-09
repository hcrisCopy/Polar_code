# 第一阶段：生成 PoLar predictor 监督数据

跨难度固定 skip-loop 结构的 train-only 挖掘、held-out 评测和图表见
[`ROBUST_SKIP_LOOP_分析说明.md`](./ROBUST_SKIP_LOOP_分析说明.md)。
MCTS 探索路径和 residual stream 的 mNN/CKA/PCA 分析见
[`REPRESENTATION_可视化说明.md`](./REPRESENTATION_可视化说明.md)。
用 smoke run 的 12 条固定路径评测五种难度各 100 道题，见
[`FIXED_12_PATHS_五难度说明.md`](./FIXED_12_PATHS_五难度说明.md)。

所有命令均在包含 `./Polar_code` 的项目根目录执行，使用远程 Linux 的 Bash。
本次交付只做静态检查，没有下载或执行模型，没有生成真实搜索结果。
原仓库文件保持不变；新增代码全部位于 `./Polar_code`，运行产物全部位于 `./Polar_data`。

## 1. 准备模型和数据

| 内容 | 来源与准确名称 | 目标相对路径 |
| --- | --- | --- |
| 示例基础模型，完整未量化权重及 tokenizer/config/generation_config | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)，需要取得访问权限 | `./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct/` |
| DART-Math 原始池的全部 5 个 parquet 分片 | [hkust-nlp/dart-math-pool-math](https://huggingface.co/datasets/hkust-nlp/dart-math-pool-math/tree/main/data)，`data/train-00000-of-00005.parquet` 至 `data/train-00004-of-00005.parquet` | `./Polar_data/raw/dart-math-pool-math/data/` |
| predictor 的冻结 embedding 模型，仅训练 predictor 时需要 | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)，完整 Hugging Face 缓存布局，包含 `refs/main` 与对应 snapshot | `./Polar_data/cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/` |
| Python 环境 | Linux Python 3.10/3.11，原 `./Polar_code/requirements.txt` 的固定版本，另外需要 PyPI 的 `pyarrow==20.0.0` 和 `matplotlib==3.10.5`；统一清单为 `./Polar_code/stage_one/requirements.txt` | 建议环境放在 `./Polar_data/environment/` |

原始池约 965 MB 压缩数据。程序不会联网下载缺失资源。准备时记录实际模型/数据 revision；下方 `local-snapshot`、`local-files` 表示本地快照来源标签，不冒充 Hugging Face commit，程序还会计算实际文件 SHA-256。

本地读取复用 `dart_math/data.py` 的字段映射，未调用其可能联网的 `load_query_dps`；该模块还顶层依赖未列入原 requirements 的 `datasets/vllm`。本阶段用 pyarrow 读本地分片，不需要为此安装 vLLM。数学判分直接复用现有模块。

也支持 `Qwen/Qwen1.5-MoE-A2.7B-Chat`、`Qwen/Qwen2.5-3B-Instruct`、`Qwen/Qwen3-8B`，路径依次为 `./Polar_data/models/` 加模型 ID。切换时显式修改 `--model-id`、`--model-path` 和 predictor 配置的 `model_path`。每张卡放一份完整模型；MoE 的 A2.7B 是活跃参数规模，不是完整权重规模。程序不自动量化、换小模型、减少生成长度或搜索预算。

**数据划分必须知道的一点：**公开池是一题多响应，字段为 `query`、`gt_ans`、`query_metadata.level`、`query_id`。它来源于 MATH 训练题，不是作者公布的每难度 2,000 道独立题清单。新增代码按题去重，冲突答案/难度报错，拒绝 `query4test=true`，再按种子和题目内容排序，默认按 `0.625/0.125/0.25` 划分 train/validation/test。这里的 test 是本项目留出集。不会复制题目凑齐论文数量。

原 `polar/train.py` 硬编码训练 `[0,1250)`、验证 `[1250,1500)`，`polar/eval.py` 测试从 `1500` 开始，通常取 500 题。**比例划分的数据必须使用下方独立训练入口，不能直接交给原 `run_polar.py` 的固定切片。**它只把 split 转成官方 `PolarDataset(indices=...)`，然后调用原 `train_polar`，不改模型、损失、优化器、模型模式或训练参数。不自动运行原评估入口。

如已经有每难度至少 2,000 道唯一题，可将 prepare 的 `--split-policy` 改成 `official`；程序严格生成 `1250/250/500` 顺序，缺题就停止。原作者的具体题目名单未公开，不能声称恢复了它。

## 2. 环境检查

先检查 GPU 数量及驱动（只查询设备）：

```bash
nvidia-smi
```

不写文件。

离线检查依赖导入、本地模型文件完整性和数据分片，不加载模型、不执行 GPU：

```bash
python -B ./Polar_code/run_stage_one.py environment --run-name mcts_formal --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --data-path ./Polar_data/raw/dart-math-pool-math
```

输出：`./Polar_data/runs/mcts_formal/environment/report.json`。脚本不安装任何依赖。该检查不证明 GPU 显存够用，实际执行兼容性由下一步检查。

## 3. 小规模正式配置测试

只选难度 1 的 8 道唯一题，按 5/1/2 划分；模型、1,024 次搜索预算、生成参数与正式配置相同。测试题只执行基线路径。

```bash
bash ./Polar_code/run_stage_one_pipeline.sh --nproc_per_node=1 --run-name mcts_smoke --data-path ./Polar_data/raw/dart-math-pool-math --source-revision local-files --model-id meta-llama/Llama-3.2-3B-Instruct --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --model-revision local-snapshot --max-questions-per-diff 8 --difficulties "1" --train-predictor false --predictor-config ./Polar_code/stage_one/predictor_config.json
```

输出：`./Polar_data/runs/mcts_smoke/{environment,prepared,search,merged,validation}/`。此步骤**真实执行基础模型**，请只在服务器运行；1,024 次预算不会因为叫“小规模”而缩减。先看 `merged/summary.md` 和 `validation/report.json`。没有找到有效路径不等于程序执行失败；不保证每题都能搜到正例。

## 4. 单卡完整搜索

先准备全部唯一题；训练与验证集独立搜索，验证路径只用于验证损失，测试集不做 MCTS。

```bash
python -B ./Polar_code/run_stage_one.py prepare --run-name mcts_formal --data-path ./Polar_data/raw/dart-math-pool-math --data-source hkust-nlp/dart-math-pool-math --source-revision local-files --difficulties 1 2 3 4 5 --seed 42 --split-policy proportional --train-fraction 0.625 --validation-fraction 0.125 --max-questions-per-diff 0
```

输出：`./Polar_data/runs/mcts_formal/prepared/manifest.json` 和 `summary.json`。`0` 明确表示不限制题数。

```bash
mkdir -p ./Polar_data/runtime/launcher
TMPDIR=./Polar_data/runtime/launcher PYTHONDONTWRITEBYTECODE=1 torchrun --standalone --nproc_per_node=1 ./Polar_code/run_stage_one.py search --run-name mcts_formal --model-id meta-llama/Llama-3.2-3B-Instruct --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --model-revision local-snapshot --seed 42 --simulations 1024 --exploration 1.4142135623730951 --length-penalty 0.1 --max-block 4 --max-repeats 4 --max-length-factor 1.15 --max-new-tokens 50 --temperature 0 --completion-timeout 604800
```

输出：`./Polar_data/runs/mcts_formal/search/`，结束后 rank 0 自动合并到 `merged/` 并验证到 `validation/`。

## 5. 单机八卡完整搜索

先执行上面的 prepare。以下命令与单卡只改变进程数，二者任选其一。

```bash
mkdir -p ./Polar_data/runtime/launcher
TMPDIR=./Polar_data/runtime/launcher PYTHONDONTWRITEBYTECODE=1 torchrun --standalone --nproc_per_node=8 ./Polar_code/run_stage_one.py search --run-name mcts_formal --model-id meta-llama/Llama-3.2-3B-Instruct --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --model-revision local-snapshot --seed 42 --simulations 1024 --exploration 1.4142135623730951 --length-penalty 0.1 --max-block 4 --max-repeats 4 --max-length-factor 1.15 --max-new-tokens 50 --temperature 0 --completion-timeout 604800
```

输出同上，逐 rank 数据在 `search/rank_00000/` 至 `search/rank_00007/`。`global_index % world_size` 分片，逐题随机种子不依赖 rank；不使用 DDP。rank 0 等到本次运行的全部完成标识才合并。没有搜索结束 barrier；异常由 torchrun 终止其他进程，缺失完成标识另有显式超时。

## 6. 合并分片

适用于搜索完成后单独重做合并；如果异常发生时所有逐题文件已经完整，也可以直接使用它。缺题或执行失败会拒绝合并，必须先恢复搜索。

```bash
python -B ./Polar_code/run_stage_one.py merge --run-name mcts_formal
```

输出：`./Polar_data/runs/mcts_formal/merged/meta-llama/Llama-3.2-3B-Instruct/dart-math-diff-{1..5}/merged_mcts_samples.json`，以及 `merged/summary.json`、`merged/summary.md`。

## 7. 验证数据

```bash
python -B ./Polar_code/run_stage_one.py validate --run-name mcts_formal
```

输出：`./Polar_data/runs/mcts_formal/validation/report.json`。检查严格 JSON、必需字段、重复 ID/题文/来源 ID、划分顺序、合法非空层路径、valid/invalid 互斥、逐题校验和、完整分片和最终文件与分片一致性；实际实例化官方 **CPU 数据加载器**并核对训练路径数，不加载模型。不同文字改写的语义重复不在这一自动检查的保证范围内。

## 8. 对接官方 predictor 训练

```bash
python -B ./Polar_code/train_stage_one_predictor.py --run-name mcts_formal --predictor-config ./Polar_code/stage_one/predictor_config.json
```

输出：`./Polar_data/runs/mcts_formal/predictor/`。JSON 配置显式列出官方全部 52 个参数，当前选择 README 的 10 epochs、batch 128、LR 5e-4、最多 50 条路径、原路径权重 0.30、cosine 和 warmup 10；训练全部五个难度。没有有效训练或验证标签的难度会停止。更换 run-name 时同步修改配置中的 `data_root`、`save_dir`，更换模型时同步修改 `model_path`。已有完成且校验和一致的 checkpoint 自动跳过；原训练函数没有 optimizer 断点机制，训练中断需显式 `--clean` 从头训练，不能把它称为训练断点续训。

## 实现依据与可调整参数

依据：[论文正文及附录 B、D](https://arxiv.org/html/2606.06574v1)、本地 `polar/data.py`、`polar/train.py`、`polar/eval.py`、`llm_depth_router/`。这是补充实现，不是作者公开的 MCTS 源码，不承诺论文精确结果。

| 设置 | 当前值 | 依据 |
| --- | --- | --- |
| 状态/根节点 | 完整可执行路径；根为 `[0,...,D-1]` | 附录 B.3/B.4 |
| 动作 | 删除或重复当前程序中的连续位置块；重复数指额外拷贝数 | 附录 B.2；额外拷贝的约定与官方 `actions_to_path` 一致 |
| 块长、额外重复次数上限 | `--max-block 4 --max-repeats 4` | 附录 B.2 的上界；均可调小，不自动改变 |
| reward | 实际生成结果的二值正确性 | 附录 B.3；直接调用官方 `_online_eval_math_single`，复用其 boxed 门槛、chat template、DART-Math 抽取和数学等价判断 |
| UCB | `R/v + c*sqrt(log(V)/v) - lambda*len(path)/D` | 附录 B.3；V 取根访问次数 |
| 搜索次数 | `--simulations 1024`，另加完整路径基线一次 | **本项目默认值，可调整；作者未公开** |
| UCB 系数 | `--exploration 1.4142135623730951` | **本项目默认值，可调整；作者未公开** |
| 长度惩罚 | `--length-penalty 0.1` | **本项目默认值，可调整；作者未公开** |
| 最大程序长度 | `floor(D * --max-length-factor 1.15)`，至少容纳完整基线 | **本项目默认值，可调整**；图 3 有 115% 预算实验，附录未给统一限制 |
| 搜索种子 | `--seed 42` | **本项目默认值，可调整**；按题/路径派生，与 rank 数无关 |
| 输出 token 上限 | `--max-new-tokens 50` | 附录 D.2 与官方在线评估默认值 |
| 温度 | `--temperature 0` | 官方在线评估调用；采样温度可显式修改，但其有效标签只表示该固定种子的一次成功，不是成功率 |
| 完成等待上限 | `--completion-timeout 604800` 秒 | **本项目工程默认值，可调整**；不是搜索预算 |

每次从未扩展动作中按题种子随机选一个子节点，执行该完整程序；相同程序只执行一次并复用实际 reward，树节点分别累计访问统计，排除祖先循环。这些是论文未细化的工程选择。没有额外 rollout 深度、早停正例数或隐藏的候选数截断。

官方 predictor 只能表达连续原始层段、段长最多 4、每段最多额外执行一次。因此搜索保留附录的较宽空间，**只把官方解析器成功解析的路径写入 final_valid/invalid_transitions**，其余真实执行记录保存在 `evaluations`，不静默丢弃。每题同时保存 `question`、`gt_ans`、`initial_score`、搜索统计和失败原因。测试题只评估完整路径。额外字段被官方加载器忽略。

缓存适配只作用于新增 ModelRunner 实例：原路由按 custom_path 执行，新增 DynamicCache 按执行位置存 KV，防止 repeat 共用同一原始层缓存；不修改原文件、原 attention 的 layer_idx、生成参数或模型模式。实现依据是仓库补丁及 [Transformers 4.52.4 的 KV 更新接口](https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/cache_utils.py)。服务器首次小规模运行仍需确认真实生成兼容性；本机没有运行验证。原 `polar/eval.py` 的独立调用没有安装此实例适配，后续用它评估 repeat 路径前需要另行对齐缓存语义。

## 恢复与清理

- 搜索原命令重跑：逐题校验已提交文件并跳过完成题；失败题重试，未完成题从该题开头重跑。每题完成即 `fsync + atomic replace`，不是每次 simulation 保存整个树。
- `.pending` 中断写入移入对应阶段 `recovery/` 留证并明确报告；损坏的已提交 JSON/校验和不匹配直接报错，不能视为完成。题目结果不会因复跑而追加重复条目。
- 同一 run-name 恢复须保持数据、代码、模型、搜索参数和 world_size 不变。更换 1/8 卡并行度用于新的正式运行没有其他参数差异；已有分片不跨 world_size 混用。
- 每阶段都有 `--clean`，只清理 `./Polar_data/runs/<run-name>/<该阶段>/`。例如 merge 的 `--clean` 不删 search。没有该参数时不删除旧产物。同一 run 使用文件锁；正在运行时不允许另一命令清理。
- 单独重新搜索并传 `--clean` 后，旧 merged 可能与新结果不同，此时自动合并会明确报错；再运行 `merge --clean` 和 validate 即可。不要拿旧 validation 报告代表新搜索结果。
- 一键脚本加 `--clean` 会依次显式清理参与的各阶段，其中旧 merged/validation 在自动合并前先清理。原始数据、模型、共享缓存、其他 run 永远不在清理范围内。
- 隐式缓存放 `./Polar_data/cache/`，Python/torchrun 临时文件放 `./Polar_data/runtime/`，运行锁放 `./Polar_data/locks/`；不启用 WandB。

静态复查命令：

```bash
python -B ./Polar_code/check_stage_one_static.py --run-name static_review
```

输出：`./Polar_data/runs/static_review/environment/static_report.json`。它只做 AST、相对导入和参数配置检查，不执行训练、推理或搜索。

一键全部运行（包含 predictor 训练；先完成资源准备并通过小规模检查；将 `--nproc_per_node=8` 改成 `1` 即为同配置单卡正式运行）：

```bash
bash ./Polar_code/run_stage_one_pipeline.sh --nproc_per_node=8 --run-name mcts_formal --data-path ./Polar_data/raw/dart-math-pool-math --source-revision local-files --model-id meta-llama/Llama-3.2-3B-Instruct --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct --model-revision local-snapshot --max-questions-per-diff 0 --difficulties "1 2 3 4 5" --train-predictor true --predictor-config ./Polar_code/stage_one/predictor_config.json
```
