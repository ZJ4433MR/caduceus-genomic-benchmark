# MLBN 消融实验交接说明

本分支用于把消融实验所需的代码、配置和运行入口交给负责消融的组员。
仓库不包含数据集、checkpoint、模型缓存、Slurm 日志或实验输出。

## 1. 实验目标

消融需要回答三个问题：

1. MLBN 的收益来自双向扫描、层内方向交互，还是最终融合？
2. normal 与 pure sequence reversal（下文简称 `flipped`）之间的鲁棒性来自结构还是数据增强？
3. 各结论在相同训练预算和相同下游协议下是否稳定？

`flipped` 指序列顺序反转，例如 `ACGT -> TGCA`，不是 reverse complement。
不要把 flipped 与 RC 结果混为一类。

## 2. 基线与关键文件

正式消融的基线为参数匹配的 MLBN-1k：

| 项目 | 默认值 | 文件 |
| --- | ---: | --- |
| `d_model` | 80 | `configs/model/mlbn_lm.yaml` |
| `n_layer` | 3 | `configs/model/mlbn_lm.yaml` |
| 预训练长度 | 1024 | `slurm_scripts/run_pretrain_mlbn_a800.sh` |
| `d_state` | 16 | `configs/model/mlbn_lm.yaml` |
| `d_conv` / kernel | 4 / 4 | `configs/model/mlbn_lm.yaml` |
| expand | 2 | `configs/model/mlbn_lm.yaml` |
| heads | 4 | `configs/model/mlbn_lm.yaml` |

核心代码位置：

- `src/models/sequence/mlbn.py`
  - `ODBC`：方向感知局部卷积；
  - `MambaBlock`：共享参数的 forward/backward Mamba；
  - `GradientEquilibrium`：双分支梯度平衡；
  - `CrossAttention_pro`：层内双方向交互；
  - `fusion_block`：最终的加和、平方差和 attention 融合；
  - `MLBN_encoder`：堆叠 MLBN block 并调用最终融合。
- `src/models/sequence/dna_embedding.py`
  - `DNAEmbeddingModelMLBN`：下游 backbone；
  - `MLBNLMHeadModel`：hg38 预训练模型；
  - `load_backbone`：从预训练 checkpoint 加载下游 backbone。
- `configs/model/mlbn_lm.yaml`：预训练配置。
- `configs/model/mlbn_dna.yaml`：GenomicBenchmarks 下游配置。
- `slurm_scripts/run_pretrain_mlbn_a800.sh`：预训练入口。
- `slurm_scripts/run_genomic_benchmark_small_l40.sh`：下游训练入口。
- `slurm_scripts/run_genomic_benchmark_reversal_test_small_l40.sh`：normal/flipped 成对评估入口。

## 3. 正文最小消融矩阵

每个结构变体只改变一项，其余训练数据、训练步数、优化器、下游超参数和随机种子保持一致。

| ID | 变体 | 实现原则 | 需要独立预训练 checkpoint |
| --- | --- | --- | --- |
| A0 | Full MLBN | 当前实现，不修改 | 是，基线 |
| A1 | Single-direction | 仅保留 forward Mamba 路径 | 是 |
| A2 | No cross-attention | 将每层 `CrossAttention_pro` 交互替换为 identity | 是 |
| A3 | No gradient equilibrium | 将 `GradientEquilibrium` 替换为 identity | 是 |
| A4 | Mean fusion | 最终融合改为方向对齐后的 `(f + b) / 2` | 是 |

补充材料可增加：

| ID | 变体 | 实现原则 |
| --- | --- | --- |
| A5 | No ODBC | 去掉方向感知局部卷积，保持残差维度不变 |
| A6 | No discrepancy branch | 最终融合仅使用 `f + b`，去掉平方差分支 |
| A7 | Untied directions | forward/backward 使用独立 Mamba 参数，并报告参数量变化 |

实现时建议为构造函数增加显式布尔或枚举参数，并把开关透传到
`configs/model/mlbn_lm.yaml` 和 `configs/model/mlbn_dna.yaml`。不要通过注释代码生成无法追踪的临时版本。
每个变体必须使用唯一名称，例如：

```text
mlbn_full
mlbn_single_direction
mlbn_no_cross_attn
mlbn_no_grad_equilibrium
mlbn_mean_fusion
```

如果删除模块导致参数量下降，表格中必须同时报告参数量。A7 参数量会增加，不能与参数匹配结果混写。

## 4. Checkpoint 规则

- A1-A7 改变了 backbone 结构，正式结果需要分别重新预训练并保存 checkpoint。
- 同一结构的 normal 与 flipped 测试复用同一个下游 checkpoint。
- 只改变测试输入方向时不需要重新训练。
- 改变 pooling/readout 时需要重新训练下游 checkpoint。
- 使用 `strict=False` 载入完整 MLBN 权重只能用于 smoke test，不能作为正式结构消融结果。
- checkpoint 不提交到 GitHub；通过集群公共目录或单独存储链接共享。

每个 checkpoint 至少记录：

```text
variant, git_commit, config, seed, pretrain_steps, checkpoint_path, sha256
```

## 5. 数据准备

hg38 预训练数据：

```bash
bash scripts/download_hg38_pretrain_data.sh
```

预期目录：

```text
<HG38_DATA_DIR>/hg38.ml.fa
<HG38_DATA_DIR>/human-sequences.bed
```

GenomicBenchmarks 数据目录见 `data/README.md`。不得提交 FASTA、parquet、数据缓存或下载压缩包。

## 6. 两阶段执行方案

### 阶段一：低成本筛选

先跑 4 个代表任务、3 个随机种子：

```text
dummy_mouse_enhancers_ensembl
demo_coding_vs_intergenomic_seqs
demo_human_or_worm
human_enhancers_cohn
```

随机种子固定为：

```text
1 2 3
```

阶段一用于检查方向是否正确、训练是否稳定以及模块效果是否有一致趋势，不用于最终论文数字。

### 阶段二：正式结果

保留 Full MLBN 和最关键的结构变体，扩展到全部 8 个 GenomicBenchmarks 任务和 5 个种子：

```text
1 2 3 4 5
```

模型选择只能依据 validation 指标；normal/flipped test 结果不能用于选择 checkpoint 或超参数。

## 7. Slurm 命令模板

先根据集群修改以下环境变量，不要把真实用户名或私有绝对路径提交回仓库：

```bash
export CONTAINER_DIR=/path/to/shared_containers
export HG38_DATA_DIR=/path/to/hg38
export GB_DATA_DIR=/path/to/genomic_benchmarks
```

### 7.1 为每个结构变体预训练

下面以 `no_cross_attn` 为例。实现开关后，通过 `EXTRA_OVERRIDES` 传入对应配置：

```bash
RUN_TAG=ablation_no_cross_attn \
MAX_STEPS=10000 \
SEQLEN=1024 \
D_MODEL=80 \
N_LAYER=3 \
NUM_HEADS=4 \
GLOBAL_BATCH_SIZE=1024 \
EXTRA_OVERRIDES="model.use_cross_attention=false" \
sbatch slurm_scripts/run_pretrain_mlbn_a800.sh
```

其他变体必须保持相同预算，只替换 `RUN_TAG` 和单个结构开关。

### 7.2 三种子下游筛选

对每个任务提交一次，`MLBN_PRETRAINED_PATH` 指向该变体自己的预训练 checkpoint：

```bash
TASK=dummy_mouse_enhancers_ensembl \
MODEL_VARIANT=mlbn_pretrained \
DISPLAY_NAME=mlbn_no_cross_attn \
SEEDS="1 2 3" \
MLBN_PRETRAINED_PATH=/path/to/ablation_no_cross_attn/last.ckpt \
sbatch slurm_scripts/run_genomic_benchmark_small_l40.sh
```

记下返回的训练 job ID，然后进行成对评估：

```bash
TASK=dummy_mouse_enhancers_ensembl \
MODEL_VARIANTS=mlbn_pretrained \
DISPLAY_NAME=mlbn_no_cross_attn \
SEEDS="1 2 3" \
SOURCE_JOB_ID=<TRAIN_JOB_ID> \
EVAL_MODES="normal_test flipped_test" \
sbatch --dependency=afterok:<TRAIN_JOB_ID> \
  slurm_scripts/run_genomic_benchmark_reversal_test_small_l40.sh
```

正式阶段把 `SEEDS` 改为 `"1 2 3 4 5"`，并覆盖全部 8 个任务。

## 8. 指标与结果表

每个变体至少报告：

- normal mean ± sample standard deviation；
- flipped mean ± sample standard deviation；
- orientation mean：`(normal + flipped) / 2`；
- flip gap：`normal - flipped`；
- 参数量；
- 训练失败、NaN 或 OOM 次数。

原始结果采用 `results/README.md` 规定的长表格式。核心字段：

```text
benchmark,task,model,comparison_role,pretrain_context,seed,orientation,metric,value
```

聚合命令：

```bash
python scripts/aggregate_results.py \
  --input results/raw/table4_ablation_runs.csv \
  --output results/tables/table4_ablation.csv
```

不要手工复制均值到论文表格；保留 seed-level 原始 CSV，并由脚本生成最终表。

## 9. 运行前检查

```bash
conda env create -f caduceus_env.yml
conda activate caduceus_env
make smoke
```

每个新变体在提交大规模任务前，至少完成：

1. 单 batch forward/backward；
2. 参数量打印；
3. 100-500 step 小规模训练无 NaN；
4. normal 和 flipped 各一次评估；
5. checkpoint 保存与重新加载；
6. 输出目录包含唯一变体名，不覆盖 Full MLBN。

## 10. 交付检查单

完成一个变体后，在组内记录：

```text
- [ ] 变体名称和对应 Git commit
- [ ] 与 A0 相比只改变一个结构因素
- [ ] 预训练配置和 checkpoint SHA256
- [ ] 下游训练 job IDs
- [ ] normal/flipped 评估 job IDs
- [ ] 3-seed 筛选结果
- [ ] 5-seed 正式结果（进入正文的变体）
- [ ] 参数量和运行成本
- [ ] 失败任务及原因
```

如需增加新的消融开关，应同时更新模型构造函数、两份 MLBN YAML、Slurm 运行名和本文件，保证配置与结果可以一一对应。
