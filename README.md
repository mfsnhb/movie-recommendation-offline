# Movie Recommendation Offline

一个纯离线电影推荐项目，基于 MovieLens-1M 数据集实现完整的推荐训练与评估流程。

## 项目结构

```text
config/
  preprocess.yaml
  retrieval.yaml
  ranking.yaml
scripts/
  run_offline_pipeline.py
  run_retrieval_preprocess.py
  run_ranking_preprocess.py
  run_retrieval_train.py
  run_ranking_train.py
  evaluate_offline.py
src/offline/
  data/
  evaluate/
  features/
  models/
  ranking/
  training/
  utils/
outputs/
  processed/
  models/
  metrics/
```

`src/offline/ranking/` 是排序链路的主目录：

- `protocol.py`：排序样本协议、item feature helper、通用字段定义
- `preprocess.py`：构造前缀展开的正样本训练集与最终评测样本
- `dataset.py`：在线负采样、train/val/test collator、inference batch 协议

`src/offline/features/ranking.py`、`src/offline/data/ranking_sequence.py`、`src/offline/models/sequence_ranker.py` 现在只保留兼容导出，方便旧入口不报错。

## 当前离线链路

1. 召回特征处理
2. 排序特征处理
3. 双塔召回训练与评估
4. 多路召回融合评估
5. DeepFM 排序训练与评估
6. 最终 Top-K 指标评估

## 数据

将 `funrec-movielens-1m` 解压到：

```text
data/raw/funrec-movielens-1m/
```

目录内应包含：

- `movies.pkl`
- `ratings.pkl`
- `users.pkl`

## 运行方式

建议在 `torch` conda 环境中执行。

配置约定：

- `config/retrieval.yaml` 与 `config/ranking.yaml` 都使用 `common + models` 的 registry 风格
- `common` 放共享训练/评测设置，`models.<model_name>` 放该模型自己的结构和训练超参数
- 排序通过 `default_model` 或 `python scripts/run_ranking_train.py --model <name>` 选择模型

全流程：

```bash
python scripts/run_offline_pipeline.py --steps all
```

分步执行：

```bash
python scripts/run_retrieval_preprocess.py
python scripts/run_ranking_preprocess.py
python scripts/run_retrieval_train.py
python scripts/run_ranking_train.py
python scripts/evaluate_offline.py
```

## 输出产物

- `outputs/processed/`：样本、词表、embedding、中间产物
  - `train_eval_sample_final.pkl`：召回训练/评测样本
  - `ranking_train_eval_sample.pkl`：排序前缀正样本与最终评测样本
  - `item_catalog.pkl`：多路召回和排序共享的 item 元数据
- `outputs/models/`：训练好的召回与排序模型
- `outputs/metrics/`：召回、多路召回、排序、最终指标

## 指标

- 召回阶段：Recall@K / HR@K / NDCG@K
- 多路召回阶段：Recall@K / HR@K / NDCG@K
- 排序阶段：AUC / GAUC / LogLoss
- 最终阶段：Recall@10/20、HR@10/20、NDCG@10/20、Coverage@10/20

## 排序样本定义

- `train` 样本按召回相同的方式做前缀展开：`[m1] -> m2`、`[m1,m2] -> m3`，只保存正样本与上下文
- `test` 样本使用每个用户按时间排序后的最后一次交互作为 `target_movie_id`
- `context_movie_id/context_genres` 是最后一次交互之前的历史
- hard negative / random negative 不再写入预处理产物，而是在训练或评测时在线采样候选

## 序列长度建议

- 召回默认 `max_seq_len=100`
- 排序默认 `max_seq_len=50`
- MovieLens-1M 里用户交互很长，若排序效果仍受限，可继续把 `ranking.max_seq_len` 提到 `100`
