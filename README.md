# Movie Recommendation Offline

基于 MovieLens-1M 的离线电影推荐项目，目标是实现一条可复现的 **召回 → 多路融合 → 排序 → 离线评估** 链路。当前实现强调：

1. 召回和排序使用不同的目标强度：召回训练更宽松，排序与评估更严格；
2. 用户历史保留所有交互，评分和时间差作为序列交互特征，而不是过滤历史；
3. 静态电影特征、序列交互特征、用户画像在模型边界上职责分离。

## 1. 项目结构

```text
config/
  preprocess.yaml        # 召回/排序预处理阈值、序列长度、多模态配置
  retrieval.yaml         # 召回模型、训练、评估、多路召回配置
  ranking.yaml           # 排序模型、训练、候选采样配置
scripts/
  run_offline_pipeline.py
  run_retrieval_preprocess.py
  run_ranking_preprocess.py
  run_retrieval_train.py
  run_ranking_train.py
  evaluate_offline.py
src/offline/
  features/              # 召回特征与样本构造
  ranking/               # 排序样本协议、预处理、dataset/collator
  models/                # TwoTower、Sequence Retrieval、DIN、DeepFM、feature encoders
  training/              # 召回与排序训练
  evaluate/              # 多路召回与最终评估
  utils/                 # 配置解析、IO、日志等
outputs/
  processed/             # 样本、词表、item catalog、多模态 embedding
  models/                # 训练好的模型与召回产物
  metrics/               # 各阶段指标
```

## 2. 运行方式

原始数据放在：

```text
data/raw/funrec-movielens-1m/
```

目录内应包含：

- `movies.pkl`
- `ratings.pkl`
- `users.pkl`

建议在 `torch` conda 环境中运行：

```bash
conda run -n torch python scripts/run_retrieval_preprocess.py
conda run -n torch python scripts/run_ranking_preprocess.py
conda run -n torch python scripts/run_retrieval_train.py
conda run -n torch python scripts/run_ranking_train.py
conda run -n torch python scripts/evaluate_offline.py
```

也可以执行全流程：

```bash
conda run -n torch python scripts/run_offline_pipeline.py --steps all
```

## 3. 样本语义

当前统一使用 `hist_*` 表示 target 之前的用户历史。历史包含所有 prior interactions，不按评分过滤；评分只作为序列 token 的交互特征或训练 loss 权重使用。

| 阶段 | Train target | Validation/Test target | History policy |
| --- | --- | --- | --- |
| Retrieval | `rating >= 3.0` | `rating >= 4.0` | target 之前所有交互 |
| Ranking | `rating >= 4.0` | `rating >= 4.0` | target 之前所有交互 |

时间特征统一命名为 `time_gap`：

```text
time_gap = target_timestamp - historical_interaction_timestamp
```

bucket 边界为：

```text
1, 3, 7, 14, 30, 90, 365 days
```

padding 位置 bucket 为 0。

## 4. 预处理

### 4.1 Retrieval preprocessing

配置来自 `config/preprocess.yaml`：

```yaml
retrieval:
  max_seq_len: 500
  positive_rating_min: 3.0
  eval_positive_rating_min: 4.0
  neutral_rating: 3.0
```

召回样本按用户时间序列构造：

1. 对每个用户按 timestamp 排序；
2. train split 使用 `rating >= 3.0` 的 target；
3. validation/test split 分别使用倒数第二个、最后一个 `rating >= 4.0` target；
4. 每个样本的输入历史是 target 之前的所有交互，最多保留最近 `500` 条，右对齐 padding；
5. 样本字段包括：
   - 静态用户特征：`user_id, gender, age, occupation, zip_code`；
   - target：`movie_id, rating`；
   - 历史序列：`hist_movie_id, hist_rating, hist_time_gap_bucket`。

最近一次构建规模：

```text
train=811012
validation=6037
test=6037
max_hist_seq_len=500
```

### 4.2 Ranking preprocessing

配置来自 `config/preprocess.yaml`：

```yaml
ranking:
  max_seq_len: 300
  positive_rating_min: 4.0
  negative_rating_max: 2.0
  neutral_rating: 3.0
  history_policy: all_ratings
```

排序样本同样是 prefix + target：

1. target 必须满足 `rating >= 4.0`；
2. validation/test 分别使用倒数第二个、最后一个高分 target；
3. `hist_movie_id, hist_rating, hist_time_gap_bucket` 保留 target 之前所有交互；
4. 当前 ranking collator 仍保留 `low_rating_movie_id`，用于从历史低分电影中采样一部分训练负样本；
5. multi-recall fused candidates 用作排序最终候选集。

## 5. Feature encoder 边界

### 5.1 MovieFeatureEncoder

`MovieFeatureEncoder` 只编码电影静态信息：

- `movie_id`
- `genres` multi-hot
- `isAdult`
- `startYear`
- `popularity`
- `averageRating`
- multimodal embedding

输出 shape：

```text
[..., emb_dim]
```

当传入 `movie_id` tensor 且模型持有 `item_feature_table` 时，encoder 会在内部 lookup 结构化 item 特征；当 ranking batch 已经携带候选/历史 item 特征时，也可以传入 dict。

### 5.2 SequenceFeatureEncoder

`SequenceFeatureEncoder` 只编码序列交互信息：

```text
movie_static_embedding + rating_embedding + time_gap_embedding + feedback_embedding -> MLP -> sequence_token_embedding
```

其中：

- 静态电影 embedding 维度为主 `emb_dim`；
- `rating_embedding`、`time_gap_embedding` 和 `feedback_embedding` 使用较小维度 `max(4, emb_dim // 2)`；
- `feedback_embedding` 显式区分 padding、负反馈、普通反馈和正反馈：`0/1/2/3 = padding/rating<=2/rating==3/rating>=4`；
- 输出仍为 `emb_dim`，供 pooling、GRU、DIN attention 或 DeepFM 使用。

这意味着 candidate/catalog item embedding 始终是静态电影 embedding；只有历史序列 token 会融合 `hist_rating` 和 `hist_time_gap_bucket`。

## 6. Retrieval 阶段

配置来自 `config/retrieval.yaml`：

```yaml
common:
  training:
    batch_size: 512
    epochs: 10
    learning_rate: 0.0005
    weight_decay: 1.0e-5
    early_stopping_patience: 2
    hard_negative_ratio: 0.1
    positive_rating_min: 3.0
    eval_positive_rating_min: 4.0
  evaluation:
    topk: 200
  multi_recall:
    enabled: true
    routes: [two_tower, sequence, item_cf, multimodal, genre, popular]
```

召回训练不再从 train 内部切 10% validation；early stopping 使用官方 validation split 的 `NDCG@200`，同时记录 `HR@200`。

### 6.1 TwoTowerRetrievalModel

配置：

```yaml
models:
  two_tower:
    architecture:
      embedding_dim: 32
      user_hidden_dims: [256, 128]
      item_hidden_dims: [256, 128]
      dropout: 0.1
      recent_history_length: 20
    training:
      two_tower_num_negatives: 200
      two_tower_temperature: 0.05
```

输入：

- user tower：静态用户画像 + `hist_movie_id, hist_rating, hist_time_gap_bucket`；
- item tower：candidate/catalog `movie_id`。

结构：

1. `MovieFeatureEncoder` 编码历史电影静态 embedding；
2. `SequenceFeatureEncoder` 把历史电影 embedding 与 `hist_rating/time_gap` 融合成序列 token；
3. 对所有有效历史做 mean pooling；
4. 对最近 `recent_history_length` 条有效历史做 recent pooling；
5. 拼接静态 user embedding、长期历史、近期历史，经 MLP 得到 user embedding；
6. item tower 只编码静态 item embedding；
7. user/item embedding 都做 L2 normalize，用点积作为相似度。

输出：

```text
encode_user(batch) -> [batch_size, emb_dim]
encode_item(item_ids) -> [num_items, emb_dim]
```

训练：

- sampled softmax / cross entropy；
- 每行正样本是当前 target `movie_id`；
- negatives 来自 unseen random negatives + multimodal-similar hard negatives；
- negatives 会排除用户全部历史和当前正样本；
- row loss 按 target rating 加权。

### 6.2 SequenceRetrievalModel

配置：

```yaml
models:
  sequence:
    architecture:
      embedding_dim: 32
      hidden_dim: 64
      num_layers: 1
      max_len: 100
      dropout: 0.1
    training:
      sequence_num_negatives: 128
      sequence_loss: softmax
```

输入：

```text
hist_movie_id, hist_rating, hist_time_gap_bucket
```

结构：

1. 从预处理的 `max_seq_len=500` 历史中截取最近 `max_len=100`；
2. compact 非 padding item 后送入 GRU；
3. 每个 token = 静态电影 embedding + rating embedding + time_gap embedding 后过 MLP；
4. GRU 输出右对齐回原序列位置；
5. `encode_user(...)` 取最后一个有效位置 hidden state 作为 user representation；
6. `encode_item(...)` 只使用静态电影 embedding。

输出：

```text
encode_sequence(...) -> hidden_states [batch_size, max_len, emb_dim]
encode_user(...) -> [batch_size, emb_dim]
encode_item(item_ids) -> [num_items, emb_dim]
```

训练：

- 当前配置为 sampled softmax；
- negatives 同样排除用户历史和当前正样本；
- hard negatives 来自与正样本 multimodal embedding 相似的 unseen movies；
- row loss 按 target rating 加权。

### 6.3 ItemCF / Multimodal / Genre / Popular

- **ItemCF**：从 train split 的历史序列构建 item-item 共现/转移图，召回目标用户历史 item 的相似电影，过滤已看。
- **Multimodal**：用用户历史电影的 multimodal embedding 做加权兴趣向量，与全量 item multimodal embedding 做相似度检索，过滤已看。
- **Genre**：根据用户历史高评分电影的 genre 分布召回同类型电影，过滤已看；受用户历史内容影响，但不建模顺序。
- **Popular**：按 train split 全局高评分交互统计电影热度，给每个用户返回未看过的热门电影；只在过滤已看时受用户历史影响。

## 7. Multi-recall 融合

多路召回 route：

```text
two_tower, sequence, item_cf, multimodal, genre, popular
```

融合方式：

```yaml
fusion_method: rrf
rrf_k: 60
allocation_metric: incremental_recall@200
route_weight_clip:
  min: 0.3
  max: 1.0
```

流程：

1. 每一路生成 Top-K candidates；
2. 在 validation split 上计算单路指标和按优先级加入后的 incremental recall；
3. 根据 incremental recall 分配 route weight；
4. 对 route weight 做 clip；
5. 在 test split 上用加权 RRF 融合；
6. 输出 fused candidates，供最终评估使用。

最近一次 route 权重：

```text
prioritized_routes = [sequence, item_cf, two_tower, genre, popular, multimodal]

clipped weights:
sequence   = 1.000000
item_cf    = 0.612636
two_tower  = 0.300000
genre      = 0.300000
popular    = 0.300000
multimodal = 0.300000
```

最近一次 test 指标：

| Route | Recall@200 | HR@200 | NDCG@200 |
| --- | ---: | ---: | ---: |
| TwoTower | 0.5784 | 0.5784 | 0.1505 |
| Sequence | 0.6801 | 0.6801 | 0.2063 |
| ItemCF | 0.6488 | 0.6488 | 0.2181 |
| Multimodal | 0.2236 | 0.2236 | 0.0513 |
| Genre | 0.4201 | 0.4201 | 0.0984 |
| Popular | 0.3545 | 0.3545 | 0.0736 |
| Fused | 0.7490 | 0.7490 | 0.2431 |

## 8. Ranking 阶段

当前默认排序模型是 DIN：

```yaml
default_model: din

common:
  training:
    train_negatives: 10
    low_rating_negative_ratio: 0.5
    random_negative_ratio: 0.5
    negative_popularity_alpha: 0.75
    validation_negatives: 300
    positive_rating_min: 4.0
    rating_weighting_enabled: true
```

训练 candidate list：

```text
[positive target, low-rating negatives, popularity-sampled negatives]
```

当前比例由 ratio 配置控制，默认 10 个训练负样本约为：

```text
low-rating negatives ≈ 5
popularity-sampled negatives ≈ 5
```

popularity-sampled negatives 按电影流行度采样：

```text
P(item) ∝ popularity(item)^0.75
```

所有训练负样本都先受同一个用户级负采样池约束：

```text
allowed_negative_pool = all_movies - user_global_positive_feedback_movies
```

其中 `user_global_positive_feedback_movies` 是该用户全局时间线里 `rating >= 4.0` 的电影。low-rating negatives 优先从用户低分历史中取，但也必须属于这个池；popularity-sampled negatives 同样不能落在用户全局正反馈电影里。

### 8.1 DINModel

配置：

```yaml
models:
  din:
    architecture:
      embedding_dim: 64
      dnn_hidden_dims: [256, 128]
      attention_hidden_dims: [128, 64]
      top_m_history: 100
      force_recent_history: 30
      dropout: 0.1
```

输入：

- candidate 静态 movie embedding；
- 历史 movie embedding + `hist_rating/time_gap/feedback` 交互 token；
- 静态 user profile embedding。

结构：

1. candidate 用 `MovieFeatureEncoder` 编码静态电影特征；
2. history 先用 `MovieFeatureEncoder` 编码静态电影特征，再用 `SequenceFeatureEncoder` 融合 `hist_rating/time_gap/feedback`；
3. 为每个 candidate 选择 top-M 历史：强制保留最近 `force_recent_history=30` 条，再补充语义相似历史；
4. Local Activation Unit 输入：

```text
[candidate, history, candidate - history, candidate * history]
```

1. attention weight 加权历史得到 candidate-aware `activated_history`；
2. DNN 输入：

```text
[candidate_embedding, activated_history, user_profile]
```

输出：

```text
forward(batch) -> logits [batch_size, candidate_size]
```

### 8.2 DeepFMModel

DeepFM 是排序 baseline：

1. candidate 使用静态 movie embedding；
2. history 使用 `SequenceFeatureEncoder` 融合评分和 time gap 后 mean pooling；
3. user profile 使用静态用户画像；
4. FM 部分建模二阶交叉，DNN 部分建模高阶非线性；
5. 输出 shape 同 DIN：

```text
forward(batch) -> logits [batch_size, candidate_size]
```

### 8.3 Ranking loss

排序训练使用 sampled softmax / cross entropy：

- candidate list 中 target index 为正样本；
- logits 先按 `candidate_mask` 屏蔽 padding；
- loss 对每行正样本做 cross entropy；
- 如果开启 rating weighting，则按 `target_rating` 调整 row weight；
- 预处理已经保证 ranking target `rating >= 4.0`，loss 内不再重复按阈值过滤 target。

## 9. 指标口径

### 9.1 Retrieval / Multi-recall

使用 Top-K 排序指标：

- `Recall@K`
- `Precision@K`
- `HR@K`
- `NDCG@K`

当前 validation/test 每个用户只有一个 held-out target，因此 `Recall@K` 和 `HR@K` 数值相同。

### 9.2 Ranking / Final evaluation

排序模型在 fused candidates 上评估：

- `Recall@10/20`
- `HR@10/20`
- `NDCG@10/20`
- `Coverage@10/20`
- `AUC`
- `GAUC`
- `LogLoss`

AUC/GAUC/LogLoss 在当前 sampled-softmax 排序设定下主要作为辅助可分性/校准参考；更核心的推荐质量指标是 `NDCG@K`、`HR@K`、`Recall@K`。

最近一次 DIN ranking 指标：

| Metric | Value |
| --- | ---: |
| Recall@10 / HR@10 | 0.0326 |
| NDCG@10 | 0.0147 |
| Recall@20 / HR@20 | 0.0583 |
| NDCG@20 | 0.0211 |
| AUC | 0.5533 |
| GAUC | 0.5428 |
| LogLoss | 0.6381 |

最近一次 final DIN 指标：

| Metric | Value |
| --- | ---: |
| Recall@10 / HR@10 | 0.0376 |
| NDCG@10 | 0.0164 |
| Recall@20 / HR@20 | 0.0754 |
| NDCG@20 | 0.0258 |
| AUC | 0.5980 |
| GAUC | 0.5931 |
| LogLoss | 0.6009 |

## 10. 主要输出产物

```text
outputs/processed/train_eval_sample_final.pkl       # 召回 train/validation/test 样本
outputs/processed/ranking_train_eval_sample.pkl     # 排序 train/validation/test 样本
outputs/processed/item_catalog.pkl                  # item 特征表和多模态 embedding
outputs/models/retrieval_model.pt                   # TwoTower checkpoint
outputs/models/sequence_model.pt                    # Sequence retrieval checkpoint
outputs/models/item_embeddings.npy                  # TwoTower item embeddings
outputs/models/sequence_item_embeddings.npy         # Sequence item embeddings
outputs/models/item_cf_model.pkl                    # ItemCF 召回模型
outputs/models/genre_model.pkl                      # Genre 召回模型
outputs/models/popular_model.pkl                    # Popular 召回模型
outputs/models/multi_recall_artifacts.pkl           # fused multi-recall candidates
outputs/metrics/retrieval_metrics.json              # TwoTower/Sequence 召回指标
outputs/metrics/multi_recall_metrics.json           # 多路召回 route + fused 指标
outputs/metrics/ranking_metrics_din.json            # DIN rerank 指标
outputs/metrics/final_metrics_din.json              # 最终推荐指标
```

## 11. 当前设计要点

1. **MovieFeatureEncoder 静态化**：电影静态特征和多模态特征只在 `MovieFeatureEncoder` 中处理。
2. **评分和时间差是序列交互特征**：`hist_rating` 和 `hist_time_gap_bucket` 只在 `SequenceFeatureEncoder` 中融合到历史 token。
3. **召回训练宽松，评估严格**：retrieval train target 使用 `rating >= 3.0`，validation/test 使用 `rating >= 4.0`。
4. **排序严格**：ranking train/validation/test target 都使用 `rating >= 4.0`。
5. **召回负样本只来自 unseen movies**：TwoTower/Sequence 训练负样本排除用户全部历史和当前正样本，hard negatives 来自多模态相似电影。
6. **早停使用真实召回指标**：TwoTower/Sequence 用官方 validation split 的 `NDCG@200` 早停，不再用 train 内部 holdout loss。
7. **多路融合看增量贡献**：route weight 按 validation incremental recall 分配，再用加权 RRF 融合。
