# Movie Recommendation Offline

基于 MovieLens-1M 的离线电影推荐项目，目标是实现一条完整、可复现的 **召回 → 多路融合 → 排序 → 离线评估** 链路。当前版本重点关注两件事：

1. 在显式评分数据里区分“召回宽松、排序严格”的样本语义；
2. 面向长历史用户，把长期兴趣、近期兴趣、序列演化和 candidate-aware attention 分别放在合适的阶段建模。

## 1. 项目结构

```text
config/
  preprocess.yaml        # 召回/排序预处理阈值、max_seq_len、多模态配置
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
  data/                  # 原始数据加载、通用 IO
  features/              # 召回特征与样本构造
  ranking/               # 排序样本协议、预处理、dataset/collator
  models/                # TwoTower、GRU4Rec、DIN、DeepFM、feature encoders
  training/              # 召回与排序训练
  evaluate/              # 多路召回与最终评估
  utils/                 # 配置解析等工具
outputs/
  processed/             # 样本、词表、item catalog、多模态 embedding
  models/                # 训练好的模型
  metrics/               # 各阶段指标
```

## 2. 数据与运行方式

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
python scripts/run_retrieval_preprocess.py
python scripts/run_ranking_preprocess.py
python scripts/run_retrieval_train.py
python scripts/run_ranking_train.py
python scripts/evaluate_offline.py
```

也可以执行全流程：

```bash
python scripts/run_offline_pipeline.py --steps all
```

## 3. 样本语义总览

当前代码统一使用 `hist_*` 表示用户在目标 item 之前的历史，不再使用旧的 `context_*` 命名，也不再保存 `hist_feedback`。历史是否作为正反馈/低分反馈，直接由 `hist_rating` 和 padding mask 动态判断。

核心评分阈值：

| 阶段 | 正样本/兴趣阈值 | 低分负反馈阈值 | 设计意图 |
| --- | --- | --- | --- |
| 召回 retrieval | `rating >= 3.0` | `rating <= 2.0` | 召回阶段更宽松，尽量扩大候选覆盖 |
| 排序 ranking | `rating >= 4.0` | `rating <= 2.0` | 排序阶段更严格，只学习高偏好行为 |

recency bucket 使用 **目标时间戳 - 历史时间戳** 的真实时间差，而不是序列位置差。bucket 边界为：

```text
1, 3, 7, 14, 30, 90, 365 days
```

padding 位置 bucket 为 0。

## 4. 预处理设计

### 4.1 召回预处理

配置来自 `config/preprocess.yaml`：

```yaml
retrieval:
  max_seq_len: 500
  positive_rating_min: 3.0
  negative_rating_max: 2.0
  neutral_rating: 3.0
  sequence_negative_pool_size: 500
```

召回样本按用户时间序列构造：

1. 对每个用户按 timestamp 排序交互；
2. 选出 `rating >= 3.0` 的正向 target 位置；
3. 每个 target 的输入历史是它之前的所有交互，最多保留最近 `500` 条，右对齐 padding；
4. 每个样本包含：
   - `user_id / gender / age / occupation / zip` 等用户特征；
   - `movie_id`：当前正样本 target；
   - `hist_movie_id`：target 之前的历史 item 序列；
   - `hist_rating`：历史评分；
   - `hist_recency_bucket`：每个历史 item 相对 target 的时间差 bucket；
   - `user_negative_movie_id`：该用户历史中 `rating <= 2.0` 的低分电影池，最多 `500` 个。

召回 split 按用户留出：

- 最后一个正向 target：`test`；
- 倒数第二个正向 target：`validation`；
- 更早的正向 target：`train`。

最近一次构建规模：

```text
train=819406
validation=6038
test=6038
max_hist_seq_len=500
```

### 4.2 排序预处理

配置来自 `config/preprocess.yaml`：

```yaml
ranking:
  max_seq_len: 300
  positive_rating_min: 4.0
  negative_rating_max: 2.0
  interest_rating_min: 4.0
```

排序阶段更严格：

1. target 正样本必须满足 `rating >= 4.0`；
2. 输入历史只保留 target 之前 `rating >= 4.0` 的兴趣行为；
3. `rating <= 2.0` 的历史 item 进入 `low_rating_movie_id`，作为排序训练时的低分负样本来源；
4. 历史最多保留最近 `300` 条，同样右对齐 padding；
5. hard negatives 不在预处理时固定写死，而是在训练 collator 中从多路召回 fused candidates 里在线取。

排序样本字段主要包括：

- `hist_movie_id`
- `hist_rating`
- `hist_recency_bucket`
- `hist_length`
- `low_rating_movie_id`
- `target_movie_id`
- `target_rating`

## 5. Item 特征与模型输入边界

item 侧特征包括：

- `movie_id`
- `genres`：multi-hot genre 特征
- `isAdult`
- `startYear`
- `popularity`
- `averageRating`
- multimodal embedding：由 CLIP/ViT 侧生成的电影多模态向量

当前召回链路已经把 item feature lookup 收敛到模型内部：TwoTower、Sequence/GRU4Rec、多路召回构建时主要传 `movie_id`，由 `MovieFeatureEncoder` 根据内部 item feature table 查结构化特征和 multimodal embedding。这样避免在训练/评估/多路召回阶段反复传大块 item feature tensor，也降低 `max_seq_len=500` 时的内存压力。

## 6. 召回阶段

召回配置来自 `config/retrieval.yaml`。共享训练配置包括：

```yaml
common:
  training:
    batch_size: 512
    epochs: 10
    learning_rate: 0.0005
    weight_decay: 1.0e-5
    negative_sampling: mixed
    hard_negative_topk: 32
    positive_rating_min: 3.0
    negative_rating_max: 2.0
    neutral_rating: 3.0
  evaluation:
    topk: 200
```

多路召回 routes：

```yaml
routes:
  - two_tower
  - sequence
  - item_cf
  - multimodal
  - genre
  - popular
```

### 6.1 TwoTower 召回

结构配置：

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
      two_tower_num_sampled_negatives: 64
      two_tower_temperature: 0.05
```

输入特征：

- user tower：
  - 静态用户画像：`user_id, gender, age, occupation, zip`；
  - `hist_movie_id`；
  - `hist_rating`；
  - `hist_recency_bucket`。
- item tower：
  - target/candidate `movie_id`；
  - item 结构化特征和 multimodal embedding 由模型内部 lookup。

模型逻辑：

1. `MovieFeatureEncoder` 编码历史电影和 target/candidate 电影；
2. user 侧编码静态用户画像；
3. 历史兴趣包含长期 pooling 与近期 pooling：
   - 长期兴趣：对满足召回兴趣阈值的历史 item 做 pooling；
   - 近期兴趣：取最近 `recent_history_length=20` 的兴趣历史做 pooling；
4. 拼接静态用户、长期历史、近期历史，经 MLP 得到 user embedding；
5. item tower 经 `MovieFeatureEncoder + MLP` 得到 item embedding；
6. user/item embedding 都做 L2 normalize，用点积作为相似度。

正负样本：

- 正样本：当前 retrieval sample 的 `movie_id`，即 `rating >= 3.0` 的 target；
- in-batch negatives：同 batch 其他用户的正样本 item；
- sampled negatives：从全局 item set 采样，排除用户已看历史和当前正样本；
- low-rating negatives：来自 `user_negative_movie_id`，即该用户历史中 `rating <= 2.0` 的电影。

Loss：

- cross entropy；
- 正样本为第 0 类，负样本包括 in-batch、随机采样和用户低分负样本；
- logits 按 `two_tower_temperature=0.05` 缩放。

### 6.2 Sequence / GRU4Rec 召回

结构配置：

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
      sequence_num_negatives: 32
      sequence_user_negative_ratio: 0.5
      sequence_loss: bce
      sequence_history_feedback: all
```

注意：预处理保留 `max_seq_len=500`，但 GRU4Rec 模型实际只消费最近 `max_len=100` 个位置。这是为了让样本产物保留更长历史给其他召回/排序模块使用，同时控制序列模型训练显存和计算成本。

输入特征：

- `hist_movie_id`；
- `hist_rating`；
- `hist_recency_bucket`；
- target/candidate `movie_id`。

模型逻辑：

1. 取历史最后 `100` 个位置；
2. 根据 mask compact 非 padding item；
3. `MovieFeatureEncoder` 编码历史 item，内部包含结构化 item 特征和 multimodal embedding；
4. 加上 rating projection：`hist_rating / 5.0 -> embedding_dim`；
5. 加上 recency bucket embedding；
6. 输入 GRU 建模短期兴趣演化；
7. 取最后一个可见位置的 hidden state 作为 user representation；
8. 与 candidate item embedding 点积得到 logit。

`sequence_history_feedback: all` 表示 GRU 输入保留所有历史评分行为；如果切换为 positive 模式，则根据 `hist_rating >= positive_rating_min` 动态过滤，不需要单独的 `hist_feedback` 字段。

正负样本：

- 正样本：当前 retrieval target `movie_id`；
- 负样本数量：`sequence_num_negatives=32`；
- 其中一部分来自用户低分池，比例由 `sequence_user_negative_ratio=0.5` 控制；
- 其余从全局 item set 随机采样并排除已看 item。

Loss：

- 当前配置为 `bce`；
- 正样本 logit 对应 label 1；
- 负样本 logit 对应 label 0；
- 代码也支持 `ce`，即正样本 + 多个负样本上的 cross entropy。

### 6.3 ItemCF 召回

输入：

- train split 中的用户历史序列；
- 目标用户的 `hist_movie_id`；
- 评分阈值使用召回阶段语义。

构建方式：

1. 只从 train split 构建 item-item 共现/转移统计，避免把 validation/test target 泄漏进召回图；
2. 对每个用户取最近若干历史 item；
3. 聚合这些历史 item 的相似 item，得到候选；
4. 过滤用户已看 item。

这一路不训练神经网络，是基于行为共现的协同过滤召回。

### 6.4 Multimodal 召回

输入：

- item multimodal embedding；
- 用户历史 item 及评分/recency 权重。

构建方式：

1. 用历史 item 的 multimodal embedding 做加权 pooling 得到用户多模态兴趣向量；
2. 与全量 item multimodal embedding 做 cosine similarity；
3. 取 Top-K 并过滤已看 item。

这一路主要补充内容相似候选，和行为协同过滤形成互补。

### 6.5 Genre 召回

输入：

- 用户历史 item 的 `genres` multi-hot 特征；
- item catalog 中的 `genres`。

构建方式：

1. 从用户近期历史统计 genre preference；
2. 对 candidate item 的 genre multi-hot 做匹配打分；
3. 取 Top-K 并过滤已看 item。

这一路是轻量内容召回，用于补充类型偏好。

### 6.6 Popular 召回

输入：

- train split 中正向交互统计。

构建方式：

1. 只基于 train split 统计 item 热度；
2. 对每个用户返回未看过的热门电影；
3. 作为兜底召回通道。

## 7. 多路召回融合

多路召回构建入口会生成：

- 各 route 的 Top-K candidates；
- 每路单独指标；
- validation 上的 route 增量贡献；
- test 上的 fused candidates 和 fused metrics。

当前融合方法：

```yaml
fusion_method: rrf
rrf_k: 60
allocation_metric: incremental_recall@200
route_weight_clip:
  min: 0.3
  max: 1.0
```

融合流程：

1. 每一路先生成 `topk=200` 个候选；
2. 对 validation split 计算每一路单独 recall 和按顺序加入时的 incremental recall；
3. 根据 `incremental_recall@200` 分配 route weight；
4. 对权重做 clip，避免某一路完全失效；
5. 在 test split 上用加权 RRF 融合：排名越靠前贡献越大，route 权重越高贡献越大；
6. 最终输出 fused Top-K candidates，供排序阶段作为 hard negatives 或最终候选集。

最近一次多路召回 route 权重：

```text
prioritized_routes = [sequence, item_cf, two_tower, genre, popular, multimodal]

clipped weights:
sequence   = 1.000000
item_cf    = 0.656340
two_tower  = 0.336666
genre      = 0.300000
popular    = 0.300000
multimodal = 0.300000
```

最近一次 test 指标：

| Route | Recall@200 | NDCG@200 |
| --- | ---: | ---: |
| TwoTower | 0.5745 | 0.1521 |
| Sequence / GRU4Rec | 0.6565 | 0.1836 |
| ItemCF | 0.6363 | 0.2088 |
| Multimodal | 0.2229 | 0.0506 |
| Genre | 0.3978 | 0.0911 |
| Popular | 0.3322 | 0.0677 |
| Fused | 0.7360 | 0.2141 |

Fused 完整指标：

```text
Recall@50  = 0.5025
NDCG@50    = 0.1785
Recall@100 = 0.6275
NDCG@100   = 0.1988
Recall@200 = 0.7360
NDCG@200   = 0.2141
```

内存控制上，多路召回构建只需要 validation/test，因此加载 retrieval sample 后会丢弃 train split；每路候选也只生成 `route_topk=topk`，避免在 `max_seq_len=500` 时额外放大候选矩阵。

## 8. 排序阶段

排序配置来自 `config/ranking.yaml`，当前默认模型是 DIN：

```yaml
default_model: din

common:
  training:
    train_negatives: 15
    low_rating_negatives: 3
    recall_hard_negatives: 10
    random_negatives: 2
    validation_negatives: 500
    positive_rating_min: 4.0
    negative_rating_max: 2.0
    rating_weighting_enabled: true
```

排序训练 candidate 构造：

1. 每条训练样本有一个正样本 `target_movie_id`；
2. 从 `low_rating_movie_id` 采样低分负样本；
3. 从 multi-recall fused candidates 采样 hard negatives；
4. 从全局 item set 采样 random negatives；
5. 组成 listwise candidate list：

```text
[positive, low-rating negatives, recall hard negatives, random negatives]
```

当前默认数量：

```text
positive = 1
low_rating_negatives = 3
recall_hard_negatives = 10
random_negatives = 2
```

### 8.1 DIN

结构配置：

```yaml
models:
  din:
    architecture:
      embedding_dim: 64
      dnn_hidden_dims: [256, 128]
      attention_hidden_dims: [128, 64]
      top_m_history: 100
      dropout: 0.1
```

输入：

- candidate movie embedding；
- `hist_movie_id` 对应的历史 item embedding；
- 静态 user profile embedding；
- context embedding。

当前 DIN 设计：

1. 从长历史里为每个 candidate 检索最相关的 `top_m_history=100` 个历史 item；
2. Local Activation Unit 输入：

```text
[candidate, history, candidate - history, candidate * history]
```

3. activation unit 直接输出 attention weight，不做 softmax；
4. 用 attention weight 对历史 embedding 加权求和，得到 candidate-aware `activated_history`；
5. `context_embedding = global_history_context + global_recency_context`；
6. `user_profile` 只包含静态用户画像，不混入历史上下文；
7. DNN 输入为：

```text
[candidate_embedding, activated_history, user_profile, context_embedding]
```

Loss：

- listwise softmax / cross-entropy 风格；
- 对同一用户的一组 candidates 做 `log_softmax`；
- target distribution 来自 candidate relevance；
- 正样本 relevance 为真实 `target_rating`，负样本 relevance 为 0；
- 可叠加 rating weight，让高分正样本权重更大。

### 8.2 DeepFM

DeepFM 作为排序 baseline：

- 使用相同的 candidate 构造和 item/user/context 特征；
- FM 部分建模二阶特征交叉；
- DNN 部分建模高阶非线性；
- 训练协议与 DIN 保持一致，便于对比。

## 9. 当前设计要点

1. **召回宽松，排序严格**  
   召回用 `rating >= 3.0` 扩大候选覆盖；排序用 `rating >= 4.0` 学更强偏好。

2. **低分样本不是简单丢弃**  
   `rating <= 2.0` 的历史 item 会进入用户低分负样本池，用作 TwoTower、GRU4Rec 和 DIN/DeepFM 的强负样本来源。

3. **长序列分阶段处理**  
   - TwoTower：长期兴趣 + 近期兴趣 pooling；
   - GRU4Rec：最近 100 个行为的短期兴趣演化；
   - DIN：从最多 300 条排序历史中检索与 candidate 最相关的 top-M 历史做 attention。

4. **recency 使用真实时间差**  
   recency bucket 基于 timestamp delta，而不是历史序列位置。

5. **item feature lookup 放在模型边界内**  
   召回训练和 multi-recall 构建尽量只传 item id，模型需要什么 item 特征就从内部表查，减少内存复制和接口噪声。

6. **多路融合看增量贡献**  
   route weight 不是按单路 recall 直接分配，而是按 validation 上的 incremental recall 分配，再用 RRF 融合。

## 10. 主要输出产物

```text
outputs/processed/train_eval_sample_final.pkl       # 召回 train/validation/test 样本
outputs/processed/ranking_train_eval_sample.pkl     # 排序样本
outputs/processed/item_catalog.pkl                  # item 元数据和特征表
outputs/models/                                     # two_tower / sequence / ranking 模型
outputs/metrics/retrieval_metrics.json              # 单模型召回指标
outputs/metrics/multi_recall_metrics.json           # 多路召回 route + fused 指标
outputs/metrics/ranking_metrics.json                # 排序指标
```

## 11. 指标口径

召回与多路召回：

- `Recall@K`
- `Precision@K`
- `HR@K`
- `NDCG@K`

排序：

- listwise candidate loss；
- ranking validation/test 可进一步看 AUC、GAUC、LogLoss 或最终 Top-K 指标。

最终评估：

- `Recall@10/20`
- `HR@10/20`
- `NDCG@10/20`
- `Coverage@10/20`
