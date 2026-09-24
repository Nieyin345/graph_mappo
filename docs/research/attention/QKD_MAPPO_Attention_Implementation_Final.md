# QKD-MAPPO Attention 实施前最终定稿
## 针对《实施前确认.md》的逐项回答

> 本文用于直接交给实现 AI / 代码 Agent。
>
> 本版本在 `docs/research/attention/QKD_MAPPO_Attention_Design_Final_v2.md` 基础上，对实施前的 6 个问题逐项定死。
>
> 核心原则：
>
> 1. **不改现有 Action Space。**
> 2. **保留现有 `edge_scorer`，新模块采用 residual/additive 方式接入。**
> 3. **新模块零初始化时，前向必须逐位等价现有模型。**
> 4. **新分数保持弧级（directed arc）输出，因为当前 sampler 的动作粒度就是有向弧。**
> 5. **物理 Key 仍然是无向共享资源；“弧级 score”只是 action representation，不表示 Key 本身有方向。**
> 6. **本次不做 request-level Attention Pool。直接使用已有 Demand Edge embedding。**
> 7. **本次不做 BFS 候选筛选。候选保持全部当前活跃物理弧。**
> 8. **Cross-Attention 的方向要修正：如果要求每条候选弧得到一个独立的 demand-aware 向量，标准 Multi-Head Attention 应采用“Edge Query、Demand Key/Value”。**
> 9. 如果一定坚持“Demand Query、Edge Key/Value”，标准 Cross-Attention 只能自然地产生“每个 Demand 一个输出”，不能直接得到逐 Edge `z_e`；因此不采用该写法。
> 10. 新模块只能作为 residual enhancement，不得破坏 BC warm start。

---

# 1. 最终总结构

本次实施的主模型：

```text
现有 GNN
    ↓
node_emb / edge_emb
    ↓
现有 directed arc representation
    ↓
旧 edge_scorer
    ↓
base_score_arc
          +
          │
          │ residual branch
          ▼
Demand Edge embeddings
    +
Arc contextual embeddings
    ↓
Demand-aware Cross-Attention
    ↓
per-arc contextual vector
    ↓
zero-initialized residual MLP
    ↓
delta_score_arc
          │
          ▼
score_arc = base_score_arc + delta_score_arc
          │
          ▼
现有 Gumbel sequential sampler
          │
          ▼
现有 action / resolver / Key Generation
```

本次**不修改**：

- Action Space；
- `_sample_matching_arrays` 的基本机制；
- `_actions_from_index_pairs`；
- `action_resolver`；
- Key Generation 数量计算；
- Service Algorithm；
- BC expert action 定义。

---

# 2. 问题 1：新分数到底替换还是相加？

## 最终答案

\[
\boxed{\text{相加，不替换}}
\]

即：

\[
s^{base}_a = EdgeScorer(pairEmb_a)
\]

新分支：

\[
\Delta s_a = ResidualMLP(z_a)
\]

最终：

\[
\boxed{
s_a=s^{base}_a+\Delta s_a
}
\]

---

# 3. 为什么必须采用 residual，而不是替换旧 edge_scorer

当前旧模型：

```python
pair_emb = torch.cat([
    node_emb[src],
    node_emb[dst],
    edge_emb_directed
], dim=-1)

base_score = edge_scorer(pair_emb)
```

这个 `edge_scorer` 已经存在于 BC checkpoint 中。

如果直接替换：

```text
old edge_scorer
        ↓
不再使用
```

那么：

- BC checkpoint 中对应的 65,921 个参数失去作用；
- 新结构在初始化时无法与旧模型保持前向一致；
- 单变量实验变差；
- 必须重新训练 BC 才能得到新的 warm start。

这与项目已有的：

> 新模块只加不减 + zero initialization + default off

实验原则冲突。

---

# 4. Residual 的正确实现方式

推荐：

```python
base_score = self.edge_scorer(pair_emb)

delta_score = self.cross_residual_head(cross_edge_ctx)

score = base_score + delta_score
```

其中：

```python
self.cross_residual_head[-1]
```

最后一层权重和 bias 全部零初始化。

于是初始化时：

\[
\Delta s_a=0
\]

因此：

\[
s_a=s^{base}_a
\]

做到：

\[
\boxed{
f_{new}(x)=f_{old}(x)
}
\]

的逐位等价。

---

# 5. 不建议使用“初始为 0 的可学习 gate”

例如不要直接做：

```python
score = base_score + alpha * delta_score
alpha = 0
```

原因：

如果 `alpha=0`，在最初反向传播阶段：

\[
\frac{\partial L}{\partial \theta_{\Delta}}
\propto \alpha
\]

新分支内部参数可能无法正常获得梯度。

而：

```text
最后一层权重 = 0
```

只会使：

> 输出为 0，但 residual branch 的最终层本身仍然可以得到梯度。

因此本次推荐：

\[
\boxed{
\text{固定 residual 系数}=1
}
\]

只对 residual head 的最后一层做 zero initialization。

---

# 6. BC Warm Start 的最终处理

采用 residual 后：

### 旧 checkpoint 中：

```text
edge_scorer
```

的参数：

> **继续完整加载。**

### 新增参数：

```text
cross_attention
residual_head
```

使用：

> 正常初始化 + residual 最后一层零初始化。

因此：

```text
旧 BC 权重
+
新 branch 初始 delta = 0
```

得到：

\[
\boxed{
\text{新模型初始化行为}=\text{旧 BC 模型行为}
}
\]

不需要为了本次结构实验重新制作 BC。

---

# 7. 问题 2：新分数到底按“弧”还是“无向边”？

## 最终答案

\[
\boxed{\text{弧级}}
\]

即：

```text
A → B
B → A
```

分别产生：

```text
score[A→B]
score[B→A]
```

---

# 8. 为什么物理 Key 无向，但 Action score 仍然必须保留方向

这里必须区分两个概念。

## 8.1 Key Resource

物理 Key：

\[
K_{AB}=K_{BA}
\]

本质上是：

> 无向共享资源。

---

## 8.2 Action Representation

当前 Actor / sampler 使用：

```text
directed arc
A → B
B → A
```

因为 Action 同时编码：

- 谁使用 Tx-out；
- 谁使用 Rx-in。

当前节点约束也是：

\[
outdeg(v)\le1
\]

\[
indeg(v)\le1
\]

所以：

> **方向是资源调度 / Action representation 的一部分，而不是 Key 本身有方向。**

因此不能因为 Key 无向，就把两个方向强行设成完全相同的 score。

---

# 9. 为什么边级 score 广播到两个方向不推荐

如果：

```text
score[A→B] = score[B→A]
```

那么模型无法表达：

> A 适合作 Tx 而 B 适合作 Rx

这种方向性。

最终只能依赖：

- Gumbel 随机噪声；
- sampler 的候选顺序；
- resolver 的后处理。

这会把本来可以学习的方向决策交给随机因素。

所以本次采用：

\[
\boxed{
s_{r,A\rightarrow B}
\neq
s_{r,B\rightarrow A}
}
\]

允许方向性存在。

---

# 10. 但如何保持“Key 本身无向”？

不是把 score 做成无向，而是：

> **底层 physical edge representation 可以共享；在生成 directed arc representation 时注入 endpoint order。**

例如：

```text
physical edge embedding:
    h_edge

arc representation:
    h_arc(A→B)
        = MLP([h_A, h_B, h_edge])

arc representation:
    h_arc(B→A)
        = MLP([h_B, h_A, h_edge])
```

这样：

```text
h_edge
```

仍然描述无向物理链路，

而：

```text
h_arc
```

描述：

> 在当前 Action semantics 下，以哪一端作为 source / Tx、哪一端作为 destination / Rx。

---

# 11. 问题 3：Cross-Attention 的 z_e 到底输入什么？

## 最终答案

不要只使用：

```text
edge_emb
```

而应该使用：

\[
\boxed{
h^{arc}=
MLP([h_{src},h_{dst},h_{edge}])
}
\]

也就是：

```text
起点节点 embedding
+
终点节点 embedding
+
物理边 embedding
```

对应当前已有 `edge_scorer` 的输入信息。

---

# 12. 为什么不能只用 edge_emb

3 层 GNN 后：

```text
edge_emb
```

确实已经吸收了两端节点的一部分信息。

但是：

> GNN 得到的是经过 message passing / aggregation 后的融合表示。

它不保证：

```text
“src 当前状态”
“dst 当前状态”
```

的方向性细节全部被无损保留。

尤其现在 Action 是 directed arc。

因此新 branch 最稳妥的做法是直接保留端点信息。

---

# 13. 推荐的 Arc Context

定义：

\[
h_a^{ctx}
=
Proj_{arc}
(
[h_{src},h_{dst},h_{edge}]
)
\]

其中：

```text
[h_src, h_dst, h_edge] = 384 dim
```

再投影到：

```text
128 dim
```

用于 Attention。

例如：

```python
arc_ctx = arc_context_proj(
    torch.cat([
        node_emb[src_t],
        node_emb[dst_t],
        edge_emb_directed[pos_t],
    ], dim=-1)
)
```

这样同时保证：

- 使用节点状态；
- 使用物理边状态；
- 保留方向；
- 不修改现有 `edge_scorer`。

---

# 14. ★ 问题 3 中真正重要的 Attention 结构修正

原先文档写的是：

```text
Demand = Query
Edge = Key/Value
```

然后声称：

> 得到逐 Edge `z_e`。

这是标准 Cross-Attention 语义下不严谨的。

如果：

```text
Q = demand
K/V = all candidate edges
```

那么：

> 一个 Demand Query 会自然地产生一个 attention 输出。

也就是说输出粒度天然是：

```text
per-demand
```

而不是：

```text
per-edge
```

---

# 15. 为了得到“逐边 demand-aware vector”，本次应该反转 Attention 方向

采用：

\[
\boxed{
Q = Edge
}
\]

\[
\boxed{
K/V = Demand
}
\]

即：

```text
每一条候选弧
    ↓ Query
所有 Demand Edges
    ↓ Key / Value
Cross-Attention
    ↓
这一条弧自己的 demand-aware context
```

于是自然得到：

\[
z_a
\]

其中：

\[
a = \text{candidate directed arc}
\]

这与最终需求完全一致：

> 每条弧最后必须得到一个独立 score。

---

# 16. 这仍然是“Demand → Edge”建模

虽然 Attention 的张量方向写成：

```text
Edge Query
Demand Key/Value
```

但模型语义仍然是：

> **让每条 Edge 根据当前 Demand context 重新评估自己的价值。**

可以称为：

```text
Demand-conditioned Edge Attention
```

而不是强调：

```text Demand Query
```

---

# 17. 推荐 Attention 数据流

设：

\[
h_a^{ctx}\in\mathbb{R}^{128}
\]

是每条候选弧的 context embedding。

设：

\[
q_d\in\mathbb{R}^{128}
\]

是第 d 个 Demand Edge embedding。

则：

\[
z_a
=
CrossAttention(
Q=h_a^{ctx},
K=\{q_d\},
V=\{q_d\}
)
\]

每个候选弧得到一个：

\[
z_a\in\mathbb{R}^{128}
\]

然后：

\[
\Delta s_a
=
MLP_{res}(z_a)
\]

最终：

\[
s_a
=
s_a^{base}
+
\Delta s_a
\]

---

# 18. Demand 有多个时，如何处理？

当前 Demand Edge 数量只有：

```text
1 ~ 12
```

因此：

```text
每条候选弧
    ×
所有 Demand Edges
```

规模非常小。

例如：

```text
205 active arcs
×
12 demands
≈
2460 pair interactions
```

所以不需要 BFS 来提前裁掉候选。

模型计算量完全可以接受。

---

# 19. 问题 4：要不要现在加入 Request-level Attention Pool？

## 最终答案

\[
\boxed{\text{本次不加}}
\]

采用：

> **现有 Demand Edge embedding 作为 Demand token。**

---

# 20. 原因

当前代码已经把同一个 BS 对的请求聚合成：

```text
1 个 Demand Edge
```

其 16 维特征已经包含：

```text
pending_amount
pending_count
min_deadline_left
mean_deadline_left
mean_wait_time
priority_sum
wait_bucket_amounts[10]
```

而且这些特征已经实际存在并被实验使用。

因此当前最合理的是：

```text
Demand Edge raw features
    ↓
existing GNN
    ↓
Demand Edge embedding
    ↓
Cross-Attention
```

不再额外拆出 request-level attention。

---

# 21. 为什么不在本次同时加 Request Attention

需要新增：

- 请求级 token；
- request → BS pair 聚合；
- request-level mask；
- batch structure；
- expiration handling；
- pooling；
- 新的张量对齐。

这样会同时改变：

```text
数据结构
+
模型结构
+
训练流程
```

不利于单变量实验。

当前实验本身已经存在：

```text
effect ≈ 0.008~0.017
```

的检测限制。

所以必须控制变量。

---

# 22. 后续若需要 Request Attention

可以作为独立 Experiment：

```text
Demand Edge embedding
        vs
Request-level Attention Pool
```

但不是本次主实验的一部分。

---

# 23. 问题 5：这次是否做 BFS 筛选？

## 最终答案

\[
\boxed{\text{不做}}
\]

本次：

\[
CandidateArcs
=
\text{全部当前活跃物理弧}
\]

即保持现有：

```text
134 ~ 205 active edges
```

范围。

如果转换成 directed arcs，则大致仍属于几百级别候选。

---

# 24. 不做 BFS 的第一个原因：服务路径与生成候选不是同一张图

当前服务算法：

```text
静态 shortest path
+
positive inventory graph BFS
```

而 Generation Candidate 依赖：

```text
当前可生成 FSO graph
```

如果现在用一个“活跃图 shortest path / BFS”去筛选 candidate：

> 很容易把“当前拓扑上看不到的潜在有价值生成边”错误过滤掉。

尤其 Key Generation 的目的不仅仅是补当前 shortest path。

---

# 25. 第二个原因：当前候选规模根本不需要 BFS 优化

当前：

```text
active physical edges ≈ 134~205
demands ≈ 1~12
```

所以即使做：

\[
205\times12
\]

级别的 edge-demand interaction：

\[
\approx2460
\]

也很小。

因此目前没有必要用 BFS 换取计算量。

---

# 26. 第三个原因：保留“无当前需求时的 proactive generation”

这是不做 BFS 最重要的一个原因。

系统允许：

```text
当前 Demand = 0
```

时仍然生成 Key。

如果：

```text
Candidate edges = 只来自需求 BFS
```

那么：

```text
没有 Demand
↓
没有 BFS candidate
↓
无法主动预生成
```

这与当前任务要求冲突。

所以：

\[
\boxed{
Candidate\ Set\ 不应该由 Demand 是否存在决定
}
\]

---

# 27. 没有 Demand 时，Residual Branch 应该怎么做？

如果：

```text
D = 0
```

那么没有 Demand token。

此时：

\[
\Delta s_a=0
\]

因此：

\[
s_a=s_a^{base}
\]

模型仍然可以按照原来的 baseline policy 做 proactive generation。

这个性质非常好：

> **新模型不会因为“没有当前请求”而把 Key Generation 强行变成 0。**

---

# 28. 多 Demand 与一条 Edge 的关系

因为本次：

```text
Q = Edge
K/V = all Demand Edges
```

所以：

> 一条物理弧可以同时观察全部 Demand Edge。

因此：

```text
Demand1
Demand2
Demand3
...
```

对该弧的综合影响由 Attention 自动形成。

相比：

```text
每个 demand 单独算 score
再人工 sum
```

这种方法更加自然。

---

# 29. 问题 6：零初始化要求是否兼容？

## 最终答案

\[
\boxed{\text{完全兼容}}
\]

而且本次应该把它作为硬性实现要求。

---

# 30. 最终 Forward 形式

建议：

```python
base_score = edge_scorer(pair_emb)

arc_ctx = arc_context_proj(pair_emb)

demand_ctx = existing_demand_edge_embeddings

cross_ctx = edge_to_demand_attention(
    query=arc_ctx,
    key=demand_ctx,
    value=demand_ctx,
)

delta_score = residual_score_head(cross_ctx)

score = base_score + delta_score
```

其中：

```python
residual_score_head[-1].weight = 0
residual_score_head[-1].bias = 0
```

所以初始化：

```text
delta_score == 0
```

得到：

```text
score == base_score
```

---

# 31. 开关行为

建议增加：

```text
use_demand_edge_cross_attention
```

默认：

```text
false
```

当：

```text
false
```

时：

```python
score = base_score
```

当：

```text
true
```

且 residual head zero-init 时：

```python
score = base_score + 0
```

因此：

\[
\boxed{
forward_{new}=forward_{old}
}
\]

逐位一致。

---

# 32. 新参数初始化策略

### 旧参数

全部保留：

```text
GNN
edge_scorer
critic
...
```

### 新参数

```text
arc_context_proj
cross_attention
residual_score_head
```

正常初始化。

但是：

```text
residual_score_head
```

最后一层：

```text
weight = 0
bias = 0
```

---

# 33. 关于 residual 分支第一轮学习

初始化时：

```text
delta_score = 0
```

因此：

```text
第 0 个 forward
=
旧模型
```

第一次反向传播时：

> residual head 最后一层会首先获得梯度。

随着最后一层权重开始偏离 0：

> 前面的 Cross-Attention / projection 也开始获得有效梯度。

这是可接受且常见的 residual adapter 式初始化。

因此不要因此引入：

```text
learnable alpha = 0
```

---

# 34. 最终实现张量粒度

## 34.1 Candidate arc

每条候选弧：

```text
a = (src, dst, physical_edge_id)
```

保留：

```text
src
dst
edge_id
```

---

## 34.2 Base arc embedding

当前已有：

```text
pair_emb[a]
=
concat(
    node_emb[src],
    node_emb[dst],
    edge_emb_directed[a]
)
```

维度：

\[
384
\]

---

## 34.3 Arc context

推荐：

```text
arc_ctx = Linear/MLP(384 → 128)(pair_emb)
```

---

## 34.4 Demand token

直接使用：

```text
demand_edge_emb
```

维度：

\[
128
\]

来自现有 GNN。

不增加 request-level pooling。

---

## 34.5 Cross-Attention

：

```text
Q = arc_ctx
K = demand_edge_emb
V = demand_edge_emb
```

输入形状概念：

```text
[batch, num_candidate_arcs, 128]
        Query

[batch, num_demand_edges, 128]
        Key / Value
```

输出：

```text
[batch, num_candidate_arcs, 128]
```

因此天然得到：

> 每条候选弧一个 demand-aware representation。

---

# 35. 特别说明：这次不是直接使用 Attention QK score

本项目仍然坚持：

> **QK Attention weight 不是最终 Action score。**

最终流程：

\[
Q/K/V
\rightarrow
z_a
\rightarrow
ResidualMLP
\rightarrow
\Delta s_a
\rightarrow
BaseScore+\Delta s_a
\]

即：

\[
\boxed{
\text{融合向量}
\rightarrow
\text{MLP}
\rightarrow
\text{Action Score}
}
\]

而不是：

\[
\boxed{
QK
\rightarrow
Action
}
\]

---

# 36. 为什么这个结构比“Demand Query → Edge KV”更合理

原写法：

```text
Demand Query
Edge Key/Value
```

更自然地解决：

> “这个 Demand 关注哪几条 Edge？”

它产生的是：

```text
per-demand representation
```

而我们的最终问题是：

> “每条 Edge 在当前全部 Demand 下应该得到什么 score？”

所以反过来：

```text
Edge Query
Demand Key/Value
```

直接计算：

> “这条 Edge 当前应该关注哪些 Demand、综合这些 Demand 后，它的价值是什么？”

输出粒度正好对应 Action。

---

# 37. 是否需要 Edge Self-Attention？

## 本次：不加入主模型。

原因仍然是：

- GNN 已经提供局部结构；
- route structure 可以由规则计算；
- 当前第一优先级是 Demand → Edge relation；
- Edge Self-Attention 的唯一合理意义是验证“候选边联合价值”。

因此：

```text
Experiment 1:
    Base + Demand-conditioned Edge Attention

Experiment 2:
    Experiment 1 + Edge Self-Attention
```

单独比较。

---

# 38. 当前版本明确不做的事情

本次代码改动不要同时：

### 不做 1

Action Space 改成：

```text
p_i→j
```

### 不做 2

Greedy decoder 替换现有 sampler。

### 不做 3

BFS candidate filtering。

### 不做 4

Request-level Attention Pool。

### 不做 5

Edge Self-Attention。

### 不做 6

修改 Service Algorithm。

### 不做 7

修改 Key Generation physics / generation amount。

这样才能保持：

\[
\boxed{
\text{单变量实验}
}
\]

---

# 39. 本次实验真正要回答的问题

本次 Experiment 1 只回答：

> **当保持现有 Action Space、Sampler、Service Algorithm、Key Generation physics、BC checkpoint 全部不变时，仅让每条候选弧根据当前 Demand context 获得一个 residual score，是否能够提高 SuccessRate？**

也就是：

\[
\boxed{
BaseScore
\quad vs \quad
BaseScore+\Delta Score_{Demand}
}
\]

这是最干净的问题。

---

# 40. 对现有 3 层 GNN 的态度

不改变层数。

仍然：

```text
3-layer GNN
```

原因：

- 现有模型已经可以产生节点/边 embedding；
- 当前实验需要控制变量；
- 不能同时改变 GNN depth 和 Attention。

---

# 41. 对现有 Demand Edge 16 维特征的态度

全部保留：

```text
pending_amount
pending_count
min_deadline_left
mean_deadline_left
mean_wait_time
priority_sum
wait_bucket_amounts[10]
```

尤其：

```text
min_deadline_left
wait_bucket_amounts
```

不得因为 Attention 而删除。

---

# 42. 对没有 Demand 的情况

如果：

```text
num_demands == 0
```

则：

```text
delta_score = 0
```

不要：

```text
score = 0
```

正确：

\[
\boxed{
score=base\_score
}
\]

这样：

> 没有当前请求时，模型仍然可以按照原策略进行 proactive Key Generation。

---

# 43. 对多个 Demand 的情况

如果：

```text
num_demands = N
```

Cross-Attention：

```text
arc query
→
N demand tokens
```

每个 arc 会根据所有当前 Demand 自动获得：

```text
demand-aware context
```

不再手工：

```text
for each demand:
    score
sum score
```

因为 Attention 本身已经提供跨 Demand 的条件化融合。

---

# 44. 为什么这比简单“逐需求 score 再 sum”更好

简单：

\[
s_a=\sum_d f(q_d,h_a)
\]

默认不同 Demand 的边际效应可以独立相加。

但是实际系统中：

> 不同 Demand 对同一条生成边的需求可能存在共享关系。

Attention 可以学习：

\[
f(h_a,\{q_d\})
\]

而不是：

\[
\sum_df(h_a,q_d)
\]

所以：

> **它更符合“一条边同时服务多个需求”的场景。**

---

# 45. 但不要把这个解释成“未来需求预测器”

本模块不是：

```text
FutureDemandPredictor
```

它只做：

> 当前状态下 Demand-conditioned Edge Value Modeling。

未来价值通过：

```text
PPO temporal credit assignment
```

学习。

---

# 46. 最终模型公式

对每个候选有向弧：

\[
a=(u,v,e)
\]

先：

\[
p_a=
[
h_u,h_v,h_e
]
\]

旧分数：

\[
s_a^{base}=MLP_{old}(p_a)
\]

Arc Context：

\[
h_a^{ctx}=Proj(p_a)
\]

Demand token：

\[
q_d=h_d^{Demand}
\]

Cross-Attention：

\[
z_a=
Attention(
Q=h_a^{ctx},
K=\{q_d\},
V=\{q_d\}
)
\]

Residual：

\[
\Delta s_a=MLP_{res}(z_a)
\]

最终：

\[
\boxed{
s_a=s_a^{base}+\Delta s_a
}
\]

其中：

\[
MLP_{res}^{last}=0
\]

初始化时：

\[
\boxed{
s_a=s_a^{base}
}
\]

---

# 47. 最终实现决策表

| 问题 | 最终决定 |
|---|---|
| Q1：新分数替换还是相加 | **相加 / residual** |
| Q1：w 怎么处理 | **不使用可学习 zero gate；固定 residual 系数 1** |
| Q2：弧级还是边级 | **弧级** |
| Q2：为什么 Key 无向还用弧级 | **Key 无向；Action representation 有方向** |
| Q3：z 输入 | **[src node, dst node, edge] → Arc Context** |
| Q3：只用 edge_emb？ | **不，只用 edge_emb 不够稳妥** |
| Q3：Cross-Attn 方向 | **Edge Query；Demand Key/Value** |
| Q4：Request-level pooling | **本次不做** |
| Q4：Demand token | **直接使用现有 Demand Edge embedding** |
| Q5：BFS candidate filtering | **本次不做** |
| Q5：candidate | **全部当前活跃物理弧** |
| Q6：zero init | **保留，Residual 最后一层 zero-init** |
| BC warm start | **完整保留旧 checkpoint** |
| Action Space | **不改** |
| Sequential sampler | **不改** |
| Service Algorithm | **不改** |
| Generation physics | **不改** |
| Edge Self-Attention | **本次不加，后续 ablation** |

---

# 48. 需要实现 AI 特别注意的一个关键点

不要机械照抄旧文档里的：

```text
Demand Query
Edge Key / Value
→ per-edge z
```

因为标准 Cross-Attention 中这会导致：

```text
per-demand output
```

而不是：

```text
per-edge output
```

本次实现必须确保 tensor shape 最终是：

```text
[B, N_arc, D]
```

即：

> 每个 batch 中每条候选弧恰好一个 demand-aware contextual vector。

---

# 49. 最终实施边界

这一次只允许改变：

```text
edge score computation
```

更具体：

```text
base_score
+
demand-conditioned residual_score
```

其他部分全部冻结。

目标：

\[
\boxed{
\text{最小结构改动}
+
\text{严格 BC 兼容}
+
\text{逐位 baseline equivalence}
+
\text{明确的 per-arc demand conditioning}
}
\]

完成后，再根据实验结果决定是否进入：

```text
Decoder ablation
Edge Self-Attention ablation
Request-level Attention ablation
```

而不是一次全部实施。
