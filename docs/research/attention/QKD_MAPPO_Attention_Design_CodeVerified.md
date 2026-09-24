# QKD-MAPPO 密钥生成调度与 Attention 设计（代码事实校正版）

> 用途：本文件作为项目当前模型设计的主参考文档，可直接供其他 AI 会话阅读、讨论和后续实现。
>
> **重要原则：**
> 1. 本文优先采用已经完成的代码事实核实结果，而不是此前的背景假设。
> 2. “代码现状”和“推荐设计”严格分开，避免把提案误写成现有实现。
> 3. 不为了增加模型复杂度而增加 Attention；每一个结构都必须对应明确的问题。
> 4. BC 暖启动属于实验稳定性的硬约束，除非另有专门实验，不应取消。

---

# 1. 任务定义

## 1.1 网络

当前任务为时变 FSO-QKD 网络中的密钥生成调度：

- 90 个节点：
  - 30 个 BS / GS
  - 30 个 HAP
  - 30 个卫星
- 所有通信链路均为 FSO。
- 卫星运动、可见性、云层遮挡、链路速率等使网络状态随时隙变化。
- 每个时隙重新确定当前可以进行 Key Generation 的物理边。

---

# 2. 当前代码中真实的职责分工

## 2.1 MAPPO 负责什么

MAPPO 负责：

> 每个时隙选择哪些当前可用的 FSO 边进行 Key Generation。

注意：

**模型不是直接输出“生成多少 Key”。**

当前代码中：

- Action 决定边是否被激活；
- 生成量由环境根据该边当前时隙的物理生成速率、时隙长度和 Key Pool 剩余容量计算。

因此当前 Action 的本质是：

\[
a_e \in \{0,1\}
\]

表示：

- `1`：本时隙激活这条边；
- `0`：本时隙不激活这条边。

## 2.2 固定服务算法负责什么

固定服务算法负责：

- 请求到达；
- 请求排队；
- FIFO / deadline 优先的服务顺序；
- 请求过期；
- 服务路径选择；
- 检查路径上的 Key；
- 沿服务路径消耗 Key。

MAPPO 不直接执行请求路由，也不直接决定某个请求采用哪条路径。

但是必须特别注意：

> **MAPPO 通过改变 Key Inventory，会间接改变固定服务算法能够找到的服务路径以及服务可行性。**

这是当前问题的核心耦合关系。

---

# 3. ★ 代码事实：一个时隙内部的真实时序

代码当前 `env.step()` 的实际顺序为：

```text
Request Arrival
      ↓
MAPPO Action Resolution
      ↓
Key Generation
      ↓
Generated Key Allocation
      ↓
Service
      ↓
Request Expiration
      ↓
Key Expiration
```

即：

\[
\boxed{
\text{Generate}_t \rightarrow \text{Serve}_t
}
\]

而不是此前某些设计中假设的：

\[
\text{Serve}_t \rightarrow \text{Generate}_t
\]

## 3.1 直接含义

本时隙刚生成的 Key：

> **本时隙就可以被服务使用。**

所以当前任务不仅是“提前为未来储备 Key”，还包括：

\[
\text{当前需求}
\rightarrow
\text{当前 Key Generation}
\rightarrow
\text{当前 Service Success}
\]

当然，未被当前请求消耗的 Key 仍然可以留在 Key Pool 中，供未来使用。

因此更准确的问题定义不是单纯的 Future Demand Prediction，而是：

\[
\boxed{
\text{当前状态下激活哪些边，能够最大化当前及未来的服务成功率}
}
\]

---

# 4. 服务路径的真实机制

当前固定服务算法不是一个“路径永远完全固定”的单一路径规则，而是两级机制。

## 4.1 第一优先：静态最短路

对于请求 `(src, dst)`：

1. 先取得静态预计算的 shortest path；
2. 如果 shortest path 上所有 hop 都存在正 Key Inventory，则使用该路径。

即：

\[
P = \text{ShortestPath}(src,dst)
\]

若：

\[
\forall e\in P,\quad K_e>0
\]

则使用该路径。

## 4.2 第二优先：正库存图上的 BFS

如果最短路径无法使用，则在：

> **当前存在正 Key Inventory 的边构成的服务邻接图**

上执行 BFS，寻找任意可用路径。

因此：

\[
\text{ServicePath}
=
\begin{cases}
\text{StaticShortestPath}, & \text{若所有 hop 有 Key}\\
\text{BFS on positive-inventory graph}, & \text{否则}
\end{cases}
\]

---

# 5. ★ Generation Graph 与 Service Graph 必须分离

这是当前设计必须明确的概念。

## 5.1 Generation Graph

定义：

\[
G_{\text{gen}}(t)
\]

表示：

> 当前时隙实际能够进行 Key Generation 的 FSO 物理边。

因此只有当前物理可用的边才能进入 Generation Candidate Set。

当前不可见的边，即使历史上有 Key，也不能因为库存存在而重新成为 Key Generation 候选。

---

## 5.2 Service Graph

定义：

\[
G_{\text{serve}}(t)
\]

表示：

> 当前服务阶段可以利用的有 Key 资源边。

它不是简单等于 `G_gen(t)`。

因为历史库存可能存在于：

- 当前不可见；
- 当前不能生成；
- 但仍有 `K_e > 0`

的边上。

这些边：

> **可以继续被服务路径使用。**

因此：

```text
Generation Candidate:
    current physical active edges only

Service-Relevant Inventory:
    positive-Key edges
    including invisible historical inventory edges
```

---

# 6. 当前 Action Space 的真实定义

代码中的每个节点动作：

```text
(tx_target, rx_source)
```

其中：

```text
tx_target ∈ neighbors ∪ {IDLE}
rx_source ∈ neighbors ∪ {IDLE}
```

节点：

- 最多选择一个 Tx-out；
- 最多选择一个 Rx-in；
- 因此同一个节点理论上可以同时参与两条边：
  - 一条作为发送端；
  - 一条作为接收端。

所以这不是标准的：

\[
\deg(v)\le1
\]

matching 约束。

更准确地说，是：

\[
outdeg(v)\le1
\]

以及：

\[
indeg(v)\le1
\]

同时成立。

因此一个节点最多参与 2 条有向激活关系。

---

# 7. 当前 Action Resolver 的 mutual-choice 机制

实际边激活要求：

```text
u 选择 v 作为 Tx
AND
v 选择 u 作为 Rx
```

即：

\[
u\rightarrow v
\]

生效必须同时满足：

\[
tx_u=v,\qquad rx_v=u
\]

之后同一无向节点对只保留一个方向代表。

当前代码在同一对方向冲突时：

> **按字典序打破平局，而不是比较 Actor 分数。**

这一点应被视为当前实现的一个独立事实。

---

# 8. 当前 PPO Action Sampling 的真实机制

当前策略并不是简单：

> 每个节点独立 softmax 一次，然后 top-k。

实际是：

```text
Arc Scores
    ↓
Gumbel noise
    ↓
Sequential sampling
    ↓
每选一条弧，更新剩余可用 Tx/Rx 端口
    ↓
继续选
    ↓
STOP / 无可用候选
```

因此它已经具有：

> **顺序、无放回、带资源占用约束的 Action Sampling**

而不是完全独立的 node-wise decision。

PPO 的 log_prob 由这些顺序决策累计，并按决策数进行归一化。

因此：

> **不能再简单地把当前模型描述成“90 个节点各自独立选边，然后靠 mutual-choice 被动拼接”。**

---

# 9. Edge Generation Capacity 的真实定义

当前代码没有独立的：

> “每节点每时隙最多生成几条边”

这种硬上限。

Key Pool capacity 是：

\[
C_e
\]

即：

> **每条 Key edge 自己有容量。**

当前生成量大致由：

\[
G_e(t)
=
\min(
R_e(t)\Delta t,\;
C_e-K_e(t)
)
\]

决定。

因此：

- Action 决定是否激活；
- 物理速率决定生成量；
- Key Pool 剩余容量决定最终能够装入多少。

后续模型设计不要再定义一个不存在的 `per-node generation budget`，除非项目未来明确添加这个约束。

---

# 10. Demand Edge 的作用

对于存在排队请求的 BS 对 `(i,j)`，构造逻辑 Demand Edge。

其作用是：

> 告诉 Key Generation Policy 当前业务压力集中在哪里。

Demand Edge 不是物理链路。

因此：

- 不能作为 Key Generation Action；
- 不能直接当成 FSO 边；
- 不能改变物理图的可见性。

但它可以作为 Heterogeneous GNN 的独立 relation。

---

# 11. 现有 Demand Features

当前已经存在：

```text
pending_amount
pending_count
min_deadline_left
mean_deadline_left
mean_wait_time
priority_sum
wait_bucket_amounts[10]
```

这些信息应保留。

特别是：

\[
\boxed{min\_deadline\_left}
\]

因为当前请求的 deadline 与 arrival time 有固定关系，而服务排序与 deadline / arrival 直接相关。

所以：

> “最急请求还剩多久”是具有明确服务语义的信号。

不要为了增加 Attention 而删除或隐去这些显式特征。

---

# 12. Attention 设计的核心目标必须重新定义

Attention 不应该承担：

> “让模型自己发现哪些物理边能够组成路径。”

因为：

- 当前图拓扑是已知的；
- shortest path / BFS 等结构事实可以由程序直接计算；
- 3 层 GNN 已经提供了局部拓扑传播。

真正值得学习的是：

\[
\boxed{
\text{某条 Key Generation Edge 对当前及未来 Service Success 的边际价值}
}
\]

尤其当前 Action 会先生成 Key，而 Service 又依据 Key Inventory 决定可服务路径，因此存在：

\[
\text{Generation}
\rightarrow
\text{Inventory}
\rightarrow
\text{Service Feasibility}
\]

的闭环。

---

# 13. Edge Self-Attention：最终结论

## 13.1 不应把它解释成“学习边连接关系”

不应该使用以下作为主要设计理由：

> “Self-Attention 让模型学习 e1-e2-e3-e4 可以串起来。”

因为这种连接关系本身属于图结构事实。

---

## 13.2 它唯一真正值得验证的价值

Edge Self-Attention 更合理的解释是：

> **学习多个候选生成边之间的联合价值/协同价值。**

例如：

```text
e1：路径前段
e2：中间瓶颈
e3：中间瓶颈
e4：路径后段
```

单独生成 `e2` 可能帮助有限；

单独生成 `e3` 也可能帮助有限；

但：

```text
e2 + e3
```

共同生成后，可能突然使某条服务路径变得可用。

因此理论上存在：

\[
V(e_2,e_3)>V(e_2)+V(e_3)
\]

Self-Attention 可以尝试建模这种候选边之间的联合上下文。

---

## 13.3 但它仍然不是主模型的必需组件

原因：

1. 3 层 GNN 已经做局部消息传播；
2. 路径连接关系属于可计算结构事实；
3. Self-Attention 会增加复杂度；
4. 当前实验统计功效较弱。

因此主模型建议：

```text
GNN
 ↓
Demand Encoder
 ↓
Demand → Edge Cross-Attention
 ↓
per-edge embedding
 ↓
Edge MLP
 ↓
Score
```

Edge Self-Attention：

> **作为独立 ablation，而不是默认主结构。**

---

# 14. Demand Attention Pool：最终结论

同一 BS 对可能对应多个请求：

\[
Q_{ij}=\{r_1,r_2,\ldots,r_n\}
\]

可以使用 request-level Attention Pool，得到：

\[
q_{attn}
\]

但：

> **不能让 Attention Pool 替代显式 urgency statistics。**

推荐：

\[
q_{ij}
=
MLP(
[q_{attn},q_{explicit}]
)
\]

其中：

```text
q_attn:
    request-level learned representation

q_explicit:
    min_deadline_left
    pending_amount
    pending_count
    mean_deadline_left
    mean_wait_time
    priority_sum
    wait_bucket_amounts
```

尤其：

```text
min_deadline_left
```

必须直接保留。

---

# 15. 为什么不能只依赖 Attention Pool

假设：

```text
r1: deadline_left = 5
r2: deadline_left = 20
r3: deadline_left = 30
```

Attention 理论上可以学到 r1 更重要，但不能保证训练过程中永远保留这一统计语义。

如果模型更偏重 demand amount：

```text
r2 + r3
```

最终可能在 pooled representation 中压过：

```text
r1
```

这对 FIFO / deadline-driven service 不利。

因此：

> Attention 是增量表示，不应成为服务紧迫性的唯一载体。

---

# 16. Demand → Edge Cross-Attention 的推荐结构

推荐：

```text
Demand representation
        │
        │ Query
        ▼
Candidate Edge representations
        │
        │ Key / Value
        ▼
Per-edge contextual representation
        │
        ▼
MLP
        │
        ▼
score_e
```

注意：

> Cross-Attention 的目标不是直接使用 QK attention weight 作为 Action Score。

正确流程是：

\[
z_e
\rightarrow
z'_e
\rightarrow
MLP(z'_e)
\rightarrow
s_e
\]

即：

> **向量 → Edge Score → Action**

而不是：

> **QK attention weight → Action**

---

# 17. Candidate Edge 的构造原则

必须区分：

### 当前可生成边

来自：

\[
G_{\text{gen}}(t)
\]

即当前物理可用 FSO edge。

这些才可以成为 Action Candidate。

### 历史库存边

如果：

```text
visibility = 0
K_e > 0
```

它：

- 不能成为新的 Key Generation Action；
- 但可以成为当前 Service Graph 的资源；
- 可以作为 Demand-to-Edge 结构关联的 context；
- 可以影响 Critic。

因此不能简单把：

```text
current physical graph
+
all inventory edges
```

重新拼成一个新的“生成物理图”。

---

# 18. BFS 在这个模型中的正确角色

BFS 不应被解释成：

> “RL 预测未来路径的算法。”

也不能让 BFS 取代 RL。

BFS 更适合承担：

### 结构候选生成 / route-related feature extraction

例如针对某个 Demand Edge：

```text
BS_i → BS_j
```

可以根据固定服务算法计算：

- 当前静态最短路；
- 服务阶段的正库存可行路径；
- 某条候选边是否处于相关路径；
- 路径位置；
- 距源端 hop；
- 距目的端 hop；
- 被多少候选路径经过。

这些属于：

> **规则可以直接计算的 route-aware structural features。**

然后交给 RL 学习：

> “生成这条边是否值得”。

---

# 19. 特别注意：当前服务 BFS 与 Generation BFS 不是同一张图

服务阶段的 BFS 使用：

\[
G_{\text{serve}}(t)
\]

即：

> 正库存边图。

而 Key Generation 候选使用：

\[
G_{\text{gen}}(t)
\]

即：

> 当前可物理生成的 FSO 边。

因此未来代码实现中必须显式区分两个 adjacency：

```text
generation_adjacency
service_inventory_adjacency
```

不要混用。

---

# 20. 当前模型推荐的总体结构

建议主模型收敛为：

```text
                         Dynamic Network
                               │
                ┌──────────────┴──────────────┐
                │                             │
         Physical FSO Graph              Demand Graph
                │                             │
                └──────────────┬──────────────┘
                               │
                         3-layer GNN
                               │
               ┌───────────────┴───────────────┐
               │                               │
         Node Embeddings                 Edge Embeddings
               │                               │
               │                         Candidate FSO Edges
               │                               │
               └──────────────┐       ┌────────┘
                              │       │
                        Demand Encoder
                              │
                  ┌───────────┴───────────┐
                  │                       │
             Attention Pool        Explicit Statistics
                  │                       │
                  └───────────┬───────────┘
                              │
                         Demand Embedding
                              │
                              ▼
                     Demand→Edge
                    Cross-Attention
                              │
                              ▼
                    Per-edge z_e
                              │
                              ▼
                         Edge MLP
                              │
                              ▼
                         score_e
                              │
                              ▼
                 Existing Action Sampler
                              │
                    Gumbel sequential
                        selection
                              │
                              ▼
                       Activated Edges
                              │
                              ▼
                      Key Generation
                              │
                              ▼
                      Key Inventory
                              │
                              ▼
                    Fixed Service Algorithm
```

---

# 21. 为什么暂时保留现有 Action Sampler

当前 Action Sampler 已经具备：

- 顺序选择；
- Gumbel exploration；
- 无放回；
- 动态占用 Tx/Rx；
- STOP；
- PPO log_prob 对应明确。

因此目前没有充分证据证明：

```text
existing sequential sampler
```

必须被替换成：

```text
edge proposal
→ edge aggregation
→ greedy decoder
```

新的 Edge Proposal 方案理论上可以解决 mutual-choice 协调问题，但会引入：

- 新的 Action distribution；
- proposal 与 executed action 的一致性问题；
- PPO log_prob 重新定义；
- BC expert action 映射问题；
- 新 decoder；
- 新训练不稳定来源。

在当前实验成本较高的前提下，不应同时做这些变化。

---

# 22. 关于 mutual-choice：应先诊断，再决定是否替换

需要先统计：

```text
mutual acceptance rate
```

以及：

```text
u prefers v
but v does not reciprocate
```

这种失败比例。

如果这类失配很高，才说明：

> mutual-choice 本身可能成为性能瓶颈。

否则不建议为了理论上更优雅的 edge-level action 而直接重构 Action Space。

---

# 23. 关于 max_weight_matching：不要再把它当作天然的“全局最优真值”

当前节点约束是：

\[
outdeg(v)\le1,\qquad indeg(v)\le1
\]

同一节点可以同时参与一条发送和一条接收关系。

所以问题并非标准无向 maximum-weight matching。

因此：

> `networkx.max_weight_matching` 的计算结果不能未经重新定义就作为当前 Action Space 的“全局最优解”。

未来若要比较 exact decoder 与 greedy decoder，应该首先准确形式化当前资源约束下的组合优化问题。

---

# 24. Greedy Decoder 的定位

Greedy：

```text
按 score 从高到低
↓
若满足资源约束则激活
↓
否则跳过
```

仍然是很合理的轻量约束解码器。

它最大的价值是：

> 计算快、规则清晰、容易嵌入 PPO。

但目前不需要因为 `max_weight_matching` 太慢就立刻切换到它。

先把真正需要优化的模型结构确定下来。

---

# 25. Centralized Critic

Critic 应表示整个系统状态：

\[
V_\phi(S_t)
\]

建议输入：

### Physical graph

- node embeddings；
- active FSO edge embeddings；
- link rate / visibility / capacity 等状态。

### Demand graph

- pending amount；
- pending count；
- min deadline；
- waiting statistics；
- priority；
- wait buckets。

### Inventory memory

包括：

```text
当前不可见但仍有库存的边
```

因为这些边可以继续影响服务。

### Agent/resource state

包括节点当前 Tx/Rx 可用状态等。

最后做 permutation-invariant pooling：

\[
z_t=
[g_V,g_{\text{phy}},g_{\text{dem}},g_{\text{inv}},g_{\text{agent}}]
\]

再：

\[
V_\phi(S_t)=MLP(z_t)
\]

---

# 26. Reward 的原则

最终目标：

\[
SuccessRate
=
\frac{\text{Successfully served requests}}
{\text{Arrived requests}}
\]

不要主要奖励：

```text
生成了多少 Key
```

否则模型可能大量生成没有服务价值的 Key。

Key Generation 的价值最终来自：

\[
\text{Key Generation}
\rightarrow
\text{Service Success}
\]

所以：

> Reward 应围绕成功服务建立，Key Generation 只是实现成功服务的手段。

如果后续发现 Credit Assignment 困难，可以做轻量 reward shaping，但不能让“生成量”成为主优化目标。

---

# 27. 预生成不能被禁止

当前即使没有请求：

> MAPPO 仍然允许在可生成边上生成 Key。

原因：

未来请求可能到达。

因此：

```text
Demand = 0
```

不等于：

```text
KeyGen = 0
```

是否生成由长期收益决定。

---

# 28. BC Warm Start：硬约束

已有实验表明：

```text
BC warm start:
std ≈ 0.0137

取消 warm start:
std ≈ 0.2
```

因此当前实验体系中：

> **BC warm start 必须保留。**

新结构必须遵循：

```text
只加不减
+
zero initialization
+
配置开关默认关闭
```

不要因为增加 Attention 而取消原有 expert warm start。

---

# 29. 实验设计：必须分阶段

当前训练成本：

```text
单条臂 ≈ 30 update
≈ 1.5 小时
```

同时：

```text
n=5 seeds
effect ≈ 0.008~0.017
```

历史上很多结构变化都低于这个量级。

因此不能一次同时改：

```text
GNN
+
Demand Encoder
+
Edge Self-Attention
+
Cross-Attention
+
Action Space
+
Decoder
```

否则无法进行因果归因。

---

# 30. 推荐实验顺序

## Experiment 0 — Baseline

保持现有：

```text
GNN
+
existing demand features
+
existing action sampler
+
BC warm start
```

获得稳定基线。

---

## Experiment 1 — Demand → Edge Cross-Attention

只增加：

```text
Demand embedding
→
Candidate Edge
Cross-Attention
→
per-edge z_e
→
MLP score
```

保留：

```text
min_deadline_left
wait statistics
wait_bucket_amounts
```

以及：

```text
existing action sampler
```

这是第一优先级实验。

---

## Experiment 2 — Route-aware explicit features

在 Experiment 1 基础上加入：

```text
on-shortest-path
route position
distance to source
distance to destination
route frequency
inventory-path relevance
```

这些 feature 由固定规则计算，而不是让 Attention 自己学习拓扑事实。

---

## Experiment 3 — Edge Self-Attention Ablation

只在已有版本基础上加入：

```text
Candidate Edge Self-Attention
```

目的仅验证：

> 多边联合上下文是否真的带来额外收益。

如果没有稳定提升，删除。

---

## Experiment 4 — New Edge Proposal Action

只有当诊断发现：

```text
mutual-choice mismatch
```

确实非常严重时，才尝试：

```text
per-edge proposal
→ edge aggregation
→ constrained decoder
```

否则不进入主模型。

---

# 31. 推荐的主模型最终版本

现阶段建议把论文主模型定义为：

\[
\boxed{
\text{3-layer Heterogeneous GNN}
+
\text{Demand Attention Pool}
+
\text{Explicit Urgency Statistics}
+
\text{Demand→Edge Cross-Attention}
+
\text{Per-edge MLP Score}
+
\text{Existing Sequential Action Sampler}
}
\]

其中：

### 保留

- 3-layer GNN；
- Demand Edge；
- min_deadline_left；
- wait statistics；
- BC warm start；
- 当前 Gumbel sequential sampler；
- centralized critic。

### 暂不作为主结构

- Edge Self-Attention；
- 新 Edge Proposal Action；
- Greedy replacement decoder；
- Max-weight matching。

---

# 32. 一句话理论解释

当前模型真正解决的问题不是：

> “识别一条路径由哪些边组成。”

而是：

\[
\boxed{
\text{在动态 FSO 网络中，通过 Key Generation 主动改变 Key Inventory，
从而提高当前及未来固定服务算法的可服务性与 SuccessRate}
}
\]

因此：

- 图结构事实交给环境 / GNN；
- 请求紧迫性通过显式 statistics 保留；
- Demand→Edge Cross-Attention 学习“需求对具体生成边的条件化价值”；
- Actor 决定激活哪些边；
- 环境根据物理速率决定生成量；
- 固定 Service Algorithm 根据 Key Inventory 执行服务；
- 最终以 SuccessRate 衡量效果。

---

# 33. 当前需要严格避免的错误描述

后续文档、代码注释和论文中不要再出现以下未经修正的说法：

### 错误 1

> “先服务再生成。”

正确：

> 当前代码是同一时隙先生成，再服务。

### 错误 2

> “模型决定每条边生成多少 Key。”

正确：

> 模型决定激活哪些边，生成量由环境根据边速率与 Key Pool capacity 决定。

### 错误 3

> “历史库存边不能参与服务。”

正确：

> 当前不可见但仍有正库存的历史边可以参与服务路径。

### 错误 4

> “服务路径始终固定。”

正确：

> 静态 shortest path 优先；若不可用，则在正库存边图上 BFS。

### 错误 5

> “每个节点最多参与一条生成边。”

正确：

> 每节点最多一条 Tx-out 和一条 Rx-in，因此最多同时参与两条有向关系。

### 错误 6

> “当前 action 是各节点完全独立采样。”

正确：

> 当前策略是带 Tx/Rx 资源占用约束的 Gumbel sequential sampling。

---

# 34. 最终结论

当前最值得继续研究的结构变化是：

\[
\boxed{
\text{Demand}
\rightarrow
\text{Edge-aware Cross-Attention}
\rightarrow
\text{Per-edge Generation Score}
}
\]

而不是继续增加：

\[
\text{更多 Self-Attention}
\]

也不是首先重构：

\[
\text{Action Space}
\]

核心原因是：

> 当前真正困难的是“需求与具体 Key Generation Edge 的价值关联”，而不是让模型重新学习图论中的基本连通关系。

同时，由于：

\[
\text{Generation}
\rightarrow
\text{Inventory}
\rightarrow
\text{Service Path/Feasibility}
\]

存在闭环，因此 Key Generation 不只是简单的 Future Path Prediction，而是在主动塑造后续服务阶段的可服务资源结构。

这应成为后续模型、实验和论文叙述的主线。
