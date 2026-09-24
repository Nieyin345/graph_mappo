# QKD-MAPPO 密钥生成调度与 Attention 设计
## 代码事实最终校正版 v2：修正 mutual_choice / Action Decoder 语义

> 本文是当前模型设计的主参考文档。
>
> 本版本在 `docs/research/attention/QKD_MAPPO_Attention_Design_CodeVerified.md` 基础上，进一步吸收最新的 `互选诊断报告.md`。
>
> **最高优先级原则：**
> - 以实际代码行为为准；
> - 区分 Action Representation、Sampling、Resolver 三个层次；
> - 不把 `mutual_choice` 错误描述为“两个 Agent 协调”；
> - 不把当前全局 sequential sampler 错误描述成“各节点独立决策”；
> - 如果要解决当前真正存在的“先到先得/组合抢占”问题，优先研究 Decoder，而不是重新定义 Action Space。

---

# 1. 当前任务

任务是：

> 在 90 节点时变 FSO-QKD 网络中，利用强化学习选择当前时隙应该激活哪些可用 FSO 边生成 Key，以最大化请求 Success Rate。

网络：

- 30 个 BS / GS
- 30 个 HAP
- 30 个卫星
- 所有链路均为 FSO
- 链路可见性、速率和拓扑随卫星运动、云层等动态变化

最终优化目标：

\[
SuccessRate=
\frac{\text{成功服务请求数}}
{\text{到达请求总数}}
\]

---

# 2. 服务算法与 RL 的职责

## 2.1 RL 负责

RL 负责：

> **当前时隙激活哪些物理 FSO 边进行 Key Generation。**

注意：

RL 当前**不直接决定生成多少 Key**。

实际代码中：

\[
Action_e \in \{0,1\}
\]

只决定边是否激活。

生成量由环境依据：

- 当前边速率；
- 时隙长度；
- Key Pool 剩余容量；

自动计算。

---

## 2.2 固定服务算法负责

固定服务算法负责：

- 请求到达；
- 请求排队；
- 请求服务顺序；
- 请求过期；
- 服务路径；
- Key 充足性检查；
- Key 消耗。

但是：

> RL 改变 Key Inventory 后，会间接改变固定服务算法能否服务，以及 fallback BFS 能找到什么路径。

因此系统存在：

\[
\boxed{
Generation
\rightarrow
Inventory
\rightarrow
Service Feasibility
\rightarrow
SuccessRate
}
\]

---

# 3. ★ 当前代码时序：先生成，再服务

当前 `env.step()` 的真实顺序：

```text
Request Arrival
      ↓
MAPPO Action Sampling / Resolution
      ↓
Key Generation
      ↓
Generated Key Allocation
      ↓
Fixed Service
      ↓
Request Expiration
      ↓
Key Expiration
```

即：

\[
\boxed{
Generate_t \rightarrow Serve_t
}
\]

而不是：

\[
Serve_t \rightarrow Generate_t
\]

因此：

> **本时隙刚生成的 Key 可以在本时隙直接服务。**

这意味着当前 RL 任务不是纯粹的 Future Key Storage。

更准确地说：

\[
\boxed{
\text{当前需求 + 当前库存 + 当前拓扑}
\rightarrow
\text{当前 KeyGen}
\rightarrow
\text{当前/未来 Success}
}
\]

---

# 4. Generation Graph 与 Service Graph

必须严格区分两类图。

## 4.1 Generation Graph

\[
G_{gen}(t)
\]

表示：

> 当前时隙可以实际生成 Key 的物理 FSO 边。

只有当前可生成边才是 Actor 的候选 Action。

不可见但历史有库存的边：

> **不能重新作为 KeyGen candidate。**

---

## 4.2 Service Graph

\[
G_{serve}(t)
\]

表示：

> 当前服务阶段可以利用的正库存 Key 边。

其中可以包含：

- 当前可见并有 Key 的边；
- 当前不可见但仍有历史库存的边。

因此：

\[
G_{gen}(t) \neq G_{serve}(t)
\]

这是代码事实，而不是模型假设。

---

# 5. 固定服务路径机制

当前服务不是单一固定路径。

## 5.1 首选静态最短路

若静态 shortest path 上所有 hop 都有正库存：

\[
\forall e\in P_{shortest}, K_e>0
\]

则直接使用该 shortest path。

## 5.2 否则执行正库存图 BFS

如果最短路不可用：

> 在正库存边构成的服务图上 BFS，寻找任意正库存可行路径。

因此：

\[
P_{service}=
\begin{cases}
P_{shortest}, & \text{若 shortest path 可用}\\
BFS(G_{serve}), & \text{否则}
\end{cases}
\]

---

# 6. ★ 当前 Action 的真实语义：不是“每个节点独立提案”

这是本版本最重要的修正。

此前文档容易产生如下错误理解：

```text
Agent i 独立选择 j
Agent j 独立选择 i
↓
mutual_choice
↓
边成立
```

**当前代码不是这样。**

实际流程是：

```text
Global candidate arc scores
        ↓
Sequential sampler
        ↓
先选择一组弧 pairs
        ↓
再把这些 pairs 翻译成各节点的
(tx_target, rx_source)
        ↓
env/action_resolver
        ↓
mutual_choice consistency check
```

---

# 7. 当前 Sampling 的真实流程

当前采样器：

1. 对候选弧计算 score；
2. 注入 Gumbel noise；
3. 找当前合法弧；
4. 选择一条弧；
5. 占用该弧两端的 Tx/Rx 资源；
6. 更新剩余合法弧集合；
7. 继续选择；
8. 直到 STOP 或不存在合法弧。

因此它是：

\[
\boxed{
\text{Sequential Gumbel Sampling}
+
\text{Resource Masking}
}
\]

而不是简单的一次性 top-k。

---

# 8. `_actions_from_index_pairs` 的关键事实

采样器已经决定：

\[
pairs=\{(u,v),...\}
\]

随后代码执行：

```python
tx_of[u] = v
rx_of[v] = u
```

所以：

> **同一条被采样的弧同时决定发送端 Tx 与接收端 Rx。**

因此：

```text
selected pair (u,v)
    ↓
u 的 Tx = v
v 的 Rx = u
```

不是：

```text
u 自己选 v
+
v 自己选 u
```

---

# 9. ★ mutual_choice 不是协调机制

Action Resolver 中的 `mutual_choice` 检查：

\[
tx_u=v
\quad\land\quad
rx_v=u
\]

在当前实现下几乎是一个：

> **对已经选定的 pair 进行一致性验证。**

因为：

\[
tx_u=v,\quad rx_v=u
\]

本来就是从同一个 `(u,v)` pair 写进去的。

因此：

> **mutual_choice 并不承担“两端独立 Agent 协调”的功能。**

最新诊断报告的实测结果也发现：

- Tx proposals 与最终 activated edges 一致；
- 所谓“双向主动率”是 0；
- 这不是发现了严重失配，而是因为该统计口径假设了一个当前实现中并不存在的“两边独立提案”过程。

---

# 10. 修正：不能再用“双方互选概率相乘”解释当前模型

以前可能写成：

\[
P(edge_{ij})
\approx
P_i(j)P_j(i)
\]

并据此解释：

> 双方偏好不同可能导致边丢失。

对于当前实现，这个解释不成立。

因为当前并不存在：

\[
i \rightarrow j
\]

和：

\[
j\rightarrow i
\]

两个独立随机决定。

当前是：

\[
\boxed{
\text{一个全局 sampler 先选择 pair}
}
\]

然后才生成：

```text
Tx/Rx action representation
```

---

# 11. 当前真正存在的问题：Sequential Greedy-like Competition

虽然 mutual_choice 不是问题，但是：

> **顺序采样本身存在组合选择问题。**

例如候选弧：

```text
e1 score = 10
e2 score = 9
e3 score = 8
```

假设：

```text
e1 与 e2 冲突
e1 与 e3 不冲突
e2 与 e3 不冲突
```

如果：

```text
e1
```

首先被采样：

```text
e2 被占用资源阻断
```

那么最终：

```text
{e1,e3}
```

可能产生：

\[
10+8=18
\]

但：

```text
{e2,e3}
```

可能产生：

\[
9+8=17
\]

这只是简单例子。

真正危险的是：

```text
高分 e1
```

虽然自身分数最高，但它可能抢占关键资源，使：

```text
多个中高分边
```

无法同时形成。

于是：

\[
\sum local\ scores
\]

并不一定等于：

\[
\text{best feasible set value}
\]

因此真正存在的竞争是：

\[
\boxed{
\text{先到先得的资源抢占}
}
\]

而不是：

\[
\boxed{
\text{双方互选失败}
}
\]

---

# 12. ★ 因此“要不要换 Action Space”需要重新回答

结论：

> **不能用 mutual_choice coordination 作为换 Action Space 的理由。**

原因是：

- 没有独立双边提案；
- 没有两边概率乘积；
- mutual_choice 基本只是 consistency check；
- 最新诊断也证明了此前的 mutual mismatch 指标不成立。

---

# 13. 真正应该考虑的是：要不要改 Decoder

当前问题是：

\[
\text{Edge Scores}
\rightarrow
\text{Sequential Sampling}
\rightarrow
\text{Final Edge Set}
\]

而可以研究：

\[
\text{Edge Scores}
\rightarrow
\text{Improved Decoder}
\rightarrow
\text{Final Edge Set}
\]

因此：

> **如果要优化当前 Action 选择过程，优先改 Decoder，而不是重定义 Action Space。**

---

# 14. 当前 Action Representation 仍然可以保留

当前：

```text
(tx_target, rx_source)
```

可以作为最终环境需要的 action representation。

同时内部也可以：

```text
edge / arc score
```

作为 decoder 的中间表示。

因此不需要因为要研究更好的组合选择，就立刻全面改成：

```text
node proposal
→ edge aggregation
```

---

# 15. Edge-level Proposal 的原始理由现在被削弱

此前方案：

\[
p_{i\rightarrow j}
\]

和：

\[
p_{j\rightarrow i}
\]

再：

\[
S_{ij}=MLP([p_{ij},p_{ji},h_i,h_j,h_{ij}])
\]

主要理由是：

> 让两端共同决定一条边，消除 mutual-choice coordination loss。

这个理由现在不成立。

如果以后仍想研究 edge-level scoring，需要换成另一个明确理由，例如：

> 用无向 edge representation 直接建模一条物理 Key link 的整体价值。

这属于：

\[
\text{better scoring representation}
\]

而不是：

\[
\text{coordination correction}
\]

---

# 16. 新的优先级判断

目前结构改动建议重新排序：

## Priority 1

\[
\boxed{
Demand
\rightarrow
Edge\ Cross\ Attention
\rightarrow
per-edge\ score
}
\]

理由：

- 当前真正有待增强的是“需求 ↔ 具体生成边”的关系；
- 不需要改变整个 Action semantics；
- 可以保留现有 BC warm start；
- 可以保留当前 sampler；
- 因此实验隔离性最好。

---

## Priority 2

加入 route-aware explicit features：

```text
on-shortest-path
route position
distance to source
distance to destination
path relevance
inventory relevance
```

这些由固定服务逻辑/图算法计算。

模型不需要重新发明图论事实。

---

## Priority 3

研究 Decoder：

```text
current sequential sampler
vs
alternative constrained decoder
```

例如：

```text
Greedy decoder
Beam-like decoder
Exact solver for correctly-formulated optimization
```

但必须先严格定义当前 Tx/Rx 资源约束对应的组合优化问题。

---

## Priority 4

Edge Self-Attention。

仅当需要验证：

> 多候选边之间的联合价值建模

时做。

不再声称它负责学习“路径连接”。

---

## Priority 5

完全更换 Action Space。

除非代码层面确认：

> 当前 action representation 本身造成了研究目标所需要的表达限制。

目前没有这个证据。

---

# 17. 关于“全局最优 matching”的再次修正

当前资源约束并非普通无向 matching：

\[
deg(v)\le1
\]

因为一个节点可以同时：

- 一个 Tx-out；
- 一个 Rx-in。

因此：

\[
outdeg(v)\le1
\]

\[
indeg(v)\le1
\]

同时存在。

所以：

> `networkx.max_weight_matching` 不能未经重新建模，就作为当前 Action Space 的“全局最优解”。

如果要研究：

```text
sequential sampler
vs
greedy
vs
exact
```

应该先明确真实组合优化目标与可行域。

---

# 18. 一个更合理的 Decoder 研究问题

令候选有向弧集合：

\[
E_t^{cand}
\]

每条弧有模型评分：

\[
s_e
\]

要求：

\[
\sum_{e:source(e)=v}x_e\le1
\]

\[
\sum_{e:destination(e)=v}x_e\le1
\]

\[
x_e\in\{0,1\}
\]

并考虑同一无向节点对的方向约束。

那么 decoder 实际解决的是：

\[
\max_x
\sum_e s_e x_e
\]

subject to above constraints.

这才是当前：

> “高分弧先选是否造成组合损失”

的正确数学抽象。

---

# 19. 重要区分：训练时 Sampling 与推理时 Decoder

这里应该区分：

## Training

需要：

\[
\log\pi_\theta(a|s)
\]

所以如果替换 decoder，需要重新考虑：

- action distribution；
- stochasticity；
- log_prob；
- PPO ratio；
- entropy；
- BC initialization。

## Evaluation / Inference

可以使用：

```text
deterministic decoder
```

例如：

```text
argmax / greedy / constrained selection
```

二者不能混写。

---

# 20. BC Warm Start 的影响更加重要

当前项目已经发现：

```text
BC warm start
std ≈ 0.0137
```

而去除 warm start：

```text
std ≈ 0.2
```

所以任何 Action/Decoder 重构都必须考虑：

> 如何把原 expert action 映射到新 policy 的 action representation。

因此：

### 不建议同时改

```text
Demand Encoder
+
Attention
+
Action Space
+
Decoder
```

否则实验结果很难解释，而且可能直接丢失 BC 的稳定优势。

---

# 21. Demand Attention 设计仍然保持原结论

请求级 Attention Pool 可以加入：

\[
q_{attn}=AttentionPool(\{r_i\})
\]

但不能删除：

```text
min_deadline_left
pending_amount
pending_count
mean_deadline_left
mean_wait_time
priority_sum
wait_bucket_amounts
```

尤其：

\[
\boxed{min\_deadline\_left}
\]

必须显式保留。

---

# 22. Edge Self-Attention 的最终定位

当前不再说：

> “Self-Attention 用于学习 e1-e2-e3 能否串起来。”

正确说法：

> **Self-Attention 用于验证候选 KeyGen edges 的联合上下文是否能改善边级价值估计。**

即：

\[
\{h_{e_1},h_{e_2},...\}
\rightarrow
\{h'_{e_1},h'_{e_2},...\}
\]

再：

\[
h'_e
\rightarrow
MLP
\rightarrow
s_e
\]

仍然不使用 QK attention weight 直接作为最终 Action score。

---

# 23. Candidate Edge 的上下文来源

针对 Demand Edge \(d\)：

可以建立：

```text
candidate current-generation edges
        +
route / service relevance features
        +
current inventory context
        +
demand context
```

需要注意：

> “库存相关性”不意味着把不可生成的历史库存边重新变成 Generation Action。

它只表示这些库存边可能影响固定服务。

---

# 24. 当前模型推荐的最终主结构

```text
                    Dynamic FSO State
                           │
              ┌────────────┴────────────┐
              │                         │
       Physical Graph              Demand Graph
              │                         │
              └────────────┬────────────┘
                           │
                      3-layer GNN
                           │
             ┌─────────────┴─────────────┐
             │                           │
      Node Embeddings              Edge Embeddings
             │                           │
             │                    Generation Candidates
             │                           │
             └──────────────┐    ┌───────┘
                            │    │
                       Demand Encoder
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Request Attention          Explicit statistics
              │                           │
              └─────────────┬─────────────┘
                            │
                     Demand Embedding
                            │
                            ▼
                  Demand → Edge
                  Cross-Attention
                            │
                            ▼
                     per-edge z_e
                            │
                            ▼
                        Edge MLP
                            │
                            ▼
                         score_e
                            │
                            ▼
              Existing Sequential Sampler
                            │
                            ▼
                    Selected Arc Set
                            │
                            ▼
                    Key Generation
                            │
                            ▼
                    Key Inventory
                            │
                            ▼
                 Fixed Service Algorithm
                            │
                            ▼
                       Success
```

---

# 25. 当前模型的正确理论叙述

不应该说：

> “MAPPO 学习固定服务路径。”

也不应该说：

> “MAPPO 预测未来最短路。”

更准确的是：

\[
\boxed{
MAPPO\ 学习在当前时隙选择哪些物理 FSO 边生成 Key，
使当前及后续固定服务机制具有更高的服务成功率。
}
\]

而服务机制又通过 Key Inventory 影响：

```text
shortest-path usability
+
fallback BFS connectivity
```

因此模型是在：

\[
\boxed{
\text{主动塑造 Key Inventory topology}
}
\]

而不是直接控制 routing。

---

# 26. 当前最值得研究的核心问题

整个模型可以进一步抽象成：

\[
s_t
\rightarrow
\text{Demand-aware edge value}
\rightarrow
\text{constrained edge selection}
\rightarrow
K_{t+1}
\rightarrow
\text{service}
\rightarrow
R_t
\]

其中最值得增强的是：

\[
\boxed{
s_t \rightarrow s_e
}
\]

即：

> **全局需求状态如何转化为每条具体物理生成边的价值。**

因此 Demand→Edge Cross-Attention 是最自然的第一步。

---

# 27. 下一阶段实验顺序

## Experiment 0 — Current baseline

保持：

- 当前 GNN；
- 当前 Demand features；
- 当前 Gumbel sequential sampler；
- 当前 resolver；
- BC warm start。

---

## Experiment 1 — Demand → Edge Cross-Attention

只改变：

```text
Demand representation
→
Cross-Attention
→
per-edge representation
→
MLP score
```

不改 Action Space。

---

## Experiment 2 — Route-aware features

增加：

```text
shortest-path relation
path position
destination distance
source distance
service relevance
inventory relevance
```

---

## Experiment 3 — Decoder ablation

比较：

```text
Current sequential sampler
vs
alternative constrained decoder
```

这一实验回答：

> “当前真正存在的顺序抢占是否损害最终 edge set 质量？”

---

## Experiment 4 — Edge Self-Attention

只回答：

> “候选边联合上下文是否提供 Cross-Attention 之外的额外信息？”

---

## Experiment 5 — Action Space redesign

只有在前面实验或代码需求证明：

> 当前 Action representation 存在表达上的硬限制

之后再研究。

---

# 28. 必须删除/修正的旧说法

以下内容以后不要再作为当前代码事实：

### 错误

> 两个节点各自独立选择对方。

### 正确

> 全局 sampler 先选择弧，再将弧映射为两端 Tx/Rx。

---

### 错误

> mutual_choice 负责解决多 Agent 协调。

### 正确

> mutual_choice 当前主要承担被采样弧对应 Tx/Rx 的一致性验证。

---

### 错误

> mutual_choice 可能导致双方偏好不一致而丢边。

### 正确

> 当前实现不存在该独立提案失配机制。

---

### 错误

> 当前主要瓶颈是 mutual-choice。

### 正确

> 当前更值得诊断的是 sequential sampler 的先到先得、资源抢占和组合选择质量。

---

### 错误

> edge-level action 可以通过消除双方协调来提高性能。

### 正确

> 如果采用 edge-level action，其价值必须来自更好的 edge value representation 或更合适的 constrained decoding，而不能以“修复 mutual-choice coordination”为主要理由。

---

# 29. 最终收敛结论

当前项目最合理的研究路线是：

\[
\boxed{
\text{先不改 Action Space}
}
\]

先做：

\[
\boxed{
Demand
\rightarrow
Edge\ Cross\ Attention
\rightarrow
Per-edge\ Score
}
\]

再验证：

\[
\boxed{
Sequential\ Decoder
}
\]

是否存在明显组合损失。

最后才考虑：

\[
\boxed{
Edge\ Self\ Attention
}
\]

和：

\[
\boxed{
Action\ Space\ Redesign
}
\]

因为最新代码事实已经证明：

> **真正存在的问题不是“两端互选失败”，而是“已经得到的边分数如何转化成高质量的可行边集合”。**

因此后续研究的核心顺序应该是：

\[
\boxed{
\text{Demand-Edge Value Modeling}
\rightarrow
\text{Constrained Decoding}
\rightarrow
\text{Action Redesign（必要时）}
}
\]

而不是：

\[
\text{Mutual-choice}
\rightarrow
\text{Edge Proposal}
\]

