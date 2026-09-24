# DemandResidual 实现后的修改与验收建议

## 目的

本文件用于交给另一个 AI / 代码 Agent，作为当前 `DemandResidual` 实现后的下一步执行依据。

当前模型主体已经基本符合设计，本阶段**不要继续扩大模型结构**。重点是：

1. 审查并确认 Reward 与最终 SuccessRate 目标严格一致；
2. 修正少量工程安全问题；
3. 完成正式 5-seed paired experiment 前的最终验收；
4. 不引入新的 Attention、GRU、BFS、Action Space 或 Decoder 变化。

---

# 1. 当前实现结论

当前 `DemandResidual` 主体原则上通过审查：

\[
s_a=s_a^{base}+\Delta s_a
\]

其中：

\[
s_a^{base}=edge\_scorer(pair\_emb_a)
\]

新分支：

\[
\Delta s_a=
DemandResidual(pair\_emb_a,demand\_emb)
\]

当前已经确认：

- 旧 `edge_scorer` 保留；
- 新 residual 最后一层 zero-init；
- 旧 BC checkpoint 可以完整复用；
- 开启 residual + zero-init 时前向与 baseline 逐位一致；
- Cross-Graph Demand 隔离已通过反事实测试；
- Edge Query / Demand Key-Value 的 Attention 方向正确；
- 输出仍为 per-arc score；
- 本次没有修改 Action Space、Sequential Sampler、Service Algorithm、Key Generation physics。

因此：

\[
\boxed{
\text{当前模型主体可以进入正式实验}
}
\]

---

# 2. 当前不建议继续修改的内容

在 Reward 审查完成之前，不要继续加入：

- Edge Self-Attention；
- Request-level Attention Pool；
- GRU / LSTM / temporal Transformer；
- BFS candidate filtering；
- 新 Action Space；
- 新 Decoder；
- 更大的 hidden dimension；
- 不必要的 attention head 搜索；
- Key Generation reward shaping；
- inventory penalty。

原因：

> 当前实验应该只回答一个问题：在完全保留 baseline 行为和 Action 机制的情况下，Demand-conditioned residual 是否提高 SuccessRate。

---

# 3. 小的代码工程改进：Demand slice metadata

当前跨图 Attention 的处理是逐图切块，这个方法本身正确，而且比构造大 block-diagonal attention mask 更节省内存。

当前潜在风险是：

> `demand_off` 的正确性依赖 batching 中 `perm` 的固定顺序。

当前代码假设：

```text
所有图的物理边按图排列
        ↓
所有图的 Demand edge 按相同图顺序排列
```

这个假设目前成立，因此当前结果可信。

但未来如果 batching / `perm` 顺序变化，很可能产生 silent misalignment。

---

## 建议

将以下信息在 batch planning 阶段显式保存：

```python
arc_slices = [
    (arc_lo_0, arc_hi_0),
    (arc_lo_1, arc_hi_1),
    ...
]

demand_slices = [
    (dem_lo_0, dem_hi_0),
    (dem_lo_1, dem_hi_1),
    ...
]
```

然后 `DemandResidual` 只使用已经准备好的 slice metadata。

不要让模型自己从：

```text
n_demand
```

反推 offset。

---

## 这不是本轮实验阻塞项

如果当前测试已经通过：

```text
改变图 B demand → 图 A score 不变
```

则不需要为了正式实验大改 batching。

可以把 metadata 显式化改进放到同一个代码 commit 中，但不能因此修改模型数学结构。

---

# 4. Residual Head：暂时保持 Linear

当前实际实现是：

\[
128\rightarrow1
\]

即：

```python
delta_score = Linear(128, 1)(ctx)
```

虽然早期设计文档曾写成：

```text
128 → 128 → 1 MLP
```

但当前 Linear 版本并非错误。

## 建议

本轮：

\[
\boxed{
保持 Linear
}
\]

并把设计文档术语统一为：

> `zero-initialized residual projection`

而不是：

> `residual MLP`

原因：

- 参数更少；
- 实验变量更少；
- 更适合验证机制本身；
- 更容易解释；
- 不需要为了结构形式而增加模型容量。

---

# 5. Attention Heads：固定 4，不要搜索

当前：

\[
128/4=32
\]

即：

```text
4 heads × 32 dim
```

完全合理。

但不要把 4 解释成理论最优值。

本轮直接固定：

```yaml
num_heads: 4
```

不要同时跑：

```text
2 heads
4 heads
8 heads
```

否则 Experiment 1 会变成：

> DemandResidual + Attention Capacity Search

而不是单变量实验。

---

# 6. Temporal Demand：当前版本暂时不加入

当前 DemandResidual 使用的是当前时隙 Demand embedding。

这已经包含：

- pending amount；
- pending count；
- min deadline left；
- mean deadline left；
- mean wait time；
- priority；
- wait buckets。

因此当前模型已经具有部分时间语义。

但是它不是历史序列模型。

当前版本不应该增加：

```text
GRU
LSTM
Temporal Transformer
过去 N 个时隙 Demand stacking
```

原因：

> 本轮实验的目标不是研究 Temporal Demand Prediction，而是研究 Demand-conditioned Edge Scoring。

未来如果发现：

- 无当前 Demand 时性能不足；
- Demand 突增时反应不足；
- 需求有明显 temporal pattern；

再单独开展 temporal experiment。

---

# 7. ★ Reward Audit：现在的最高优先级

当前模型验收已经较充分。

真正还没有在实现报告中被完整证明的是：

\[
\boxed{
Reward \leftrightarrow SuccessRate
}
\]

因此正式 5-seed 实验开始前，必须完成一次独立 Reward Audit。

---

# 8. Reward Audit 必须回答的 9 个问题

请直接从代码读取，不要根据文档猜。

## 8.1 Reward 精确公式

给出：

```text
reward_t = ?
```

以及对应的代码位置。

---

## 8.2 Reward 在 env.step() 哪个阶段计算？

必须确认是：

```text
Request Arrival
→ Action
→ Generate
→ Allocate
→ Service
→ Reward
```

还是其他顺序。

本项目当前代码事实是：

\[
Generate_t\rightarrow Serve_t
\]

所以如果某条边本时隙生成的 Key 直接促成服务成功：

> 该成功应该体现在本时隙 reward 中。

---

## 8.3 Reward 是否基于“完整请求成功”

必须确认：

```text
一个请求只有 remaining_amount == 0
才算成功？
```

还是：

```text
只要 served_amount > 0
就产生 reward？
```

这是最关键的问题。

---

## 8.4 Partial Service 的 reward

构造一个最小测试：

```text
request amount = 100
本时隙只服务 20
剩余 80
```

检查：

```text
reward = ?
```

如果目标 SuccessRate 定义为：

> 必须完整服务才算成功，

那么推荐：

```text
partial service → reward 0
```

而不是：

```text
partial service → reward 20
```

否则 reward objective 与 SuccessRate 不一致。

---

# 9. 最推荐的主 Reward

如果代码中的 Success 定义确实是：

> 请求被完整满足并结束服务。

那么推荐：

\[
\boxed{
r_t=\#\text{completed requests at }t
}
\]

即：

```text
本时隙完成 0 个请求 → reward 0
本时隙完成 1 个请求 → reward 1
本时隙完成 5 个请求 → reward 5
```

---

# 10. 为什么推荐“完成请求数”而不是“每时隙成功率”

不推荐：

\[
r_t=
\frac{success_t}{arrival_t}
\]

原因：

### 问题 1：不同规模时隙权重不正确

例如：

```text
t1: 1 / 1 = 1.0
t2: 10 / 10 = 1.0
```

reward 一样，但实际成功请求数不同。

### 问题 2：没有到达请求时：

```text
arrival_t = 0
```

需要额外处理。

### 问题 3：容易让 reward 与最终 episode SuccessRate 的统计口径不一致。

因此推荐：

\[
\boxed{
reward = successful\ completed\ request\ count
}
\]

SuccessRate 只作为最终 evaluation metric。

---

# 11. Reward 与 SuccessRate 的数学对齐

假设 episode 内到达请求总数：

\[
N_{arrived}
\]

成功完成数量：

\[
N_{success}
\]

如果：

\[
r_t=N_{success,t}
\]

则：

\[
\sum_t r_t=N_{success}
\]

而：

\[
SuccessRate=
\frac{\sum_t r_t}
{N_{arrived}}
\]

只要：

\[
N_{arrived}
\]

不依赖 RL action，那么：

\[
\boxed{
\max \mathbb E[\sum_t r_t]
\Longleftrightarrow
\max SuccessRate
}
\]

这是最干净的 reward-objective 对齐。

---

# 12. 不建议当前加入“生成 Key 奖励”

不要使用：

\[
+\alpha\cdot GeneratedKeys
\]

作为主 reward。

否则会产生：

> “只要生成 Key 就有即时奖励。”

这会诱导模型偏向：

- 过度生成；
- 在无需求边上囤积；
- 追求 Key throughput 而不是 SuccessRate。

当前任务最终目标是：

\[
SuccessRate
\]

所以第一版不要奖励 Key generation 本身。

---

# 13. 不建议当前加入 Inventory Penalty

不要第一版就做：

\[
-\lambda K_{stored}
\]

虽然它可以抑制无用囤积，但也会惩罚：

> 对未来请求有价值的 proactive inventory。

让 PPO 通过长期 SuccessRate 自己学习库存价值即可。

---

# 14. 不建议第一版奖励“激活一条边”

不要：

```text
activate edge → +c
```

因为这奖励的是 Action 本身，而不是 Action 的服务价值。

---

# 15. 过期请求的 penalty

推荐第一版不要额外设置：

```text
expired request → -1
```

而采用：

```text
completed request → +1
otherwise → 0
```

原因：

如果同时：

```text
success = +1
expire = -1
```

就不再是纯 SuccessRate objective，而变成：

\[
success-\lambda failure
\]

如果后续发现 reward sparsity 很严重，再单独实验 failure penalty。

---

# 16. Reward 的最小验证实验

不要只读取代码。

建议加一个 tiny deterministic test。

## Case A：成功

```text
request amount = 100

路径：
e1, e2

e1 inventory = 100
e2 inventory = 100
```

Service：

```text
success = 1
```

期望：

```text
reward = 1
```

---

## Case B：部分服务

```text
request amount = 100

e1 = 100
e2 = 20
```

如果服务允许 partial consume：

```text
served = 20
remaining = 80
```

期望：

```text
reward = 0
```

前提：

> SuccessRate 只认完整请求。

---

## Case C：过期

请求到达并一直没有满足：

```text
deadline reached
```

期望：

```text
reward = 0
```

如果当前代码有额外 penalty，必须明确写出来并判断是否真的需要。

---

## Case D：没有请求

```text
arrival = 0
```

期望：

```text
reward = 0
```

不能 NaN。

---

# 17. Reward 时间归属测试

必须特别测试：

```text
本时隙开始时：
Key = 0

本时隙：
Action 激活 edge
    ↓
Generate 100
    ↓
Service request amount 100
    ↓
成功
```

期望：

```text
本时隙 reward = 1
```

如果本时隙 reward 没有统计到这个 success：

> reward timing 就与真实环境的 Generate → Serve 顺序不一致。

---

# 18. Reward Audit 输出格式

建议另一个 AI 最终给出：

```text
## Reward Audit

reward formula:
    ...

code location:
    ...

calculation order:
    ...

complete request reward:
    ...

partial service reward:
    ...

expired request reward:
    ...

idle / no-demand reward:
    ...

generated-key reward:
    ...

inventory penalty:
    ...

alignment with SuccessRate:
    PASS / FAIL

evidence tests:
    Case A:
    Case B:
    Case C:
    Case D:
```

---

# 19. 对正式 5-seed 实验的要求

Reward Audit 通过后再运行：

```text
baseline:
u0ctl_s42 ... u0ctl_s46

experiment:
dminj_s42 ... dminj_s46
```

保持：

- 相同 seed；
- 相同环境；
- 相同 expert/BC warm start；
- 相同训练窗口；
- 唯一变量：

```yaml
model.actor.demand_residual.enabled
```

---

# 20. 实验报告必须同时报告

不要只报告：

```text
mean SuccessRate
p-value
```

至少报告：

\[
\Delta SR_i
=
SR_i^{attn}-SR_i^{baseline}
\]

每个 seed 的：

```text
seed
baseline SR
experiment SR
difference
```

以及：

```text
mean difference
std
95% CI
paired t-test
```

因为当前项目历史上已经存在：

> 改动真实存在但 effect 太小，5 seeds 不足以稳定检出。

所以：

\[
p>0.05
\]

不能直接写成：

> “没有效果。”

应该写：

> “在当前样本量下未检测到统计显著差异。”

---

# 21. 当前模型的两个明确限制

正式论文/报告中要准确描述：

## 限制 1：没有历史 Demand memory

当前 residual 使用：

```text
current Demand only
```

不直接利用：

```text
Demand(t-1), Demand(t-2), ...
```

所以它不是 temporal predictor。

---

## 限制 2：没有 Demand 时 residual 不改变 baseline

：

\[
D_t=0
\Rightarrow
\Delta s=0
\]

因此 proactive generation 仍由 baseline scorer 决定。

这不是 bug，而是当前 residual 的设计边界。

---

# 22. 当前实现的最终建议

## 保留

```text
GNN
+
existing edge_scorer
+
DemandResidual
+
existing sequential sampler
+
existing service algorithm
+
BC warm start
```

## 暂不增加

```text
Self-Attention
Request Attention
GRU
BFS filtering
new action
new decoder
reward shaping
inventory penalty
```

---

# 23. 最终执行顺序

```text
Step 1
Reward Audit
        ↓
Step 2
Tiny deterministic reward tests
        ↓
Step 3
如果 reward 与 SuccessRate 对齐
        ↓
Step 4
保持当前 DemandResidual 不再改结构
        ↓
Step 5
5-seed paired experiment
        ↓
Step 6
根据结果决定是否进入 Decoder / Self-Attention / Temporal Demand 实验
```

---

# 24. 最终判定标准

如果：

### Model

```text
zero-init equivalence = PASS
cross-graph isolation = PASS
BC compatibility = PASS
```

### Reward

```text
completed request = +1
partial service = 0
expired = 0
no request = 0
reward calculated after service
```

并满足：

\[
\sum_t r_t=N_{success}
\]

那么：

\[
\boxed{
当前实现可以进入正式实验
}
\]

否则：

> **先修 reward，再跑实验。**

不要在 reward 没有确认前继续增加模型结构。
