# Reward 审计后的最终处理建议
## 针对《Reward审计报告.md》的评审结论

> 本文件用于直接交给另一个 AI / 代码 Agent。
>
> **核心结论先说：**
>
> 1. **暂时不要把 reward 改成“完成请求数”。**
> 2. 但也**不能简单写成“当前总 reward 与 SuccessRate 严格等价”**。
> 3. 当前代码中的 `success_rate` 实际是**密钥量口径**，不是“成功请求数 / 到达请求数”。
> 4. 当前 reward 的 `served_keys` 主项与这个指标同源，但 reward 还有其他与策略有关的 shaping 项，因此严格数学等价并不成立。
> 5. **正式 5-seed 实验之前，必须先把“论文目标到底是请求级 SuccessRate 还是密钥量 SuccessRate”这一语义问题定死。**
> 6. 在没有改变 reward 配置的情况下，可以做当前 DemandResidual 的 architecture experiment；如果决定改变 reward，则 baseline 与 experiment 必须一起重跑。

---

# 1. 先纠正 Reward Audit 中一个过强的结论

审计报告写：

> reward 与指标“同向且同口径”。

其中“主项同口径”是对的，但：

\[
\boxed{\text{总 reward 与 SuccessRate 严格等价}}
\]

这个结论目前不能直接成立。

原因是当前 reward 实际为：

\[
r_t
=
r^{served}_t
+
r^{storage}_t
-
r^{failed}_t
-
r^{expired}_t
+
r^{keepactive}_t
\]

具体生效配置为：

\[
r_t=
50\frac{served\_keys}{10^5}
+
0.5\frac{storage}{10^6}
-
5\frac{failed\_keys}{10^5}
-
0.01\frac{expired\_keys}{10^5}
+
0.001\,keep\_active\_count
\]

最后再乘：

\[
reward\_scale=0.002
\]

因此不能只看：

\[
served\_keys
\]

就推出：

\[
\max E[\sum r_t]
\Longleftrightarrow
\max SuccessRate
\]

因为其他项也可能随 policy 改变。

---

# 2. 当前代码真正的 SuccessRate 定义

根据 Reward Audit：

```python
success_rate = served_keys / arrived_keys
```

并且：

```python
request_completion_rate =
    completed_requests / arrived_requests
```

两个指标同时存在。

但实验目前使用：

```text
success_rate
```

而不是：

```text
request_completion_rate
```

因此当前实验实际优化/评估的是：

\[
\boxed{
SR_{key}=
\frac{\text{served key amount}}
{\text{arrived key amount}}
}
\]

而不是：

\[
\boxed{
SR_{request}=
\frac{\text{completed requests}}
{\text{arrived requests}}
}
\]

---

# 3. ★ 这是现在真正需要在论文语义上解决的问题

项目最早的任务描述如果写的是：

> SuccessRate = successful requests / arrived requests

那么它与当前代码指标存在直接冲突。

现在必须二选一。

## 方案 A：把研究目标正式定义为“Key-volume Success Rate”

定义：

\[
SuccessRate=
\frac{\sum served\_keys}
{\sum arrived\_keys}
\]

优点：

- 与当前 metrics 一致；
- 与当前 reward 主项一致；
- 不需要重写现有训练目标；
- 与当前 partial service 语义天然一致；
- 不需要重新跑已有 baseline。

如果业务上确实更关心：

> 总体有多少 QKD key demand 被满足，

那么这是合理定义。

---

## 方案 B：真正采用“Request Completion Rate”

定义：

\[
SuccessRate=
\frac{\#completed\ requests}
{\#arrived\ requests}
\]

那么：

- metrics 要改；
- reward 要改；
- baseline 要重跑；
- DemandResidual experiment 也要重跑；
- 论文里所有结果都要重算。

不能只把 reward 改成 completed request count，而继续使用旧的 key-volume `success_rate`。

---

# 4. 当前不建议现在直接改成完成请求数

原因不是说 request-level metric 不合理。

而是：

> 当前整个项目的代码、历史实验和训练 reward 已经围绕 key-volume metric 建立。

如果现在突然改：

```text
served_keys
↓
completed_requests
```

那么前面的：

- baseline；
- BC；
- reward；
- 历史效果；
- 训练稳定性；
- 已有 SR；

都不再是同一实验体系。

因此：

\[
\boxed{
\text{不要为了当前 DemandResidual 实验临时更换 metric}
}
\]

---

# 5. 但必须在论文中停止使用“成功请求率”这个容易误导的定义

如果决定保持当前指标，那么正式文档/论文应该明确写：

> Success Rate is defined as the ratio of served key amount to arrived key amount.

例如：

\[
SR_{key}
=
\frac{K_{served}}
{K_{arrived}}
\]

不要继续写：

\[
\frac{\#successful\ requests}
{\#arrived\ requests}
\]

否则实验结果的语义和数学定义会不一致。

---

# 6. 当前 reward 与 SuccessRate 的真正关系

令：

\[
SR_{key}
=
\frac{\sum_t served_t}
{\sum_t arrived_t}
\]

由于：

\[
\sum_t arrived_t
\]

由 RequestGenerator、seed、时间决定，而不由 policy 决定，所以：

\[
\max \sum_t served_t
\]

确实与：

\[
\max SR_{key}
\]

一致。

**但前提是 reward 只包含 served 项。**

当前 reward 还有：

```text
storage
failed
expired
keep_active
```

所以严格来说：

\[
\boxed{
\max \sum_t r_t
\neq
\max SR_{key}
\quad\text{必然成立}
}
\]

只能说：

> reward 的主要优化方向与 SR_key 一致，且 served 项是其最核心的正向项。

---

# 7. 必须检查 reward shaping 的实际影响大小

这是当前最值得补的一项，而不是马上修改 reward。

根据生效配置，乘上 `reward_scale=0.002` 后：

## Served

每增加 1 key：

\[
50/10^5\times0.002
=
1\times10^{-6}
\]

---

## Failed

每增加 1 failed key：

\[
5/10^5\times0.002
=
1\times10^{-7}
\]

即：

> 失败一个 key 的惩罚量约为成功服务一个 key 奖励量的 10%。

---

## Expired

每增加 1 expired key：

\[
0.01/10^5\times0.002
=
2\times10^{-10}
\]

相对于 served：

\[
\frac{2\times10^{-10}}{10^{-6}}
=
2\times10^{-4}
\]

即：

> expired-key penalty 相对极小。

---

## Storage

每增加 1 storage key：

\[
0.5/10^6\times0.002
=
1\times10^{-9}
\]

相对于 served：

\[
10^{-3}
\]

即：

> 单位 storage reward 只有 served reward 的约 0.1%。

因此 storage 项单个 key 很小，但如果 storage 数量非常大，它仍可能产生可见影响。

---

## Keep Active

每额外保持 1 条 active edge：

\[
0.001\times0.002
=
2\times10^{-6}
\]

而：

\[
1\ served\ key
=
10^{-6}
\]

所以：

\[
\boxed{
1\ 条 keep-active edge
\approx
2\ 个 served\ keys
}
\]

按这个量级看，keep-active 项反而比 storage 和 expired 项更值得关注。

例如：

```text
100 条 active edge
```

一时隙就贡献：

\[
2\times10^{-4}
\]

相当于：

\[
200\ served\ keys
\]

的 reward。

因此：

\[
\boxed{
keep\_active\_reward
\text{ 必须重点审查}
}
\]

---

# 8. 对当前 Reward 的最终建议：先不改，先做 decomposition audit

当前不要直接删：

```text
failed_penalty
storage_reward
keep_active_reward
```

因为这是一个已经使用过的 baseline reward。

如果现在临时修改：

> baseline 与 DemandResidual 实验就不再是纯模型结构比较。

更合理的是：

### 先冻结当前 reward

保持：

```yaml
served_weight
storage_reward_weight
failed_weight
expired_key_weight
keep_active_reward
```

完全不变。

然后对 rollout 做 reward decomposition：

```text
served contribution
storage contribution
failed contribution
expired contribution
keep_active contribution
```

分别统计：

- 每步均值；
- 每 episode 总量；
- 占 total reward 的比例；
- 与 SuccessRate 的相关性。

---

# 9. 特别检查 keep_active reward

如果发现：

\[
R_{keepactive}
\]

在总 reward 中占比很高，或者在没有服务收益时也能明显推动 return：

> 这就是一个潜在的 reward hacking / objective distortion 来源。

此时不要在 DemandResidual 实验中直接修改。

而应该另做：

```text
Reward Ablation:
baseline reward
vs
without keep_active_reward
```

并对 baseline 单独重跑。

---

# 10. 对 failed penalty 的判断

当前：

\[
failed\_penalty
\]

相对 served reward 的单位量级约为：

\[
10\%
\]

这并不一定有问题。

它可以起到：

> 对服务失败造成的资源浪费 / 未满足需求进行轻度反向反馈。

但它已经意味着：

\[
Reward
\neq
纯 SuccessRate
\]

所以论文中不要写：

> reward 就等于 SuccessRate。

更准确：

> reward is a shaped objective whose dominant positive component is served key volume, with additional failure/storage/resource shaping.

---

# 11. 对 expired_key penalty 的判断

当前：

\[
expired\_key\_weight=0.01
\]

从单位系数看极小。

因此目前不认为它是主要问题。

而且如果它只占 reward 极少部分：

> 没有必要为了追求数学形式纯净而单独修改它。

保持 baseline 一致更重要。

---

# 12. 对 storage reward 的判断

当前：

\[
0.5/10^6
\]

相对 served：

\[
50/10^5
\]

小约 1000 倍。

因此单个 key 的 storage reward 很弱。

它更像：

> 一个很轻的 proactive storage shaping。

这与当前任务允许：

> 没有当前请求也可以预生成 Key

这一要求是基本一致的。

当前不建议删除。

但仍应做 episode-level decomposition，确认 storage 总量是否巨大到能够抵消 served 主项。

---

# 13. Partial Service 现在不是 bug

Audit 发现：

```text
request demand = 100
served = 20
```

会得到：

```text
served_keys += 20
reward > 0
```

如果当前正式指标定义是：

\[
SR_{key}
=
served\_keys/arrived\_keys
\]

那么这其实是**一致的**。

因此不要按照之前“完整请求才 reward +1”的假设去修改。

这不是实现错误。

真正的问题只是：

> **当前 SuccessRate 的语义是 Key-volume，而不是 Request-completion。**

---

# 14. Generate → Serve → Reward 的顺序是正确的

Audit 已确认：

```text
Generate
→ Allocate
→ set_slot_new
→ Serve
→ Expire
→ Reward
```

所以：

> 本时隙生成的 Key 本时隙能够产生服务结果，本时隙对应的 reward 能够记录这个收益。

这一点：

\[
\boxed{PASS}
\]

不要修改时序。

---

# 15. 为什么当前 reward 不应该简单改成 completed_requests

如果直接：

\[
r_t=\#completed\_requests_t
\]

会出现：

### 大请求和小请求等权

```text
10000-key request = 1
100-key request = 1
```

而当前业务可能更关心：

> 实际交付了多少 Key。

同时 reward 会明显稀疏。

而当前 audit 还指出每步到达请求数量较多、完整完成数量相对少，因此会进一步增加 sparse reward 风险。

因此：

\[
\boxed{
暂时保持 key-volume reward
}
\]

是更稳妥的工程选择。

---

# 16. 但是论文目标必须明确

现在最重要的问题不是：

> reward 要不要改？

而是：

> **QKD-SAGIN 这个项目最终到底要优化“请求完成率”，还是“Key demand fulfillment ratio”？**

如果是：

### Key fulfillment

保持当前：

\[
SR_{key}
=
\frac{K_{served}}{K_{arrived}}
\]

完全合理。

如果是：

### Request completion

那最终必须改成：

\[
SR_{request}
=
\frac{N_{completed}}{N_{arrived}}
\]

并全面重做 reward / baseline。

不能两个概念混着用。

---

# 17. ★ 当前建议的正式实验策略

在你尚未决定论文指标语义前：

\[
\boxed{
不要开始最终 5-seed 正式实验
}
\]

因为：

> 如果先跑 5 seeds，之后决定把 SuccessRate 改成 request-level，就得整个 baseline + experiment 重跑。

---

# 18. 但不要重新设计 DemandResidual

DemandResidual 当前结构已经通过：

- zero-init；
- BC compatibility；
- cross-graph isolation；
- per-arc output；

等验收。

所以现在**不要回头改模型**。

需要确定的是：

```text
metric semantics
+
reward shaping influence
```

---

# 19. 这一步建议做一个很轻量的 Reward Decomposition Probe

不需要训练。

直接用现有 BC checkpoint + 固定验证轨迹跑一遍，输出：

```text
Episode:
    SuccessRate_key

Reward total:
    R_total

Components:
    R_served
    R_storage
    R_failed
    R_expired
    R_keep_active

Ratios:
    R_served / R_total
    R_storage / R_total
    R_failed / R_total
    R_expired / R_total
    R_keep_active / R_total
```

再计算：

\[
\rho_i=
\frac{|R_i|}
{\sum_j|R_j|}
\]

注意不要只看 signed ratio，因为正负项可能互相抵消。

---

# 20. 再计算“等效 served keys”

把所有 reward component 转换成：

> 相当于多少个 served key reward。

因为：

\[
1\ served\ key
=
10^{-6}
\]

reward（在当前 scale 下）。

因此：

\[
K^{equiv}_{component}
=
\frac{|R_{component}|}
{10^{-6}}
\]

例如：

```text
keep_active contribution
= 0.0002
```

等价：

```text
200 served keys
```

这样非常直观。

---

# 21. 对 reward decomposition 的判断标准

不要提前硬设一个绝对百分比阈值。

重点看：

### 情况 A

```text
served dominates
其他项很小
```

→ 当前 shaping 可以继续保留。

### 情况 B

```text
keep_active / storage
与 served 同量级
甚至更大
```

→ reward objective 已经明显不是纯服务成功，应单独做 reward ablation。

### 情况 C

```text
失败 penalty
大幅抵消 served reward
```

→ 需要检查是否导致 PPO 更关心避免失败而不是提升服务量。

---

# 22. 当前最重要的最终决策

我建议：

### 决策 1：先保留当前 reward

不要为了迎合“completed request reward”的旧建议去改。

---

### 决策 2：修正论文 / 设计文档的指标定义

二选一：

```text
Key-volume Success Rate
```

或：

```text
Request Completion Rate
```

当前代码显然使用前者。

---

### 决策 3：做 reward decomposition probe

验证：

```text
served
storage
failed
expired
keep_active
```

到底谁在真正推动 return。

---

### 决策 4：只有 reward decomposition 发现 shaping 严重扭曲时，才另做 reward ablation

而不是把 reward 修改与 DemandResidual 模型实验混在一起。

---

# 23. 对当前 Reward Audit 报告的最终评价

| 结论 | 审查意见 |
|---|---|
| `success_rate` 是 key-volume | ✅ 正确 |
| `request_completion_rate` 另存 | ✅ 正确 |
| reward 主项是 served_keys | ✅ 正确 |
| reward 在 Serve 后计算 | ✅ 正确 |
| partial service 有 reward | ✅ 正确 |
| 不应该简单改成 completed_requests | ✅ 同意 |
| reward 与指标“完全等价” | ⚠️ **说得过强** |
| 当前 reward 主方向与指标一致 | ✅ |
| keep_active 需要特别审查 | ✅ **重点** |
| failed penalty 需要立即删除 | ❌ 不建议立即改 |
| expired penalty 需要立即删除 | ❌ 不建议立即改 |
| storage reward 需要立即删除 | ❌ 不建议立即改 |
| 当前可以直接进入最终 5-seed | ⚠️ **先做 metric 定义 + reward decomposition** |

---

# 24. 最终执行清单

当前不要改 DemandResidual。

先执行：

```text
① 明确论文最终指标：
   SR_key
   或
   SR_request

② 如果保持 SR_key：
   不修改现有 reward 主结构

③ 跑一次 reward decomposition probe：
   served
   storage
   failed
   expired
   keep_active

④ 输出各项：
   raw contribution
   absolute contribution
   equivalent served keys

⑤ 确认 reward 没有明显被 shaping 主导

⑥ 冻结 reward + metric

⑦ 再跑 DemandResidual 的 5-seed paired experiment
```

---

# 25. 最终建议

当前最稳妥的选择是：

\[
\boxed{
\text{保持现有 Key-volume reward}
}
\]

但同时：

\[
\boxed{
\text{正式承认 SuccessRate 是 Key-volume fulfillment ratio}
}
\]

并且：

\[
\boxed{
\text{先做一次 reward decomposition，再开始最终实验}
}
\]

**不要为了理论上的“请求成功率”去临时改 Reward。**

如果论文真正要求的是：

\[
\frac{\#completed\ requests}{\#arrived\ requests}
\]

那么应把它作为一次独立的“目标定义重构”：

```text
Metric
+
Reward
+
Baseline
+
DemandResidual
+
所有历史对照
```

全部重新跑。

不能在现有实验体系上半途切换。
