# v3 实验结果台账

> **本文件由脚本生成，不要手改表格。** 生成命令：
>
> ```bash
> python scripts/diag/backfill_manifest.py --repo-root . --apply
> ```
>
> 数据源：`outputs/model_ledger.jsonl`（中心台账，由原始产物重建）。
> 手写内容写在文末「设计要点」「备注」段，重新生成时原样保留。

## 测试环境

**`env_fp` 不同 = 数字不可直接配对比较。**

| env_fp | 训练窗(激活) | 回合步 | 起始模式 | 拓扑 gs/hap/sat | 请求 | 配对热度 | 路由 | 速率源 |
|---|---|---|---|---|---|---|---|---|
| 9091810f | 0–295 | 240 | random_day | 4/1/1 | poisson rate=0.8 amt=100000.0 dl=30 | zipf s=1.0 | link_path/shortest_available_path | h5 |

## 验证结果 · 窗口 A

协议：第 330–365 天，15 个请求种子 `[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]`，每集 240 步，起始模式 `random_day`。

| 批次 | arm | 模型 | 训练种子 | 成功率 | 密钥比 | 隔离 | env_fp | 重复 |
|---|---|---|---|---|---|---|---|---|
| 20260924-frozen-eval | expert | expert | — | **0.7674** | 206.7894 | — | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-isolated-s42 | v3 | 42 | **0.7478** | 160.8036 | 是 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-isolated-s43 | v3 | 43 | **0.7498** | 160.2040 | 是 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-isolated-s44 | v3 | 44 | **0.7533** | 158.7943 | 是 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-s42 | v3 | 42 | **0.7691** | 148.6056 | 否 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-s43 | v3 | 43 | **0.7600** | 151.1327 | 否 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-s44 | v3 | 44 | **0.7593** | 150.6941 | 否 | 9091810f |  |
| 20260924-iso-attribution | expert | expert | — | **0.7674** | 206.7894 | — | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-isolated-s42 | v3 | 42 | **0.7478** | 160.8036 | 是 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-isolated-s43 | v3 | 43 | **0.7498** | 160.2040 | 是 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-isolated-s44 | v3 | 44 | **0.7533** | 158.7943 | 是 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-s42 | v3 | 42 | **0.7691** | 148.6056 | 否 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-s43 | v3 | 43 | **0.7600** | 151.1327 | 否 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-s44 | v3 | 44 | **0.7593** | 150.6941 | 否 | 9091810f | 是 |
| 20260924-iso-attribution | v3-iso-s42 | v3 | 42 | **0.7456** | 192.2320 | 是 | 9091810f |  |
| 20260924-iso-attribution | v3-iso-s43 | v3 | 43 | **0.7534** | 189.9352 | 是 | 9091810f |  |
| 20260924-iso-attribution | v3-iso-s44 | v3 | 44 | **0.7545** | 192.4333 | 是 | 9091810f |  |
| 20260924-iso-attribution | v3-v3full-s42 | v3 | 42 | **0.7558** | 193.6706 | 否 | 9091810f |  |
| 20260924-iso-attribution | v3-v3full-s43 | v3 | 43 | **0.7508** | 186.7326 | 否 | 9091810f |  |
| 20260924-iso-attribution | v3-v3full-s44 | v3 | 44 | **0.7472** | 196.9729 | 否 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-iso-s42 | v3 | 42 | **0.7460** | 191.8142 | 是 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-iso-s43 | v3 | 43 | **0.7561** | 189.3007 | 是 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-iso-s44 | v3 | 44 | **0.7554** | 192.5703 | 是 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-retry-v3full-s42 | v3 | 42 | **0.7584** | 192.4691 | 否 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-v3full-s43 | v3 | 43 | **0.7502** | 188.1278 | 否 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-v3full-s44 | v3 | 44 | **0.7459** | 196.4553 | 否 | 9091810f |  |
| 20260925-v3-diagnostics | rl-outputs-diagnostics-v3-set-bc-20260925-ordered | v3 | — | **0.7524** | 191.8368 | 否 | — |  |
| 20260925-v3-diagnostics | rl-outputs-diagnostics-v3-set-bc-20260925-set | v3 | — | **0.7369** | 200.8029 | 否 | — |  |
| 20260925-v3-diagnostics | v3-probe-base-s42 | v3 | 42 | **0.7584** | 192.4691 | 否 | — |  |
| 20260925-v3-diagnostics | v3-probe-high-s42 | v3 | 42 | **0.7571** | 192.2534 | 否 | — |  |

> 「重复 = 是」表示该条与更早一条是同一次测量（同 `evaluated_at`、同数值）被复制到两个目录，**不构成独立证据**。

## 验证结果 · 窗口 B

协议：第 300–329 天，15 个请求种子 `[200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214]`，每集 240 步，起始模式 `random_day`。

| 批次 | arm | 模型 | 训练种子 | 成功率 | 密钥比 | 隔离 | env_fp | 重复 |
|---|---|---|---|---|---|---|---|---|
| 20260924-frozen-eval | expert | expert | — | **0.8674** | 197.8869 | — | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-isolated-s42 | v3 | 42 | **0.8413** | 157.5977 | 是 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-isolated-s43 | v3 | 43 | **0.8350** | 159.2591 | 是 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-isolated-s44 | v3 | 44 | **0.8392** | 157.2529 | 是 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-s42 | v3 | 42 | **0.8593** | 148.3643 | 否 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-s43 | v3 | 43 | **0.8572** | 149.6655 | 否 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-s44 | v3 | 44 | **0.8511** | 151.3319 | 否 | 9091810f |  |
| 20260924-iso-attribution | expert | expert | — | **0.8674** | 197.8869 | — | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-isolated-s42 | v3 | 42 | **0.8413** | 157.5977 | 是 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-isolated-s43 | v3 | 43 | **0.8350** | 159.2591 | 是 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-isolated-s44 | v3 | 44 | **0.8392** | 157.2529 | 是 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-s42 | v3 | 42 | **0.8593** | 148.3643 | 否 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-s43 | v3 | 43 | **0.8572** | 149.6655 | 否 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-s44 | v3 | 44 | **0.8511** | 151.3319 | 否 | 9091810f | 是 |
| 20260925-retrain-v3-8ep | v3-retrain8-iso-s42 | v3 | 42 | **0.8292** | 188.2836 | 是 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-iso-s43 | v3 | 43 | **0.8250** | 189.4181 | 是 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-iso-s44 | v3 | 44 | **0.8382** | 189.6705 | 是 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-retry-v3full-s42 | v3 | 42 | **0.8403** | 190.8742 | 否 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-v3full-s43 | v3 | 43 | **0.8268** | 184.0750 | 否 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-v3full-s44 | v3 | 44 | **0.8333** | 193.6009 | 否 | 9091810f |  |
| 20260925-v3-diagnostics | v3-probe-base-s42 | v3 | 42 | **0.8426** | 190.0961 | 否 | — |  |
| 20260925-v3-diagnostics | v3-probe-high-s42 | v3 | 42 | **0.8423** | 189.2997 | 否 | — |  |

> 「重复 = 是」表示该条与更早一条是同一次测量（同 `evaluated_at`、同数值）被复制到两个目录，**不构成独立证据**。

## 训练记录

`PPO 更新` 列为「否」= 全程 `kl=0` 且 `clip_frac=0`，即 PPO 从未真正更新（预算配置错误），该次训练不构成架构结论。

| 批次 | arm | 模型 | 轮数 | 是否多 minibatch | PPO 更新 | 每轮集数 | minibatch |
|---|---|---|---|---|---|---|---|
| 20260924-frozen-eval | frozen-s42 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | frozen-isolated-s42 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | frozen-isolated-s43 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | frozen-isolated-s44 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | frozen-s42 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | frozen-s43 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | frozen-s44 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | iso-s42 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | iso-s43 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | iso-s44 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | v3full-s42 | v3 | 20 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | v3full-s43 | v3 | 20 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | v3full-s44 | v3 | 20 | 否 | **否** | 1 | 1024 |
| 20260925-retrain-v3-8ep | retrain8-iso-s42 | v3 | 10 | 是 | 是 | 8 | 1024 |
| 20260925-retrain-v3-8ep | retrain8-iso-s43 | v3 | 10 | 是 | 是 | 8 | 1024 |
| 20260925-retrain-v3-8ep | retrain8-iso-s44 | v3 | 10 | 是 | 是 | 8 | 1024 |
| 20260925-retrain-v3-8ep | retrain8-retry-v3full-s42 | v3 | 10 | 是 | 是 | 8 | 1024 |
| 20260925-retrain-v3-8ep | retrain8-v3full-s43 | v3 | 10 | 是 | 是 | 8 | 1024 |
| 20260925-retrain-v3-8ep | retrain8-v3full-s44 | v3 | 10 | 是 | 是 | 8 | 1024 |
| 20260925-v3-diagnostics | probe-base-s42 | v3 | 30 | 是 | 是 | 8 | 1024 |
| 20260925-v3-diagnostics | probe-high-s42 | v3 | 30 | 是 | 是 | 8 | 1024 |
| v3-from-dminj30 | pairhist-v3-s42-from-dminj30 | v3 | 7 | 是 | 是 | — | — |
| v3-from-dminj30 | pairhist-v3-s43-from-dminj30 | v3 | 7 | 是 | 是 | — | — |
| v3-from-dminj30 | pairhist-v3-s44-from-dminj30 | v3 | 7 | 是 | 是 | — | — |

## 设计要点

> 来源：`docs/v3注意力评分诊断_20260924.md`、`docs/当前状态与优化方向.md`、
> `docs/模型优化历程与下一步实验.md`、本目录 `config.yaml`。共享的测量规范
> 与 PPO 机制不在此重复，见文末「相关文档」。

**归属**：本模型 = `pair_path_attention` 主导的 actor（`attention_only: true`，
即去掉基础 scorer 残差，候选弧分数完全由注意力给出）。`arc_query: true`、
`num_heads: 4`、`max_hops: 4`。顺序全局匹配采样器（双端约束 + STOP）**未改动**。

### 已知实现 bug（已修）

`PairPathAttentionScorer.forward()` 在候选边不属于任何正权重请求路径时，
先算了备用分数，随后**仍**执行 `result / norm.clamp_min(1e-6)`；此时 `norm=0`，
备用分数被放大一百万倍，旧 checkpoint 的路径外分数约 `-1648`，策略基本不选这些边。
旧 BC 打分头 ReLU 输出 **99.45% 为零**。

- 只改前向、用**同一批旧权重**重评：留出均值 `0.7211 → 0.7528`（+3.2 点）
- 修复后同样新训一天 BC 的边分数 p10/p90 约 `-0.67 / 2.52`（原先 p10≈p50≈p90）
- 最小回归测试 `tests/test_v3_attention_fallback.py`

### 结构限制：注意力在**站对内部**排不出序

同状态诊断（80 个留出状态、13366 个候选查询）：within-pair 方差仅占 **15%**；
消融 attention context 对分数的影响只有 `arc_emb` 的 **1/2400**。
后果：attention **结构上无法给同一站对内的候选弧排序**，而匹配采样器要的正是这个排序。
`arc_query` 就是为此加的（query = `[arc_emb, 该弧物理边向量, 该 pair demand]`，7H），
但注释里也承认逐弧离散度仍只有 `2.3e-4`。

### 动作级补诊（2026-09-25）

150 个留出专家状态中命中 **6559/8854** 条专家边。2385 条错选边里，
**2004 条发生时仍有可行的专家边**；提前 STOP 只漏掉 **4** 条
⟹ **STOP 不是问题**，问题是中途选错边。固定 BC 起点把 PPO 延长到 30 轮、
actor LR 从 `1e-5` 提至 `3e-5`，A 窗最终仅 `0.7590/0.7594`、B 窗 `0.8426/0.8423`，
30 轮后专家边命中几乎不变 ⟹ 单靠加大学习率无效。
原始数据：[诊断报告](../../../server_results_node342/diagnostics_v3_20260925/诊断报告.md)。

### 关闭的线索（不要再走）

- **需求传播消融不转正**：`fuse_demand_to_node: false` 与 v3 在统计上**不可区分**
  （BC 后 0.7512 vs 0.7512，PPO 后 0.7527 vs 0.7515）。旧读数的 +1.9/+3.0 点
  未能复现，方向已反转；保持显式配置，不作为默认。
- **「让模型看见需求」四连败**：`demand_edge` / `v1` / `v2` / `DemandResidual` 均测不出。

### 未完成

**注意力消融链**（已规划、未开跑）：温度实验（2 vs 1）→ 移除 cross-attention 独立消融
→ 四头饱和判定。基线取普通 v3。四头最大注意力权重中位数
`0.969 / 0.999 / 0.502 / 0.320`（两头尖锐、两头分散）；**尖锐不等于坏，需消融判定**。

### 相关文档（共享内容，不在此复制）

| 主题 | 文档 |
|---|---|
| 测量规范（配对、种子数、两个 regime） | [`测试规范.md`](../../../docs/测试规范.md) |
| 产物格式与环境指纹 | [`实验产物规范.md`](../../../docs/实验产物规范.md) |
| 原始数字与推导（只增不删） | [`训练诊断记录.md`](../../../docs/训练诊断记录.md) |
| 全项目现状与下一步 | [`当前状态与优化方向.md`](../../../docs/当前状态与优化方向.md) |
| v2→v3→v4 演进时间线 | [`模型优化历程与下一步实验.md`](../../../docs/模型优化历程与下一步实验.md) |

## 备注

**env_fp `9091810f`** = 删 Stockholm 后的现环境（专家锚 0.7674；删前为 0.6979，旧值不可引用）。


