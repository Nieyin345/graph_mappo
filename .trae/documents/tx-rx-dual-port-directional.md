# 严格 Tx/Rx 双端口方向链路重构

## Context（为什么做这个改动）

现有 QKD 调度模型是"无向共享池 + 每节点每时隙一条链路"：动作空间每节点选 1 个邻居，resolver 的 4 种 matching 都保证每个节点至多出现在 1 条被激活的边；MILP 里也有对应的匹配约束 `Σ incident x ≤ 1`。

用户希望更符合物理：**每个节点有固定的 1 个发射端(Tx) 与 1 个接收端(Rx)**。发射端去连接其他节点的接收端形成一条有向链路；因此每节点每时隙可同时有 **一条"我发"的链路 + 一条"我收"的链路**（Tx-出≤1、Rx-进≤1），两条链路对端必须不同。好处是节点当期能给两条链路的密钥池生密钥，作为中继骨架，提升可达成功率。

- 密钥池仍按**无向 pair** 共享，可信中继/消耗端**保持不变**（用户明确：消耗不影响）。
- 方向只决定"哪个物理口在用"，不改变密钥池归属。
- **默认即新语义，不做向后兼容开关**（当前仓库无已训练 RL checkpoint，动作空间改变本就需要重训）。

## 目标模型

一条有向链路 `u→v` = 使用 u 的 Tx 指向 v 的 Rx。被激活 ⇔ 双方达成（u 的 Tx 选择 v，且 v 的 Rx 开放给 u）。
每个节点动作 = `(tx_target, rx_source)`：
- `tx_target ∈ 邻居 ∪ {IDLE}`：产出有向边 `me → tx_target`；
- `rx_source ∈ 邻居 ∪ {IDLE}`：产出有向边 `rx_source → me`。

每个节点每时隙的容量约束 = **Tx-出 ≤1 且 Rx-进 ≤1**，且不能 `u→v` 与 `v→u` 同时被选（对端不同）。
密钥生成：`v` 到 pair `(u,v)` 池的生密钥率 = 该时隙 `u→v` 与 `v→u` 两向激活之和 × 速率 × slot。

## 关键实现手法：双部图匹配

resolver 与 policy 采样的最快捷径 = 把"单部匹配(每节点≤1)"换成**双部匹配**：
- 左部 = 发送节点，右部 = 接收节点；
- 一张有向候选 `u_L → v_R`（u≠v）；
- 双部匹配天然保证 Tx-出≤1、Rx-进≤1、以及不会同时选 `u→v` 与 `v→u`（冲突时由分数裁决）。

## 改动清单

### A. 数据结构 / 动作形状
- `qkd_rl/core/types.py`：`Edge` 保持无向 pair 不变（池仍按 pair）。`KeyRequest` 不变。
- `qkd_rl/env/action_space.py` `NodeActionSpace`：
  - `actions` 对外改为 `dict[str, tuple[str,str]]`（`(tx_target, rx_source)`，各可 IDLE）。
  - 新增 `tx_candidates_for_node` / `rx_candidates_for_node`（均 = 邻居 + IDLE）。
  - 保留 `candidates_for_node`、`action_to_edge`（无向归一，供解析/池用）；新增"有向边 ↔ (pair edge_id)"映射。

### B. 观测 / mask / 图构建
- `qkd_rl/env/masks.py` `ActionMaskBuilder`：`build()` 输出拆成每节点 `(tx_mask, rx_mask)`；`_cand_offsets/_splits/_flat_pos/_edge_pos` 按方向各自维护；边缘掩码按**有向** `(u→v)` 记账，`edge_windows` 速率仍按无向 edge_id 取。
- `qkd_rl/env/graph_builder.py`：`action_candidates[node]` 存 `(tx_cands, rx_cands)`；`_active_edges` 沿用；`_mask_total` 相应翻倍。

### C. action_resolver 4 种 mode（核心）
统一把"每节点 used ≤1"改为两套容量 `used_tx`（其 Tx 已被占用）、`used_rx`（其 Rx 已被占用），每条候选为**有向** `(u→v)`：
- `_resolve_mutual_choice`：有向候选 `u→v` 生效 ⇔ u 的 tx_target==v 且 v 的 rx_source==u；命中后 `used_tx∪{u}; used_rx∪{v}`。
- `priority_matching` / `max_weight_matching`：候选改成有向；用**双部图** `nx.max_weight_matching`（左=发送、右=接收），权重用既有 edge_score（一个有向边一个分）。`_prune_matching_candidates` 按 (u,tx / v,rx) 分别计数。
- `greedy_rate_matching`：同上改用 `used_tx/used_rx` 桶的有向贪心。
- `_greedy_match`、`_edge_is_legal` 适配（src 检查 Tx 位、dst 检查 Rx 位）。

### D. env.step / qkp
- `qkd_rl/env/env.py` `step()`：解包 `(tx_target, rx_source)`；`resolved.activated_edges` 为**有向边列表**；`_generate_keys` 里 `qkp.add_keys` 仍按 pair 的 edge_id（无向归一）——`LinkQKPPool` **零改动**。`_active_edge_ids/_baseline_edge_scores` 遍历有向边即可。

### E. policy 匹配采样（RL）
- `qkd_rl/algos/policy.py` `_sample_matching` / `_matching_log_prob_entropy`：`used_nodes` → `used_tx / used_rx`；合法候选需 `src∉used_tx and dst∉used_rx`；选中后占用 `src` 发射位与 `dst` 接收位。`matched_edges` 存**有向** `u→v`；`actions[node]` 回填 `(tx=其发射目标或 IDLE, rx=指向其者或 IDLE)`。
- `PolicyStep.actions` 类型由 `dict[str,str]` 变 `dict[str, tuple[str,str]]`。

### F. 模型 actor（RL）
- `qkd_rl/models/graph_mappo.py`：actor 每节点输出 1 行 → **2 行**（tx 行 + rx 行）；`_full_lengths/_cand_offsets` 按 tx/rx 两段摊平，`max_n` 取两者最大。`edge_scorer` 直接复用已存在的**有向** edge 行（`pos_t` 取 `src→dst` 那份）；`_edge_score_map` 按**有向** `(u→v)→score`。`actor.logits` 字典 value 为 `(tx_cat, rx_cat)`。

### G. 配置
- 新增 `features.action.roles: [tx, rx]`（默认即 2）。
- `action_resolver.*`：新增 `directional: true`；`max_candidates_per_node/max_candidate_edges` 语义改为按**有向候选**计数。
- 新增 `env.dual_role_ports: true`（写死默认开）。

### H. MILP 最优基线
- `qkd_rl/baselines/receding_horizon_milp.py` `solve_window()`：
  - 激活变量 `x[e,tau]`（无向边、二进制）改为**有向弧** `a[u→v,tau]`（二进制）。
  - 替换 L501-513 的匹配约束为：
    - `Σ_v a[u→v,tau] ≤ 1`（Tx-出）
    - `Σ_v a[v→u,tau] ≤ 1`（Rx-进）
    - `a[u→v,tau] + a[v→u,tau] ≤ 1`（对端不同）
  - 无向 pair 池的库存等式：`s[e]` 的生成项 = `rate *(a[u→v]+a[v→u]) * slot`（两向累加）；`row[xv]` 系数相应改为两向和。
  - 其余（availability、容量、serve、path flow、final_inventory）保留。

### I. 受影响调用方（需一并适配动作形状）
`scripts/train_graph_mappo.py`、`supervised_*.py`、`run_baselines.py` 以及 `greedy_*` / `random` 等基线策略、`rollout_buffer` / `rollout_workers`、`tests/test_action_*.py`、`test_model_forward.py`。旧的 `.pth` checkpoint 结构不兼容，需重训。

## 落地顺序（降低风险）
1. C+E 的双部匹配 + 两套 used（先让 env 能跑有向动作、resolver/采样一致）。
2. B/D/A 的动作形状、mask、graph_builder、env.step。
3. F 模型双行 logits。
4. G/H 配置与 MILP。
5. 收敛受影响的测试/基线脚本与新增单个有向边 smoke 测试。

## 风险
- **reward/metrics 激活边统计口径**：每节点最多 2 边，switch_count/keep_active_count 语义变化，需复核 reward 里"激活边数"统计。
- **大量调用依赖旧 `dict[str,str]` 形状**，一次全改（无兼容开关），需修测试与脚本。
- **MILP 求解规模**：有向弧变量约为原来 2 倍且加对端=1 约束，单窗口规模上升，注意 `max_edges/max_requests` 与超时。
- 双部匹配下 `u→v.sum`+`v→u`≤1 只在 MILP 显式加；resolver/policy 由双部匹配隐式保证。

## 验证
1. **单元/环境冒烟**：构造 env，用固定 `(tx_target, rx_source)` 动作走 `step()`，assert `activated_edges` 全是被激活的有向边、每节点 Tx-出≤1 且 Rx-进≤1、对端不重复、密钥按 pair 入池。
2. **mask 一致性**：`test_action_masks.py` 通过；非法动作（同节点双 Tx / 同 pair 冲突）被拒。
3. **policy 采样**：单 obs 采样一轮，断言每个节点 `(tx, rx)` 的 Tx/Rx 容量与 matched_edges 一致；`evaluate_actions` 的 joint log prob 可用有向 matched_edges 重建。
4. **RL 冒烟训练**：`python scripts/train_graph_mappo.py --num-updates 5` 可跑完、无形状报错。
5. **MILP**：`solve_window` 在 seed 7 单窗口求解，打印有向弧 RD 段，断言生成量 = `rate*(a[u→v]+a[v→u])*slot`，且能找到比"每节点1链路"更高/相当的最优服务量。
6. **对比**：用新配置跑 `generate_milp_demos.py` 与 `eval_long_horizon.py`（同一批 validation episode、seed 7、240 步），给出 MILP `optimal_sr` 与后续 RL 的 served/arrived 对照。