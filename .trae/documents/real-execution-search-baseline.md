# 真实执行最优解搜索基线（替代失效的 MILP）

## Context / 背景

用户需要在 QKD-SAGIN 场景下找到一个**真实执行成功率最高的链路选择策略**，作为 RL 对照的"准最优解"参考。

已确认的失败事实（同一验证场景 seed 7, t=496422, 240 槽）：

| 模型 | plan SR | 真实执行 exec SR |
|---|---|---|
| MILP 可执行（锁单路径） | 0.090 | 0.017 |
| MILP 上界（多路径自由流） | 0.995 | 0.194 |
| Greedy V3（现实基站） | — | **0.287** |

**MILP 失败的根因**：它用"密钥可自由流动"的松弛模型规划，激活的边太少；而真实 env 服务一条请求需要 **src→dst 每条 hop 都有正库存的全连通路径**。计划时以为"流可调配"，真实执行时密钥堆在边上、请求找不到连通路径，吃不到。greedy 反而因广撒网激活形成连通结构而更强。

**关键前提（已验证）**：env 确定性 —— `self.rng` 只在 reset 时抽起始时隙（[env.py](d:\destop\work_space\learning_space\论文\QKD-SAGIN生产端调度\qkd_rl\qkd_rl\env\env.py#L51)），step 阶段路由/请求流完全固定。**给定时隙动作序列，env 重放结果唯一**。这使"把 env 当黑盒评估器做搜索"成为可能且评估无噪声。

## 目标与方法

新方法范式：**不再建模求解，直接在真实 env 上搜索链路选择动作序列**，使重放到成功率最大化。

环境评估机制（复用，不修改）：
- [env.py](d:\destop\work_space\learning_space\论文\QKD-SAGIN生产端调度\qkd_rl\qkd_rl\env\env.py)  `step(actions)` 推进一个时隙
- [routing.py](d:\destop\work_space\learning_space\论文\QKD-SAGIN生产端调度\qkd_rl\qkd_rl\env\routing.py#L212-L237) `partial_consume_for_request` 瓶颈 hop 单路径服务
- [request.py](d:\destop\work_space\learning_space\论文\QKD-SAGIN生产端调度\qkd_rl\qkd_rl\env\request.py#L148-L207) EDF 逐条服务
- [greedy_relay_diffusion.py](d:\destop\work_space\learning_space\论文\QKD-SAGIN生产端调度\qkd_rl\qkd_rl\baselines\greedy_relay_diffusion.py) `GreedyRelayDiffusionPolicyV3.score_edges` 多因子边评分（rate/importance/completion/keep/switch），作为候选动作生成器和热启底座，**必须复用**

## 推荐方案（等待用户确认方向后二选一落地）

### 方案 B（推荐，改动小）：rollout 前瞻 + local search / hill climbing
1. **热启**：用 greedy V3 跑完整段，得基准动作序列与 exec SR（0.287）。
2. **逐槽 local search**：对每个时隙 t，用 `score_edges` 生成该槽多组候选（扰动权重 / 取 top-k 变体 / 强制额外激活边），或尝试"多激活一条未在基准里的高 completion 边"。
3. **前瞻评估**：对每个候选，从 clone 的 env 状态前瞻 rollout 后续 H 槽（H≈10~30，因确定性无噪声），用累计 `served_keys` 打分，替换若更优。
4. **迭代**：轮次间传入上一轮 greedy 解作为新底座，逐步爬山（约数轮）。
5. **输出**：最优动作序列 + 真实重放 SR，存为新基线，供 RL 对比。

可行性要点：env 克隆需保存 `qkp.levels/batches/positive`、`requests.pending`、`request_history`、`last_activated_edges`、`t`、`_prev_waiting_keys` —— 建议封装一个 `clone_env_state`/`rollout` 工具复用。

### 方案 A（更硬核）：进化/遗传算法直接搜序列
- 个体 = 整段每槽激活边集（或每槽参与匹配的候选边）；适应度 = 真实重放 SR
- greedy 解作初始种子，交叉/变异，多代逼近连通结构最优
- 计算量大（每代 N×240 重放），但能更强地逼近全局最优。

### 方案 C（最快）：以 greedy 为 expert，直接引导 RL
- 不追求上界，把 greedy V3 的 per-slot 边级选择作为监督信号喂给 RL（已有 supervised_train_milp_demos 的 BC 管线可改）。
- 最稳，但不算"找真实最优"。

## 待修改/新增文件
- 新增 `qkd_rl/baselines/...search.py`（或 `scripts/_search_real_opt.py`）：实现 rollout + clone_env + local search / GA。
- 新增 `scripts/find_real_optimal.py`：CLI，参数（seed/start/steps/horizon/rounds/policy），跑搜索并打印真实 SR 曲线，对比 greedy。
- 复用 `_diag_compare.py` 的环境构建逻辑（demo seed 对齐）。

## 验证
1. 先在 seed 7 / demo 场景（t=496422, 240 槽）跑搜索。
2. 断言：搜索所得序列真实重放 SR **> greedy 的 0.287**。
3. 若方案 A/B 提不上去，回退方案 C（greedy expert 引导 RL）。
4. 用 `_diag_compare.py` 同口径复核。

## 待用户确认
- 选 A / B / C 哪个方案落地。
- 时间/算力预算（决定 GA 种群规模与 rollouts 数）。