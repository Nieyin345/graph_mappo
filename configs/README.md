# configs 配置目录索引

本目录的 yaml 分三类：**默认加载链**（任何入口都会加载）、**场景/窗口覆盖**（训练与基线脚本显式加载）、**独立工作流**。所有文件都被代码或脚本引用，**不要随意删除**（删除会破坏引用它的脚本/测试）；如需精简请先确认引用方（见下表"主要使用者"）。

## 加载机制

- **默认链**（`qkd_rl/env/factory.py::DEFAULT_CONFIG_FILES`，任何入口/测试先加载这 6 个，按顺序合并）：
  `default.yaml → rate_provider.yaml → features.yaml → env_small.yaml → graph_mappo.yaml → train_mappo.yaml`
- **RL 训练主入口** `scripts/train_graph_mappo.py`：
  默认链 → 显式覆盖 `env_full.yaml` → `global.yaml`（训练/验证窗口）→ `train_profiles.yaml`（`--mode` 选训练模式）→ 可选 `--configs` 追加
- **基线** `scripts/run_baselines.py`：`global.yaml`（`--config`）+ `baselines.yaml`
- **监督预热** `scripts/supervised_train_bfs_greedy.py`：`supervised_train.yaml`

## 文件清单

| 文件 | 用途 | 加载位置 | 主要使用者 | 备注 |
|---|---|---|---|---|
| `default.yaml` | project / seed / runtime 基础 | 默认链① | 全部入口 | 必留 |
| `rate_provider.yaml` | H5 数据源、可用性口径、越界策略 | 默认链② | 全部入口 | 必留 |
| `features.yaml` | 节点/物理边/需求边特征开关与归一化、relay importance | 默认链③ | 全部入口 | 必留 |
| `env_small.yaml` | 小场景（4 GS/1 HAP/1 SAT）：环境、请求流、QKP、路由、奖励 | 默认链④ | 测试与默认实验 | 与 `env_full.yaml` 是同一组键的不同取值（二选一场景），不是重复 |
| `graph_mappo.yaml` | 模型结构：GNN 层数/隐藏维、actor/critic、mask 分布 | 默认链⑤ | 训练/评估 | 必留 |
| `train_mappo.yaml` | 训练超参：rollout、GAE、PPO、optimizer、logging | 默认链⑥ | RL 训练默认 | `--mode` 时会被 profile 覆盖 |
| `env_full.yaml` | 全规模场景（真实 H5）：请求/奖励/QKP 标定 | `train_graph_mappo.py` 显式覆盖 | RL 训练主流程 | 注意：deadline_steps=30 与 env_small 的 960 尺度不同，参数族不同 |
| `global.yaml` | 全局训练/验证时间窗口、请求种子 | `train_graph_mappo.py` / `run_baselines.py` | 训练、基线 | 全局实验窗口，所有算法共用 |
| `train_profiles.yaml` | `--mode` 训练模式：`random_episode`/`continuous`/`fixed_day`/`curriculum`/`demand_edge` | `train_graph_mappo.py --mode` | RL 训练 | 覆盖 `train` 段（含 `value_target`、`replay_days`、PPO 参数） |
| `baselines.yaml` | 基线策略开关与参数（greedy 系列、ILP） | `run_baselines.py` / `compute_milp_reference.py` / `supervised_train_bfs_greedy.py` | 基线对比 | |
| `supervised_train.yaml` | 监督预热：BFS+greedy expert 的评估窗口与输出 | `supervised_train_bfs_greedy.py` | 可选预热工作流 | 独立工作流，与 RL 训练无关 |

## 调参提醒

- 改 `train` 相关参数前先确认生效文件：默认链加载后，`train_graph_mappo.py` 会用 `env_full.yaml`、`global.yaml`、`train_profiles.yaml`（`--mode`）**依次覆盖**——例如 `entropy_coef` 在 `train_mappo.yaml` 为 0.01，而各 profile 统一覆盖为 0.001。
- `env_small.yaml` 与 `env_full.yaml` 的请求参数族不同（deadline 960 vs 30 步），跑实验时不要混用两套参数的经验值。
- `value_target` 默认 `gae`（推荐）；`mc` 仅保留用于消融。`replay_days` 默认 0（PPO 保持 on-policy）。
