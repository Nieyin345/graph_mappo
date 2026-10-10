# 非学习策略与算法对照（2026-10-09）

这里放的是**非神经网络/非 checkpoint 的运行时策略**，不是 `model_zoo/v2–v8` 的旧版本。命名中的 `v2`、`v3` 代表某一启发式自身迭代，不应与神经网络版本号对齐。

## 代码分类

| 分组 | 文件与用途 |
|---|---|
| 随机 | `random_policy.py`：随机基线 |
| 速率/库存/需求贪心 | `greedy_rate.py`、`greedy_qkp.py`、`greedy_demand.py` |
| 端口与匹配 | `greedy_matching.py`、`_matching.py`（辅助逻辑） |
| 中继策略 | `greedy_relay.py`、`greedy_relay_diffusion.py`、`greedy_relay_v2.py`、`greedy_static_relay.py` |
| 路径专家/BC | `path_greedy.py`、`path_greedy_pair.py`、`serve_probe.py`；论文专家与行为克隆的数据生成可能使用这些组件 |
| 滚动优化 | `receding_horizon_milp.py`；注意不要与 `scripts/milp/` 中脱离环境回放的理论离线上界混淆 |

## 哪里执行？

- `scripts/baselines/run_baselines.py`：公共随机、速率/库存/需求贪心、中继 diffusion V3、匹配等基线，开关见 `configs/baselines.yaml`。
- `scripts/eval/frozen_eval.py`：冻结协议下的 `PathScoreGreedy` 专家与 RL checkpoint 对照。
- `scripts/eval/eval_expert.py`：论文专家策略对照入口。
- `scripts/baselines/eval_long_horizon.py`：**RL checkpoint 的长窗口评估入口**；虽然放在 `scripts/baselines/`，它不是单独一种启发式算法。
- `scripts/milp/`：MILP 理论上界、演示与其他离线规划；不等于可以直接与运行时端口策略比较的同条件上界。

## 对比实验的硬约束

1. 所有算法使用同一场景配置、H5 速率数据、请求种子、时间窗、端口约束和有限密钥口径；保留配置哈希与 checkpoint 身份。
2. 必须分别记录 `mean_request_fulfillment`、按比特加权的服务率、产钥/用钥量、等待时间和可执行性；不要仅报一个“成功率”。
3. 不要因为类位于 `qkd_rl/baselines/` 就认定所有策略都已加入 `run_baselines.py` 的自动注册清单。
4. 2026-10-09 已修复两个通用 RL 评估入口按 v2 强制构造模型的问题；历史评估如经过该路径，需要检查 checkpoint 类型和复评需求。
5. 保留未直接注册的历史策略源码以支持复现；新增论文对照时在对应评估脚本中显式注册并加测试，暂不移动或重命名。

完整可学习模型版本见 [`model_zoo/README.md`](../model_zoo/README.md)，统一审查报告见 [`docs/模型与对比策略目录审查_20261009.md`](../../docs/模型与对比策略目录审查_20261009.md)。
