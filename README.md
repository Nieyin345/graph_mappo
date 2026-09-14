# QKD-SAGIN 生产端调度（Graph-MAPPO）

面向 QKD-SAGIN（量子密钥分发-天地一体化网络）生产端的**密钥生成链路调度**项目。
三层 FSO-QKD 网络：30 地面站（GS）、30 高空平台（HAP）、30 卫星（SAT），1978 条候选物理链路；
每个时隙（1 分钟）智能体要为每个节点决定一条密钥生成链路，约束是**双端口**（Tx-out ≤ 1、
Rx-in ≤ 1、同一对链路不能双向同时开）。链路速率 / LOS 全部从 H5 数据集读取（`H5RateProvider`）。

项目核心是**一条启发式预训练 + 强化学习微调**的训练管线：

```text
BFS 需求扩散启发式（PG-Phased 专家）
        │  行为克隆（BC 预热）
        ▼
Graph-MAPPO 强化学习（warm-start 自 BC 权重）
```

---

## 1. 强化学习模型：Graph-MAPPO

**架构**：共享 GNN 编码器（GraphSAGE，3 层，128 维）→ 共享 actor（edge scorer）→ 全局 critic。
`mixed` 模式把节点/物理边/需求边一起编码，需求信息沿物理边传播；critic 用 typed-mean 池化输出
每时隙一个全局 value。

**动作空间：全局匹配（关键设计）**。早期版本是"每个节点独立采样一条边 + resolver 贪心消解"，
PPO 优化的提议分布与环境真正执行的匹配不一致，成功率长期停在约 30%；当前版本：

1. actor 对每条合法**定向弧** `(tx_target, rx_source)` 打分；
2. 策略从全部合法弧出发，每步只考虑两端点仍空闲的弧，按 softmax 采样一条（含 STOP 选项），
   直到没有可用弧——**采样的匹配就是环境执行的匹配**；
3. PPO 优化的是该匹配的联合 log-prob（`_matching_log_prob_entropy_fast` 向量化实现）。

**训练**：多进程 rollout（worker 池，权重+温度随任务下发保证探索温度同步）→ GAE →
PPO（clip、advantage 归一化、KL 早停、按角色梯度裁剪）。

**RL 效果（实测）**：
- 固定场景（day 0 + 固定请求种子）20 轮：**0.728 → 0.819**，超过启发式（0.7765）✓
- 标准验证协议（5 种子）：30 轮评估均值 **0.713 ± 0.033**，无上升趋势（详见 §4 困境）

## 2. 启发式算法：BFS 需求扩散 + 分阶段路径调度

启发式由两部分组成：一个**边打分器**（谁值得开）和一个**路径级调度器**（按什么顺序、开整条路）。
它同时是强化学习的**行为克隆专家**。

- **需求扩散重要性（relay importance）**：对每个等待中的请求，用其两端 GS 当起点，在当前合法
  可见物理边构成的图上做 BFS，得到两端到各节点的最短跳数，据此给中继节点/链路打分；
- **分阶段路径调度（PG-Phased）**：三阶段策略——①存量密钥网络（先用已持有的密钥库存服务）、
  ②混用库存的辅助路径、③全新路径（生成新密钥服务）；配合 ServeProbe 路由按瓶颈跳部分服务。
- 网络原始生成能力约 4.7×10⁶ 密钥/时隙，而需求约 6–10×10⁴ 密钥/时隙——约束不在产量，
  而在**稀疏时变拓扑上的可达性**（每时隙仅约 9.6% 的 (链路,时隙) 组合速率非零）。

**启发式效果（实测）**：`path_score_greedy_phased` 成功率 **0.8104 ± 0.0575**（标准协议），
是最强的非学习方法，比第二名 `greedy_relay`（0.435）高 1.86 倍。

## 3. 效果对比一览（实测）

| 策略 | 协议 | 成功率 |
|---|---|---|
| 启发式 `path_score_greedy_phased` | 标准验证（5 种子，确定性） | **0.8104** |
| 启发式 `path_score_greedy_phased` | 固定场景（12 种子） | **0.7765** |
| BC 预热权重（克隆启发式） | 标准验证（5 种子，确定性） | **0.7581** |
| BC 预热权重 | 固定场景（12 种子，确定性） | **0.7280** |
| RL（固定场景 20 轮） | 固定场景 | **0.819**（从 0.728 涨起） |
| RL（全局训练 30 轮） | 标准验证 | **0.713 ± 0.033**（无趋势） |
| 随机策略 | — | ≈ 0.18 |

要点：
- BC 克隆会丢 5.2 个点（0.8104 → 0.7581），但把 RL 起点从"随机 ≈0.18"抬到"≈0.73–0.86"；
- **RL 在固定场景能显著超过启发式，但全局训练卡住**。

## 4. 当前困境：RL 全局训练成功率没有提升

全局训练（30+ 轮）在标准验证协议上出现**成功率停滞**：

- 6 次评估 = 0.706 / 0.715 / 0.759 / 0.711 / 0.729 / 0.659，均值 0.713 ± 0.033，
  与"纯噪声围绕 0.713 波动"一致；低于 BC 起点（0.758）与启发式（0.810）；
- 训练侧指标全平：8 局采样均值 0.858 → 0.842，reward 92 → 91，无趋势；
- **critic 全程健康**（corr(V,R) = 0.89），说明问题在 actor 侧；
- 固定场景 smoke 验证同样观察到：60 轮后 RL checkpoint 同口径评估 0.8574，
  **低于** BC 基线 0.8622，且策略熵从 3.33 升到 4.01（向随机化漂移）。

**假设方向**（待验证）：探索温度过高（1.2 起步）把策略推向高熵；PPO 每轮更新量太小
（2880 步 ÷ minibatch 1024 ≈ 3 次梯度/轮）学不动；BC 起点已接近局部最优，继续 PPO 反而
被高熵样本轻微污染。当前正在做：降低探索温度、加大每轮更新量、同口径评估验证。

## 5. 代码结构

```text
qkd_rl/                  # 核心库：env / rl / link / data / baselines / evaluation
configs/                 # 所有 YAML 配置（索引见 configs/README.md）
scripts/rl/              # 训练入口：BC 预训练、MAPPO 训练、评估
scripts/baselines/       # 启发式 / 基线
scripts/milp/            # MILP 最优解与演示数据生成
tests/                   # pytest 测试
docs/                    # 算法说明与汇报（含全部实测数字与复现协议）
```

## 6. 快速开始

```powershell
# 生成速率归一化参考值（缺失时 p99 退化为常量 10.0）
conda run -n pytorch python scripts/estimate_rate_stats.py

# 冒烟测试环境
conda run -n pytorch python scripts/smoke_test_env.py

# 跑测试
conda run -n pytorch python -m pytest
```

### 训练管线

```powershell
# ① 启发式专家预训练（行为克隆，默认边收集边训练）
conda run -n pytorch python scripts/rl/supervised_train_pg_phased.py `
  --run-name supervised_pg_phased

# ② Graph-MAPPO 强化学习（从 BC checkpoint warm-start）
conda run -n pytorch python scripts/rl/train_graph_mappo.py `
  --mode curriculum --run-name exp1 `
  --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt `
  --device cuda

# ③ 固定场景 smoke 验证（调参 / 验证奖励设计）
conda run -n pytorch python scripts/rl/train_graph_mappo.py `
  --configs train_mappo_smoke.yaml `
  --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt `
  --run-name rl_smoke_day0

# ④ 同口径评估 checkpoint（与训练 success_rate 直接可比）
conda run -n pytorch python scripts/rl/eval_fixed_scenario.py `
  --checkpoint outputs/rl_smoke_day0/checkpoint_final.pt `
  --steps 1440 --seeds 7-14 --device cuda
```

### 数据与产物（均不上传到仓库）

```text
dataset/global/*.h5     # 全年链路物理数据（525600 分钟 × 1978 链路）
outputs/                # 训练产物：轨迹 / BC 权重 / RL checkpoint / 指标
  trajs_pg_phased/      #   启发式引导数据（BC 训练用轨迹，pkl）
  supervised_pg_phased/ #   BC 权重
  rl_smoke_day0*/       #   RL checkpoint 与 metrics.jsonl
weather/                # 天气数据
```

> 仓库只保留**代码 + 配置 + 文档 + 测试**。所有 `.h5`、`*.pkl` 轨迹、`*.pt` checkpoint、
> `outputs/`、`dataset/global/*`（除 `rate_stats.json` 参考值）、`weather/` 均由 `.gitignore` 排除。

## 7. 配置索引

重要配置见 [configs/README.md](configs/README.md)。要点：

- `configs/global.yaml`：全局训练/验证窗口与共享请求种子。
- `configs/features.yaml`：节点、物理边、需求边特征开关与解析维度。
- `configs/env_small.yaml` / `configs/env_full.yaml`：场景规模、链路 QKP 容量、请求、路由、奖励、resolver 模式。
- `configs/graph_mappo.yaml`：encoder、共享 actor、critic、masked categorical 设置。
- `configs/train_mappo.yaml`：rollout 长度、GAE/PPO 超参、学习率、日志与 checkpoint 间隔。
- `configs/train_mappo_smoke.yaml`：固定场景 smoke 训练（固定 day + 固定请求种子）。
- `configs/train_profiles.yaml`：训练模式（`continuous`、`fixed_day`、`curriculum`、`demand_edge`）。
- `configs/baselines.yaml`：启发式基线开关与参数。

## 8. 文档

- [configs/README.md](configs/README.md)：配置索引
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md)：完整训练操作手册
- [docs/启发式与强化学习算法说明.md](docs/启发式与强化学习算法说明.md)：算法原理、基线、效果对比、RL 训练过程（含全部实测数字）
- [docs/BFS引导强化学习预训练汇报.md](docs/BFS引导强化学习预训练汇报.md)：预训练汇报
