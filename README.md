# QKD RL

FSO-QKD link scheduling code scaffold.

Current implementation target:

- Training reads link rates / LOS exclusively from the H5 dataset via `H5RateProvider` (no mock/fake data generator).
- QKP is stored per link with YAML-configurable capacity.
- GS-GS communication demand is encoded as non-action demand edges in the graph.
- Demand edges store only currently pending requests. Pending demand is bucketed
  by already-queued time (`wait_bucket_count`), served/expired requests are
  removed immediately, and the bucket vector is converted into physical-link
  relay importance with an optional `wait_decay_tau`.
- Demand amount/count features use `log1p` normalization; deadline-left features are scaled to `[0, 1]`
  by `requests.deadline_steps`, so accumulated demand does not dominate the input scale.
- Two model modes: `mixed` keeps the old node/edge GNN fusion; `demand_edge`
  scores physical edges directly from edge features plus relay importance and
  only keeps demand-side messages for the critic.
- `ActionResolver` supports multiple YAML-selected matching modes.
- Tests exercise the same `H5RateProvider` code path against a small generated H5 dataset (`tests/helpers.py`).
- `qkd_rl.env.factory` is the package-level environment assembly entry.
- `qkd_rl.models.graph_mappo` contains the pure-PyTorch GNN actor-critic.
- `qkd_rl.algos.policy.MAPPOPolicy` samples a **global matching** action directly: the actor scores every
  legal edge, a sequential sampler picks disjoint edges until no free endpoints remain, and the executed
  action is exactly that matching. PPO optimizes the joint log probability of the sampled matching.
- `qkd_rl.algos.mappo_trainer` implements the full MAPPO loop: rollout -> GAE -> PPO update -> checkpoint.
- `qkd_rl.algos.mappo_trainer` supports a curriculum schedule that starts with short episodes and gradually
  lengthens to the full-day rollout.

Important config files: see [configs/README.md](configs/README.md) for the full
index (every yaml's purpose, load order, and who references it). Highlights:

- `configs/global.yaml`: global training/validation windows and shared request seeds.
- `configs/features.yaml`: node, physical edge, and GS-GS demand edge feature switches and resolved dimensions.
- `configs/env_small.yaml` / `configs/env_full.yaml`: scenario size, link-level QKP capacity, requests, routing, reward, and action resolver mode (small vs full scale).
- `configs/graph_mappo.yaml`: encoder, shared actor, critic, and masked categorical settings.
- `configs/train_mappo.yaml`: rollout length, GAE/PPO hyper-parameters, learning rates, logging and checkpoint intervals.
- `configs/train_profiles.yaml`: all training modes (`continuous`, `fixed_day`, `curriculum`, `demand_edge`).
- `configs/baselines.yaml`: heuristic / ILP baseline switches and parameters.

Run from this folder:

```powershell
conda run -n pytorch python scripts/smoke_test_env.py
conda run -n pytorch python -m pytest
```

## 训练（Graph-MAPPO）

训练入口是 `scripts/train_graph_mappo.py`，默认加载 `default -> rate_provider -> features -> env_full -> graph_mappo -> train_mappo` 配置（数据只从 H5 读取）：

```powershell
# 连续训练（默认）
conda run -n pytorch python scripts/train_graph_mappo.py `
  --mode continuous --run-name exp1 --seed 7 --device cuda

# 课程训练（短局起步，逐步加长到全天）
conda run -n pytorch python scripts/train_graph_mappo.py `
  --mode curriculum --run-name exp1 --seed 7 --device cuda

# demand_edge 模式（需求分桶 -> 物理边 relay 权重 -> Actor 直连）
conda run -n pytorch python scripts/train_graph_mappo.py `
  --mode demand_edge `
  --run-name exp1_demand_edge --seed 7 --device cuda

# 覆盖更新次数
conda run -n pytorch python scripts/train_graph_mappo.py `
  --mode curriculum `
  --run-name exp1 --num-updates 200 --seed 7 --device cuda

# 从 checkpoint 续训
conda run -n pytorch python scripts/train_graph_mappo.py `
  --mode curriculum `
  --run-name exp1 --checkpoint outputs/exp1/checkpoint_update_0100.pt
```

训练循环由 `qkd_rl/algos/mappo_trainer.py` 驱动：

- `RolloutBuffer`（`algos/rollout_buffer.py`）按 episode 存全图观测，每个时隙一条样本；`compute_gae`（`algos/gae.py`）在 episode 结束时计算 return/advantage。
- 每个样本一次完整 GNN 前向：共享 GNN 编码器输出节点/边嵌入，actor 的 edge scorer 给出每条合法边的分数，
  策略按分数顺序采样一个全局匹配；全局 critic 给每个时隙一个 value，所有节点共享同一 advantage。
- PPO 更新包含 clip、advantage 归一化、KL 早停、梯度裁剪；actor 与 critic 使用不同学习率（一个 optimizer 两个参数组）。
- 课程机制：`train.curriculum.stages` 按 update 数切换 `rollout_steps / episodes_per_update`；
  短局阶段使用 `value_target: gae`，用 Critic bootstrap 接续局外未来，避免短视。
- 输出目录为 `outputs/<run_name>/`：`resolved_config.yaml`、`metrics.jsonl`、`checkpoint_update_XXXXXX.pt`、`checkpoint_final.pt`。
- `env.reset(seed=...)` 会重播种请求流；`episode_start_mode: random_day` 时每个 episode 从随机日边界开始。

## 动作表示

早期版本是“每个节点独立采样一条边 + resolver 贪心消解”，PPO 优化的提议分布和环境真正执行的匹配不一致，
导致成功率长期停在约 30%。当前版本改为全局匹配动作：

1. `mixed` 模式用 GNN 编码后对每条合法物理边输出分数；`demand_edge` 模式直接由
   物理边特征 + relay 重要性经 MLP 输出分数；
2. 策略从全部合法边出发，每步只考虑两端点仍空闲的边，按 softmax 采样一条，直到没有可用边；
3. 采样序列的联合 log-prob / entropy 作为该步动作的概率，PPO 优化的是“环境实际执行的匹配”；
4. 确定性评估时按分数 argmax 依次选取，得到零冲突的调度。

## 特征归一化

- 链路速率：`log_p99` 归一化（`rate_stats.json`）。
- 需求金额（节点 pending demand、recent_demand 窗口、需求边等待分桶量）：`log1p(x)/log1p(1e6)`。
- 需求计数（queue pressure、pending_count）：`log1p(x)/log1p(100)`。
- deadline 剩余：按 `requests.deadline_steps` 归一到 `[0,1]`，调低排队时长时无需改特征代码。
- 需求边等待分桶：`deadline_steps` 按 `wait_bucket_count` 均匀分桶，桶内剩余量
  用 `log1p` 归一化；物理边 relay 重要性是桶权重的指数衰减加权和，再做 `log1p`。

完整操作手册见 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。

## H5 速率数据接入（H5RateProvider）

`build_global_tensor.py` 把全年链路物理数据写到 `dataset/global/`：

- `link_data.h5`：`link_registry`（link_id/node_u/node_v/link_type）、`distance`、`los`(int8)、`zenith`、`k_max`，形状均为 `(T, L)`，`k_max` 单位 bps
- `node_registry.csv` / `link_registry.csv` / `build_metadata.json`：节点与链路注册表

`qkd_rl.link.rate_provider.H5RateProvider` 已实现对应读取，通过配置切换：

```yaml
# configs/rate_provider.yaml
rate_provider:
  provider: h5            # training/evaluation read the H5 dataset only
  h5:
    dataset_dir: dataset/global
    availability_source: los_and_rate   # los | rate | los_and_rate
    out_of_range_policy: zero           # zero | pad_last | raise
    node_name_map: {}                   # env node id -> H5 node name
```

- 场景节点名与 H5 `node_registry.csv` 的名字保持一致（默认 identity 匹配）；不一致时用 `node_name_map` 映射。
- 全年末越界：`zero`（补 0 且不可用）、`pad_last`（沿用末值）、`raise`（报错）。
- 链路类型自动映射：`HAP-GS -> gs_hap`、`SAT-GS -> gs_sat`、`SAT-HAP -> hap_sat`、`SAT-SAT -> sat_sat`。
- `scenario.mode: full` 时 `ScenarioBuilder.build_full()` 从同一份 registry 构建全规模场景，保证场景与 provider 一致。

使用示例（在项目根目录）：

```powershell
# 检查 H5 provider 接口并统计速率分布（写 outputs/rate_stats.json）
conda run -n pytorch python scripts/inspect_rate_provider.py --scenario full --samples 2000

# 全规模场景训练（需要 dataset/global/ 已生成）
conda run -n pytorch python scripts/train_graph_mappo.py --configs env_full.yaml --run-name full_exp1 --num-updates 500
```

`configs/env_full.yaml` 提供全规模场景的配套配置；加载顺序建议：

```text
default.yaml -> rate_provider.yaml -> features.yaml -> env_full.yaml -> graph_mappo.yaml -> train_mappo.yaml
```
