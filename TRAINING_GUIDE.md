# QKD RL 训练与基线操作手册

本文档说明如何训练主模型、如何运行基线、结果保存在哪里，以及常用参数的含义。所有命令默认在项目根目录
`Research/qkd_rl/` 下执行，Python 环境为 conda 的 `pytorch`。

## 1. 训练主模型

### 1.1 命令模板

```powershell
conda run -n pytorch python scripts/train_graph_mappo.py `
  --mode continuous `
  --run-name my_run `
  --num-updates 30 `
  --seed 7 `
  --device cuda

# demand_edge 模式：需求等待分桶 -> 物理边 relay 权重 -> Actor 直连
conda run -n pytorch python scripts/train_graph_mappo.py `
  --mode demand_edge `
  --run-name my_run_demand_edge `
  --num-updates 30 `
  --seed 7 `
  --device cuda
```

### 1.2 参数说明

| 参数 | 含义 |
| --- | --- |
| `--mode` | 训练模式，见 `configs/train_profiles.yaml` |
| `--run-name` | 输出子目录名，结果写到 `outputs/<run-name>/` |
| `--num-updates` | 总更新次数 |
| `--seed` | 随机种子，训练与环境共用 |
| `--device` | `cuda` 或 `cpu` |
| `--checkpoint` | 从已有 `.pt` 恢复训练 |

### 1.3 配置加载顺序

不传 `--mode` 时默认使用 `continuous`：

```text
default.yaml -> rate_provider.yaml -> features.yaml -> env_small.yaml
-> graph_mappo.yaml -> train_mappo.yaml
```

然后脚本会强制叠加 `env_full.yaml`（训练只读 H5 数据集），再叠加 `train_profiles.yaml` 中对应模式。

训练模式：

- `continuous`：连续 30 天，随机起点后不 reset
- `curriculum`：短局到长局的课程安排，`value_target: gae`
- `fixed_day`：固定第 0 天跑满一天
- `demand_edge`：`demand_edge` 模式（默认关 LSTM）

## 2. 训练结果保存在哪里

所有训练输出都在 `outputs/<run-name>/`，例如 `outputs/train_curriculum_v1/`：

| 文件 | 内容 |
| --- | --- |
| `resolved_config.yaml` | 本次训练最终生效的全部参数 |
| `metrics.jsonl` | 每个 update 一行指标 |
| `checkpoint_update_000025.pt` | 按 `checkpoint_interval` 保存的模型 |
| `checkpoint_final.pt` | 训练结束时的模型 |
| `figures/<run-name>_learning_curve.png/svg/pdf` | 学习曲线 |

`metrics.jsonl` 每行字段：

```text
update, actor_loss, critic_loss, entropy, kl,
mean_reward, mean_return, mean_abs_advantage,
mean_success_rate, mean_served_keys,
rollout_s, update_s, elapsed_s
```

其中 `mean_success_rate` 是训练窗口内的即时统计，随机日起点下波动很大；判断模型真实水平请用固定日评估
（见第 5 节）。

## 3. 主要训练参数

以下参数都在 `outputs/<run-name>/resolved_config.yaml` 中可查，修改时编辑 `configs/*.yaml`。

### 3.1 `train` 段

| 参数 | 含义 |
| --- | --- |
| `num_updates` | 更新次数 |
| `rollout_steps` | 每次更新采样的步数 |
| `episodes_per_update` | 每次更新并行跑几局 |
| `value_target` | `gae`（截断局用 bootstrap，推荐）或 `mc` |
| `gamma` | 折扣因子 |
| `gae_lambda` | GAE 参数 |
| `curriculum.stages` | 课程阶段，每阶段含 `until_update / rollout_steps / episodes_per_update` |

### 3.2 `train.ppo` 段

| 参数 | 含义 |
| --- | --- |
| `epochs` | PPO 每批数据重复训练轮数 |
| `minibatch_size` | 小批量大小 |
| `clip_eps` | PPO 裁剪范围 |
| `entropy_coef` | 熵奖励系数 |
| `value_coef` | Critic loss 权重 |
| `max_grad_norm` | 梯度裁剪 |
| `target_kl` | KL 早停阈值 |
| `value_norm_tau` | 收益统计 EMA 系数 |

### 3.3 `train.optimizer` 段

| 参数 | 含义 |
| --- | --- |
| `actor_lr` | Actor/Encoder 学习率 |
| `critic_lr` | Critic 学习率 |

### 3.4 其他常用配置段

- `features.demand_edge`：`wait_bucket_count` 把过期时间均匀分成等待分桶；
  `wait_decay_tau` 控制 `exp(-wait/tau)` 衰减；`0` 表示不衰减
- `features.edge.relay_importance`：双向最短跳数候选、`max_path_links` 多跳上限、
  `hop_decay_factor` 长通道衰减、`capacity_decay_strength` 容量衰减强度；
  每条链路按自己的 `(容量-库存)/容量` 加权，累积后用 `log1p` 归一化
- `model`：`mode: mixed`（旧 GNN 混合）或 `mode: demand_edge`（Actor 直接看物理边）；
  GNN 层数、隐藏维度、Actor/Critic 结构
- `env`：环境步数、随机日起点、切换代价
- `requests`：请求到达率、金额、期限
- `qkp`：链路容量、密钥 TTL
- `reward`：默认 `mode: shaped`（dense-flow），即每步奖励 = `served_keys` 奖励 + 小权重
  `added_keys × relay_importance` 密集奖励 - 失败/等待/切换/过期惩罚；`success_rate` 模式仍保留，
  但训练默认使用 shaped 以提供非稀疏的每步信号

## 4. Baseline 怎么跑

### 4.1 命令模板

```powershell
conda run -n pytorch python scripts/run_baselines.py `
  --episodes 5 `
  --seeds 1000,1001,1002,1003,1004 `
  --out outputs/eval/my_baseline `
  --name my_baseline `
  --rl-checkpoint outputs/train_curriculum_v1/checkpoint_update_000100.pt `
  --rl-name rl_model `
  --episode-start-mode fixed `
  --time-limit-days 1 `
  --device cuda
```

### 4.2 参数说明

| 参数 | 含义 |
| --- | --- |
| `--episodes` | 每个策略跑几局 |
| `--seeds` | 逗号分隔的种子列表 |
| `--out` | 输出目录（相对项目根目录） |
| `--name` | 图表名称 |
| `--rl-checkpoint` | 可选，把训练好的 RL 模型加入对比 |
| `--rl-name` | RL 模型在结果中的名字 |
| `--episode-start-mode` | `fixed`（从 t=0）或 `random_day` |
| `--time-limit-days` | 只跑前 N 天 |
| `--device` | RL 模型用 `cuda`/`cpu` |

### 4.3 基线策略与参数

基线开关和参数在 `configs/baselines.yaml`：

- `random`：随机策略
- `greedy_rate`：按当前速率全局匹配，`use_future_mean_rate` 可选
- `greedy_qkp`：按库存/速率混合
- `greedy_demand`：按需求/速率混合
- `greedy_relay`：按请求路径补库存，参数含 `rate_weight / demand_weight / completion_multiplier / keep_weight / deadline_window`
- `greedy_matching`：速率/库存/需求/保持的综合匹配
- `ilp_optimal`：ILP 参考，参数含 `max_requests / max_paths_per_request / time_limit_s / mip_rel_gap`

每个策略都有 `enabled: true/false`，不需要的策略可以直接关闭。

## 5. 单独评估某个 RL Checkpoint

测试参数默认写在 `configs/global.yaml`，一般不需要每次改命令行：

```powershell
conda run -n pytorch python scripts/eval_long_horizon.py
```

需要临时覆盖时再传参数，例如：

```powershell
conda run -n pytorch python scripts/eval_long_horizon.py `
  --window-start-day 0 --window-end-day 30 `
  --episode-days 1 --episodes 3
```

结果是一个 JSON，包含：

```text
checkpoint, update, mean_success_rate,
mean_request_completion_rate, mean_served_keys, mean_arrived_keys,
min/max success_rate, per-episode details
```

`configs/global.yaml` 参数说明：

- `window.start_day`：激活窗口起始天
- `window.end_day`：激活窗口结束天
- `episode.days` / `episode.steps`：一局时长
- `episodes` 或 `seeds`：局数 / 指定随机种子
- `start_mode`：`random_day` 随机起点，`fixed` 固定起点
- `checkpoints`：要评估的模型列表
- `out`：结果 JSON 路径

推荐用多个随机起点、多天评估不同 checkpoint，不要只看固定一天。

## 6. Baseline 结果保存在哪里

输出目录为 `outputs/<out>/`，例如 `outputs/eval/my_baseline/`：

| 文件 | 内容 |
| --- | --- |
| `episodes.csv` | 每局汇总 |
| `summary.json` | 每个策略的平均 reward / served_keys / failed_keys / success_rate / conflict_count |
| `steps_<policy>.csv` | 每个策略的逐步明细 |
| `figures/my_baseline_policy_comparison.*` | 策略对比图 |
| `figures/my_baseline_episode_timeline.*` | 单局时间线 |

## 7. 注意事项

- 训练日志里的 `mean_success_rate` 是随机日窗口内的即时统计，波动大；最终判断以固定日评估为准。
- 修改特征归一化、奖励或动作表示后，旧 checkpoint 不能直接续训或复用，需要重新训练。
- 后台训练进程的输出会重定向到 `outputs/<xxx>.log`，可随时查看进度。
