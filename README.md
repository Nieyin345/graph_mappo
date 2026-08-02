# QKD RL

FSO-QKD link scheduling code scaffold.

Current implementation target:

- `RateProvider` is an interface; real H5 reading is intentionally deferred.
- QKP is stored per link with YAML-configurable capacity.
- GS-GS communication demand is encoded as non-action demand edges in the graph.
- Request history is tracked with rolling windows for arrived, served, and failed demand.
- `ActionResolver` supports multiple YAML-selected matching modes.
- The first runnable check is a small mock-rate environment.
- `qkd_rl.env.factory` is the package-level environment assembly entry.
- `qkd_rl.models.graph_mappo` contains the pure-PyTorch GNN actor-critic skeleton.
- `qkd_rl.algos.policy.MAPPOPolicy` converts actor logits into env-compatible node actions and action scores.

Important config files:

- `configs/features.yaml`: node, physical edge, and GS-GS demand edge feature switches and resolved dimensions.
- `configs/env_small.yaml`: scenario size, link-level QKP capacity, requests, routing, reward, and action resolver mode.
- `configs/graph_mappo.yaml`: encoder, shared actor, critic, and masked categorical settings.
- `configs/train_mappo.yaml`: rollout length, PPO epochs, learning rates, and advantage settings.

Run from this folder:

```powershell
conda run -n pytorch python scripts/smoke_test_env.py
conda run -n pytorch python -m pytest
```

## H5 速率数据接入（H5RateProvider）

`build_global_tensor.py` 把全年链路物理数据写到 `dataset/global/`：

- `link_data.h5`：`link_registry`（link_id/node_u/node_v/link_type）、`distance`、`los`(int8)、`zenith`、`k_max`，形状均为 `(T, L)`，`k_max` 单位 bps
- `node_registry.csv` / `link_registry.csv` / `build_metadata.json`：节点与链路注册表

`qkd_rl.link.rate_provider.H5RateProvider` 已实现对应读取，通过配置切换：

```yaml
# configs/rate_provider.yaml
rate_provider:
  provider: h5            # mock | h5
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
conda run -n pytorch python scripts/inspect_rate_provider.py --provider h5 --scenario full --samples 2000
```

`configs/env_full.yaml` 提供全规模场景的配套配置；加载顺序建议：

```text
default.yaml -> rate_provider.yaml -> features.yaml -> env_full.yaml -> graph_mappo.yaml -> train_mappo.yaml
```