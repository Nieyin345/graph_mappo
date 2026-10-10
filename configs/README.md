# configs/：配置文件使用与索引

> 更新于 2026-10-09。此目录含正式运行、历史复现、消融和临时诊断参数。**不要凭文件数量或前缀批量移动/删除 YAML**；实验脚本和 CLI 可能直接使用文件名。

## 1. 加载机制

默认环境配置文件列表由 `qkd_rl/env/factory.py::DEFAULT_CONFIG_FILES` 确定，按顺序为：

```text
default.yaml
  -> rate_provider.yaml
  -> features.yaml
  -> env_small.yaml
  -> graph_mappo.yaml
  -> train_mappo.yaml
```

其他配置通过具体训练/评估脚本显式加载。多数入口采用后加载覆盖前加载的配置合并语义，但具体顺序应以入口脚本、冻结运行配置及其验证测试为准。

## 2. 常用配置族

| 用途 | 文件 | 注意事项 |
|---|---|---|
| 通用运行/速率/特征 | `default.yaml`、`rate_provider.yaml`、`features.yaml` | 默认链的一部分 |
| 小/全规模环境 | `env_small.yaml`、`env_full.yaml` | 不同规模、业务参数不可混同 |
| RL 模型/训练 | `graph_mappo.yaml`、`train_mappo.yaml`、`train_profiles.yaml` | 训练入口、模式可能再次覆盖 |
| RL 正式配置栈（已有历史约定） | `rl_algorithm.yaml`、`train_full_rl.yaml`、`train_ent01.yaml`、`train_joint_ppo_fix.yaml`、`train_stocked_graph.yaml` | **仅代表配置栈文档约定**，运行前以脚本、对应测试及实验日志复核 |
| v7.5 论文场景 | `env_full.yaml` → `env_full_demand_v2.yaml` → `dataset_v7.yaml` → `scenario_v7.yaml` | 此叠加顺序写在 `scenario_v7.yaml` 文件头 |
| 模型数据集 | `dataset_v7.yaml`、`dataset_v2*.yaml` | v2 为历史兼容/复现；不能按版本号删除 |
| 云与有限密钥对照 | `qkd_block_eval_60s.yaml`、`qkd_block_eval_300s.yaml`、`qkd_block_eval_continuous.yaml`、`qkd_cloud_ungated_robustness.yaml` | 对照假设必须进入实验元数据 |
| 未来天气 oracle | `features_oracle_v7.yaml`、`features_weather_oracle_v7.yaml` | 仅作明确标记的上界对照，不能当作因果策略 |
| 诊断/消融 | `diag_*.yaml`、`train_*.yaml` 等 | 部分被脚本或历史训练结果引用，保留 |
| 历史归档 | `archive/` | 仅用于过去实验复现；不作为默认生效配置 |
| 其他实验资料 | `park/`、场景规模变体、请求 pair CSV 等 | 均可能由指定运行参数引用 |

## 3. 论文 v7.5 主线要点

- 参考 [信道与系统模型规范](../docs/信道与系统模型规范_定稿草稿_20261005.md)：60 SAT / 45 HAP / 45 GS，调度槽 60 s，信道模型与训练块协议分开。
- 论文主场景入口 [scenario_v7.yaml](scenario_v7.yaml) 使用 45 个固定请求对、`fixed_shortest_path`、300 s 训练库存代理块；最终有限密钥评估必须使用其对应 overlay。
- `dataset_v7.yaml` 与真实 H5 来源要对应；存在名为 v7 的配置不等于本地已经生成、核验了年度 v7 数据集。
- 实验比较必须保存**所有实际叠加的配置路径、配置内容哈希及 CLI**，不能只保存最后一个 overlay 的名字。

## 4. 为什么没有清理掉几百份诊断配置

首次盘点发现顶层配置约 168 项，其中不少消融/旧版本文件。部分可被脚本直接引用，部分仅由服务器命令行手工指定，单纯搜索代码引用不能证明“不再使用”。本轮不移动或删除配置，避免破坏 checkpoint、实验日志和启动脚本的复现能力。

整理前的旧版索引因正文存在乱码，已原样备份为 `docs/归档/过程文档/configs_README_整理前乱码备份_20261009.md`。项目文档总入口参见 [docs/00_索引.md](../docs/00_索引.md)。
