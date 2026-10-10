# 可学习模型注册表（2026-10-09）

本目录维护**神经网络模型的结构版本**，与信道计算模块 `tools/channel_v7/`（物理模型 v7.5）不是同一种版本概念。

## 哪个是论文主模型？

依据当前 `scripts/batch/run_paper_experiments.sh` 的论文实验配方，**拟研究的主方法**是 **v8 + self_new 预备偏置**，而不是“所有带 v8 文件夹的运行”都自动属于主方法：

- 基础架构：`qkd_rl/model_zoo/v8/`，在 v3 的 pair-path attention 之上加入 usage-forecast head。
- 研究方法覆盖：`configs/v8_bias.yaml`、`configs/diag_head_readout.yaml`、`configs/diag_selfnew_h3_w04.yaml`，并要求加载 v7.5 论文场景参数。
- 当前批处理配方：`scripts/batch/run_paper_experiments.sh` 的 `arm selfnew`。
- 以上只是**源码与实验配方的身份判断**，不代表训练已完成、优于对照或已完成论文统一评估。

## 模型矩阵

| 版本 | 作用 | 文件/状态 | 对比解释 |
|---|---|---|---|
| **v2** | 基础 Graph-MAPPO edge scorer；没有 pair-path attention | `v2/model_impl.py` + `graph_builder.py` + `config.yaml` | 基础神经网络对照；**不是最新论文方法** |
| **v3** | v2 + 站对路径注意力主导打分 | `v3/` 独立快照 | 注意力架构对照；可用作 BC 起点，仍需匹配输入维度 |
| **v4** | 需求隔离、库存边与路径注意力的**组合改变** | `v4/` 独立快照 | 历史组合消融，不能当作“只关闭需求融合”的单变量实验 |
| **v5** | 每站对因果 LSTM 历史 | 仅 `v5/model.py` 占位和 README；源码在旧服务器分支 `codex/pairhist-v4` | **本仓库不能直接实例化/训练/读取 v5 checkpoint**，保留占位便于识别历史结果 |
| **v6** | 基于 v3，只隔离需求边进入 GNN 的一个改变 | `v6/` 独立实现；`graph_builder.py` 当前与 v3 完全一致 | 单变量需求隔离消融 |
| **v7** | 基于 v3，第二阶段剩余端口准备策略 | `v7/model_impl.py` 继承 v3；`graph_builder.py` 复用 v3 | 早期双阶段 RL 方法实验；**与信道 v7.5 无关** |
| **v8** | 基于 v3 的预测辅助头/预备偏置 | `v8/` 独立实现；通过 overlay 指定 `self_new` | 当前论文配方中的预测/准备方法及其多种对照 |

历史成绩见各模型 `RESULTS.md`、`docs/实验日志.md` 和运行产物。不同数据集、请求分布、奖励口径、训练预算或评测协议的数值**不能直接相互排名**。

## 正确的导入、训练与评估

- **统一注册接口**：`qkd_rl.model_zoo.build_model(action_space, config)`、`graph_builder_class(config)`；它们按 `config["experiment"]["model"]` 选择模型版本。
- **没有明确模型身份时回退 v2**：这是历史兼容机制，不是“系统默认主论文模型是 v2”的声明。
- `qkd_rl/rl/models/graph_mappo.py` 是 **v2 兼容性 re-export**，不是 v3/v7/v8 通用实现。新代码不要从这个位置构造多版本模型。
- 训练：`scripts/experiment.py train --model v8 ...`；它自动合并 `model_zoo/v8/config.yaml`。另一入口 `scripts/train/train_graph_mappo.py` 读取已解析配置或默认链，必须明确核对 `experiment.model`。
- 评估：`scripts/experiment.py evaluate`、`scripts/eval/frozen_eval.py`、`scripts/baselines/run_baselines.py`、`scripts/baselines/eval_long_horizon.py`；**后两个已在 2026-10-09 修正为按 checkpoint 模型版本构造网络**。
- 保存真实的 config/checkpoint/输入特征定义与版本指纹。不能把 v3 checkpoint 直接按 v8 结构严格加载当作成功；是否允许部分参数迁移由显式的 checkpoint 兼容规则决定。

## 目录为什么暂时不合并代码

目前 v2、v3、v4、v6、v8 的 `model_impl.py` 有不少同源重复，而 v7 则继承 v3。**重复代码是长期维护负担，但也保留了历史实验的结构隔离和可复现性。** 现阶段强行抽成一个文件，会改变训练入口、checkpoint 键及实验对照的代码指纹，收益不及风险。

建议未来新模型使用可组合的公共 encoder/scorer/head，但**把旧 v2–v8 版本冻结为可复现参考实现**，而不是整批迁移或覆盖。新架构首先创建新模型目录和对应测试。

## 其他基线

不属于神经网络模型的启发式、随机及局部优化算法统一放在 [`../baselines/`](../baselines/)；离线 MILP 上界和相关求解代码位于 `scripts/milp/`，不应与可执行的在线策略混为一类。详情见 [baseline 索引](../baselines/README.md)。

更详细的审查见 [`docs/模型与对比策略目录审查_20261009.md`](../../docs/模型与对比策略目录审查_20261009.md)。
