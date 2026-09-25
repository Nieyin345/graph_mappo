# v5 model contract

> **v5 不是本仓库的代码。** 它的实现在服务器分支 `codex/pairhist-v4` 上，
> 本仓库只保存了它的产物快照 `server_results/pairhist_v5/`。
> 要复现必须切到那个分支，当前工作区**载不动 v5 的 checkpoint**。

## 命名来历（2026-09-26 改）

这个模型原本也叫 **v4**，与仓库里的 [`model_zoo/v4`](../v4/) 撞名。两者是完全不同的模型：

| | 卖点 | 代码位置 |
|---|---|---|
| [`model_zoo/v4`](../v4/) | 需求隔离 + 4 跳站对注意力 | **本仓库** |
| **v5**（本目录） | 每站对**因果请求时间序列**（LSTM，窗口 32 步） | 服务器分支 `codex/pairhist-v4` |

撞名造成过实际事故：一份描述 v5 的架构文档被当成 v4 的架构说明，诊断建立在错误的模型上。
按 [`实验产物规范.md`](../../../docs/实验产物规范.md) §3.4「一个版本号只指一个模型」，
LSTM 那个后移为 **v5**，`v4` 保持唯一含义。

## 契约（据 `server_results/pairhist_v5/` 产物与 train.log）

- **新增**：每个站对维护长度 ≤32 步的序列（新到达/已服务/待服务三通道，`log1p` 归一），
  共享单层 LSTM（hidden 32），末态拼到对应需求边特征再进 GNN。
- **边界**：策略在 t 只能用 t 之前的序列，不读未来。
- **继承 v3**：每站对 BFS 路径集合、共享 pair-path 注意力、`max_hops=4`。
- **起点**：三种子均从 `outputs/dminj_s{42,43,44}/checkpoint_final.pt` 的 update 30 续训，
  计划到 update 60。

## 已知限制

- 训练被 `poll.log` 末行的 `ERROR: codex queue failed: 1`（2026-09-23 17:46）**中断**：
  s42 到 7 轮、s43 到 2 轮、s44 **0 轮**。三种子轮数不齐 ⟹ 不可按均值比较。
- **从未做过冻结评测** ⟹ 无可与 v2/v3 并排的成功率。
- 回传产物里**没有配置文件**（无 `resolved_config.yaml` / `manifest.json`），
  只有裸 `metrics.jsonl` + `train.log` ⟹ 测试环境不可恢复。
