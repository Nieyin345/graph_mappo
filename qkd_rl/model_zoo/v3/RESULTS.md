# v3 实验结果台账

> **本文件由脚本生成，不要手改表格。** 生成命令：
>
> ```bash
> python scripts/diag/backfill_manifest.py --repo-root . --apply
> ```
>
> 数据源：`outputs/model_ledger.jsonl`（中心台账，由原始产物重建）。
> 手写说明请写在文末「备注」段，生成时会保留该段。

## 测试环境

**`env_fp` 不同 = 数字不可直接配对比较。**

| env_fp | 训练窗(激活) | 回合步 | 起始模式 | 拓扑 gs/hap/sat | 请求 | 配对热度 | 路由 | 速率源 |
|---|---|---|---|---|---|---|---|---|
| 9091810f | 0–295 | 240 | random_day | 4/1/1 | poisson rate=0.8 amt=100000.0 dl=30 | zipf s=1.0 | link_path/shortest_available_path | h5 |

## 验证结果 · 窗口 A

协议：第 330–365 天，15 个请求种子 `[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]`，每集 240 步，起始模式 `random_day`。

| 批次 | arm | 模型 | 训练种子 | 成功率 | 密钥比 | 隔离 | env_fp | 重复 |
|---|---|---|---|---|---|---|---|---|
| 20260924-frozen-eval | expert | expert | — | **0.7674** | 206.7894 | — | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-isolated-s42 | v3 | 42 | **0.7478** | 160.8036 | 是 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-isolated-s43 | v3 | 43 | **0.7498** | 160.2040 | 是 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-isolated-s44 | v3 | 44 | **0.7533** | 158.7943 | 是 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-s42 | v3 | 42 | **0.7691** | 148.6056 | 否 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-s43 | v3 | 43 | **0.7600** | 151.1327 | 否 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-s44 | v3 | 44 | **0.7593** | 150.6941 | 否 | 9091810f |  |
| 20260924-iso-attribution | expert | expert | — | **0.7674** | 206.7894 | — | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-isolated-s42 | v3 | 42 | **0.7478** | 160.8036 | 是 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-isolated-s43 | v3 | 43 | **0.7498** | 160.2040 | 是 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-isolated-s44 | v3 | 44 | **0.7533** | 158.7943 | 是 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-s42 | v3 | 42 | **0.7691** | 148.6056 | 否 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-s43 | v3 | 43 | **0.7600** | 151.1327 | 否 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-s44 | v3 | 44 | **0.7593** | 150.6941 | 否 | 9091810f | 是 |
| 20260924-iso-attribution | v3-iso-s42 | v3 | 42 | **0.7456** | 192.2320 | 是 | 9091810f |  |
| 20260924-iso-attribution | v3-iso-s43 | v3 | 43 | **0.7534** | 189.9352 | 是 | 9091810f |  |
| 20260924-iso-attribution | v3-iso-s44 | v3 | 44 | **0.7545** | 192.4333 | 是 | 9091810f |  |
| 20260924-iso-attribution | v3-v3full-s42 | v3 | 42 | **0.7558** | 193.6706 | 否 | 9091810f |  |
| 20260924-iso-attribution | v3-v3full-s43 | v3 | 43 | **0.7508** | 186.7326 | 否 | 9091810f |  |
| 20260924-iso-attribution | v3-v3full-s44 | v3 | 44 | **0.7472** | 196.9729 | 否 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-iso-s42 | v3 | 42 | **0.7460** | 191.8142 | 是 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-iso-s43 | v3 | 43 | **0.7561** | 189.3007 | 是 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-iso-s44 | v3 | 44 | **0.7554** | 192.5703 | 是 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-retry-v3full-s42 | v3 | 42 | **0.7584** | 192.4691 | 否 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-v3full-s43 | v3 | 43 | **0.7502** | 188.1278 | 否 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-v3full-s44 | v3 | 44 | **0.7459** | 196.4553 | 否 | 9091810f |  |
| 20260925-v3-diagnostics | rl-outputs-diagnostics-v3-set-bc-20260925-ordered | v3 | — | **0.7524** | 191.8368 | 否 | — |  |
| 20260925-v3-diagnostics | rl-outputs-diagnostics-v3-set-bc-20260925-set | v3 | — | **0.7369** | 200.8029 | 否 | — |  |
| 20260925-v3-diagnostics | v3-probe-base-s42 | v3 | 42 | **0.7584** | 192.4691 | 否 | — |  |
| 20260925-v3-diagnostics | v3-probe-high-s42 | v3 | 42 | **0.7571** | 192.2534 | 否 | — |  |

> 「重复 = 是」表示该条与更早一条是同一次测量（同 `evaluated_at`、同数值）被复制到两个目录，**不构成独立证据**。

## 验证结果 · 窗口 B

协议：第 300–329 天，15 个请求种子 `[200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214]`，每集 240 步，起始模式 `random_day`。

| 批次 | arm | 模型 | 训练种子 | 成功率 | 密钥比 | 隔离 | env_fp | 重复 |
|---|---|---|---|---|---|---|---|---|
| 20260924-frozen-eval | expert | expert | — | **0.8674** | 197.8869 | — | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-isolated-s42 | v3 | 42 | **0.8413** | 157.5977 | 是 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-isolated-s43 | v3 | 43 | **0.8350** | 159.2591 | 是 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-isolated-s44 | v3 | 44 | **0.8392** | 157.2529 | 是 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-s42 | v3 | 42 | **0.8593** | 148.3643 | 否 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-s43 | v3 | 43 | **0.8572** | 149.6655 | 否 | 9091810f |  |
| 20260924-frozen-eval | v3-frozen-s44 | v3 | 44 | **0.8511** | 151.3319 | 否 | 9091810f |  |
| 20260924-iso-attribution | expert | expert | — | **0.8674** | 197.8869 | — | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-isolated-s42 | v3 | 42 | **0.8413** | 157.5977 | 是 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-isolated-s43 | v3 | 43 | **0.8350** | 159.2591 | 是 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-isolated-s44 | v3 | 44 | **0.8392** | 157.2529 | 是 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-s42 | v3 | 42 | **0.8593** | 148.3643 | 否 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-s43 | v3 | 43 | **0.8572** | 149.6655 | 否 | 9091810f | 是 |
| 20260924-iso-attribution | v3-frozen-s44 | v3 | 44 | **0.8511** | 151.3319 | 否 | 9091810f | 是 |
| 20260925-retrain-v3-8ep | v3-retrain8-iso-s42 | v3 | 42 | **0.8292** | 188.2836 | 是 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-iso-s43 | v3 | 43 | **0.8250** | 189.4181 | 是 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-iso-s44 | v3 | 44 | **0.8382** | 189.6705 | 是 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-retry-v3full-s42 | v3 | 42 | **0.8403** | 190.8742 | 否 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-v3full-s43 | v3 | 43 | **0.8268** | 184.0750 | 否 | 9091810f |  |
| 20260925-retrain-v3-8ep | v3-retrain8-v3full-s44 | v3 | 44 | **0.8333** | 193.6009 | 否 | 9091810f |  |
| 20260925-v3-diagnostics | v3-probe-base-s42 | v3 | 42 | **0.8426** | 190.0961 | 否 | — |  |
| 20260925-v3-diagnostics | v3-probe-high-s42 | v3 | 42 | **0.8423** | 189.2997 | 否 | — |  |

> 「重复 = 是」表示该条与更早一条是同一次测量（同 `evaluated_at`、同数值）被复制到两个目录，**不构成独立证据**。

## 训练记录

`PPO 更新` 列为「否」= 全程 `kl=0` 且 `clip_frac=0`，即 PPO 从未真正更新（预算配置错误），该次训练不构成架构结论。

| 批次 | arm | 模型 | 轮数 | 是否多 minibatch | PPO 更新 | 每轮集数 | minibatch |
|---|---|---|---|---|---|---|---|
| 20260924-frozen-eval | frozen-s42 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | frozen-isolated-s42 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | frozen-isolated-s43 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | frozen-isolated-s44 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | frozen-s42 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | frozen-s43 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | frozen-s44 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | iso-s42 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | iso-s43 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | iso-s44 | v3 | 10 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | v3full-s42 | v3 | 20 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | v3full-s43 | v3 | 20 | 否 | **否** | 1 | 1024 |
| 20260924-iso-attribution | v3full-s44 | v3 | 20 | 否 | **否** | 1 | 1024 |
| 20260925-retrain-v3-8ep | retrain8-iso-s42 | v3 | 10 | 是 | 是 | 8 | 1024 |
| 20260925-retrain-v3-8ep | retrain8-iso-s43 | v3 | 10 | 是 | 是 | 8 | 1024 |
| 20260925-retrain-v3-8ep | retrain8-iso-s44 | v3 | 10 | 是 | 是 | 8 | 1024 |
| 20260925-retrain-v3-8ep | retrain8-retry-v3full-s42 | v3 | 10 | 是 | 是 | 8 | 1024 |
| 20260925-retrain-v3-8ep | retrain8-v3full-s43 | v3 | 10 | 是 | 是 | 8 | 1024 |
| 20260925-retrain-v3-8ep | retrain8-v3full-s44 | v3 | 10 | 是 | 是 | 8 | 1024 |
| 20260925-v3-diagnostics | probe-base-s42 | v3 | 30 | 是 | 是 | 8 | 1024 |
| 20260925-v3-diagnostics | probe-high-s42 | v3 | 30 | 是 | 是 | 8 | 1024 |
| v3-from-dminj30 | pairhist-v3-s42-from-dminj30 | v3 | 7 | 是 | 是 | — | — |
| v3-from-dminj30 | pairhist-v3-s43-from-dminj30 | v3 | 7 | 是 | 是 | — | — |
| v3-from-dminj30 | pairhist-v3-s44-from-dminj30 | v3 | 7 | 是 | 是 | — | — |

## 备注

（本段由人工维护，脚本重新生成时保留。）

**人工测注**：env_fp `9091810f` = 删 Stockholm 后的现环境（专家锚 0.7674）。
