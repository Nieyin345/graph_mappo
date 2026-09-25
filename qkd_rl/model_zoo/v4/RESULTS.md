# v4 实验结果台账

> **本文件由脚本生成，不要手改表格。** 生成命令：
>
> ```bash
> python scripts/diag/backfill_manifest.py --repo-root . --apply
> ```
>
> 数据源：`outputs/model_ledger.jsonl`（中心台账，由原始产物重建）。
> 手写说明请写在文末「备注」段，生成时会保留该段。

## 测试环境

**无配置存档，环境不可恢复**（该批次的 `resolved_config.yaml` 未回传）。
引用这些数字前必须补记环境。

## 训练记录

`PPO 更新` 列为「否」= 全程 `kl=0` 且 `clip_frac=0`，即 PPO 从未真正更新（预算配置错误），该次训练不构成架构结论。

| 批次 | arm | 模型 | 轮数 | 是否多 minibatch | PPO 更新 | 每轮集数 | minibatch |
|---|---|---|---|---|---|---|---|
| v4-lstm-from-dminj30 | pairhist-v4-s42-from-dminj30 | v4 | 7 | 是 | 是 | — | — |
| v4-lstm-from-dminj30 | pairhist-v4-s43-from-dminj30 | v4 | 2 | 是 | 是 | — | — |

## 备注

（本段由人工维护，脚本重新生成时保留。）
