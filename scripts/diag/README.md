# scripts/diag —— 诊断与运维工具

这些脚本原先散在 `.tmp/` 里，而 `.tmp/` 是 **gitignored** 的，于是
`docs/训练诊断记录.md` 里"用 `.tmp/xxx.py` 测出……"这类引用对读者是**死链**：
结论在版本库里，得出它的工具不在。

放在这里的东西判据是：**日志引用了它，或者它会被反复用**。一次性的探针
仍然留在 `.tmp/`，写完就丢。

全部**在服务器上跑**（本机有 hook 拦 `python`，且本机内存装不下）：

```bash
bash scripts/sync.sh                       # 或 deployment/sync.sh
ssh qkd 'cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python scripts/diag/val_align.py --prefix ent01'
```

| 脚本 | 用途 |
|---|---|
| `val_align.py` | 打印每个 run 的验证曲线，**按轮对齐**（验证行自身不带 update 号，靠自己前面最近的训练行定位）。诊断任何"某轮好/差"的问题都从这里开始 |
| `local_summary.py` | 从 `outputs/`（或本地 `server_results/`）生成配对对照表 + 全部曲线 + 配置差异。`--roots outputs --out x.md` |
| `pss_trend.py` | 采样每个 run 的 PSS 随时间/轮数的变化并做线性拟合，量化内存增长 |
| `reap.py` | 回收孤儿 worker 与卡住的探针父进程。**默认只报告**，`--apply` 才杀；`--protect` 保护在跑的探针 |
| `reap_now.sh` | `reap.py` 的包装：自动识别并保护当前在跑的探针，然后执行 |
| `fetch_all.sh` | 把服务器结果抓到本地（**排除权重 .pt**，只留 metrics/config/rollout_debug） |
| `wake_poll.sh` | 唤醒链（本地跑）：轮询服务器，**仅在有新信息时**输出一行。指纹已排除易变数字，否则会每 2 分钟误唤醒 |
| `wake_parse.py` | `wake_poll.sh` 调用的服务器端探针；顺带在内存余量够时补起被误杀的实验臂 |
| `probe_loss_ab.py` | 严格 A/B 测 `_loss_for_batch` 新旧实现的 update 耗时（从 git HEAD 取旧模块并存对比） |
| `verify_loss_refactor.py` | 验证 loss 重构数值/梯度等价 |

## 写这类脚本的三条纪律（都是踩过的）

1. **测"每 X 变化 Y"至少要 4 个点、用拟合**，不能心算。3 个点拟合出的
   0.43 实际是 0.21，量级差一倍，直接改变并发上限的结论。
2. **测速度必须 A/B 同负载交替**，不能拿"现在"比"历史"。历史数字是机器空闲时
   测的，和并发时不可比 —— chunk 扫描就是被这个坑误导成"chunk=128 快 1.23x"。
3. **不许用 `pkill -f <模式>`**：远程 `bash -c` 的命令行自身含该模式，
   `pkill -f` 会杀掉自己（exit 255，后续命令一条都不跑）。先 `pgrep` 取 PID。
