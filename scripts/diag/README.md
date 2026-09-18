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
| `mem_audit.py` | **内存现状按 PSS 分组汇总**：trainer / worker / 孤儿各占多少，并解释 MemAvailable 为什么低于预期。答"还能不能再开一个 run"。诊断内存一律从这里开始（RSS 会把共享页重复计数，误导性极强） |
| `reap.py` | 回收孤儿 worker 与卡住的探针父进程。**默认只报告**，`--apply` 才杀；`--protect` 保护在跑的探针。被 `--min-gb`/`--min-age-min` 筛掉的会**单独报计数**（阈值按旧数量级设就会静默漏报，已验证过一次） |
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

## 怎么送到节点

```bash
scp scripts/diag/* qkd:/opt/qkd/graph_mappo/scripts/diag/
```

**注意**：`deployment/sync.sh` 走 git push 到节点的 `deploy` 分支，要求节点工作区
**干净**（`receive.denyCurrentBranch=updateInstead`）。而节点上有我 scp 上去、
尚未进入节点 git 历史的文件（`configs/train_ent01*.yaml` 等），于是推送会被拒：

```
! [remote rejected] ... -> deploy (Working directory has unstaged changes)
```

处理原则：**先逐字核对节点文件与本地已提交版本是否一致**，一致才能丢弃
（`git checkout --` / `git clean -fd`）。这次的核对结果：
`mappo_trainer.py`、`train_graph_mappo.py`、`train_ent01.yaml`、
`train_ent01_g999.yaml` 的 sha256 与本地**完全相同**；
`train_window_329.yaml` 的 sha256 不同但 `diff` **逐字相同**——只是行尾符
（本地 CRLF / 节点 LF）所致，`sha256` 会被行尾符骗过。

不想动节点 git 状态时，`scp` 新脚本上去是**完全安全**的替代方案（本次就走这条）。
