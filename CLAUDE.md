# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 本地不跑 Python——一切在服务器上

本机（15.7 GB 内存）跑一次 1920 步基准就涨到 6–8 GB，已两次压满机器。`.claude/hooks/no_local_python.sh` 强制拦截本地 `python`/`pytest`（exit 2），经 `ssh`/`scp`/`rsync` 的调用放行。

- SSH 别名：`qkd`（CloudLab/Emulab 节点；**地址是动态的**，每次换节点直接替换，旧地址不保留）
- 服务器路径：`/opt/qkd/graph_mappo`（代码）、`/opt/qkd/venv/bin/python`（venv）
- 同步代码（含未提交修改，git 工作区快照方式）：`bash deployment/sync.sh`；新节点用 `bash deployment/bootstrap.sh <user@host>`
- 长任务必须 `setsid nohup ... &`，否则 ssh 通道被挂住
- 探针/临时脚本写在 `.tmp/`（gitignored），scp 到服务器 `/tmp/` 再跑。**禁止内联命令**（`python -c "..."`）——Windows 三层转义 + GBK 编码几乎必败，永远先写脚本文件

## 并行跑 run 前先算内存，不是先看 load

> ⚠ **本节所有绝对数都是「当次节点的实测」，不是常数。** CloudLab 节点随时换，
> 每次换 CPU/内存/磁盘都变（历史上就换过：AMD EPYC 7543 128 vCPU →
> Intel Xeon 8360Y）。**别引用绝对数，按「先测再算」现测。**
> 值得留下的是**判据与比例**——那些与机器无关。

**判据是 `MemAvailable / 30`**（按跑满算），**不是 CPU 核数**，也不是
启动时的内存。每个 run 只吃约 3 核，核多"看起来"能跑几十个——
照 load 判断铺 6 个，被 OOM 杀掉 4 个。
- `load average` 和 `%CPU` 都看不见这个墙。过载的症状是**每轮耗时上涨**
  （`rollout_s`、`update_s` 一起涨），不是进程变慢
- 分波启动用 `.tmp/wave_launch.sh`（按内存预算排队，不是一次全铺）
- **并发本身不吃速度**：同一份 buffer 在 5 个训练并发下测得 12.41 ms/步，
  与机器空闲时的完全相同。**约束是内存，不是核心数**
- **★ 磁盘是常被忘掉的硬约束，且常比内存先见底**。专家轨迹很占地方
  （带 pair_path masks 的 60 天 ≈ 2 GB，外推 295 天 ≈ 11 GB）⟹
  **铺全量轨迹前先 `df -h /opt/qkd`**

**★★ 每 run 的 PSS 是「配置相关」的 —— 这是比例关系，与机器无关，是稳的：**

| 配置 | 相对基线 | 说明 |
|---|---|---|
| `minibatch_size: 256`（基线） | **1.0×** | 参照点（绝对值须现测） |
| `minibatch_size: 512` | **≈1.19×** | 缓存分配器留下更高的激活高水位 |
| **`history_encoder.enabled: true`** | **≈2.3×** | LSTM 每步对全部实体跑一次 |

### ★ 先测再算（唯一可靠的做法）

```bash
# 1) 机器余量
grep MemAvailable /proc/meminfo && df -h /opt/qkd && nproc
# 2) 一个在跑的 run 的真实 PSS（父 + 所有 worker）
#    ★ 用 PSS 不用 RSS —— RSS 把共享页重复计数，误导性极强
for p in <run父pid> $(pgrep -P <run父pid>); do
    awk '/^Pss:/{s+=$2} END{print s}' /proc/$p/smaps_rollup
done
# 3) 并发上限 = (MemAvailable − 安全余量) / PSS(该配置)
```

**同型重臂不能全铺**：三条 hist 就能把并发推到 OOM。
停损判据是**已投入轮数 + 未来增长量**，**不是**启动时间
（"杀最晚启动的"被证明是破坏性的，见下）。

⚠ **`seq_len` 是环境窗口与 LSTM 回看共读的同一个键**
（`env/history_buffer.py` + `rl/models/history_encoder.py`）。
默认 **240** 在「按环境步调度」下是灾难：每步只新增 1 个时间点却重算 240 步
⟹ 12 倍慢、2.4 倍重、**会 OOM**。降到 **32** ⟹ 稳态约 **2.1× 慢**、**2.3× 重**。

**★★ 门/判据类代码必须打印它自己的输入，不能只打印 verdict。**
实测：内存门里的 `live_runs()` **恒返回空** —— `/proc/<pid>/cmdline` 用
**NUL 字节**分隔 argv，而正则写的是 `--run-name\s+(\S+)`，**`\s` 不匹配 `\x00`**。
⟹ `n=0`、待涨=0 ⟹ **条件恒真、无条件启动**（它一直打印「在跑 0」，
而实际有 6 条 run）。**恒真的门不报错**，伪装成"资源不够"。
读 `cmdline` 必须 `.replace("\0", " ")`；写门要打印输入或对已知答案做断言。

**不要用"杀最晚启动的"看门狗**。`.tmp/mem_watchdog.py --threshold 20 --apply`
曾在阈值设错（按旧的 13 GB 记录）时，精确杀掉了唯一想保的对照臂。
自动保护措施建在错误数量级上就是**破坏措施**，比没有更糟——没有它时 OOM
至少是全员平等地崩。阈值类工具上线前必须用实测校准。

**OOM 会遗留孤儿 worker，但孤儿是结果不是原因**：killer 用 SIGKILL 杀的是
**trainer 父进程**，`multiprocessing.spawn` 的 8 个 rollout worker 被 reparent
到 init 后**继续空转，永不退出**，每个 1.1 GB。每发生一次 OOM 就新增约 9 GB
孤儿，只清孤儿不解决复发（真正的做法是别把内存排满）。
`pkill -f train_graph` **只杀父进程，会制造新的孤儿**。

**`pkill -f <模式>` 在 `ssh host '...'` 里会杀掉自己**：远程 `bash -c` 的
命令行自身含该模式，`pkill -f` 匹配整条命令行 → 连同后面的命令一起被杀
（表现为 exit 255，后续命令一条都没跑）。先 `pgrep` 取 PID 再 `kill`。

清理用 `.tmp/reap.py`（默认只报告，`--apply` 才杀；判据 `PPid==1` 且 cmdline
含 `multiprocess`，另收**卡在 `do_wait` 的 `.tmp/probe_*.py` 父进程**——它们
命令行里没有 `multiprocess`，看门狗看不到，实测单个能占 5 GB 以上；
`--protect <pid>` 保护在跑的探针）。**worker 自身 cmdline 里没有 run-name**，
别试图从它自证归属。诊断内存现状用 `.tmp/mem_audit.py`（按 PSS 分组，RSS 会把
共享页重复计数，误导性极强）。

## 常用命令（都在服务器上执行）

**★ 起 run 必须带齐这两项，缺任一项都会静默出事**（2026-09-24 两条都实测踩过）：

```bash
ulimit -n 65535                                          # 默认只有 1024
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup setsid \
    /opt/qkd/venv/bin/python -u scripts/train/train_graph_mappo.py ...
```

- **漏 `ulimit -n`**：8 个 worker + torch 线程撞 `Errno 24`。症状**不是干脆地崩**
  ——一个 worker 变 zombie，父进程卡死在 `pipe_read`，**永不退出也永不报错**，
  `metrics.jsonl` 冻在 update 1（实测三条 run 各卡 8 小时）。此时 `update_s`
  会是 4900s 之类的大数，那是**死锁时长，不是计算量**
- **漏 `OMP/MKL_NUM_THREADS`**：分到 **72 个 torch 线程**（`nproc` 的一半），
  三组 run 的线程总数就能把整台机器压到过载。run 的 `launch.out` 里会打印
  `Threads: torch=72 OMP=None MKL=None`——**这一行要读**
- ⚠ `configs` 里的 `runtime.num_threads: 4` **会被启动入口的环境变量逻辑覆盖**，
  写配置文件里没用（这正是被误导的地方：以为配置管住了）
- 不必手写：**从能跑通的脚本 `cp` 再改**（如
  `pairhist-v3-attention-only-20260923/run_matched_screen.sh`，两项都有）。
  **不要凭印象重写**——2026-09-24 我重写时正是漏掉了这两项

```bash
# 训练（主入口；build_config() 可被探针复用）
scripts/train/train_graph_mappo.py --configs rl_algorithm.yaml train_full_rl.yaml \
    --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
    --seed 7 --num-updates 20 --run-name <名字>

# 测试（服务器上）
/opt/qkd/venv/bin/python -m pytest tests/ -x -q          # 全部
/opt/qkd/venv/bin/python -m pytest tests/test_gae.py -q   # 单文件

# 专家基线（唯一重要的对照；baselines.yaml 里没有它）
python scripts/eval/eval_expert.py --seeds 7-21           # 留出协议

# 配对比较两个策略（必须配对，见下）
python scripts/eval/compare_policies.py ...
```

结果都在 `outputs/<run-name>/`：`resolved_config.yaml`（生效配置）、`metrics.jsonl`（每轮一行）、`rollout_debug.jsonl`（每轮 rollout 分解，免费的高价值诊断数据）、`checkpoint_*.pt`。**`outputs/` 是 gitignored 的，只增不删。**

## 测量规范（判断"有没有效果"的硬规矩）

详见 `docs/测试规范.md` §4，两条最容易踩：

1. **配对不是可选**：逐种子成功率跨 0.29–0.88，不配对时 15 种子 SE≈0.062，配对后 ≈0.012。0.02 的差距必须配对才能测出。
2. **比较训练配置要 ≥3 个训练种子**（⑦）：单种子分辨率只有 ~0.035，单次单点差异一律不算结论。

## 架构

### 配置链（谁覆盖谁）

`load_default_config` 起底 → 各 yaml 依次深合并，**后面的覆盖前面的**。训练时不传 `--mode` 的实际链：

```
default → rate_provider → features → env_small → graph_mappo → train_mappo
       → env_full（强制叠加）→ --mode 对应 profiles
```

注意 `env_full.yaml` 的 reward 段**几乎全部覆盖**早期文档描述的旧奖励：实际生效的是 `mode: shaped`（`served_weight: 50` 稠密服务奖励 + storage/keep_active + failed 惩罚；`success_delta_enabled: false`）。文档比代码旧，以 `resolved_config.yaml` 为准。

### 两个测量 regime（别混）

| | 训练 regime | 验证 regime（留出协议） |
|---|---|---|
| 天数窗口 | 0–295（`activation_window` 限制回合**起始日**，回合可跑出窗） | 330–365 |
| 回合长度 | 1440 步 | 240 步 |
| 种子 | 训练种子 `--seed` | 15 个请求种子（**但见下方警告**） |

> ⚠ **`global.yaml` 里的种子数因树而异，不要假定是 15 个。** 2026-09-24 实测：
> `pairhist-v3-attention-only-20260923/configs/global.yaml` 只有 **7–11（5 个）**，
> 而对照 `matched240` 用的是 **100–114**。在别处直接跑会拿到 5 个不同种子的读数、
> **无法配对**。跨臂比较前先 `grep -A8 '^  validation:' configs/global.yaml`，
> 不一致就在臂自己的配置里显式覆盖 `validation`。

训练窗口侧成功率 ~0.86 已近饱和（专家 0.869）—— **训练侧确实看不出差距**，差距在验证 regime。⚠ 专家锚的值**取决于环境**：删 Stockholm 前是 **0.6979**，删后（现环境）是 **0.7674**（2026-09-21 实测，见 `实验日志.md` §2.2）。**引用旧值会得到错误结论**。

验证侧的方向取决于 `entropy_coef`：

- `0.001`（旧的错误值，如 `r6_base`）：RL ≈ **0.653**，比专家**低** 4.5 点 —— 就是这一条常被引用成"RL 打不过专家"
- `0.01`（现用值，`configs/train_ent01.yaml`）：RL ≈ **0.7178**

后者 3/3 种子同向，但 n=3（df=2，临界值 4.303）下 t=3.46、**p=0.074，未达显著**——只能说"方向一致、未测出"。判据与预注册（补种子 45/46 到 n=5）见 `docs/训练诊断记录.md` 的对应节。训练器内置 `eval_interval` 轮的 `evaluate_validation` 并存 `checkpoint_best_val.pt`。

**★ 2026-09-24 补：同协议下的现役参考点**（15 种子，240 步，`matched240` 一套配置）

| 策略 | 验证 SR |
|---|---|
| v3-fixed 的 **BC 产物（ep-32 快照）** | **0.7971** |
| v3-fixed BC 训满 60 轮 | 0.7822（**反而更差**） |
| `matched240_v2_cold`（RL 20 轮） | 0.7707 |
| 专家锚 | 0.7674 |

> 这几个数**不是机器相关的**（验证 regime 固定 15 种子 / 240 步），但**是配置相关的**：
> 只在这套 `matched240` 配置与协议下可比。换环境（如再删节点）或换 config 就要重测。
>
> ⚠ **BC 不是越久越好**：ep32 = 0.7971、ep60 = 0.7822（15 种子里 12 个下降）
> ⟹ 第 32 轮后对 60 天专家数据过拟合。起 RL 用 ep-32 快照是最优截断。
> 评测产物在 `pairhist-v3-attention-only-20260923/experiments/v3fixed/`。

### 核心数据流

- `qkd_rl/env/`：环境。`env.py`（reset/step、起日逻辑）→ `action_resolver.py`（每槽全局有向匹配，双端约束，Gumbel 顺序采样 + STOP）→ `qkp.py`（链路密钥池）→ `routing.py`（服务路由）→ `reward.py`（shaped 稠密奖励）→ `metrics.py`（`success_rate = served/arrived`）
- `qkd_rl/rl/algos/`：`mappo_trainer.py`（核心，~1200 行：rollout 收集、`update()` 里的 PPO 损失与 `clip_per_role`、验证选点）· `gae.py`（`non_terminal = 1 - terminated` 是**刻意选择**且实测更优，勿改 truncated）· `rollout_buffer.py`（minibatch 是跨回合随机子集，非顺序切分）· `rollout_workers.py`（并行 rollout，worker 各自播种 RNG）
- `qkd_rl/baselines/`：`path_greedy.py` 的 `PathScoreGreedy(phased=True)` 是**专家**（BC 起点即模仿它）；`greedy_relay` 系弱得多（0.443 vs 0.708），别拿它当对照
- `qkd_rl/evaluation/`：`test_protocol.py` 定义验证协议的规范构造（`load_validation_profile`/`build_validation_env_config`），evaluator 与 trainer 的 `evaluate_validation` 走同一调用路径

### 评测调用路径（探针要 token 级一致才可比）

RL 策略 step 环境时必须传 `edge_scores=st.edge_scores, expected_matched_edges=list(st.matched_edges or [])`（`resolver_mode != "max_weight_matching"` 时）；专家是 `env.step(actions, scores)`。`eval_expert.py`、`Evaluator._act`、`trainer.evaluate_validation` 三处已对齐——写新探针时照抄其中一处。

## 已知坑（实测，记了数的）

- **线程数影响训练结果**（OMP_NUM_THREADS 确定性影响，差 0.018）：A/B 对比必须固定线程数；评测不受影响
- **CloudLab 节点地址每次重启都变**：换新主机名直接替换，不要"同步文档"
- 6 个 md 文档（README、算法说明、方法总结等）有**不可逆**的双重编码损坏（UTF-8 被按 GBK 读后再存，1357 个 `?` 是丢失字节的墓碑，PUA 字符可逆但表不可构造）。**别信任这些文件的可读性，以代码和 git 历史为准**；重写比还原可行
- 诊断结论与方法都在 `docs/训练诊断记录.md`（每条带原始数字）——做训练侧改动前先读它，很多直觉方案（降 critic_lr、改终止语义）已被实验否决

## 红线

- 训练/评测/探针一律服务器上跑；本机 hook 会拦
- `outputs/` 只增不删
- 服务器上改了 governor/系统设置，测完恢复原值
- `.bib` 与图表样式等规范见上层 `learning_space/CLAUDE.md`
