# 服务器连接与使用

> 这份文档记录 **2026-09-14** 定下来的、实际可用的一套连接方式，以及之前折腾很久都没通的原因。
> 换服务器时看第 4 节。

---

## 1. 快速连接

配置已经写好，直接用别名：

```bash
ssh qkd 'hostname; nproc; free -g | head -2'
```

`~/.ssh/config` 里：

```
Host qkd
  HostName amd276.utah.cloudlab.us
  User qinglong
```

**换服务器只改 `HostName` 这一行。**

平台每次重启实验都会发一个**新的主机名**（`d7525-10s10319.wisc.cloudlab.us` →
`apt053.apt.emulab.net` → `amd276.utah.cloudlab.us`），所以这个别名是会被反复改的那一处。

> **`deployment/sync.sh` 里写的是别名 `qkd`，不是主机名** —— 它让 ssh 在连接时自己去解析
> `~/.ssh/config`。所以换节点时只要改这一行，同步脚本不用动。

---

## 2. 认证是怎么工作的

### 2.1 事实（2026-09-18 起）

| 项 | 状态 |
|---|---|
| **唯一钥匙** | `~/.ssh/qkd_rl/qkd_rl_deploy`（ed25519，**无密码短语**） |
| 公钥指纹 | `SHA256:OI42hlsHPR2WsclffBT7OPPxnN3BloVLn9BVjAXhvPQ` |
| 平台侧 | 已在 CloudLab 门户注册，**每台新节点自动认它** |
| agent | Windows ssh-agent 里只装这一把 |
| 服务器端 | 只接受 `publickey`，**不接受密码登录** |
| 其余旧钥匙 | 已归档到 `~/.ssh/_retired/`（45THX 默认钥匙、9/13 的 id_ed25519_qkd），确认无用后可删 |

### 2.2 历史教训（为什么换过钥匙）

- 旧默认钥匙 `id_ed25519`（指纹 45THX，**有密码短语**）：9/14 前后是平台
  注册钥匙，但密码短语让一切非交互调用（BatchMode / 脚本 / git push）签
  不了名，当时靠 Windows ssh-agent 缓解。
- 9/17 实验重实例化（Clemson → Utah）后，门户里注册的钥匙换成了
  `qkd_rl_deploy`，旧钥匙从此对任何新节点都不被接受。由于服务器不开密码
  回退、verbose 确认钥匙已递出仍被拒，一度误判为"服务器端 authorized_keys
  失效"，直到用户指出钥匙用错了才破案。
- 教训：**先对"平台注册的是哪把钥匙"，再排查别的。** `ssh -v` 看递出的
  指纹，和门户 Account > SSH Keys 里的指纹一对，一眼定案。

### 2.3 现在的日常状态

`ssh qkd 'hostname'` 直接能连（双保险：Windows ssh-agent 里装了钥匙，
`~/.ssh/config` 的 `Host qkd` 也写了 `IdentityFile`，哪边生效都行）。
实验换新实例后若突然 Permission denied，先怀疑**门户注册的钥匙变了或
没注册**，其次才是本机。

---

## 3. 几个 Windows / Git Bash 特有的坑（都踩过）

### 3.1 conda 劫持了 `ssh` 和 `ssh-add`

`(base)` 环境把 `D:\anaconda1\Library\usr\bin` 排在 PATH 前面，于是：

```
Get-Command ssh-add  →  D:\anaconda1\Library\usr\bin\ssh-add.exe   ← conda 的 MSYS 版本
```

它找 Unix socket，**不认 Windows 的命名管道**，所以永远报
`Could not open a connection to your authentication agent`。

**正确做法**：写全路径 `C:\Windows\System32\OpenSSH\ssh.exe` / `scp.exe` / `ssh-add.exe`。
（`deployment/upload_data.sh` 和 `deployment/sync.sh` 里都已经自动优先用系统 OpenSSH；
后者还额外把它通过 `GIT_SSH_COMMAND` 交给 git，否则 git 会用 Git for Windows 自带的
那个 MSYS `ssh`，同样连不上 agent。）

### 3.2 Git Bash 自带的 `ssh` 连不上 Windows agent

Git Bash 的 MSYS `ssh` 同样不认 Windows 命名管道 —— 即使密钥已经在 Windows agent 里，
`ssh qkd`（Git Bash 版本）依然会失败。

**所以：服务器操作一律用 Windows 的 `ssh.exe`。** 文档里的命令都按这个写。

### 3.3 管理员 PowerShell 里加的密钥，普通会话看不见

之前用管理员窗口 `ssh-add` 成功了、但普通会话里 `ssh-add -l` 显示 `The agent has no
identities` —— 提权会话和平常会话看到的是**不同的 agent 实例**。

**必须用普通（非管理员）PowerShell。**

### 3.4 密码短语那一步要真的输入

有一次 `ssh-add` 弹了 `Enter passphrase`，回车直接过去，结果
`The agent has no identities` —— 密码短语没交上去，等于没加。

### 3.5 不要走的路

- **不要给我密码**：命令会进历史记录和日志；而且服务器根本不开密码认证，给了也没用。
- **不要手改服务器上的 `authorized_keys`**：只对那一台有效，换服务器就失效（你早就指出了这点）。
  正确的位置是**平台门户的 SSH Keys**，它会下发到每台新节点。

---

## 4. 换服务器时要做什么

### 4.1 本机：只改一处

```bash
# ~/.ssh/config  把 qkd 的 HostName 改成新地址
```

然后 `ssh qkd 'hostname'` 验证。

### 4.2 新节点：一条命令装环境

`deployment/setup.sh` 会装 `python3.10-venv`（裸 Emulab 镜像没有）、建 venv、
装 CPU 版 torch 和全部依赖，并**先报告这台机器有没有显卡**：

```bash
scp deployment/setup.sh qinglong@<新地址>:~/
ssh qinglong@<新地址> 'bash ~/deployment/setup.sh'
```

（在 Git Bash 里跑的话把 `scp`/`ssh` 写成 `/c/Windows/System32/OpenSSH/scp.exe` 全路径。）

### 4.3 同步代码：一条命令（走 git 通道）

改完代码要传到节点上，用这个。它**不是复制文件，是 git 推送**：

```bash
bash deployment/sync.sh                         # 同步（含未提交的改动）
bash deployment/sync.sh --dry-run               # 先看会改哪些文件，不推送
bash deployment/sync.sh --tmp-file .tmp/probe.py # 只同步一个指定探针（推荐）
bash deployment/sync.sh --with-tmp              # 批量同步 .tmp 顶层 *.py / *.sh
```

**为什么不用 rsync**：Windows 的 Git Bash 不带 rsync，而 git 两端都有。

顺带解决三件事：

| 好处 | 说明 |
|---|---|
| **增量** | 只发变化的对象，几秒完事，不用每次重打整包 |
| **可回溯** | 节点上 `git log` 就是部署历史，出问题能定位到是哪一版 |
| **不被 CRLF 污染** | 见下方警告：以前用 tar 打包，整棵树的 LF 被转成了 CRLF |

> **同步的是工作区快照，不是某个提交。** 未提交的改动照样会传过去 —— 否则每次改完
> 都得先 commit 才能测，那是倒退。快照用临时索引（`GIT_SSH_COMMAND` + `GIT_INDEX_FILE`）
> 生成，**不动你的 HEAD、暂存区和工作区**。

**一次性装配**（每个新节点做一次，幂等，重复跑没副作用）：

```bash
bash deployment/sync.sh --setup
```

它会在 `/opt/qkd/graph_mappo/.git` 初始化仓库、把 `receive.denyCurrentBranch` 设为
`updateInstead`（这样 `git push` 直接更新节点上的工作区，不需要额外汇合步骤），
并以节点现有内容打一个基线提交。

节点上跟的是 **`deploy` 分支**，不是 `main` —— `main` 属于 GitHub，节点只跟 `deploy`。

> **`dataset/` 归节点所有。** 同步树里根本不含它，所以 374 MB 的数据永远不受影响。
> 装配时往 `.git/info/exclude` 写 `dataset/` 就是为了兜住这件事：`.gitignore` 里那条
> `!dataset/global/rate_stats.json` 会让 `git add -A` 把 `rate_stats.json` 收进版本库，
> 而下次同步的树里没有它 —— `git checkout` 就会把这个文件删掉，训练直接静默退化成
> `p99 = 10.0`。

> **`.tmp/` 同步规则**：普通同步不会强制加入未跟踪的 `.tmp/` 探针；仓库中原本已跟踪的
> `.tmp` 文件仍会按 Git 正常同步。日常调试优先用 `--tmp-file .tmp/xxx.py` 精确带一个脚本。
> `--with-tmp` 只保留为批量兼容模式：它会加入 `.tmp` **顶层**全部 `*.py` / `*.sh`，候选超过
> 100 个时会警告；`_archive/`、`nodecode/` 等子目录不会被递归打进部署快照。日志、yaml 和
> 运行产物仍不带。

> **之前用 tar 打包上传的 CRLF 问题**：本机 git 的 `core.autocrlf=true`（装在
> `C:/Program Files/Git/etc/gitconfig`），工作区是 CRLF，但 git 里存的 blob 是 LF。
> tar 是把工作区直接打包，所以节点上整棵源码树变成了 CRLF。走 git 的话节点拿到的
> 是 LF，和仓库里一致。这不是审美问题 —— 它让「本机和节点上的文件是否一致」重新
> 变成一个可以用 `md5sum` 直接回答的问题。

**这套机制取代了 `deployment/upload_data.sh --code-only` 的代码传输部分。** `deployment/upload_data.sh`
现在只负责**数据**（`dataset/`、BC 权重）和**首次装环境**。

### 4.4 传数据：一条命令

`dataset/` 被 `.gitignore` 排除，`git clone` 拿不到，必须单独传：

```bash
bash deployment/upload_data.sh qinglong@<新地址>
```

它会：打包代码（**含未提交的改动**）→ 传注册表和 BC 权重 → 传 374 MB 的
`link_data.h5` → **校验 md5** → 在节点上生成 `rate_stats.json`。

> `rate_stats.json` 不能省。缺了它 `RateNormalizer` 会静默退化成常量 `p99 = 10.0`，
> 而真实全局 p99 是 **12,655 bps** —— 边特征里 11 列速率特征的量级全错，
> 结果和之前测过的任何数字都不可比。

要重训 BC 才需要 4.3 GB 的 `outputs/trajs_pg_phased`，加 `--full`。

### 4.5 做镜像：让新节点自带环境（推荐）

每次换服务器都要重装环境 + 重传 374 MB 数据，很费时间。**做一次磁盘镜像就一劳永逸。**

**⚠️ 家目录不会进镜像 —— 这是最容易踩的坑。** 平台在创建镜像的对话框里明确警告：

> The contents of your home directory is **NOT** saved and will be deleted during
> the imaging process. You should install your software and data files in
> standard locations like `/usr/local` or `/opt`.

也就是说，**不管家目录在哪个分区上，成像时都会被剔除**。用 `df -h ~` 看到
`/dev/sda3` 并不能说明它会被保存 —— 我一开始就是被这一点误导，判断错了。

**所以环境必须装在 `/opt`。** 现在已经搬好了：

| 路径 | 大小 | 说明 |
|---|---|---|
| `/opt/qkd/venv` | 1.3 GB | 全部依赖（torch 2.14.0+cpu、numpy、scipy、h5py…） |
| `/opt/qkd/graph_mappo/dataset/global/link_data.h5` | 374 MB | 训练唯一数据源 |
| `/opt/qkd/graph_mappo/dataset/global/rate_stats.json` | 887 B | **必须校验它存在**（见 §4.4 的警告） |
| `/opt/qkd/graph_mappo/outputs/supervised_pg_phased/*.pt` | 12 MB | BC 预热权重 |
| `/opt/qkd/graph_mappo/{qkd_rl,scripts,configs,docs,tests}` | 小 | 代码 |

合计 **约 1.7 GB**，`/opt/qkd` 属主已 `chown` 给你（属主信息会存进镜像，所以新节点上
你直接可写）。家目录里只剩下点文件和一份 `deployment/setup.sh`，成像时丢掉无所谓。

> **搬迁已验证**：venv 从 `~` 移到 `/opt` 后**不需要重建**，因为我们一直是直接调
> `venv/bin/python`（不走那些 shebang 写死了旧路径的 `bin/pip` 等脚本）。
> 需要 pip 时用 `python -m pip` 即可。搬迁后从新位置跑训练，成功率 0.779、
> rollout 44.6 s / update 77.0 s，与搬迁前完全一致。
> （那两个耗时是 **2026-09-15 修复前**的数字，见 §5.2；成功率不受影响。）

**做镜像前清掉运行产物**（几百 MB 的 `outputs/par*`、`outputs/diag_*`、`__pycache__`、
日志），只留上表那些。

**镜像之后**，新节点的流程：

```bash
# 1. 改 ~/.ssh/config 里 qkd 的 HostName，然后确认连上
ssh qkd 'hostname'

# 2. 装配同步通道（一次性），然后把代码推上去（几秒）
bash deployment/sync.sh --setup
bash deployment/sync.sh

# 3. 只有换到【带 GPU】的节点时才需要这一步
```

**第 3 步：换到 GPU 节点的处理。** 镜像里的 torch 是 CPU 版（做镜像的机器没显卡），
在 GPU 节点上会让显卡完全闲置。换掉它一条命令：

```bash
ssh qkd '/opt/qkd/venv/bin/python -m pip install --upgrade torch \
    --index-url https://download.pytorch.org/whl/cu124'
```

然后验证 `torch.cuda.is_available()` 返回 True：

```bash
ssh qkd '/opt/qkd/venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
```

> 注意顺序：**先传代码再换 torch**。镜像里冻结的是当时的 `scripts/`，如果顺序反了，
> 跑的是旧脚本。
>
> 不想敲这条命令也行：`deployment/setup.sh` 现在会自动检测显卡并选对应的 torch 版本，
> 但它会把整套依赖重装一遍，比上面那条慢。

> **⚠️ 图像兼容性**：从 `c6220` 节点做的镜像不一定能直接跑在别的节点类型上
> （尤其是有 GPU 的 `c4130`）。新建实验时如果平台报镜像不兼容，就需要在那台机器上
> 重新装一遍环境再单独做个镜像 —— 但那时已经知道 `/opt` 这个坑了，成本很低。

`--code-only` 会先**逐项校验**镜像里 `link_data.h5` / `rate_stats.json` / BC 权重 /
`venv` 都在，缺任何一样就直接报错退出（避免静默退回 `p99 = 10.0` 让结果不可比），
然后只传代码包 —— 从 7 分钟变成几秒。

> **搬到 `/opt` 的额外好处**：venv 里写死的绝对路径现在是 `/opt/qkd/venv`，
> **和用户名无关了**。以前放在 `~/venv` 时，只要新节点换了用户名 venv 就失效；
> 现在同一个镜像在任何用户名下都能直接用。

### 4.6 创建镜像时对话框怎么选

对话框三个选项（"What would you like to do with this image?"）：

| 选项 | 含义 | 建议 |
|---|---|---|
| Update the current profile to use it | **就地修改**现有 profile，之后每次实例化都用新镜像 | 除非你确定镜像没问题，否则别选 |
| Make a copy of the current profile, updated to use the new image | **复制**一份 profile，原 profile 不动 | **推荐**：既有能直接启动的 profile，又不动原来的 |
| Just create a disk image, I will decide what to do with it later | 只创建镜像，不碰任何 profile | 最保守；之后要手动建/改 profile 才能用 |

**推荐第二个**（复制 profile）：拿到一个开箱即用的 profile，原 profile 保持完好，
万一镜像有问题也只影响副本。

**"Did you add any accounts or groups to your node?" 复选框 —— 不勾。**
我们全程用的是节点上已有的 `qinglong` 账号，没有新建账号或组（`/opt/qkd` 的属主是
`qinglong:rdmaopt-PG0`，都是已存在的）。勾了反而会让 Emulab 尝试记录并重建账号。

**"Please briefly describe your image" 建议填**，例如：

```
qkd_rl 2026-09-14: python venv + 374MB dataset + BC checkpoint at /opt/qkd
```

之后在镜像列表里一眼能认出哪个是哪个。

---

## 5. 服务器现状与定位

> **2026-09-15 重测。** 这一节的数据在「切片反向」修复之后重新测过，修复前后的数字
> 并列给出。旧数字已经不代表当前代码。

### 5.1 硬件

| | **当前 `amd276`**（CloudLab Utah, `c6525-100g`） | `apt053`（Emulab Apt） | `d7525`（CloudLab Wisconsin） | 本机笔记本 |
|---|---|---|---|---|
| CPU | **EPYC 7402P**（2019, Zen 2） | Xeon E5-2650 v2（2013） | EPYC 7302（2019） | Raptor Lake（2023） |
| 核心 | **48 线程 @2.8GHz** | 16 核 / 32 线程 @2.6GHz | 32 核 / 64 线程 @3.0GHz | 16 核 |
| **GPU** | **无** | **无** | **无** | **RTX 4060** |
| 内存 | **125 GB** | 62 GB | 125 GB | 15.7 GB |

sudo 免密可用，用户在 `root` 组。

### 5.2 实测性能：一次 update 从 47 s 降到 21 s

**2026-09-15 的修复。** `EdgeConditionedGraphLayer` 和 critic 的逐图池化都在切片一个
形状 `[批量边数, 128]` 的张量，而 PyTorch 里**切片的反向会 `zeros_like(整个输入)`**
再把切片的梯度贴回去 —— 每次 update 光这一项写零就有 **32.7 GB**（profiler 归因
11.2 s / 165 s，chrome trace 的父子链是
`aten::zero_ <- aten::zeros <- aten::slice_backward <- SliceBackward0`）。

改法是让物理边和需求边**全程保持两个独立张量**，逐图池化也换成 `index_add_` 的
分段求和 —— 反向里就不再存在"对整张量切片"这个操作。LayerNorm 是逐行的、池化是
普通均值，所以拆开算**数学上完全等价**：改动前后 log-prob / entropy 逐位一致，
value 只差 6e-8（池化求和的浮点结合律）。

amd276 实测（diag 规模：240 rollout 步 × 8 episode = 1920 步）：

| 阶段 | 48 线程 | 单线程 |
|---|---|---|
| rollout | 29 s | 34 s |
| update（修复前） | 47.4 s | 174.7 s |
| update（**修复后**） | **21.0 s** | **69.2 s** |
| 一轮合计 | 76 s → **50 s** | 209 s → **103 s** |

**加速 2.26×（48 线程）/ 2.53×（单线程）。**

#### 第二个瓶颈：rollout 的搬运负载（2026-09-15）

修完切片反向之后，用**真实训练配置**（`--mode random_episode --configs train_mappo.yaml`，
1440 步 × 8 episode、minibatch 1024、4 个 rollout worker）在 amd276 上重测：

| 阶段 | 修复前 | 切片反向修复后 | **搬运负载修复后** |
|---|---|---|---|
| rollout | 173–196 s | 227–231 s | **71–72 s** |
| update | 381–475 s | 96–99 s | 95–97 s |
| **一轮** | **553–654 s** | 327 s | **166–169 s** |

**根因不在并行度，在序列化。** 证据链：

1. 单个 episode（1440 步、单线程）实测 **28.9 s**；8 个 episode 的 rollout 却是 227 s
   ≈ 8 × 28.9 —— 看起来像没并行。
2. 但 `ps` 显示 8 个 worker **都在跑**，只是每个只吃 42% 的 CPU。
3. 关键对照：**用 8 个独立进程各跑 2 个 episode，16 个 episode 只用 64 秒**（7.4 倍
   并行）—— 说明 OS 层并行完全正常，`bats` 那套本来就够用。
4. 差别只有一处：worker 会把每个 episode **pickle 成文件**传回主进程。
5. 实测负载：**115 KB/步**，其中观测本身只占 58 KB。1440 步 × 8 episode =
   **每次 update 搬运 1.3 GB**，pickle 速度只有 **10 MB/s**，主进程要**串行**反序列化
   ~150 s —— 这就是那 227 秒。

多出来的 57 KB/步来自 `RolloutStep.log_probs` / `entropies`：它们是
`dict[str, torch.Tensor]`，**每个节点一个 tensor**（90 节点 ×2 = 180 个小 tensor），
而代码注释里明确写着每个节点存的是**同一个标量**（PPO 只读其中一个）。

**修法**：给 `RolloutStep` 加 `__getstate__`/`__setstate__`，序列化时把这种"全节点
同值"的字典压成 `(keys, value)`，反序列化时展开回同样的字典（各节点共享同一个
tensor 对象）。字段语义完全不变。

| | 改前 | 改后 |
|---|---|---|
| 200 步的负载 | 23.1 MB | **12.8 MB** |
| pickle | 2.30 s | **0.17 s（13.5×）** |
| unpickle | 3.05 s | **0.31 s（9.8×）** |

字节只少了一半，**速度却快了 10 倍** —— 因为瓶颈是对象数量，不是字节数。

> 顺带一提：`torch.set_num_threads(1)`（`rollout_workers.py`）看着像"48 核只用 1 个"
> 的元凶，但**不是**。实测单 episode 从 1 线程到 24 线程只快 **1.18 倍** ——
> rollout 是 Python/numpy 受限，不是 torch 受限，给 worker 加线程没有意义。

**并行度恢复了。** 搬运负载修复前后，同一组 worker 伸缩测试：

| workers | 修复前 rollout | **修复后 rollout** |
|---|---|---|
| 4 | 227 s | 71.3 / 71.5 s |
| 8 | 238 s（**反而更慢**） | **46.8 / 46.1 s（1.55×）** |

修复前加 worker 完全无效，因为关键路径是主进程那 ~150 s 的串行反序列化，
与 worker 数量无关；修掉之后 8 个 worker 对 8 个 episode 才真正并行起来。

#### 第三个杠杆：`batch_chunk` 与线程数

update 是内存带宽受限的，两个参数都印证了这一点：

| `batch_chunk` | update 耗时（24 线程） |
|---|---|
| 512（`train_mappo.yaml` 现值） | 95.07 s |
| 256 | 90.44 s |
| **128** | **67.02 s** |

chunk 决定一次前向里装多少步：512 步时边张量约 `512 × 393 × 128 × 4B ≈ 103 MB`，
**等于这台 CPU 的全部 L3**；降到 128 就是 ~26 MB，能待在缓存里。

| torch 线程数 | update 耗时（chunk=512） |
|---|---|
| 24（默认 = 物理核数） | 74.4 s |
| 16 | 82.1 s |
| 12 | 78.6 s |

**12 到 24 线程只差 6%** —— update 在 ~12 线程就饱和了。

#### rollout 那一步到底花在哪（profile 结论，供以后想再压的人参考）

300 步、单线程、cProfile（7.48 s，24.9 ms/步）：

| 函数 | self 占比 | 调用数 |
|---|---|---|
| `torch._C._nn.linear` | **22%** | 13216（**每步 44 次**） |
| `policy._sample_matching` | 5% | 301 |
| `routing.prepare_serve` | 4% | 300 |
| `env._compute_storage_pathness` | 4% | 300 |
| `masks.build` | 3.5% | 301 |
| `numpy.fromiter` | 2.8% | 5108 |
| `graph_builder.build` | 2% | 301 |
| `dict.get` | 1.9% | **1,508,277（每步 5000 次）** |

**`_read_dataset_rows` / H5 / rate_provider 一个都没进前 30。** 直觉上像瓶颈的
"每时隙重新加载图"其实不是 —— `_read_dataset_rows` 的分块缓存（`_chunk_cache_size=8`）
已经把重复解压挡住了。

真实分布大致是 **模型前向 43% / 环境侧图构建 39% / 其余零碎**：

- 模型侧是**每步 44 个小 Linear**（3 层 GNN × 5 个 MLP × 2 Linear + 投影 + scorer），
  输入只有 90 节点 / 282 边 —— 是小算子启动开销，不是算力。
- 环境侧摊在十来个函数上，外加 `dict.get` 每步 5000 次、`setdefault`/`append` 各
  3500 次的容器开销 —— 大多是**对 1978 条链路逐条走 Python**，而模型真正看到的
  只有其中约 282 条有向边。

**意味着没有一招制胜**：要再大幅提速得同时「把 44 个小算子合起来」和「把逐链路的
Python 向量化」，两件都是重构，各自约 20–30%。**不建议在没有明确需求时动**。

> **已经试过并且否掉的一条路：锁步批前向。** 直觉是"一个 worker 一次跑 K 个
> episode，块对角前向能把 44 个算子摊到 K 个环境上"。实测（canonical 配置，
> 11520 环境步）：
>
> | 配置 | 墙钟 | 每环境步的 CPU |
> |---|---|---|
> | 8 worker（各 1 个 episode） | **46.1 s** | 32 ms |
> | 1 进程锁步批 8 个 episode | 151–159 s | **13.1 ms** |
>
> **批前向确实把 CPU 效率提高了 2.4 倍**（算子被摊薄了，证实了诊断），**但它只用
> 一个核，墙钟反而慢 3.3 倍**。原因是墙钟 = 1440 × 单次锁步耗时，而单次耗时随 K
> 增长（K=1 是 32 ms，K=8 是 105 ms）。
>
> **所以对「快一个臂」而言，8 个 worker 各跑 1 个 episode 已经是最优；批前向只在
> 「一个核要伺候很多环境」时才有价值。** 别再往这个方向改了。
>
> 真正剩下的结构性空间是**让 rollout 和 update 重叠**（update 跑时 8 个 worker 全
> 闲着，反之亦然），理论上能从 108 s 降到 max(46, 62) ≈ 62 s。但那是异步 PPO 的
> 架构改动，风险不小，**没做**。





修复前的历史数字（不同机器、不同配置，仅供量级参考）：

| 阶段 | 笔记本 GPU | 笔记本纯 CPU | 服务器 |
|---|---|---|---|
| rollout | 25 s | 40 s | 32 s |
| update | 15 s | 67 s | 50–74 s |
| 合计 | 40 s | 108 s | 85–106 s |

**GPU 的优势被这次修复大幅削弱了。** 修复前 CPU update 比 GPU 慢 3–5 倍，现在
amd276 的 21 s 对笔记本 GPU 的 15 s 只差 **1.4 倍**。也就是说「没有显卡的服务器
单跑一定更慢」这个结论**在当前代码上已经不成立** —— 一轮 50 s vs 40 s。

> 笔记本那一列仍是修复前的数字。这个修复是纯代码层面的，两边都受益；笔记本上的
> 新数字没有复测。

**当前最大的单项开销已经变成 rollout（29 s，占一轮的 58%）。** rollout 是 Python
受限、基本单线程的，加核和加显卡都没用 —— 这直接改变了下面的优先级排序。

（**注意**：上面这几段是 diag 规模的数字。真实训练配置经过下面第二个瓶颈的修复后，
一轮 167 s 里 update 占 95 s、rollout 占 71 s —— 见 §5.4 的重排。）

### 5.3 那它有什么用：批量并行（吃满这台机器）

**内存大是它的真正价值。** 每个训练进程 RSS 约 **3.6 GB**，125 GB 装得下几十个；
瓶颈在 CPU 核数。

**2026-09-15 重测**（amd276，8 个 rollout worker + 12 个 update 线程每臂，
`batch_chunk=64`，各跑 1 个 update）：

| 并发臂数 | 每臂一轮 | 聚合吞吐 | 相对单臂 |
|---|---|---|---|
| 1 | 131 s | 0.0076 更新/秒 | — |
| 2 | 138 s | 0.0145 更新/秒 | **1.91×** |
| 3 | 158–159 s | 0.0189 更新/秒 | **2.49×** |

**2 个臂几乎白送**（每臂只慢 5%），**3 个臂每臂慢 21%，但聚合仍是 2.49 倍**。
按物理核算，3 臂 × 159 s 已接近 100% 占满 24 个物理核 —— 这就是这台机器的实际上限。
边际收益 1→2 是 +0.0069、2→3 是 +0.0044，所以 **2 臂是甜点，3 臂是上限，再多不值**。

三个臂的成功率 0.8635 / 0.8662 / 0.8632，与单臂基线 0.865 一致 —— 并发本身不影响
数值。

> **单臂要多线程，多臂才切 12。** update 在 ~12 线程就饱和（见 §5.2），所以：
>
> | 场景 | update 线程 | rollout worker | 每臂一轮 |
> |---|---|---|---|
> | 单臂（最快） | 24 | 8 | **≈110 s** |
> | 2–3 臂（吃满） | 12 | 8 | 131–159 s |
>
> 单臂给 12 线程是浪费（update 75.4 s vs 24 线程的 61.6 s）。

> **每个进程必须显式限制线程数**，否则各自开 24 线程互相抢核：
> ```bash
> OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 /opt/qkd/venv/bin/python scripts/train/train_graph_mappo.py ...
> ```


### 5.4 换服务器时该挑什么

按对训练速度的**实际影响**排序（**两次修复后重排**）。按真实训练配置，
一轮 167 s = **update 95 s（57%）+ rollout 71 s（43%）**：

| 优先级 | 指标 | 为什么 | 实测依据 |
|---|---|---|---|
| **1** | **核数多** | update 现在是大头且并行良好；核多直接换算力 | update：1 线程 69 s → 48 线程 21 s（diag 规模） |
| **2** | **单核性能强** | rollout 占 43%，是 Python/numpy 受限的，加线程没用只能靠单核 | rollout：24 线程只比 1 线程快 1.18 倍 |
| 3 | 内存大 | 只对"同时跑多个实验臂"有用，不加快单跑 | 6 路并行 = 3.2× 吞吐 |
| 4 | 有 GPU | update 15 s（笔记本）vs CPU 21 s（diag 规模），只快 1.4 倍 —— **不再是决定性因素** | 见 §5.2 |

**所以：优先核数，其次单核性能。** 带 GPU 仍然更好，但权重从"决定性"降到了
"锦上添花" —— 而 GPU 节点通常核数少、排队久，这笔账要重新算。

> 一个仍然成立的思路：**笔记本跑少量单跑，服务器用来跑并行臂**。修复后两边单跑
> 已经接近（40 s vs 50 s），但服务器能同时开多路，需要对比多个方案时优势明显。

**上机后的判定方法**：跑 §7 的自检，看 `metrics.jsonl` 里的 `rollout_s` / `update_s`，
对照 §5.2 的表。**修复后 diag 规模的 update 应显著低于 50 s**（amd276 上是 21 s），
如果还在 45 s 以上，说明节点上的代码不是最新的 —— 用 `bash deployment/sync.sh`
同步一下。

---

## 6. 故障速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `Permission denied (publickey)`，但日志里有 `Server accepts key` | agent 空了 / 没在用 Windows agent | 重跑 §2.3 的 `ssh-add`；确认用的是 `C:\Windows\System32\OpenSSH\ssh.exe` |
| `Could not open a connection to your authentication agent` | 用到 conda 的 `ssh-add` | 写全路径 `C:\Windows\System32\OpenSSH\ssh-add.exe` |
| `The agent has no identities`（刚输过密码） | 密码短语没真正提交 / 用了管理员窗口 | 普通 PowerShell 重来，确认提示后**实际输入** |
| `Connection timed out` | 实验被回收了 | 去平台门户重启实验，拿新主机名，按 §4 走一遍 |
| `ssh-add -l` 报 `Bad file descriptor`，`/tmp/*.sock` 变成普通文件 | Git Bash 里的 agent 死了 | 不再用 Git Bash 的 agent，改用 Windows 服务（§2.3） |
| 训练结果比以前差一截 | 忘了生成 `rate_stats.json` | 在节点上跑 `python scripts/estimate_rate_stats.py` |

---

## 7. 上机第一条自检

```bash
ssh qkd 'cd /opt/qkd/graph_mappo && /opt/qkd/venv/bin/python scripts/train/train_graph_mappo.py \
    --configs rl_algorithm.yaml train_diag_fast.yaml \
    --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt \
    --run-name diag_smoke --num-updates 3'
```

看 `metrics.jsonl` 里的 `rollout_s` / `update_s`，和 §5.2 的表对照，确认机器性能符合预期。
`scripts/show_metrics.py` 可以一行一轮地打印：

```bash
ssh qkd '/opt/qkd/venv/bin/python /opt/qkd/graph_mappo/scripts/show_metrics.py \
    /opt/qkd/graph_mappo/outputs/diag_smoke/metrics.jsonl'
```

**修复后（2026-09-15）的预期值**，amd276 上实测：

| 指标 | 期望 |
|---|---|
| `update_s` | **~21 s**（修复前 47 s；若仍在 45 s 以上说明代码不是最新的，跑 `bash deployment/sync.sh`） |
| `rollout_s` | ~29 s |
| `nb`（每轮 minibatch 数） | 应约等于 `rollout_steps × episodes_per_update ÷ minibatch_size`：diag 配置（1920 步 / 256）是 **8**，完整配置（11520 步 / 256）约 **45**。明显偏小才说明 KL 早停被过早触发 |
