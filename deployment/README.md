# Server deployment

本项目的训练在 CloudLab / Emulab 服务器执行。本目录只放服务器基础设施脚本。

## Workflow

```text
new node
  -> deployment/bootstrap.sh
       -> setup.sh       安装 Python/venv/依赖
       -> sync.sh        Git 工作区快照同步源码
       -> upload_data.sh 传输训练数据与 BC 权重
       -> smoke test     验证训练入口
```

## Fresh node

```bash
bash deployment/bootstrap.sh <user@host>
```

数据已经存在于镜像时：

```bash
bash deployment/bootstrap.sh <user@host> --skip-data
```

跳过 smoke test：

```bash
bash deployment/bootstrap.sh <user@host> --no-smoke
```

## Daily development

代码同步（包含未提交修改）：

```bash
bash deployment/sync.sh
bash deployment/sync.sh --dry-run
```

数据单独同步：

```bash
bash deployment/upload_data.sh <user@host> --data-only
```

## Important separation

- `qkd_rl/`、`configs/`、`scripts/`、`tests/`：源码与实验入口，由 `sync.sh` 管理。
- `dataset/`：训练数据，不进入 Git 部署快照，由 `upload_data.sh` 管理。
- `outputs/`：训练产物，默认不作为源码同步对象。
- `/opt/qkd/venv`：服务器 Python 环境。
- `/opt/qkd/graph_mappo`：服务器工作树。

不要用代码 tar 包覆盖已经由 `sync.sh` 管理的服务器工作树；这样会破坏 Git 工作树并可能引入 CRLF。

详细连接、SSH agent、镜像和性能说明见 `docs/deployment/server.md`。
