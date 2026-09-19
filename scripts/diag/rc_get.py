"""按**点分路径**读 `resolved_config.yaml`，一行一个值。

### 为什么需要它

`run_mode_de.sh` 第一版的自检用

    grep -E "^  mode:" resolved_config.yaml | head -1

去核对 `model.mode`，结果抓到的是**文件里第一个两空格缩进的 `mode:`** ——
那是 `env.action_resolver.mode`（值 `mutual_choice`），不是 `model.mode`。
三个**健康**的 run 因此全被判定成"结果不可用"，监管进程 exit 1，`.go` 标记
永不生成，唤醒句柄悬空。

而且 `model:` 段在 `resolved_config.yaml` 的**最后**（约第 281 行），
所以"取最后一个"这类取巧同样不可靠 —— 换个 yaml 就失效。

**「一个会把成功报成失败的判据，会把真失败一起淹掉。」** 这是本项目第三次
「判据落在错误位置」，只是方向相反（前两次报坏消息，这次把好消息报成坏）。

按**路径**取值而不是按**行**取值，这一类错误就不再可能。

用法（服务器上）：

    /opt/qkd/venv/bin/python /tmp/rc_get.py outputs/<run>/resolved_config.yaml \
        model.mode train.ppo.entropy_coef train.ppo.minibatch_size

输出（每个查询一行，缺路径打印 `<缺失>`，便于直接比对）：

    model.mode=mixed
    train.ppo.entropy_coef=0.01

退出码：0 = **全部**路径都存在；1 = 至少一个缺失（调用方据此判失败）。
"""
from __future__ import annotations

import sys

try:
    import yaml
except ImportError:                                   # pragma: no cover
    print("缺 pyyaml", file=sys.stderr)
    sys.exit(2)

MISSING = "<缺失>"


def get(cfg, dotted: str):
    """按点分路径取值；任何一段不存在就返回 MISSING（不抛异常）。"""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(f"用法: {argv[0]} <resolved_config.yaml> <点分路径> [点分路径 ...]",
              file=sys.stderr)
        return 2
    path, queries = argv[1], argv[2:]
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except Exception as exc:                          # noqa: BLE001
        print(f"读取失败 {path}: {exc}", file=sys.stderr)
        return 2

    bad = 0
    for q in queries:
        v = get(cfg, q)
        if v is MISSING:
            bad += 1
        print(f"{q}={v}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
