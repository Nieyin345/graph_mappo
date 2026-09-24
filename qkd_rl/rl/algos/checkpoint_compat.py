"""Checkpoint/model/optimizer compatibility helpers for MAPPO.

Kept separate from the training loop because these routines only participate in
checkpoint construction/loading, not rollout collection or PPO loss math.
"""
from __future__ import annotations

from collections import defaultdict

import torch


def _reset_module(module: torch.nn.Module) -> None:
    """Re-initialize a module's parameters in place (identity preserved)."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def _expand_last_dim(t: torch.Tensor, new_dim: int) -> torch.Tensor:
    """把张量最后一维加宽到 ``new_dim``，**新增的列/行初始化为零**。

    专为「给已有模型的 encoder 输入接上一段新特征」而写：新特征的投影权重
    置零 ⟹ 前向传播逐位等于旧模型（新分支贡献恰好为 0），于是**旧 checkpoint
    仍然等价可用**，新分支再从零开始学。

    只处理最后一维变宽的情形；其余形状差异直接交给调用方判定为"不可升级"。
    """
    if t.dim() == 0 or t.size(-1) >= new_dim:
        raise ValueError("_expand_last_dim: %s 无法加宽到 %d" % (tuple(t.shape), new_dim))
    pad = list(t.shape[:-1]) + [new_dim - t.size(-1)]
    return torch.cat([t, t.new_zeros(pad)], dim=-1)


def _upgrade_state_dict_for_model(
    model: torch.nn.Module, state: dict
) -> tuple[dict, list[str], list[str], list[str], list[str]]:
    """把 checkpoint 的 state_dict 适配到**当前**模型的结构上。

    返回 ``(新 state, 已加宽, 已丢弃, 缺失, 多余)``。

    为什么需要它：本项目里「改结构」与「沿用权重」一直被当成互斥的两件事
    （`docs/当前状态与优化方向.md` §4：改形状 = 丢 BC 暖启动）。但其实只有当
    **形状变了且无法零填**时才必须丢。若某个权重只是**最后一维变宽**
    （典型：encoder 的输入维度因为接了新特征而变大），把它按
    ``[旧权重 | 0]`` 加宽就得到**逐位等价**的模型 ⟹ 暖启动可以保留。

    实测（2026-09-20，`history_encoder` 只开节点通道）：
    101 个键里 **100 个形状完全相同**，**只有 1 个**变宽
    （``encoder.node_proj.0.weight (128,17) → (128,81)``）⟹ 正好落在这个函数能救的范围。

    ⚠ **本函数放松了 `load_state_dict(strict=True)` 的两项检查**（缺失键、
    多余键），所以它**必须**把这两项回传给调用方**显式打印**——静默放宽是
    本项目反复吃过的亏（记忆 `failed-launch-must-be-loud`）。
    """
    cur = model.state_dict()
    out: dict = {}
    widened: list[str] = []
    dropped: list[str] = []
    missing: list[str] = []
    for k, v in cur.items():
        old = state.get(k)
        if old is None:
            # 新增的模块（如 history_encoder.*）：用当前初始化值，等价于从零学。
            out[k] = v
            missing.append(k)
        elif tuple(old.shape) == tuple(v.shape):
            out[k] = old
        elif old.dim() >= 1 and old.shape[:-1] == v.shape[:-1] and old.size(-1) < v.size(-1):
            out[k] = _expand_last_dim(old, v.size(-1))
            widened.append("%s %s→%s" % (k, tuple(old.shape), tuple(v.shape)))
        else:
            out[k] = v
            dropped.append("%s %s→%s" % (k, tuple(old.shape), tuple(v.shape)))
    unexpected = sorted(k for k in state if k not in cur)
    return out, widened, dropped, missing, unexpected


def _optimizer_param_names(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> list[str]:
    """优化器**每个参数槽**对应的限定名，顺序与 ``torch.optim`` 的 state 索引一致。

    这是修「参数个数变了之后按位置查表全错」的地基（见
    ``_upgrade_optimizer_state_for_model``）：要按名字重映射，先得有一张
    「索引 → 名字」的表，而且它必须**逐位对齐**优化器真实的收集顺序。

    做法是**拿参数对象查名字**，不假设任何顺序：``named_parameters()`` 给出
    模型里每个参数的唯一名字，优化器的每个槽位再按 ``id()`` 反查。

    ★ 为什么**不**按「组的顺序 = encoder/actor/critic」或「模型的注册顺序」推：
      · 组 0 是 ``encoder + history_encoder`` **两个**子模块 —— 按「组的第一个
        参数找 owner」只会列出 encoder 的，实测少 8 个（当场被长度守卫拦下）；
      · 模型的**注册顺序**是 ``history_encoder`` 在前（``__init__`` 先建它），
        与组 0 内部的拼接顺序**相反**。
      两张"看起来合理"的表都错位 —— 这恰是本函数要根治的那类错误。

    ★ 口径与优化器一致：``nn.Module.parameters()``（优化器收到的）与
      ``named_parameters()`` 默认都按 ``remove_duplicate=True`` 去重，两边相同；
      优化器 state 的索引只收 **parameter**（不收 buffer），所以这里也只看参数。

    认不出的参数（被手工塞进优化器）⟹ 返回空表，调用方据此**拒绝改写**。
    宁可不修，也不能拿错位的名字去改写 state。
    """
    id2name = {id(p): n for n, p in model.named_parameters()}
    names: list[str] = []
    for g in optimizer.param_groups:
        for p in g["params"]:
            name = id2name.get(id(p))
            if name is None:
                return []
            names.append(name)
    return names


def build_param_groups(model: torch.nn.Module, policy_lr: float, critic_lr: float):
    """构造优化器的 ``param_groups``，并**守住「模型每个参数都被注册」**。

    为什么是模块级函数而不是 ``__init__`` 里的一段：本仓库曾经有**四处**
    手写了同一份三组列表（``MAPPOTrainer.__init__`` + 三个
    ``scripts/train/supervised_train_*.py``），2026-09-20 发现它们**同时**
    漏掉了 ``history_encoder`` —— 同一份逻辑的四个副本、bug 也四份
    （记忆 ``duplicate-implementation-drifts``：正解是让第二份**上岗失败**，
    不是把四份都改对）。收敛到这里之后，漏注册只可能发生在一个地方，且有守卫。

    分组语义（**保持 3 组不变**，为了与既有 checkpoint 的 ``param_groups``
    结构对齐；改成 2 组会让 ``load_state_dict`` 抛错 ⟹ 静默丢掉热身优化器状态）：
      · 组 0 = encoder（若开了 ``history_encoder``，它并入此组——它只喂
        encoder/actor，不喂 critic，属策略侧，与 encoder 共享 lr）
      · 组 1 = actor（策略侧）
      · 组 2 = critic（价值侧）

    ★ 守卫是**必须**的：漏注册的症状（参数永不更新）与「配置没开那个子模块」
      在读数上**完全一样**，没有任何东西会报错。这个 bug 静默活了一整条 hist
      实验线，就是因为没人检查它。

    返回 ``(param_groups, policy_params, value_params)``。
    """
    hist = getattr(model, "history_encoder", None)
    enc_params = list(model.encoder.parameters())
    if hist is not None:
        enc_params = enc_params + list(hist.parameters())
    param_groups = [
        {"params": enc_params, "lr": float(policy_lr)},
        {"params": list(model.actor.parameters()), "lr": float(policy_lr)},
        {"params": list(model.critic.parameters()), "lr": float(critic_lr)},
    ]

    registered = {id(p) for g in param_groups for p in g["params"]}
    unreg = [p for p in model.parameters() if id(p) not in registered]
    if unreg:
        unreg_ids = {id(p) for p in unreg}
        owners = sorted(
            n for n, m in model.named_children()
            if any(id(q) in unreg_ids for q in m.parameters())
        )
        raise RuntimeError(
            "★ 有 %d 个模型参数（%d 个张量）未被注册进优化器：%s —— "
            "它们将永不更新，而症状与「配置没开」无法区分。"
            "请在 build_param_groups 里把该子模块并入对应的一组。"
            % (sum(p.numel() for p in unreg), len(unreg), owners or "顶层参数"))

    policy_params = list(param_groups[0]["params"]) + list(param_groups[1]["params"])
    return param_groups, policy_params, list(param_groups[2]["params"])


def _upgrade_optimizer_state_for_model(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, opt_state: dict
) -> list[str]:
    """把 checkpoint 的**优化器**状态适配到当前结构上（就地修改 ``opt_state``）。

    为什么必须有这个：``Optimizer.load_state_dict`` 只校验 **param_groups** 的
    结构（组数、每组参数个数），**完全不校验每个 state 张量的形状**。于是
    模型被 ``_upgrade_state_dict_for_model`` 加宽之后（如
    ``encoder.node_proj.0.weight (128,17)→(128,81)``），Adam 的
    ``exp_avg``/``exp_avg_sq`` 仍是 ``(128,17)``，**静默通过加载**，
    直到**第一次 ``optimizer.step()``** 才抛
    ``RuntimeError: The size of tensor a (17) must match the size of tensor b (81)``。

    实测（2026-09-20）：这个 bug 在 ``seq_len: 240`` 时**被 OOM 掩盖**——
    第一个 update 之前进程就被杀了，从没走到 ``step()``；把 ``seq_len`` 降到 32
    跑得够快，才第一次走到那里并暴露出来。

    加宽规则与模型侧完全一致：**新增的列置零**。这既是数学上正确的
    （新输入的梯度贡献初始为 0 ⟹ 一阶矩本就该是 0），也顺带让
    ``exp_avg`` 与加宽后的权重对齐。**无动量近似**：某一槽位形状差得无法
    靠补零救回时，该槽位丢弃并打印。

    ★★ 2026-09-20 二次修正：**本函数原先按「扁平位置」查表，这在参数个数
    变化时全错**。`history_encoder` 的参数并入组 0 的**中间**（位置 84..91），
    把 actor/critic 整体后移 8 位 ⟹ 旧 state 的 85..92 被拿去和 hist 的形状比
    （实测把 encoder 尾部的 `(128,384)` 与 `(64,)` 相比），而旧 idx 92 及以上
    在新表里**没有落点**、加载后静默消失。更要命的是组大小校验随后抛
    ValueError、整个优化器状态被丢，而**对照臂是带 Adam 矩热启动的**。
    这里已改为**按参数名重映射**（`_optimizer_param_names` 提供名字表）；
    另外**必须**把 `param_groups` 的 `params` 键删掉 —— `load_state_dict`
    会填入当前优化器的 params，留着旧的长度就是那个组大小校验。
    ⚠ 旧 checkpoint **里的参数名不可得**（模型已被换成新结构，且从未存过
    `param_names`），所以「参数个数变了」这种情形只能按名字配、配不上一律丢弃；
    只有参数个数**相同**时才能保住每个槽位（此时位置与名字两套口径等价）。

    返回 ``(state, msgs)``：``state`` 是可直接赋给 ``optimizer.state`` 的
    ``defaultdict``（按**参数对象**索引），``msgs`` 是人类可读的处理说明
    （调用方必须打印）。**绝不允许静默跳过。**
    """
    msgs: list[str] = []
    if not opt_state or "state" not in opt_state or "param_groups" not in opt_state:
        return None, msgs
    groups = opt_state["param_groups"]
    try:
        old_groups = [list(g["params"]) for g in groups]
    except (TypeError, KeyError):
        return None, ["!! 优化器状态结构无法解析（param_groups 里没有 params），整体丢弃"]
    old_flat = [int(p) for g in old_groups for p in g]
    if len(set(old_flat)) != len(old_flat):
        return None, ["!! 优化器状态索引有重复 ⟹ 无法可靠映射，整体丢弃"]
    # 当前优化器的扁平参数表：state 最终要按**这些对象**索引。
    flat_live = [p for g in optimizer.param_groups for p in g["params"]]

    # ★★ 为什么**不能**按「扁平位置」直接查表。
    #
    #    `Optimizer.load_state_dict` 的映射是
    #    `id_map = dict(zip(saved_params, live_params))` —— **纯按位置对齐**。
    #    而 `history_encoder` 的参数被并入**组 0 的中间**（`build_param_groups`：
    #    组 0 = `encoder + hist`，即位置 84..91），把 actor/critic 的索引
    #    **整体后移 8 位**。于是按位置查表时，`target[85]`（hist 的第一个参数）
    #    拿去和旧 state 的 85 号（actor 的）比 —— 实测正是这一条：把 encoder
    #    尾部的 `(128,384)` 拿去和 `(64,)` 比。
    #
    #    更糟的是**旧 idx 92 及以上在新表里没有落点**：`id_map` 的定义域只有
    #    旧参数个数（101）那么长，`zip(strict=True)` 静默截断 ⟹ 那 9 个参数的
    #    Adam 状态在加载后**直接消失**（既非清零、也非报错，是"没了"）。
    #    而 `load_state_dict` 的组大小校验先一步抛 ValueError，
    #    于是整个优化器状态被丢弃、从零开始 Adam —— 而**对照臂是带 Adam 矩
    #    热启动的**（对照日志：`形状相同 180`）⟹ 臂与对照差两个变量。
    #    实测 2026-09-20：`hist32_fix_s42` 就是被这条静默毁掉的。
    #
    #    ⟹ 逐槽修补**结构上不可能**修好它：位置映射一旦参数个数变了就失效。
    #    PyTorch 对这种情况给的正式接口就是 load_state_dict 的 pre-hook
    #    （见其 docstring：「To use the parameters' names for custom cases ...
    #    a custom register_load_state_dict_pre_hook should be implemented」）。
    #    这里就按**参数名**重映射：名字 → 索引的两张表都从**模型本身**取。
    live_names = _optimizer_param_names(model, optimizer)
    live_shapes = [
        tuple(p.shape) for g in optimizer.param_groups for p in g["params"]
    ]
    if len(live_names) != len(live_shapes):
        return None, ["!! 名字表与参数表长度不一致（%d vs %d）⟹ 拒绝改写"
                      % (len(live_names), len(live_shapes))]

    # ★★ 索引对不齐的两条路，**参数个数变了**时走第二条。
    #
    #   (a) 个数相同 ⟹ 名字表逐位相同（结构未变，或只换了同形的模块）。
    #       实测 hist=off 走这条：`形状相同 180 / 加宽 0 / 丢弃 0`，与对照臂日志相符。
    #
    #   (b) 个数变了（典型：checkpoint 没开 `history_encoder` 而当前开了）。
    #       **旧参数名不可得** —— 模型已被换成新结构，旧结构的参数名无处可取
    #       （checkpoint 里也没存 `param_names`）。所以改成**组内位置**对齐：
    #       新第 j 组的第一个参数 = 叠加到基础模型组 j 上的「额外参数」的起点。
    #       判据是「同位置 = 同一参数」对**本项目的**结构改动成立 ——
    #       `build_param_groups` 对某一组的改动总是**在组尾追加/移除**
    #       （组 0 = `encoder + hist`，hist 接在 encoder 之后；实测
    #        hist=off 组 0 = 84 个，hist=on = 92 个，**前 84 个逐位同名**），
    #       而模型的**注册顺序**（`history_encoder` 在 `encoder` 之前）只影响
    #       `named_parameters()`，**不影响**优化器 state 的索引 —— 那跟着的是
    #       `param_groups` 的顺序。
    #
    #       ★ 猜错也不静默：位置对错了，形状比对会**大批失败**并逐条打印
    #         （「无法适配 … ⟹ 丢弃该槽位」），绝不会安静地装错。
    #         正对照就靠这一点成立。所以这不是"假设"，是**带自检的启发式**：
    #         对了 → 全中；错了 → 全丢 + 满屏说明。
    n_extra = len(live_names) - len(old_flat)
    old_direct: dict = {}                 # 未变组的旧扁平索引 → 当前扁平索引
    if len(old_flat) == len(live_names):
        pass                              # 恒等映射：下面直接用 key 本身
    else:
        new_off = old_off = 0
        for j, g in enumerate(optimizer.param_groups):
            n_live = len(g["params"])
            n_old = len(old_groups[j]) if j < len(old_groups) else 0
            n_common = min(n_live, n_old)
            for i in range(n_common):
                old_direct[old_off + i] = new_off + i
            # ★ 两个偏移各自按**本组的真实大小**推进。曾经在这里把 new_off
            #   也按 `n_common` 推进 —— 组 0 一变宽（84→92），后面每一组就
            #   **整体少偏 8 位**，把 actor 的 Adam 矩装到**同名同形**的
            #   参数上。形状检查抓不到（都是 actor 的参数，形状一样）⟹
            #   会静默装错。探针的「交集逐名字比对」就是为这条设的。
            new_off += n_live
            old_off += n_old
        msgs.append("    索引重映射策略：组内位置对齐（旧 %d 参数 / 新 %d，差 %d）"
                    % (len(old_flat), len(live_names), n_extra))

    # Adam 的 state 是混合的：`step` 是**标量计数器**（优化器超参数，不属于参数形状），
    # `exp_avg` / `exp_avg_sq` 才与参数同形。把前者当参数张量去比形状，
    # 会把**每一个**参数都判成「无法适配」并整槽丢弃 —— 那就等于静默清空 Adam。
    # 实测（2026-09-20）第一版就是这个错：打印出「形状相同 3 / 丢弃 90」。
    _HPARAMS = ("step",)

    n_ok = n_widen = n_move = n_drop = 0
    new_state: dict = {}
    for key, st in opt_state["state"].items():
        key = int(key)
        if key >= len(old_flat):
            n_drop += 1
            msgs.append("    - 旧槽位 %d 越界（旧优化器只有 %d 个参数）⟹ 丢弃"
                        % (key, len(old_flat)))
            continue
        if len(old_flat) == len(live_names):
            # 个数相同：名字表逐位对齐，旧槽位 k 就是新槽位 k。
            idx = key
            name = live_names[key]
        else:
            # 个数变了：只能按**组内位置**对齐（见上面的策略说明）。
            idx = old_direct.get(key)
            name = live_names[idx] if idx is not None else "<旧槽位 %d>" % key
        if idx is None:
            n_drop += 1
            msgs.append("    - 旧槽位 %d（%s）在当前结构里没有对应参数 ⟹ 丢弃"
                        % (key, name))
            continue
        cur_shape = live_shapes[idx]
        fixed = {}
        bad = None
        for k, t in st.items():
            if k in _HPARAMS or not torch.is_tensor(t) or t.dim() == 0:
                fixed[k] = t          # 超参数/标量：与参数形状无关，原样保留
            elif tuple(t.shape) == cur_shape:
                fixed[k] = t
                n_ok += 1
            elif (t.dim() >= 1 and t.shape[:-1] == cur_shape[:-1]
                  and t.size(-1) < cur_shape[-1]):
                fixed[k] = _expand_last_dim(t, cur_shape[-1])
                n_widen += 1
                msgs.append("    + %s.%s %s→%s（新增列置零 ⟹ 无动量近似）"
                            % (name, k, tuple(t.shape), cur_shape))
            else:
                bad = "    - %s.%s %s 无法适配 %s ⟹ 丢弃该槽位" % (
                    name, k, tuple(t.shape), cur_shape)
                break
        if bad is not None:
            n_drop += 1
            msgs.append(bad)
            continue
        if idx != key:
            n_move += 1
        new_state[idx] = fixed

    # 「参数个数不变」时 n_move 必为 0；一变则**所有后移的参数都要搬家**。
    # 两个数都打出来，免得日后把「映射成功」误读成「位置没动过」。
    msgs.insert(0, "    索引重映射：搬运 %d 槽 / 原地 %d 槽"
                % (n_move, len(new_state) - n_move))

    # 组内超参（betas/eps/weight_decay/...）沿用 checkpoint 里的那份 —— 与
    # `load_state_dict` 的正常语义一致（它把保存的 param_groups 覆盖到当前组上，
    # 只强制回填 `params`）。**学习率随后由调用方按配置覆盖**，这里不动它。
    for live_g, old_g in zip(optimizer.param_groups, groups):
        for k, v in old_g.items():
            if k != "params":
                live_g[k] = v

    # ★★ 为什么**不**用 `optimizer.load_state_dict()`：一旦参数个数发生变化，
    #    它就**结构上不可能**接受任何合法输入 ——
    #      · 它校验 `len(g["params"])` 必须逐组相等 ⟹ 个数变了必抛 ValueError；
    #      · 而那个 `len()` 又要求 `"params"` 键存在 ⟹ 把键删掉去绕校验，
    #        会直接 KeyError（`saved_lens = (len(g["params"]) ...)`，实测已复现）。
    #    两条路都堵死。而 `optimizer.state` 本来就是**按参数对象**索引的
    #    （`state_dict()` 只存 `id(p)` 的**位置**再反查），所以这里直接构造它 ——
    #    这也正是 PyTorch 给这种场景指的逃生口（其 docstring 推荐的
    #    `register_load_state_dict_pre_hook`，本质就是自己重写 state）。
    # 新参数（如刚开启的 `history_encoder.*`）本来就没有优化器状态 —— 它们
    # 已经进了 `param_groups`，Adam 会在第一次 `step()` 时从零起点补齐。
    direct_state: defaultdict = defaultdict(dict)
    for idx, st in new_state.items():
        direct_state[flat_live[idx]] = st
    msgs.insert(0, "优化器状态适配（按参数名重映射）：形状相同 %d / 加宽 %d / 丢弃 %d"
                % (n_ok, n_widen, n_drop))
    return direct_state, msgs
