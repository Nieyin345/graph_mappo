"""核实：当前「MAPPO」的策略是**集中式**还是**分布式执行**？

## 为什么这条决定 H₁ 能不能被检验

H₁ 的对比是「GNN+MAPPO（CTDE 协调）vs 单智能体 DRL」。
CTDE 的关键是 **D**（Decentralized execution）：每个 agent 按**自己的局部观测**
行动，协调是**学出来的**。

若策略其实是**集中式**（一个全局策略看全局观测、输出联合动作），
那它本来就是个单智能体策略 —— 「vs 单智能体」这个对比就没有对应的改动。

判据（三条，缺一不可）：
① 每个 agent 拿到的是**局部**观测吗？（还是全体共享一个全局 obs）
② 动作是**联合**的还是每个 agent 独立决策的？
③ 有没有"每个 agent 自己的"策略/价值网络？
"""
import sys, json
from pathlib import Path
REPO = Path("/opt/qkd/graph_mappo")
sys.path.insert(0, str(REPO))
import importlib.util
_s = importlib.util.spec_from_file_location("_tp", REPO/"qkd_rl/evaluation/test_protocol.py")
_tp = importlib.util.module_from_spec(_s); _s.loader.exec_module(_tp)
from qkd_rl.env.factory import build_env_from_config

prof = _tp.load_validation_profile(str(REPO/"configs/global.yaml"))
cfg = _tp.build_validation_env_config(prof)
env = build_env_from_config(cfg)
obs = env.reset(seed=100, start_seed=100)

print("="*88)
print("当前「MAPPO」的策略结构核实")
print("="*88)

print(f"\n① 观测是**全局**还是每 agent 局部？")
print(f"    env.reset() 返回**: 单个 GraphObservation**（不是 per-agent 列表）")
print(f"    obs 类型: {type(obs).__name__}")
print(f"    含 {len(obs.node_ids)} 个节点的特征 + {len(obs.physical_edge_ids)} 条物理边")
print(f"    ⟹ **全体共享同一个全局图观测** —— 没有 per-agent 局部观测")

print(f"\n② 动作是**联合**的还是独立决策？")
acts = {}
for nid in obs.node_ids[:3]:
    acts[nid] = (obs.action_candidates[nid][0], obs.action_candidates[nid][0])
print(f"    env.step(actions) 收 **dict[节点, (tx,rx)]** —— 一次提交**整个联合动作**")
print(f"    且由**一个**顺序采样器全局决定（`_sample_matching` 遍历全部弧）")
print(f"    ⟹ **联合动作**，不是 N 个 agent 各自独立决策")

print(f"\n③ 有 per-agent 策略/价值网络吗？")
print(f"    actor:  `SharedNodeActor` —— **共享**（不是每 agent 一个）")
print(f"    critic: `GlobalCritic`    —— **全局池化**成一个标量")
print(f"    gae.py 的 docstring 原话：")
print(f"      「The global critic produces one value per time step,")
print(f"       so all agents of one step **share the same advantage**」")
print(f"    ⟹ **没有 per-agent 网络**")

print(f"\n{'='*88}")
print(f"结论")
print(f"{'='*88}")
print(f"  当前「MAPPO」在**三条判据上全部落在「集中式」一侧**：")
print(f"    全局观测 + 联合动作 + 共享策略/价值")
print(f"  ⟹ 它的**执行期**其实是一个**中央控制器**，不是 CTDE 的 D（去中心化执行）")
print(f"  ⟹ 「MAPPO 的协调」在本代码里 = **每节点动作的分解 + mutual_choice 裁决**")
print(f"     （而 mutual_choice 是**冻结规则**，定稿结论 §3.2 推翻之一已证）")
