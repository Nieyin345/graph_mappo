# configs 閰嶇疆鐩綍绱㈠紩

鏈洰褰曠殑 yaml 鍒嗕笁绫伙細**榛樿鍔犺浇閾?*锛堜换浣曞叆鍙ｉ兘浼氬姞杞斤級銆?*鍦烘櫙/绐楀彛瑕嗙洊**锛堣缁冧笌鍩虹嚎鑴氭湰鏄惧紡鍔犺浇锛夈€?*鐙珛宸ヤ綔娴?*銆傛墍鏈夋枃浠堕兘琚唬鐮佹垨鑴氭湰寮曠敤锛?*涓嶈闅忔剰鍒犻櫎**锛堝垹闄や細鐮村潖寮曠敤瀹冪殑鑴氭湰/娴嬭瘯锛夛紱濡傞渶绮剧畝璇峰厛纭寮曠敤鏂癸紙瑙佷笅琛?涓昏浣跨敤鑰?锛夈€?
## 鍔犺浇鏈哄埗

- **榛樿閾?*锛坄qkd_rl/env/factory.py::DEFAULT_CONFIG_FILES`锛屼换浣曞叆鍙?娴嬭瘯鍏堝姞杞借繖 6 涓紝鎸夐『搴忓悎骞讹級锛?  `default.yaml 鈫?rate_provider.yaml 鈫?features.yaml 鈫?env_small.yaml 鈫?graph_mappo.yaml 鈫?train_mappo.yaml`
- **RL 璁粌涓诲叆鍙?* `scripts/train/train_graph_mappo.py`锛?  榛樿閾?鈫?鏄惧紡瑕嗙洊 `env_full.yaml` 鈫?`global.yaml`锛堣缁?楠岃瘉绐楀彛锛夆啋 `train_profiles.yaml`锛坄--mode` 閫夎缁冩ā寮忥級鈫?鍙€?`--configs` 杩藉姞
- **鍩虹嚎** `scripts/baselines/run_baselines.py`锛歚global.yaml`锛坄--config`锛? `baselines.yaml`
- **鐩戠潱棰勭儹** `scripts/train/supervised_train_bfs_greedy.py`锛歚supervised_train.yaml`

## 鏂囦欢娓呭崟

| 鏂囦欢 | 鐢ㄩ€?| 鍔犺浇浣嶇疆 | 涓昏浣跨敤鑰?| 澶囨敞 |
|---|---|---|---|---|
| `default.yaml` | project / seed / runtime 鍩虹 | 榛樿閾锯憼 | 鍏ㄩ儴鍏ュ彛 | 蹇呯暀 |
| `rate_provider.yaml` | H5 鏁版嵁婧愩€佸彲鐢ㄦ€у彛寰勩€佽秺鐣岀瓥鐣?| 榛樿閾锯憽 | 鍏ㄩ儴鍏ュ彛 | 蹇呯暀 |
| `features.yaml` | 鑺傜偣/鐗╃悊杈?闇€姹傝竟鐗瑰緛寮€鍏充笌褰掍竴鍖栥€乺elay importance | 榛樿閾锯憿 | 鍏ㄩ儴鍏ュ彛 | 蹇呯暀 |
| `env_small.yaml` | 灏忓満鏅紙4 GS/1 HAP/1 SAT锛夛細鐜銆佽姹傛祦銆丵KP銆佽矾鐢便€佸鍔?| 榛樿閾锯懀 | 娴嬭瘯涓庨粯璁ゅ疄楠?| 涓?`env_full.yaml` 鏄悓涓€缁勯敭鐨勪笉鍚屽彇鍊硷紙浜岄€変竴鍦烘櫙锛夛紝涓嶆槸閲嶅 |
| `graph_mappo.yaml` | 妯″瀷缁撴瀯锛欸NN 灞傛暟/闅愯棌缁淬€乤ctor/critic銆乵ask 鍒嗗竷 | 榛樿閾锯懁 | 璁粌/璇勪及 | 蹇呯暀 |
| `train_mappo.yaml` | 璁粌瓒呭弬锛歳ollout銆丟AE銆丳PO銆乷ptimizer銆乴ogging | 榛樿閾锯懃 | RL 璁粌榛樿 | `--mode` 鏃朵細琚?profile 瑕嗙洊 |
| `env_full.yaml` | 鍏ㄨ妯″満鏅紙鐪熷疄 H5锛夛細璇锋眰/濂栧姳/QKP 鏍囧畾 | `train_graph_mappo.py` 鏄惧紡瑕嗙洊 | RL 璁粌涓绘祦绋?| 娉ㄦ剰锛歞eadline_steps=30 涓?env_small 鐨?960 灏哄害涓嶅悓锛屽弬鏁版棌涓嶅悓 |
| `train_mappo_smoke.yaml` | 鍥哄畾鍗曟棩鍐掔儫锛氫竴涓満鏅笂楠岃瘉妯″瀷+濂栧姳 | `train_graph_mappo.py --configs` | 璋冭瘯 | 涓?`train_diag_fast.yaml` 鐢ㄩ€旈噸鍙?|
| `train_diag_fast.yaml` | 蹇€熻瘖鏂璁撅細240姝ッ?灞€銆佸浐瀹氱瀛?鍥哄畾娓╁害銆佸崟杩涚▼鎵归噺 rollout | `train_graph_mappo.py --configs` | 濂栧姳/绠楁硶杩唬 | 淇濈暀 128脳3 鐨勬ā鍨嬪昂瀵镐互渚夸粠 BC checkpoint 缁锛涘彧鏈夎繖鏉¤矾寰勪細鍐?`rollout_debug.jsonl` 鐨勫鍔卞垎瑙?|
| `global.yaml` | 鍏ㄥ眬璁粌/楠岃瘉鏃堕棿绐楀彛銆佽姹傜瀛?| `train_graph_mappo.py` / `run_baselines.py` | 璁粌銆佸熀绾?| 鍏ㄥ眬瀹為獙绐楀彛锛屾墍鏈夌畻娉曞叡鐢?|
| `train_profiles.yaml` | `--mode` 璁粌妯″紡锛歚random_episode`/`continuous`/`fixed_day`/`curriculum`/`demand_edge` | `train_graph_mappo.py --mode` | RL 璁粌 | 瑕嗙洊 `train` 娈碉紙鍚?`value_target`銆乣replay_days`銆丳PO 鍙傛暟锛?|
| `baselines.yaml` | 鍩虹嚎绛栫暐寮€鍏充笌鍙傛暟锛坓reedy 绯诲垪锛?| `run_baselines.py` / `supervised_train_bfs_greedy.py` | 鍩虹嚎瀵规瘮 | 绂荤嚎鐞嗘兂涓婄晫鐢?`scripts/milp/compute_milp_upper_bound.py` 鍗曠嫭杩愯锛圲I "milp" 鍕鹃€夛級 |
| `supervised_train.yaml` | 鐩戠潱棰勭儹锛欱FS+greedy expert 鐨勮瘎浼扮獥鍙ｄ笌杈撳嚭 | `supervised_train_bfs_greedy.py` | 鍙€夐鐑伐浣滄祦 | 鐙珛宸ヤ綔娴侊紝涓?RL 璁粌鏃犲叧 |

| `rl_algorithm.yaml` | RL 共享 PPO/optimizer 参数 | `--configs` 显式叠加 | 正式 RL/诊断 | 必须排在具体训练 preset 前 |
| `train_full_rl.yaml` | 1440×8 正式训练窗口与 15-seed validation | `--configs` 显式叠加 | 正式 RL | 排在算法参数后 |
| `train_ent01.yaml` | `entropy_coef=0.01` 消融后保留的探索覆盖 | `--configs` 显式叠加 | 当前正式栈 | 后加载覆盖前值 |
| `train_joint_ppo_fix.yaml` | joint matching log-prob 修正后的 `actor_lr=5e-5` / joint KL 标尺 | `--configs` 显式叠加 | 当前正式栈 | 旧 3e-4 只适用于错误的 mean-logprob 标尺 |
| `train_stocked_graph.yaml` | 库存边进入观测图、但不进入生成动作集 | `--configs` 显式叠加 | 当前正式栈 | 仅改变可观测性，不放宽 action mask |

### 当前正式联合 PPO 配置栈

按下面顺序加载；本项目使用后加载覆盖前加载的 deep-merge 语义：

```bash
--configs rl_algorithm.yaml train_full_rl.yaml train_ent01.yaml \
          train_joint_ppo_fix.yaml train_stocked_graph.yaml
```

这组顺序由 `tests/test_formal_training_stack.py` 锁定关键值，避免以后调整 preset 时静默把
`actor_lr`、`entropy_coef`、joint `target_kl` 或 `include_stocked_edges` 覆盖回旧值。

## 璋冨弬鎻愰啋

- 鏀?`train` 鐩稿叧鍙傛暟鍓嶅厛纭鐢熸晥鏂囦欢锛氶粯璁ら摼鍔犺浇鍚庯紝`train_graph_mappo.py` 浼氱敤 `env_full.yaml`銆乣global.yaml`銆乣train_profiles.yaml`锛坄--mode`锛?*渚濇瑕嗙洊**鈥斺€斾緥濡?`entropy_coef` 鍦?`train_mappo.yaml` 涓?0.01锛岃€屽悇 profile 缁熶竴瑕嗙洊涓?0.001銆?- `env_small.yaml` 涓?`env_full.yaml` 鐨勮姹傚弬鏁版棌涓嶅悓锛坉eadline 960 vs 30 姝ワ級锛岃窇瀹為獙鏃朵笉瑕佹贩鐢ㄤ袱濂楀弬鏁扮殑缁忛獙鍊笺€?- `value_target` 榛樿 `gae`锛堟帹鑽愶級锛沗mc` 浠呬繚鐣欑敤浜庢秷铻嶃€俙replay_days` 榛樿 0锛圥PO 淇濇寔 on-policy锛夈€?

## 閰嶇疆缁存姢瑙勫垯

- `train_profiles.yaml` 鏄敮涓€鐨?RL 璁粌妯″紡娉ㄥ唽琛紱CLI 涓庢闈?UI 鍏辩敤瀹冦€?- `configs/archive/` 鍙繚瀛樺巻鍙查厤缃紝涓嶅弬涓庨粯璁ゅ姞杞介摼锛屼篃涓嶄綔涓哄疄楠屽叆鍙ｃ€?- UI 淇濆瓨妯″紡鏃跺彧鏇存柊鐣岄潰鏆撮湶鐨勮缁冨弬鏁帮紝淇濈暀杩炵画璁粌銆佽绋嬪涔犮€侀渶姹傝竟绛夋ā寮忎笓灞炲瓧娈点€?- `env.episode_steps` 涓?`train.rollout_steps` 涓嶅啀鐢?UI 寮鸿浜掔浉瑕嗙洊锛涜繛缁ā寮忕瓑閰嶇疆鍙互淇濇寔鑷繁鐨勯暱浼氳瘽璇箟銆?

## Config lifecycle

- Top-level `configs/*.yaml` contains default configs plus **active/recent experiment overlays**. A preset can be active even when no source file references it; many experiments are launched manually.
- `configs/archive/` contains reproducibility-only presets. They are not recommendations for new training runs.
- Archived presets must be referenced with their explicit relative path, for example `archive/experiments_legacy/var2_smoke.yaml`. Passing an old top-level basename produces a migration hint instead of silently loading legacy settings.
- Do not archive a preset based on reference count alone. Require explicit evidence in its header/log that the experiment is obsolete or superseded.

The two generic presets moved to `archive/experiments_legacy/` are intentionally narrow cleanup: an old CPU/16GB speed overlay and a one-off validation smoke overlay. Current PPO/server tuning remains in `rl_algorithm.yaml` and the formal stack above.
