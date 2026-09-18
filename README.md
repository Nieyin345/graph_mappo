# QKD-SAGIN 鐢熶骇绔皟搴︼紙Graph-MAPPO锛?
闈㈠悜 QKD-SAGIN锛堥噺瀛愬瘑閽ュ垎鍙?澶╁湴涓€浣撳寲缃戠粶锛夌敓浜х鐨?*瀵嗛挜鐢熸垚閾捐矾璋冨害**椤圭洰銆?涓夊眰 FSO-QKD 缃戠粶锛?0 鍦伴潰绔欙紙GS锛夈€?0 楂樼┖骞冲彴锛圚AP锛夈€?0 鍗槦锛圫AT锛夛紝1978 鏉″€欓€夌墿鐞嗛摼璺紱
姣忎釜鏃堕殭锛? 鍒嗛挓锛夋櫤鑳戒綋瑕佷负姣忎釜鑺傜偣鍐冲畾涓€鏉″瘑閽ョ敓鎴愰摼璺紝绾︽潫鏄?*鍙岀鍙?*锛圱x-out 鈮?1銆?Rx-in 鈮?1銆佸悓涓€瀵归摼璺笉鑳藉弻鍚戝悓鏃跺紑锛夈€傞摼璺€熺巼 / LOS 鍏ㄩ儴浠?H5 鏁版嵁闆嗚鍙栵紙`H5RateProvider`锛夈€?
椤圭洰鏍稿績鏄?*涓€鏉″惎鍙戝紡棰勮缁?+ 寮哄寲瀛︿範寰皟**鐨勮缁冪绾匡細

```text
BFS 闇€姹傛墿鏁ｅ惎鍙戝紡锛圥G-Phased 涓撳锛?        鈹? 琛屼负鍏嬮殕锛圔C 棰勭儹锛?        鈻?Graph-MAPPO 寮哄寲瀛︿範锛坵arm-start 鑷?BC 鏉冮噸锛?```

---

## 1. 寮哄寲瀛︿範妯″瀷锛欸raph-MAPPO

**鏋舵瀯**锛氬叡浜?GNN 缂栫爜鍣紙GraphSAGE锛? 灞傦紝128 缁达級鈫?鍏变韩 actor锛坋dge scorer锛夆啋 鍏ㄥ眬 critic銆?`mixed` 妯″紡鎶婅妭鐐?鐗╃悊杈?闇€姹傝竟涓€璧风紪鐮侊紝闇€姹備俊鎭部鐗╃悊杈逛紶鎾紱critic 鐢?typed-mean 姹犲寲杈撳嚭
姣忔椂闅欎竴涓叏灞€ value銆?
**鍔ㄤ綔绌洪棿锛氬叏灞€鍖归厤锛堝叧閿璁★級**銆傛棭鏈熺増鏈槸"姣忎釜鑺傜偣鐙珛閲囨牱涓€鏉¤竟 + resolver 璐績娑堣В"锛?PPO 浼樺寲鐨勬彁璁垎甯冧笌鐜鐪熸鎵ц鐨勫尮閰嶄笉涓€鑷达紝鎴愬姛鐜囬暱鏈熷仠鍦ㄧ害 30%锛涘綋鍓嶇増鏈細

1. actor 瀵规瘡鏉″悎娉?*瀹氬悜寮?* `(tx_target, rx_source)` 鎵撳垎锛?2. 绛栫暐浠庡叏閮ㄥ悎娉曞姬鍑哄彂锛屾瘡姝ュ彧鑰冭檻涓ょ鐐逛粛绌洪棽鐨勫姬锛屾寜 softmax 閲囨牱涓€鏉★紙鍚?STOP 閫夐」锛夛紝
   鐩村埌娌℃湁鍙敤寮р€斺€?*閲囨牱鐨勫尮閰嶅氨鏄幆澧冩墽琛岀殑鍖归厤**锛?3. PPO 浼樺寲鐨勬槸璇ュ尮閰嶇殑鑱斿悎 log-prob锛坄_matching_log_prob_entropy_fast` 鍚戦噺鍖栧疄鐜帮級銆?
**璁粌**锛氬杩涚▼ rollout锛坵orker 姹狅紝鏉冮噸+娓╁害闅忎换鍔′笅鍙戜繚璇佹帰绱㈡俯搴﹀悓姝ワ級鈫?GAE 鈫?PPO锛坈lip銆乤dvantage 褰掍竴鍖栥€並L 鏃╁仠銆佹寜瑙掕壊姊害瑁佸壀锛夈€?
**RL 鏁堟灉锛堝疄娴嬶級**锛?- 鍥哄畾鍦烘櫙锛坉ay 0 + 鍥哄畾璇锋眰绉嶅瓙锛?0 杞細**0.728 鈫?0.819**锛岃秴杩囧惎鍙戝紡锛?.7765锛夆湏
- 鏍囧噯楠岃瘉鍗忚锛? 绉嶅瓙锛夛細30 杞瘎浼板潎鍊?**0.713 卤 0.033**锛屾棤涓婂崌瓒嬪娍锛堣瑙?搂4 鍥板锛?
## 2. 鍚彂寮忕畻娉曪細BFS 闇€姹傛墿鏁?+ 鍒嗛樁娈佃矾寰勮皟搴?
鍚彂寮忕敱涓ら儴鍒嗙粍鎴愶細涓€涓?*杈规墦鍒嗗櫒**锛堣皝鍊煎緱寮€锛夊拰涓€涓?*璺緞绾ц皟搴﹀櫒**锛堟寜浠€涔堥『搴忋€佸紑鏁存潯璺級銆?瀹冨悓鏃舵槸寮哄寲瀛︿範鐨?*琛屼负鍏嬮殕涓撳**銆?
- **闇€姹傛墿鏁ｉ噸瑕佹€э紙relay importance锛?*锛氬姣忎釜绛夊緟涓殑璇锋眰锛岀敤鍏朵袱绔?GS 褰撹捣鐐癸紝鍦ㄥ綋鍓嶅悎娉?  鍙鐗╃悊杈规瀯鎴愮殑鍥句笂鍋?BFS锛屽緱鍒颁袱绔埌鍚勮妭鐐圭殑鏈€鐭烦鏁帮紝鎹缁欎腑缁ц妭鐐?閾捐矾鎵撳垎锛?- **鍒嗛樁娈佃矾寰勮皟搴︼紙PG-Phased锛?*锛氫笁闃舵绛栫暐鈥斺€斺憼瀛橀噺瀵嗛挜缃戠粶锛堝厛鐢ㄥ凡鎸佹湁鐨勫瘑閽ュ簱瀛樻湇鍔★級銆?  鈶℃贩鐢ㄥ簱瀛樼殑杈呭姪璺緞銆佲憿鍏ㄦ柊璺緞锛堢敓鎴愭柊瀵嗛挜鏈嶅姟锛夛紱閰嶅悎 ServeProbe 璺敱鎸夌摱棰堣烦閮ㄥ垎鏈嶅姟銆?- 缃戠粶鍘熷鐢熸垚鑳藉姏绾?4.7脳10鈦?瀵嗛挜/鏃堕殭锛岃€岄渶姹傜害 6鈥?0脳10鈦?瀵嗛挜/鏃堕殭鈥斺€旂害鏉熶笉鍦ㄤ骇閲忥紝
  鑰屽湪**绋€鐤忔椂鍙樻嫇鎵戜笂鐨勫彲杈炬€?*锛堟瘡鏃堕殭浠呯害 9.6% 鐨?(閾捐矾,鏃堕殭) 缁勫悎閫熺巼闈為浂锛夈€?
**鍚彂寮忔晥鏋滐紙瀹炴祴锛?*锛歚path_score_greedy_phased` 鎴愬姛鐜?**0.8104 卤 0.0575**锛堟爣鍑嗗崗璁級锛?鏄渶寮虹殑闈炲涔犳柟娉曪紝姣旂浜屽悕 `greedy_relay`锛?.435锛夐珮 1.86 鍊嶃€?
## 3. 鏁堟灉瀵规瘮涓€瑙堬紙瀹炴祴锛?
| 绛栫暐 | 鍗忚 | 鎴愬姛鐜?|
|---|---|---|
| 鍚彂寮?`path_score_greedy_phased` | 鏍囧噯楠岃瘉锛? 绉嶅瓙锛岀‘瀹氭€э級 | **0.8104** |
| 鍚彂寮?`path_score_greedy_phased` | 鍥哄畾鍦烘櫙锛?2 绉嶅瓙锛?| **0.7765** |
| BC 棰勭儹鏉冮噸锛堝厠闅嗗惎鍙戝紡锛?| 鏍囧噯楠岃瘉锛? 绉嶅瓙锛岀‘瀹氭€э級 | **0.7581** |
| BC 棰勭儹鏉冮噸 | 鍥哄畾鍦烘櫙锛?2 绉嶅瓙锛岀‘瀹氭€э級 | **0.7280** |
| RL锛堝浐瀹氬満鏅?20 杞級 | 鍥哄畾鍦烘櫙 | **0.819**锛堜粠 0.728 娑ㄨ捣锛?|
| RL锛堝叏灞€璁粌 30 杞級 | 鏍囧噯楠岃瘉 | **0.713 卤 0.033**锛堟棤瓒嬪娍锛?|
| 闅忔満绛栫暐 | 鈥?| 鈮?0.18 |

瑕佺偣锛?- BC 鍏嬮殕浼氫涪 5.2 涓偣锛?.8104 鈫?0.7581锛夛紝浣嗘妸 RL 璧风偣浠?闅忔満 鈮?.18"鎶埌"鈮?.73鈥?.86"锛?- **RL 鍦ㄥ浐瀹氬満鏅兘鏄捐憲瓒呰繃鍚彂寮忥紝浣嗗叏灞€璁粌鍗′綇**銆?
## 4. 褰撳墠鍥板锛歊L 鍏ㄥ眬璁粌鎴愬姛鐜囨病鏈夋彁鍗?
鍏ㄥ眬璁粌锛?0+ 杞級鍦ㄦ爣鍑嗛獙璇佸崗璁笂鍑虹幇**鎴愬姛鐜囧仠婊?*锛?
- 6 娆¤瘎浼?= 0.706 / 0.715 / 0.759 / 0.711 / 0.729 / 0.659锛屽潎鍊?0.713 卤 0.033锛?  涓?绾櫔澹板洿缁?0.713 娉㈠姩"涓€鑷达紱浣庝簬 BC 璧风偣锛?.758锛変笌鍚彂寮忥紙0.810锛夛紱
- 璁粌渚ф寚鏍囧叏骞筹細8 灞€閲囨牱鍧囧€?0.858 鈫?0.842锛宺eward 92 鈫?91锛屾棤瓒嬪娍锛?- **critic 鍏ㄧ▼鍋ュ悍**锛坈orr(V,R) = 0.89锛夛紝璇存槑闂鍦?actor 渚э紱
- 鍥哄畾鍦烘櫙 smoke 楠岃瘉鍚屾牱瑙傚療鍒帮細60 杞悗 RL checkpoint 鍚屽彛寰勮瘎浼?0.8574锛?  **浣庝簬** BC 鍩虹嚎 0.8622锛屼笖绛栫暐鐔典粠 3.33 鍗囧埌 4.01锛堝悜闅忔満鍖栨紓绉伙級銆?
**鍋囪鏂瑰悜**锛堝緟楠岃瘉锛夛細鎺㈢储娓╁害杩囬珮锛?.2 璧锋锛夋妸绛栫暐鎺ㄥ悜楂樼喌锛汸PO 姣忚疆鏇存柊閲忓お灏?锛?880 姝?梅 minibatch 1024 鈮?3 娆℃搴?杞級瀛︿笉鍔紱BC 璧风偣宸叉帴杩戝眬閮ㄦ渶浼橈紝缁х画 PPO 鍙嶈€?琚珮鐔垫牱鏈交寰薄鏌撱€傚綋鍓嶆鍦ㄥ仛锛氶檷浣庢帰绱㈡俯搴︺€佸姞澶ф瘡杞洿鏂伴噺銆佸悓鍙ｅ緞璇勪及楠岃瘉銆?
## 5. 浠ｇ爜缁撴瀯

```text
qkd_rl/                  # 鏍稿績搴擄細env / rl / link / data / baselines / evaluation
configs/                 # 鎵€鏈?YAML 閰嶇疆锛堢储寮曡 configs/README.md锛?scripts/train/              # 璁粌鍏ュ彛锛欱C 棰勮缁冦€丮APPO 璁粌銆佽瘎浼?scripts/baselines/       # 鍚彂寮?/ 鍩虹嚎
scripts/milp/            # MILP 鏈€浼樿В涓庢紨绀烘暟鎹敓鎴?tests/                   # pytest 娴嬭瘯
docs/                    # 绠楁硶璇存槑涓庢眹鎶ワ紙鍚叏閮ㄥ疄娴嬫暟瀛椾笌澶嶇幇鍗忚锛?```

## 6. 蹇€熷紑濮?
```powershell
# 鐢熸垚閫熺巼褰掍竴鍖栧弬鑰冨€硷紙缂哄け鏃?p99 閫€鍖栦负甯搁噺 10.0锛?conda run -n pytorch python scripts/estimate_rate_stats.py

# 鍐掔儫娴嬭瘯鐜
conda run -n pytorch python scripts/smoke_test_env.py

# 璺戞祴璇?conda run -n pytorch python -m pytest
```

### 璁粌绠＄嚎

```powershell
# 鈶?鍚彂寮忎笓瀹堕璁粌锛堣涓哄厠闅嗭紝榛樿杈规敹闆嗚竟璁粌锛?conda run -n pytorch python scripts/train/supervised_train_pg_phased.py `
  --run-name supervised_pg_phased

# 鈶?Graph-MAPPO 寮哄寲瀛︿範锛堜粠 BC checkpoint warm-start锛?conda run -n pytorch python scripts/train/train_graph_mappo.py `
  --mode curriculum --run-name exp1 `
  --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt `
  --device cuda

# 鈶?鍥哄畾鍦烘櫙 smoke 楠岃瘉锛堣皟鍙?/ 楠岃瘉濂栧姳璁捐锛?conda run -n pytorch python scripts/train/train_graph_mappo.py `
  --configs train_mappo_smoke.yaml `
  --checkpoint outputs/supervised_pg_phased/supervised_pg_phased_latest.pt `
  --run-name rl_smoke_day0

# 鈶?鍚屽彛寰勮瘎浼?checkpoint锛堜笌璁粌 success_rate 鐩存帴鍙瘮锛?conda run -n pytorch python scripts/evaluate/eval_fixed_scenario.py `
  --checkpoint outputs/rl_smoke_day0/checkpoint_final.pt `
  --steps 1440 --seeds 7-14 --device cuda
```

### 鏁版嵁涓庝骇鐗╋紙鍧囦笉涓婁紶鍒颁粨搴擄級

```text
dataset/global/*.h5     # 鍏ㄥ勾閾捐矾鐗╃悊鏁版嵁锛?25600 鍒嗛挓 脳 1978 閾捐矾锛?outputs/                # 璁粌浜х墿锛氳建杩?/ BC 鏉冮噸 / RL checkpoint / 鎸囨爣
  trajs_pg_phased/      #   鍚彂寮忓紩瀵兼暟鎹紙BC 璁粌鐢ㄨ建杩癸紝pkl锛?  supervised_pg_phased/ #   BC 鏉冮噸
  rl_smoke_day0*/       #   RL checkpoint 涓?metrics.jsonl
weather/                # 澶╂皵鏁版嵁
```

> 浠撳簱鍙繚鐣?*浠ｇ爜 + 閰嶇疆 + 鏂囨。 + 娴嬭瘯**銆傛墍鏈?`.h5`銆乣*.pkl` 杞ㄨ抗銆乣*.pt` checkpoint銆?> `outputs/`銆乣dataset/global/*`锛堥櫎 `rate_stats.json` 鍙傝€冨€硷級銆乣weather/` 鍧囩敱 `.gitignore` 鎺掗櫎銆?
## 7. 閰嶇疆绱㈠紩

閲嶈閰嶇疆瑙?[configs/README.md](configs/README.md)銆傝鐐癸細

- `configs/global.yaml`锛氬叏灞€璁粌/楠岃瘉绐楀彛涓庡叡浜姹傜瀛愩€?- `configs/features.yaml`锛氳妭鐐广€佺墿鐞嗚竟銆侀渶姹傝竟鐗瑰緛寮€鍏充笌瑙ｆ瀽缁村害銆?- `configs/env_small.yaml` / `configs/env_full.yaml`锛氬満鏅妯°€侀摼璺?QKP 瀹归噺銆佽姹傘€佽矾鐢便€佸鍔便€乺esolver 妯″紡銆?- `configs/graph_mappo.yaml`锛歟ncoder銆佸叡浜?actor銆乧ritic銆乵asked categorical 璁剧疆銆?- `configs/train_mappo.yaml`锛歳ollout 闀垮害銆丟AE/PPO 瓒呭弬銆佸涔犵巼銆佹棩蹇椾笌 checkpoint 闂撮殧銆?- `configs/train_mappo_smoke.yaml`锛氬浐瀹氬満鏅?smoke 璁粌锛堝浐瀹?day + 鍥哄畾璇锋眰绉嶅瓙锛夈€?- `configs/train_profiles.yaml`锛氳缁冩ā寮忥紙`continuous`銆乣fixed_day`銆乣curriculum`銆乣demand_edge`锛夈€?- `configs/baselines.yaml`锛氬惎鍙戝紡鍩虹嚎寮€鍏充笌鍙傛暟銆?
## 8. 鏂囨。

- [configs/README.md](configs/README.md)锛氶厤缃储寮?- [TRAINING_GUIDE.md](TRAINING_GUIDE.md)锛氬畬鏁磋缁冩搷浣滄墜鍐?- [docs/鍚彂寮忎笌寮哄寲瀛︿範绠楁硶璇存槑.md](docs/鍚彂寮忎笌寮哄寲瀛︿範绠楁硶璇存槑.md)锛氱畻娉曞師鐞嗐€佸熀绾裤€佹晥鏋滃姣斻€丷L 璁粌杩囩▼锛堝惈鍏ㄩ儴瀹炴祴鏁板瓧锛?- [docs/BFS寮曞寮哄寲瀛︿範棰勮缁冩眹鎶?md](docs/BFS寮曞寮哄寲瀛︿範棰勮缁冩眹鎶?md)锛氶璁粌姹囨姤
