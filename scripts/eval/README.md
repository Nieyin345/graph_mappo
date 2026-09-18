# 璇勪及宸ュ叿

鍥炵瓟涓や釜闂锛?*RL 绛栫暐鐩稿鍐荤粨鍩虹嚎澶勫湪浠€涔堜綅缃?*锛屼互鍙?*涓ゆ潯鏇茬嚎鏈夋病鏈夌湡鐨勫垎寮€**銆?
娴佺▼瑙勮寖瑙?`docs/娴嬭瘯瑙勮寖.md`锛涜繖閲屽彧璇存槑姣忎釜鑴氭湰鍋氫粈涔堛€佷粈涔堟椂鍊欑敤鍝釜銆?
## 鍐荤粨鍩虹嚎锛堝彧璺戜竴娆★級

| 鑴氭湰 | 鐢ㄩ€?|
|---|---|
| `eval_expert.py` | 璺戦」鐩笓瀹?`PathScoreGreedy(phased=True)` + `ServeProbe`銆?*瀹冩湁鐙珛鑴氭湰灏辨槸鍥犱负涓撳涓嶅湪 `configs/baselines.yaml` 閲?*锛宍run_baselines.py` 姘歌繙璺戜笉鍒板畠 鈥斺€?婕忔帀瀹冧細璁╅棬妲涜浣庝及 1.6 鍊嶏紙0.443 瀵?0.708锛夈€傞€愮瀛愮殑缁撴灉鍐欐垚 JSON锛屼緵閰嶅姣旇緝銆?|
| `build_frozen_baselines.py` | 鎶?`random` + 6 涓?`greedy_*` + 涓撳鍚堝苟鎴?`outputs/eval/frozen_baselines.json`銆?*鍙惈闈炲涔犵瓥鐣?* 鈥斺€?RL checkpoint 鏄娴嬪璞★紝涓嶅喕缁撱€?|

## 璇勪及涓€涓?RL checkpoint

鍏堣窇鍐呯疆鐨勫熀绾垮叆鍙ｏ紙`--policies __none__` 鍏虫帀宸茬粡鍐荤粨鐨勫惎鍙戝紡锛夛細

```bash
python scripts/baselines/run_baselines.py --config configs/global.yaml \
    --episodes 15 --seeds 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21 \
    --out outputs/eval/rl_<鏍囩> --policies __none__ \
    --rl-checkpoint <ckpt.pt> --rl-name rl_<鏍囩>
```

鍐嶇敤 `compare_policies.py` 鍒よ銆?
## 姣旇緝涓庡垽璇?
| 鑴氭湰 | 鐢ㄩ€?|
|---|---|
| `compare_policies.py` | **涓诲姏**銆備换鎰忓涓瓥鐣ユ寜 seed **閰嶅**姣旇緝锛岀粰鍑洪厤瀵瑰樊鐨勫潎鍊笺€佹爣鍑嗚銆乼 鍊笺€傞厤瀵逛笉鏄彲閫夌殑锛氶€愮瀛愭垚鍔熺巼璺?0.29鈥?.88锛屼笉閰嶅鐨勬爣鍑嗚绾?0.06锛屾祴涓嶅嚭 0.02 鐨勫樊璺濓紱閰嶅鍚庣害 0.012銆?|
| `compare_arms.py` | 涓や釜 run 鎸?update 閰嶅锛堣緭鍏ユ槸 `metrics.jsonl`锛夛紝鐢ㄤ簬璁粌涓湡鐨勮噦闂村姣斻€?|
| `analyze_arms.py` | 鍗曟潯鏇茬嚎**鑷韩**鐨勮秼鍔匡細鏈€灏忎簩涔樻枩鐜?+ t 鍊?+ 鍓嶅悗鍗婂潎鍊笺€傚垽鏂?鍒板簳娑ㄦ病娑?銆?|
| `show_evals.py` | 浠?`metrics.jsonl` 閲屽垎鍑?`eval_validation` 搴忓垪 鈥斺€?璇ユ枃浠舵贩鐫€璁粌璁板綍鍜岄獙璇佽褰曪紝**鎸夎鏁版暟浼氭暟閿?*銆?|
| `summarize_baselines.py` | 鎶?`run_baselines.py` 鐨?`summary.json` 姹囨€绘垚姣忕瓥鐣ヤ竴琛岀殑琛ㄣ€?|
| `show_reward_breakdown.py` | 涓や釜 run 鐨勫鍔卞垎椤瑰苟鎺掑姣斻€俙rollout_debug.jsonl` 鐨勯敭閮芥槸 `mean_` 鍓嶇紑鐨勯€愭鍧囧€硷紱**鍔犱簡鏉冮噸鍚?`mean_reward` 璺ㄨ噦涓嶅彲姣?*锛岃兘姣旂殑鍙湁鎴愬姛鐜囦笌鐗╃悊閲忋€?|
