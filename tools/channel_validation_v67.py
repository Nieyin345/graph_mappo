# -*- coding: utf-8 -*-
"""四类链路速率验证：vs 仰角 / vs 昼夜 / vs 文献锚点。

输出：docs/images/channel_validation_v67/ 下的 md 报告 + csv。
不改任何模型参数，纯读取当前 rate_model_v2 + finite_key。
运行（服务器）：/opt/qkd/venv/bin/python tools/channel_validation_v67.py
"""
from __future__ import annotations
import math, os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "channel_v2"))
sys.path.insert(0, os.path.join(ROOT, "tools", "dataset_v2"))
from rate_model_v2 import ParamsV2, RateModelV2, airmass_kasten_young, rayleigh_tau  # noqa
from finite_key import FiniteKeyParams, slot_rate, slot_rate_states, key_length  # noqa

R_E = 6371e3
H_SAT = 550e3
H_HAP = 20e3
fk = FiniteKeyParams()
p = ParamsV2()
model = RateModelV2(p, sat_alt_km=H_SAT / 1e3, hap_alt_km=H_HAP / 1e3)

def slant_to_sat(elev_deg, h=H_SAT):
    e = np.radians(np.asarray(elev_deg, dtype=float))
    return np.sqrt((R_E + h) ** 2 - (R_E * np.cos(e)) ** 2) - R_E * np.sin(e)

def sat_sat_geom(elev_deg, h=H_SAT):
    """给定卫星星下点单侧仰角，返回两端星间斜距（同高度、对地心张角 2*(90-elev)）。"""
    e = np.radians(np.asarray(elev_deg, dtype=float))
    # 星下点仰角 elev -> 卫星对地心的张角 alpha = 90 - elev
    alpha = np.pi / 2 - e
    d = 2 * (R_E + h) * np.sin(alpha)   # 弦长
    return d

def hap_alt_for_elev(elev_deg, h=H_HAP):
    """HAP-GS：由仰角反推水平距离（HAP 在地面站上空 20km 的球面关系）。"""
    e = np.radians(np.asarray(elev_deg, dtype=float))
    return np.sqrt((R_E + h) ** 2 - (R_E * np.cos(e)) ** 2) - R_E * np.sin(e)

def sat_hap_dist(elev_deg, h_sat=H_SAT, h_hap=H_HAP):
    """SAT-HAP 近似：用卫星到 HAP 的直线距离（HAP 在地面站上方）。"""
    # 简化：卫星仰角 elev（相对 HAP 所在站点），斜距 ~ (h_sat-h_hap)/sin(elev)
    e = np.radians(np.asarray(elev_deg, dtype=float))
    return np.maximum((h_sat - h_hap) / np.sin(e), h_sat - h_hap)

elevs = np.array([5., 10., 15., 20., 30., 45., 60., 75., 90.])

def rate_finite(ltype, L, elev, *, sun_rx, aod, acn2, isl=None, graze=None, night=True):
    """用块内汇总（新规范）算有限密钥速率，含闪烁 GH 期望。"""
    kw = dict(sun_rx_deg=np.full_like(np.asarray(elev, dtype=float), sun_rx),
              aod550=np.full_like(np.asarray(elev, dtype=float), aod),
              a_cn2=np.full_like(np.asarray(elev, dtype=float), acn2))
    if isl is not None:
        kw["isl_illumination"] = np.array([isl] * len(elev))
    if graze is not None:
        kw["grazing_alt_km"] = np.full_like(np.asarray(elev, dtype=float), graze)
    _, parts = model.link_rate(ltype, np.asarray(L, float), np.asarray(elev, float),
                               return_parts=True, **kw)
    eta = parts["eta"]; y0 = parts["y0"]
    sc_h = parts.get("scint_h"); sc_w = parts.get("scint_w")
    if sc_h is None:
        return np.asarray(slot_rate(eta, y0, fk), float)
    eta_arr = np.asarray(eta, float)
    eta_s = np.stack([eta_arr * sc_h[k] for k in range(len(sc_w))], 0)
    y0_s = np.broadcast_to(y0, eta_s.shape)
    n_s = np.stack([sc_w[k] * np.full_like(eta_arr, fk.n_pulses) for k in range(len(sc_w))], 0)
    return np.asarray(slot_rate_states(eta_s, y0_s, n_s, fk), float)

# ---------------- 四类链路 @ 夜、晴、AOD=0.1 ----------------
res = {}
res["SAT-GS"] = rate_finite("SAT-GS", slant_to_sat(elevs), elevs, sun_rx=-10, aod=0.1, acn2=p.a_night)
res["HAP-GS"] = rate_finite("HAP-GS", hap_alt_for_elev(elevs), elevs, sun_rx=-10, aod=0.1, acn2=p.a_night)
res["SAT-HAP"] = rate_finite("SAT-HAP", sat_hap_dist(elevs), elevs, sun_rx=-10, aod=0.0, acn2=p.a_night)
# SAT-SAT：星间链路，直接扫斜距（4500km 门限内），固定仰角无关；背景取 umbra（地影）
ss_L = np.array([500e3, 1000e3, 2000e3, 3000e3, 4500e3])
res["SAT-SAT"] = (ss_L, rate_finite("SAT-SAT", ss_L, np.full_like(ss_L, 90.0), sun_rx=-10, aod=0.0,
                                     acn2=p.a_night, isl="umbra", graze=200e3))

# ---------------- 昼夜对比 (SAT-GS, HAP-GS @ 仰角集合) ----------------
day_night = {}
for lt, Lfun in (("SAT-GS", slant_to_sat), ("HAP-GS", hap_alt_for_elev)):
    r_n = rate_finite(lt, Lfun(elevs), elevs, sun_rx=-10, aod=0.1, acn2=p.a_night)
    r_d = rate_finite(lt, Lfun(elevs), elevs, sun_rx=+30, aod=0.1, acn2=p.a_day)
    day_night[lt] = (r_n, r_d)

# ---------------- 清除云/天气：纯 clear ----------------
print("=" * 78)
print("四类链路 slot 有限密钥速率 [bit/s]（夜、晴、AOD=0.1、50% 占空无云）")
print("=" * 78)
hdr = "elev(deg): " + "".join(f"{e:>10.0f}" for e in elevs)
print(hdr)
for lt in ("SAT-GS", "HAP-GS", "SAT-HAP"):
    print(f"{lt:10s}: " + "".join(f"{v:>10.1f}" for v in res[lt]))
print()
print("SAT-SAT（星间，扫斜距，仰角=90°，地影背景）：")
print("  L(km)    : " + "".join(f"{L/1e3:>10.0f}" for L in res["SAT-SAT"][0]))
print("  rate(bps): " + "".join(f"{v:>10.1f}" for v in res["SAT-SAT"][1]))
print()
print("=" * 78)
print("昼夜对比（SAT-GS / HAP-GS）：夜 vs 白天(太阳高度+30°)")
print("=" * 78)
for lt in ("SAT-GS", "HAP-GS"):
    rn, rd = day_night[lt]
    print(f"{lt:10s} 夜: " + "".join(f"{v:>10.1f}" for v in rn))
    print(f"{'':10s} 昼: " + "".join(f"{v:>10.1f}" for v in rd))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rn > 0, rd / rn, np.nan)
    print(f"{'':10s} 昼/夜比: " + "".join(f"{v:>10.2f}" for v in ratio))
print()

# ---------------- 文献锚点对比 ----------------
print("=" * 78)
print("文献锚点对比（数量级）")
print("=" * 78)
# SAT-GS @ 天顶 vs 文献
zen = np.array([90.0])
r_zen = rate_finite("SAT-GS", slant_to_sat(zen), zen, sun_rx=-10, aod=0.1, acn2=p.a_night)[0]
r_zen_sift = r_zen * 2.0   # 粗略 sifted = 2x final? 仅数量级
print(f"本模型 SAT-GS 天顶有限密钥 = {r_zen:.0f} bit/s")
print(f"  文献: Micius sifted 40.2 kbps @530km (Liao2017); final 1.1 kbps;  "
      f"Chen2021 final 47.8 kbps 均值; Khmelev2023 sifted 25.3 kbps(600mm)")
# 30 度
e30 = np.array([30.0])
r30 = rate_finite("SAT-GS", slant_to_sat(e30), e30, sun_rx=-10, aod=0.1, acn2=p.a_night)[0]
print(f"本模型 SAT-GS @30° = {r30:.0f} bit/s（文献带 10^3 bit/s 量级）")
# HAP-GS 天顶
rh = rate_finite("HAP-GS", hap_alt_for_elev(zen), zen, sun_rx=-10, aod=0.1, acn2=p.a_night)[0]
print(f"本模型 HAP-GS 天顶 = {rh:.0f} bit/s（预期 10^5-10^6）")
# 衍射损耗对比 Liao2017 22dB（我们 1550nm）
L1200 = np.array([1200e3])
_, parts = model.link_rate("SAT-GS", L1200, np.array([19.8]), sun_rx_deg=np.array([-10.]),
                           aod550=np.array([0.1]), a_cn2=np.array([p.a_night]), return_parts=True)
bp = float(parts["bp"][0]); mol = float(parts["mol"][0]); aer = float(parts["aer"][0]); phi = float(parts["phi"][0])
print(f"本模型 @1200km/elev19.8°: η_bp={-10*math.log10(bp):.2f}dB, η_mol={-10*math.log10(mol):.2f}dB, "
      f"η_aer={-10*math.log10(aer):.2f}dB, η_φ={-10*math.log10(phi):.2f}dB")
print(f"  文献 Liao2017（848.6nm, 300mm发/1m收）: 衍射 22dB + 指向<3dB + 大气3-8dB")

# ---------------- 存盘 ----------------
OUT = os.path.join(ROOT, "docs", "images", "channel_validation_v67")
os.makedirs(OUT, exist_ok=True)
json.dump({lt: (res[lt].tolist() if isinstance(res[lt], np.ndarray) else [res[lt][0].tolist(), res[lt][1].tolist()])
           for lt in res}, open(os.path.join(OUT, "rates.json"), "w"), indent=1)
print("\nsaved ->", OUT)
