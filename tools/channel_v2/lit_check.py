import math, numpy as np, copy
from rate_model_v2 import ParamsV2, RateModelV2
R=6371.0
def elev_from_dist(L,h):
    # L^2 = R^2 + (R+h)^2 - 2R(R+h)cos(g); sin(e) = ((R+h)^2 - R^2 - L^2)/(2 R L)
    return math.degrees(math.asin(((R+h)**2-R**2-L**2)/(2*R*L)))
def db(x): return 10*np.log10(x)

print("== 1. Micius (Liao 2017): 848.6 nm, 10 urad full divergence, 1 m RX, multimode, h=500 km")
lam=848.6e-9; w0=2*lam/(math.pi*10e-6)
p=ParamsV2(); p.lam=lam; p.w0={"SAT":w0,"HAP":w0}; p.d_rx=dict(p.d_rx,GS=1.0); p.sigma_theta=dict(p.sigma_theta,SAT=1.2e-6)
m=RateModelV2(p,sat_alt_km=500)
for L in (645,1200):
    e=elev_from_dist(L,500)
    for nm,A in (("night",1.1e-14),):
        bp,T=m.eta_bp(np.array([L*1e3]),"SAT","GS",np.array([e]),A,m.hv_sat)
        mol=m.eta_mol(np.array([e]),1e3); aer=m.eta_aer(np.array([e]),0.2)
        ch=bp*mol*aer
        print(f"  L={L} km elev={e:.1f}: geometric+pointing {db(bp)[0]:.1f} dB, atm {db(mol*aer)[0]:.1f} dB, "
              f"channel {db(ch)[0]:.1f} dB, +RX optics 0.5 -> {db(ch*0.5)[0]:.1f} dB (w0={w0*100:.1f} cm)")
print("  measured link efficiency (transmitter output -> front of detectors): ~22 dB @645 km, ~35 dB @1200 km")

print("== 2. v2 base, SAT-GS total loss incl. SMF coupling and detector")
m2=RateModelV2(ParamsV2(),sat_alt_km=550)
for e in (90,60,45,30,20,15,10):
    s=math.radians(e); L=math.sqrt((R+550)**2-(R*math.cos(s))**2)-R*math.sin(s)
    for nm,A,sun in (("night",1.1e-14,-10),("day",2.75e-14,30)):
        r,pt=m2.link_rate("SAT-GS",np.array([L*1e3]),np.array([float(e)]),sun_rx_deg=np.array([sun]),aod550=np.array([0.15]),a_cn2=np.array([A]),return_parts=True)
        print(f"  elev {e:2d} L={L:6.0f} km {nm:5s}: eta {db(pt['eta'])[0]:6.1f} dB  bp {db(pt['bp'])[0]:6.1f}  smf {db(pt['phi'])[0]:5.1f}  r0 {pt['r0'][0]*100:5.1f} cm  SKR {r[0]:9.0f} bps  ({r[0]/1e8:.2e} b/pulse)")
print("== 3. HAP-GS zenith / 30 deg")
for e in (90,30,15):
    L=20/math.sin(math.radians(e))
    r,pt=m2.link_rate("HAP-GS",np.array([L*1e3]),np.array([float(e)]),sun_rx_deg=np.array([-10]),aod550=np.array([0.15]),a_cn2=np.array([1.1e-14]),return_parts=True)
    print(f"  elev {e} L={L:.0f} km: eta {db(pt['eta'])[0]:.1f} dB (bp {db(pt['bp'])[0]:.2f}, smf {db(pt['phi'])[0]:.1f}), SKR {r[0]/1e6:.2f} Mbps")
print("== 4. Chu 2020 HAP style tolerance: max loss with SPAD 25%, 500 cps equivalent")
print("== 5. Karakosta 2025 Table 3 geometry: balloon 35 km, W0 10 cm, D_rx 0.4 m, 1 urad, Cn2(0)=9.6e-14, SNSPD .85")
import math
p=ParamsV2(); p.d_rx=dict(p.d_rx,GS=0.4); p.sigma_theta=dict(p.sigma_theta,HAP=1e-6)
m5=RateModelV2(p,sat_alt_km=550,hap_alt_km=35)
for e,lit in ((90,112.01e3),(20.5,2.21e3)):
    L=35/math.sin(math.radians(e))
    r,pt=m5.link_rate("HAP-GS",np.array([L*1e3]),np.array([float(e)]),sun_rx_deg=np.array([-10]),aod550=np.array([0.1]),a_cn2=np.array([9.6e-14]),return_parts=True)
    eta=pt['eta'][0]; h=lambda x:-x*math.log2(x)-(1-x)*math.log2(1-x)
    skr_their=80e6*0.01*eta*(1-2*h(0.04))
    eta_lit=lit/(80e6*0.01*(1-2*h(0.04)))
    print(f"  elev {e} L={L:.0f} km: our eta {db(eta):.1f} dB (smf {db(pt['phi'])[0]:.1f}, bp {db(pt['bp'])[0]:.1f}) -> SKR with their protocol {skr_their/1e3:.1f} kbps; "
          f"their {lit/1e3:.2f} kbps (implied eta {db(eta_lit):.1f} dB); our own protocol {r[0]/1e3:.0f} kbps")
print("   same, but AO to radial order 6 (J=28, Noll Delta_J = 0.2944 J^-0.866 = %.4f) as in Karakosta:" % (0.2944*28**(-math.sqrt(3)/2)))
p.delta3=0.2944*28**(-math.sqrt(3)/2); p.smf_model='core_halo'  # high-order AO: s2 << 1, Marechal core valid
m6=RateModelV2(p,sat_alt_km=550,hap_alt_km=35)
for e,lit in ((90,112.01e3),(20.5,2.21e3)):
    L=35/math.sin(math.radians(e))
    r,pt=m6.link_rate("HAP-GS",np.array([L*1e3]),np.array([float(e)]),sun_rx_deg=np.array([-10]),aod550=np.array([0.1]),a_cn2=np.array([9.6e-14]),return_parts=True)
    eta=pt['eta'][0]; skr_their=80e6*0.01*eta*(1-2*h(0.04))
    print(f"  elev {e}: our eta {db(eta):.1f} dB (smf {db(pt['phi'])[0]:.1f}) -> {skr_their/1e3:.1f} kbps vs their {lit/1e3:.2f} kbps")
