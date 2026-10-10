import numpy as np
from rate_model_v2 import ParamsV2, RateModelV2

p = ParamsV2()
m = RateModelV2(p, sat_alt_km=550.0)

print("=== 1. Ground Elevation Cut-off (elev_min_deg = 15.0 deg) ===")
for el in [90.0, 30.0, 20.0, 15.0, 14.9, 10.0, 5.0]:
    r = m.link_rate("SAT-GS", 1000e3, el, sun_rx_deg=-10.0, aod550=0.1, a_cn2=p.a_night)
    status = "ACTIVE" if r > 0 else "CUT-OFF MASKED (0 bps)"
    print(f"  Elev = {el:>4.1f} deg -> Rate = {r:>8.1f} bps ({status})")

print("\n=== 2. ISL Range Cut-off (isl_max_dist_m = 4500 km) ===")
for dist_km in [1000.0, 3000.0, 4400.0, 4500.0, 4600.0, 5000.0]:
    r = m.link_rate("SAT-SAT", dist_km * 1e3, 0.0, isl_illumination="umbra")
    status = "ACTIVE" if r > 0 else "RANGE CUT-OFF (0 bps)"
    print(f"  ISL Dist = {dist_km:>6.1f} km -> Rate = {r:>8.1f} bps ({status})")

print("\n=== 3. ISL Earth Occultation / Atmosphere Cut-off (grazing_alt >= 85 km) ===")
for g_alt in [500.0, 150.0, 90.0, 85.0, 84.9, 50.0, 0.0, -100.0]:
    r = m.link_rate("SAT-SAT", 2000e3, 0.0, isl_illumination="umbra", grazing_alt_km=g_alt)
    status = "ACTIVE" if r > 0 else "EARTH OCCULTATION MASKED (0 bps)"
    print(f"  Grazing Alt = {g_alt:>6.1f} km -> Rate = {r:>8.1f} bps ({status})")
