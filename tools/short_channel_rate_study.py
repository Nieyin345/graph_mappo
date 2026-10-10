"""Controlled 24-hour rate-scale study; synthetic weather, production v7 optics.

Run on the test server, from the repository root. This writes diagnostic H5/CSV
files, not a scheduling/training dataset. Every link is evaluated independently;
network sums are potential capacity, not simultaneously achievable throughput.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.channel_v7.model import (
    ParamsV7, RateModelV7, asymptotic_components, asymptotic_from_components,
)
from tools.channel_v7.dataset import compute_block, generate_station_cloud_state
from tools.dataset_v2.build_dataset_v2 import (
    BuildConfig, read_stations, satellite_configs, build_nodes, candidate_links,
    geometry, sat_positions,
)
from tools.dataset_v2.finite_key import FiniteKeyParams, key_length_gains

SCENARIOS = {
    "clear": dict(A=1.1e-14, aod=0.1, high=0., lowmid=0.),
    "aerosol": dict(A=1.1e-14, aod=0.8, high=0., lowmid=0.),
    "strong_turbulence": dict(A=1e-13, aod=0.1, high=0., lowmid=0.),
    "cirrus": dict(A=1.1e-14, aod=0.1, high=1., lowmid=0.),
    "broken_cloud": dict(A=1.1e-14, aod=0.1, high=0., lowmid=0.5),
    "overcast": dict(A=1.1e-14, aod=0.1, high=0., lowmid=1.),
}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def slant(elevation, top_km, bottom_km=0.):
    r = 6371. + bottom_km
    angle = np.radians(elevation)
    return (np.sqrt((6371. + top_km)**2 - (r*np.cos(angle))**2) - r*np.sin(angle))*1000


def static_grid(model, fk):
    rows = []
    for name, weather in SCENARIOS.items():
        for lt in ("HAP-GS", "SAT-GS", "SAT-HAP", "SAT-SAT"):
            for day in (False, True):
                for value in ([500, 1000, 2000, 4000] if lt == "SAT-SAT" else [15, 30, 60, 90]):
                    elev = 90 if lt == "SAT-SAT" else value
                    distance = value*1000 if lt == "SAT-SAT" else slant(
                        elev, 20 if lt == "HAP-GS" else 550, 20 if lt == "SAT-HAP" else 0)
                    kw = dict(sun=np.array([30 if day else -30]))
                    ground = lt in ("HAP-GS", "SAT-GS")
                    if ground:
                        kw.update(A=np.array([weather["A"]]), aod=weather["aod"],
                                  high_cloud=weather["high"], tau_cirrus=0.3)
                    st = model.optical_states(lt, np.array([distance]), np.array([elev]), **kw)
                    q, e, v, y, t = asymptotic_components(st["eta"], st["y0"], st["weights"], fk)
                    compatible = bool(st["moment_compatible"][0])
                    keep = 1-weather["lowmid"] if ground else 1.
                    row = dict(scenario=name, link_type=lt, day=day, distance_km=distance/1000,
                               elevation_deg=elev, rytov=float(st["rytov"][0]),
                               domain_valid=bool(st["domain_valid"][0]), moment_compatible=compatible,
                               mean_eta=float((st["eta"]*st["weights"]).sum(axis=0)[0]),
                               qber_signal=float(e[0, 0]/q[0, 0]), clear_probability=keep)
                    for duration in (60, 300):
                        rate = float(key_length_gains(q, e, fk.f_rep*duration, fk)[0]/duration)
                        row[f"conditional_finite_{duration}_bps"] = rate if compatible else float("nan")
                    proxy = float(asymptotic_from_components(q, e, v, y, t, fk)[0]*fk.f_rep)
                    row["conditional_proxy_bps"] = proxy if compatible else float("nan")
                    row["expected_gated_proxy_bps"] = keep*proxy if compatible else float("nan")
                    rows.append(row)
    return rows


def released_keys(stats, first, visibility, gates, fk, duration):
    """Independent continuous contacts, 5 s PAT only on contact acquisition.

    Close on geometric loss, duration limit, or study end. Cloud outages advance
    elapsed time but contribute zero retained pulses (main beacon-gated model).
    """
    T, L = visibility.shape
    slot_s = float(fk.slot_s)
    if not np.isfinite(duration) or duration <= 0 or not float(duration/slot_s).is_integer():
        raise ValueError("duration must be a positive integer multiple of slot_s")
    output = np.zeros((T, L))
    n = np.zeros(L)
    gains = np.zeros((L, 3))
    errors = np.zeros((L, 3))
    elapsed = np.zeros(L, dtype=int)
    previous = np.zeros(L, dtype=bool)

    def close(mask, step):
        if np.any(mask):
            q = np.divide(gains[mask], n[mask, None], out=np.zeros_like(gains[mask]), where=n[mask, None] > 0)
            e = np.divide(errors[mask], n[mask, None], out=np.zeros_like(errors[mask]), where=n[mask, None] > 0)
            valid = np.isfinite(n[mask]) & np.isfinite(gains[mask]).all(axis=1) & np.isfinite(errors[mask]).all(axis=1)
            released = np.full(int(mask.sum()), np.nan)
            if np.any(valid):
                released[valid] = key_length_gains(q[valid], e[valid], n[mask][valid], fk)
            output[step, mask] += released
            n[mask], gains[mask], errors[mask], elapsed[mask] = 0, 0, 0, 0

    for step in range(T):
        active = visibility[step].astype(bool)
        close(previous & ~active, step)
        new = active & ~previous
        q = np.where(new[:, None], first["gain"][step], stats["gain"][step])
        e = fk.e_d*q + (0.5-fk.e_d)*q[:, -1, None]
        fraction = np.where(new, first["usable_fraction"][step], stats["usable_fraction"][step])
        # An incompatible slot invalidates its whole retained block. Preserve
        # NaNs in the output rather than treating invalid physics as an outage.
        pulses = np.where(active & (gates[step] > 0), fraction*fk.n_pulses, 0.)
        n += pulses
        gains += np.where(pulses[:, None] > 0, q*pulses[:, None],
                          np.where(np.isnan(pulses[:, None]), np.nan, 0.))
        errors += np.where(pulses[:, None] > 0, e*pulses[:, None],
                           np.where(np.isnan(pulses[:, None]), np.nan, 0.))
        elapsed += active
        close(active & (elapsed >= int(duration/slot_s)), step)
        previous = active
    close(elapsed > 0, T-1)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    p, fk = ParamsV7(), FiniteKeyParams()
    model = RateModelV7(p, sat_alt_km=550, hap_alt_km=20)
    grid = static_grid(model, fk)
    write_csv(args.out/"static_grid.csv", grid)
    cfg = BuildConfig(channel_model="v7", hap_mobility="static", sat_constellation="walker60",
                      cloud_model="station_markov", alpha_min_deg=15, h_min_km=85,
                      domain_policy="extrapolate", intra_slot_samples=4)
    stations = [s for s in read_stations(ROOT/"tools/channel_v2/stations_45_with_elevation.csv")
                if s["name"] in ("London", "Beijing", "Denver")]
    satellites = satellite_configs("walker60")
    nodes, G, S = build_nodes(stations, cfg, satellites)
    links, _ = candidate_links(nodes, G, S, cfg)
    # Retain all 1770 satellite pairs: adjacent in-plane pairs alone sit near
    # 4277 km and would miss the much shorter inter-plane contacts.
    write_csv(args.out/"links.csv", [dict(link_id=i,u=u,v=v,link_type=lt) for i,(u,v,lt) in enumerate(links)])
    write_csv(args.out/"nodes.csv", nodes)
    T, L = 1440, len(links)
    state = dict(cfg=cfg, fk=fk, model=model, G=G, sat_cfgs=satellites,
                 gs_lat=np.array([s["lat"] for s in stations]), gs_lon=np.array([s["lon"] for s in stations]),
                 gs_alt_km=np.array([s["alt_km"] for s in stations]),
                 link_u=np.array([x[0] for x in links]), link_v=np.array([x[1] for x in links]),
                 link_type=np.array([x[2] for x in links]))
    summary = []
    for name, weather in SCENARIOS.items():
        print(f"starting {name}: {T} slots, {L} links", flush=True)
        cloud, _ = generate_station_cloud_state(np.full((G,24),weather["lowmid"]), T, seed=2023)
        state.update(cloud_state=cloud, c_hi=np.full((G,24),weather["high"]),
                     aod=np.full((G,24),weather["aod"]), a_cn2=np.full((G,24),weather["A"]))
        chunks = [compute_block(state, start, min(start+120,T), geometry, sat_positions)
                  for start in range(0,T,120)]
        rate = np.concatenate([x[1] for x in chunks])
        visible = np.concatenate([x[3] for x in chunks])
        stats = {key:np.concatenate([x[-2][key] for x in chunks]) for key in chunks[0][-2]}
        first = {key:np.concatenate([x[-1][key] for x in chunks]) for key in chunks[0][-1]}
        gates = np.ones((T,L))
        for col,(u,v,lt) in enumerate(links):
            if lt in ("HAP-GS","SAT-GS"):
                gates[:,col] = cloud[u]
        r60 = released_keys(stats,first,visible,gates,fk,60)
        r300 = released_keys(stats,first,visible,gates,fk,300)
        proxy = stats["proxy_per_pulse"]*stats["usable_fraction"]*fk.f_rep
        with h5py.File(args.out/f"{name}.h5","w") as f:
            for key,array in dict(conditional_finite60_bps=rate, conditional_proxy_bps=proxy,
                                  geometry_visible=visible, cloud_gate=gates, released60_bits=r60,
                                  released300_bits=r300, domain_fraction=stats["model_domain"]).items():
                f.create_dataset(key,data=array,compression="gzip")
            f.attrs["weather_kind"] = "synthetic controlled scenario, NOT observed weather"
            f.attrs["scenario_json"] = json.dumps(weather)
        for lt in sorted(set(state["link_type"])):
            cols = state["link_type"] == lt
            selected = rate[:,cols][visible[:,cols].astype(bool)]
            positive = selected[np.isfinite(selected)&(selected>0)]
            domain_total = stats["domain_visible_substeps"][:,cols].sum()
            summary.append(dict(scenario=name,link_type=lt,links=int(cols.sum()),
                visible_slots=int(visible[:,cols].sum()),positive_fraction_visible=float(np.mean(selected>0)),
                invalid_fraction_visible=float(np.mean(~np.isfinite(selected))),
                outside_domain_fraction=float(1-stats["domain_valid_substeps"][:,cols].sum()/domain_total),
                conditional_positive_p10_bps=float(np.percentile(positive,10)) if len(positive) else 0.,
                conditional_positive_median_bps=float(np.median(positive)) if len(positive) else 0.,
                conditional_positive_p90_bps=float(np.percentile(positive,90)) if len(positive) else 0.,
                conditional_max_bps=float(np.nanmax(selected)),
                cloud_clear_fraction_visible=float(gates[:,cols][visible[:,cols].astype(bool)].mean()),
                total_released60_bits=float(r60[:,cols].sum()),total_released300_bits=float(r300[:,cols].sum())))
        write_csv(args.out/"daily_summary.csv",summary)
    (args.out/"metadata.json").write_text(json.dumps(dict(
        weather_kind="controlled synthetic, not real weather validation", scenarios=SCENARIOS,
        config=dataclasses.asdict(cfg),optics=dataclasses.asdict(p),protocol=fk.as_dict(),
        independent_links=True, time_start="2023-01-01T00:00:00Z",slots=T,stations=stations,
        satellites=S,links=L,
        source_sha256={str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            for path in (Path("tools/short_channel_rate_study.py"),
                         Path("tools/channel_v7/model.py"), Path("tools/channel_v7/dataset.py"),
                         Path("tools/channel_v2/rate_model_v2.py"),
                         Path("tools/dataset_v2/build_dataset_v2.py"), Path("tools/dataset_v2/finite_key.py"))}),indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
