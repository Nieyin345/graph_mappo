import pandas as pd, h5py, numpy as np, os
ROOT = r'd:\destop\work_space\learning_space\论文\QKD-SAGIN生产端调度\qkd_rl'
full = pd.read_csv(os.path.join(ROOT, 'dataset', 'global', 'link_registry.csv'))
pos = {int(lid): i for i, lid in enumerate(full['link_id'])}
f = h5py.File(os.path.join(ROOT, 'dataset', 'global', 'link_data.h5'), 'r')
kmax = f['k_max']  # (525600, 2378)

hg = full[full['link_type'] == 'HAP-GS'].copy()
gs = hg['node_u'].copy(); hap = hg['node_v'].copy()
col = (hg['node_u'] == hg['node_v'] - 30) | (hg['node_v'] == hg['node_u'] - 30)
print('total HAP-GS =', len(hg), ' colocated =', int(col.sum()), ' cross =', int((~col).sum()))

cross = hg[~col]
stats = []
for _, r in cross.iterrows():
    p = pos[int(r['link_id'])]
    v = kmax[:, p].astype(np.float64)
    nz = v > 0
    g = int(r['node_u']) if r['node_u'] < 30 else int(r['node_v'])
    h = int(r['node_v']) if r['node_v'] < 30 else int(r['node_u'])
    stats.append((g, h, float(v.max()), float(v[nz].mean()) if nz.any() else 0.0, float(nz.mean())))
print('cross HAP-GS eff-time ratio distribution:')
vals = np.array([s[4] for s in stats])
print('  eff-time(>0) fraction mean=%.3f min=%.4f max=%.4f' % (vals.mean(), vals.min(), vals.max()))
peaks = np.array([s[2] for s in stats])
print('  peak-rate k_bps: min=%.2f median=%.2f mean=%.2f max=%.2f' % (peaks.min(), np.median(peaks), peaks.mean(), peaks.max()))
reliable = [s for s in stats if s[4] >= 0.1]
print('  cross pairs with eff-time>=0.1 :', len(reliable), '/', len(stats))
print('sample cross pairs (gs, hap, peak, mean_nz, eff_frac):')
for s in sorted(stats, key=lambda x: -x[2])[:12]:
    print('   gs%d-hap%d peak=%.0f mean_nz=%.0f eff=%.2f' % s)