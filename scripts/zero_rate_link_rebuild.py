import h5py
import numpy as np
import pandas as pd
import os

ROOT = r'd:\destop\work_space\learning_space\论文\QKD-SAGIN生产端调度\qkd_rl'
ds = os.path.join(ROOT, 'dataset', 'global')
link_path = os.path.join(ds, 'link_data.h5')
xml = h5py.File(link_path, 'r')
kmax = xml['k_max']
T, L0 = kmax.shape
chunk = kmax.chunks[0]
old_reg = xml['link_registry'][:]

zero = pd.read_csv(os.path.join(ds, '_zero_rate_links.csv'))
zero_ids = set(int(l) for l in zero['link_id'])
keep = np.array([i for i in range(L0) if i not in zero_ids], dtype=np.int64)
new_reg = old_reg[keep].copy()
# renumber link_id to consecutive 0..N'-1 (order preserved)
new_reg['link_id'] = np.arange(len(new_reg), dtype=np.int32)

tmp_path = link_path + '.tmp'
with h5py.File(tmp_path, 'w') as wf:
    dw = wf.create_dataset('k_max', shape=(T, len(new_reg)), dtype=np.float32, chunks=(chunk, len(new_reg)), compression='gzip')
    wf.create_dataset('link_registry', data=new_reg)
    for k, v in xml.attrs.items():
        wf.attrs[k] = v
    r0 = 0
    while r0 < T:
        r1 = min(r0 + chunk, T)
        blk = np.asarray(kmax[r0:r1])  # (rows, L0)
        dw[r0:r1] = blk[:, keep]
        r0 = r1
        if r0 % (chunk * 32) == 0:
            print('  wrote %d / %d rows' % (r0, T), flush=True)
xml.close()

os.replace(tmp_path, link_path)
print('rewrote link_data.h5: L %d -> %d' % (L0, len(new_reg)))

# --- update CSVs ---
reg = pd.read_csv(os.path.join(ds, 'link_registry.csv'))
kept_reg = reg.iloc[list(keep)].copy()
kept_reg['link_id'] = np.arange(len(kept_reg))
kept_reg.to_csv(os.path.join(ds, 'link_registry.csv'), index=False)

zero_rows = reg.iloc[list(zero_ids)].copy()
zero_rows['reason'] = 'zero_rate_all_year'
sk_path = os.path.join(ds, 'link_registry_skipped.csv')
sk = pd.read_csv(sk_path) if os.path.exists(sk_path) else None
zero_rows = zero_rows[['link_id', 'node_u', 'node_v', 'link_type', 'reason']]
out = pd.concat([sk, zero_rows], ignore_index=True) if sk is not None else zero_rows
out.to_csv(sk_path, index=False)
print('link_registry.csv: %d -> %d rows' % (len(reg), len(kept_reg)))
print('skipped.csv += %d zero-rate rows (total %d)' % (len(zero_rows), len(out)))
os.remove(os.path.join(ds, '_zero_rate_links.csv'))
print('done')