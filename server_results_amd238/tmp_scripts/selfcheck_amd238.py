"""amd238 装配自检：代码可导入、数据可读、BC 权重可载。"""
import torch
import h5py
import qkd_rl  # noqa: F401

ck = torch.load(
    "outputs/supervised_pg_phased/supervised_pg_phased_latest.pt",
    map_location="cpu",
    weights_only=False,
)
print("ckpt_keys", len(ck))
with h5py.File("dataset/global/link_data.h5", "r") as f:
    print("h5_keys", list(f.keys())[:6])
print("SELF_CHECK_OK")
