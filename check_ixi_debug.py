import time
import numpy as np
from pathlib import Path

t0 = time.time()
print("[0] script start", flush=True)

from data.ixi_singlecoil_dataset import IXISinglecoilSliceDataset
print(f"[1] import done, cost={time.time()-t0:.2f}s", flush=True)

def dummy_mask_func():
    H = W = 256
    mask = np.ones((1, H, W), dtype=np.float32)
    mask_fold = np.ones((1, H, W), dtype=np.float32)
    return mask, mask_fold

root = "/mnt/SSD/wsy/data/train"
files = sorted(Path(root).glob("*.nii.gz"))
print(f"[2] found {len(files)} files", flush=True)
if len(files) > 0:
    print(f"[3] first file = {files[0]}", flush=True)

t1 = time.time()
train_ds = IXISinglecoilSliceDataset(
    root=root,
    mask_func=dummy_mask_func,
    image_size=256,
    num_skip_slice=20,
    normalize_mode="max",   # 先别用 percentile
)
print(f"[4] dataset built, len={len(train_ds)}, cost={time.time()-t1:.2f}s", flush=True)

t2 = time.time()
sample = train_ds[0]
print(f"[5] first sample loaded, cost={time.time()-t2:.2f}s", flush=True)

kspace, mask, mask_fold = sample
print("[6] kspace shape:", kspace.shape, flush=True)
print("[7] mask shape:", mask.shape, flush=True)
print("[8] mask_fold shape:", mask_fold.shape, flush=True)
print("[9] all done", flush=True)