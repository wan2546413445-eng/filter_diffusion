from torch.utils.data import DataLoader
import time

from check_ixi_debug import train_ds

loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
)

t0 = time.time()
batch = next(iter(loader))
print("first batch time =", time.time() - t0)

kspace, mask, mask_fold = batch
print("batch kspace:", kspace.shape)
print("batch mask:", mask.shape)
print("batch mask_fold:", mask_fold.shape)