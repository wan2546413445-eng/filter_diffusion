
from pathlib import Path
import nibabel as nib

root = Path("/mnt/SSD/wsy/data")

for split in ["train", "val", "test"]:
    files = sorted((root / split).glob("*.nii.gz"))
    nums = []

    for f in files:
        nii = nib.load(str(f))
        nums.append(nii.shape[2])

    print(f"\n[{split}]")
    print("num_files =", len(nums))
    print("min_slices =", min(nums))
    print("max_slices =", max(nums))
    print("mean_slices =", sum(nums) / len(nums))