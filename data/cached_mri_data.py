import torch
from pathlib import Path
from tqdm import tqdm
import pickle
import os


class CachedSliceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        original_dataset,
        cache_root: str,
        slice_info_pkl: str,
        force_rebuild: bool = False,
        num_skip_slice: int = 6,
    ):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.original_dataset = original_dataset
        self.slice_info_pkl = slice_info_pkl
        self.num_skip_slice = int(num_skip_slice)

        # 读取切片数量信息（用于验证缓存完整性）
        with open(slice_info_pkl, "rb") as f:
            self.slice_dict = pickle.load(f)

        self.cached_samples = []
        self._init_cache(force_rebuild)

    def _init_cache(self, force_rebuild):
        cache_meta = self.cache_root / "cache_meta.pt"
        expected_total = sum(max(0, v - self.num_skip_slice) for v in self.slice_dict.values())

        if force_rebuild or not cache_meta.exists():
            self._build_cache()
        else:
            meta = torch.load(cache_meta)
            self.cached_samples = meta["samples"]

            if len(self.original_dataset) != expected_total:
                print(
                    f"Warning: original_dataset len ({len(self.original_dataset)}) "
                    f"!= expected_total from pkl ({expected_total})"
                )

            if len(self.cached_samples) != len(self.original_dataset):
                print(
                    f"Cache incomplete ({len(self.cached_samples)} vs {len(self.original_dataset)}), rebuilding..."
                )
                self._build_cache()
            else:
                print(f"Loaded {len(self.cached_samples)} cached samples from {self.cache_root}")
    def _build_cache(self):
        print(f"Building cache in {self.cache_root}...")
        self.cached_samples = []
        for idx in tqdm(range(len(self.original_dataset)), desc="Caching"):
            sample = self.original_dataset[idx]  # (kspace, mask, mask_fold)
            cache_path = self.cache_root / f"{idx:06d}.pt"
            torch.save(sample, cache_path)
            self.cached_samples.append(str(cache_path))

        # 保存元信息
        torch.save({"samples": self.cached_samples}, self.cache_root / "cache_meta.pt")
        print(f"Cache built: {len(self.cached_samples)} samples.")

    def __len__(self):
        return len(self.cached_samples)

    def __getitem__(self, idx):
        cache_path = self.cached_samples[idx]
        kspace, mask, mask_fold = torch.load(cache_path, weights_only=False)
        return kspace, mask, mask_fold