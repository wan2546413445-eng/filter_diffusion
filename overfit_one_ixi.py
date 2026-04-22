import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import fastmri

from data.ixi_singlecoil_dataset import IXISinglecoilSliceDataset


def build_fixed_cartesian_mask_func(H=256, W=256, acc=4, acs=24, seed=1234):
    rng = np.random.RandomState(seed)

    center = W // 2
    left = center - acs // 2
    right = left + acs

    sampled_cols = set(range(left, right))
    num_keep = max(acs, W // acc)

    candidates = [i for i in range(W) if i not in sampled_cols]
    rng.shuffle(candidates)
    extra = max(0, num_keep - len(sampled_cols))
    sampled_cols.update(candidates[:extra])

    col_mask = np.zeros(W, dtype=np.float32)
    col_mask[list(sampled_cols)] = 1.0

    mask_2d = np.tile(col_mask[None, :], (H, 1))
    mask = mask_2d[None, ...].astype(np.float32)   # [1, H, W]
    mask_fold = mask.copy()

    def mask_func():
        return mask.copy(), mask_fold.copy()

    return mask_func


class SmallUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=32):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bot = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)
        return out


def psnr_torch(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return -10.0 * torch.log10(mse + 1e-8)


def prepare_one_sample(dataset, idx, device):
    kspace, mask, _ = dataset[idx]   # kspace:[1,H,W,2], mask:[1,H,W]

    kspace = kspace.to(device)
    mask = mask.to(device)

    # GT magnitude
    img_gt = fastmri.ifft2c(kspace)          # [1,H,W,2]
    img_gt = fastmri.complex_abs(img_gt)     # [1,H,W]

    # ZF
    kspace_us = kspace * mask.unsqueeze(-1)  # [1,H,W,2]
    img_zf = fastmri.ifft2c(kspace_us)       # [1,H,W,2]

    # 输入改成 [B,C,H,W]
    x_in = img_zf.permute(0, 3, 1, 2).contiguous()      # [1,2,H,W]
    y_gt = img_gt.unsqueeze(1).contiguous()             # [1,1,H,W]
    zf_mag = fastmri.complex_abs(img_zf).unsqueeze(1)   # [1,1,H,W]

    return x_in, y_gt, zf_mag


@torch.no_grad()
def save_vis(model, x_in, y_gt, zf_mag, save_path):
    model.eval()
    pred = model(x_in)
    pred = pred.clamp(0.0, 1.0)

    pred_img = pred[0, 0].cpu()
    gt_img = y_gt[0, 0].cpu()
    zf_img = zf_mag[0, 0].cpu()
    err_img = torch.abs(pred_img - gt_img)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(gt_img.numpy(), cmap="gray")
    plt.title("GT")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(zf_img.numpy(), cmap="gray")
    plt.title("ZF")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred_img.numpy(), cmap="gray")
    plt.title("Pred")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(err_img.numpy(), cmap="gray")
    plt.title("|Pred-GT|")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def get_grad_mean_sum(model):
    grad_mean_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_mean_sum += p.grad.abs().mean().item()
    return grad_mean_sum


def main():
    os.makedirs("./debug_overfit_ixi", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device, flush=True)

    mask_func = build_fixed_cartesian_mask_func(
        H=256, W=256, acc=4, acs=24, seed=1234
    )

    print("building dataset...", flush=True)
    t0 = time.time()
    dataset = IXISinglecoilSliceDataset(
        root="/mnt/SSD/wsy/data/train",
        mask_func=mask_func,
        image_size=256,
        num_skip_slice=20,
        normalize_mode="max",
    )
    print(f"dataset len = {len(dataset)}, build cost = {time.time()-t0:.2f}s", flush=True)

    sample_idx = 1000
    x_in, y_gt, zf_mag = prepare_one_sample(dataset, sample_idx, device)

    print("x_in shape =", x_in.shape, flush=True)   # [1,2,H,W]
    print("y_gt shape =", y_gt.shape, flush=True)   # [1,1,H,W]

    zf_psnr = psnr_torch(zf_mag, y_gt).item()
    print(f"ZF PSNR = {zf_psnr:.2f} dB", flush=True)

    model = SmallUNet(in_ch=2, out_ch=1, base=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_steps = 3000
    print_every = 100
    save_every = 500

    best_psnr = -1.0

    # 保存 step 0，可直观看初始输出
    save_vis(model, x_in, y_gt, zf_mag, "./debug_overfit_ixi/step_0000.png")
    print("[saved] ./debug_overfit_ixi/step_0000.png", flush=True)

    for step in range(1, num_steps + 1):
        model.train()

        # 训练时不要 clamp
        pred = model(x_in)

        loss_l1 = F.l1_loss(pred, y_gt)
        loss_mse = F.mse_loss(pred, y_gt)
        loss = 0.8 * loss_l1 + 0.2 * loss_mse

        optimizer.zero_grad()
        loss.backward()

        grad_mean_sum = get_grad_mean_sum(model)

        optimizer.step()

        if step % print_every == 0 or step == 1:
            with torch.no_grad():
                pred_eval = model(x_in).clamp(0.0, 1.0)
                cur_psnr = psnr_torch(pred_eval, y_gt).item()
                pred_min = pred.min().item()
                pred_max = pred.max().item()
                pred_mean = pred.mean().item()

                if cur_psnr > best_psnr:
                    best_psnr = cur_psnr

            print(
                f"step={step:04d} | "
                f"loss={loss.item():.6f} | "
                f"L1={loss_l1.item():.6f} | "
                f"PSNR={cur_psnr:.2f} dB | "
                f"best={best_psnr:.2f} dB | "
                f"pred[min,max,mean]=({pred_min:.4f}, {pred_max:.4f}, {pred_mean:.4f}) | "
                f"grad_mean_sum={grad_mean_sum:.6e}",
                flush=True
            )

        if step % save_every == 0 or step == num_steps:
            save_path = f"./debug_overfit_ixi/step_{step:04d}.png"
            save_vis(model, x_in, y_gt, zf_mag, save_path)
            print(f"[saved] {save_path}", flush=True)

    torch.save(model.state_dict(), "./debug_overfit_ixi/overfit_one_ixi.pth")
    print("done.", flush=True)


if __name__ == "__main__":
    main()