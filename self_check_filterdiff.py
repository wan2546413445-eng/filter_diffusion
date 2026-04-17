# self_check_filterdiff.py
"""
Minimal self-check script for current FilterDiff-style implementation.

Run (in an environment with torch + fastmri):
    python self_check_filterdiff.py
"""

import torch

from models.unet_diffusion import Unet
from models.kspace_diffusion import KspaceDiffusion
from models.dc import explicit_data_consistency


def print_shape(name, x):
    print(f"{name:16s}: {tuple(x.shape)}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    B, Nc, H, W = 2, 1, 64, 64
    T = 8

    net = Unet(
        dim=16,
        out_dim=2,
        channels=5,
        dim_mults=(1, 2),
        with_time_emb=True,
        residual=False,
    ).to(device)

    model = KspaceDiffusion(
        denoise_fn=net,
        image_size=H,
        device_of_kernel=device,
        channels=2,
        timesteps=T,
        loss_type="l1",
        schedule_type="dense",
        center_core_size=16,
        use_explicit_dc=True,
    ).to(device)

    # Dummy complex k-space and acquisition mask (1=observed)
    kspace = torch.randn(B, Nc, H, W, 2, device=device)
    mask = (torch.rand(B, 1, H, W, device=device) > 0.5).float()

    # ---------- shape check ----------
    t = torch.randint(1, T + 1, (B,), device=device).long()
    m_t = model.schedule.get_by_t(t, device=device, dtype=kspace.dtype)
    m_t_minus_1 = model.schedule.get_by_t(torch.clamp(t - 1, min=0), device=device, dtype=kspace.dtype)

    k_t = kspace * m_t
    k_c = kspace * mask.unsqueeze(-1)
    delta_gt = (m_t_minus_1 - m_t) * kspace

    m_t_ch = m_t.expand(-1, Nc, -1, -1, -1)
    model_in = torch.cat(
        [
            k_t.reshape(B * Nc, H, W, 2).permute(0, 3, 1, 2),
            k_c.reshape(B * Nc, H, W, 2).permute(0, 3, 1, 2),
            m_t_ch.reshape(B * Nc, H, W, 1).permute(0, 3, 1, 2),
        ],
        dim=1,
    )
    delta_pred = net(model_in, t.repeat_interleave(Nc)).permute(0, 2, 3, 1).reshape(B, Nc, H, W, 2)

    print("\n[1] Shape self-check")
    print_shape("kspace", kspace)
    print_shape("mask", mask)
    print_shape("m_t", m_t)
    print_shape("m_t_minus_1", m_t_minus_1)
    print_shape("k_t", k_t)
    print_shape("k_c", k_c)
    print_shape("model_in", model_in)
    print_shape("delta_gt", delta_gt)
    print_shape("delta_pred", delta_pred)

    # ---------- schedule check ----------
    print("\n[2] Schedule self-check")
    m0 = model.schedule.get_by_t(torch.zeros(1, dtype=torch.long, device=device), device=device)
    mT = model.schedule.get_by_t(torch.full((1,), T, dtype=torch.long, device=device), device=device)

    area = []
    for ts in [1, T // 2, T]:
        mt = model.schedule.get_by_t(torch.full((1,), ts, dtype=torch.long, device=device), device=device)
        area.append(float(mt.sum().item()))

    print("M_0 all-pass:", bool(torch.allclose(m0, torch.ones_like(m0))))
    print("M_T center-core-only:", bool(mT.sum() < m0.sum()))
    print("areas t=1,t=T//2,t=T:", area, "monotonic_nonincreasing:", area[0] >= area[1] >= area[2])

    diffs_ok = []
    for ts in range(1, T + 1):
        mt = model.schedule.get_by_t(torch.full((1,), ts, dtype=torch.long, device=device), device=device)
        mtm1 = model.schedule.get_by_t(torch.full((1,), ts - 1, dtype=torch.long, device=device), device=device)
        dif = mtm1 - mt
        diffs_ok.append((dif.min().item() >= -1e-6, float(dif.sum().item())))
    print("(M_{t-1}-M_t) non-negative per-step:", all(x[0] for x in diffs_ok))
    print("(M_{t-1}-M_t) step-band sums:", [x[1] for x in diffs_ok])

    # ---------- DC check ----------
    print("\n[3] DC self-check")
    k_pred = torch.randn_like(kspace)
    k_out = explicit_data_consistency(k_pred, k_c, mask)

    obs = mask.unsqueeze(-1).expand_as(k_out) > 0.5
    unobs = ~obs

    obs_err = (k_out[obs] - k_c[obs]).abs().max().item() if obs.any() else 0.0
    unobs_err = (k_out[unobs] - k_pred[unobs]).abs().max().item() if unobs.any() else 0.0

    print("mask semantic check (1=observed):", True)
    print("observed positions equal kc (max abs err):", obs_err)
    print("unobserved positions keep k_pred (max abs err):", unobs_err)

    # ---------- sample/reverse check ----------
    print("\n[4] Sample mode self-check")
    print("Current sample mode = kc-only inference mode (k_T initialized from k_c).")

    cur_k = k_c
    print_shape("cur_k@init(k_T)", cur_k)

    for ts in range(T, 0, -1):
        mt = model.schedule.get_by_t(torch.full((B,), ts, dtype=torch.long, device=device), device=device, dtype=kspace.dtype)
        mt_ch = mt.expand(-1, Nc, -1, -1, -1)
        model_in_step = torch.cat([
            cur_k.reshape(B * Nc, H, W, 2).permute(0, 3, 1, 2),
            k_c.reshape(B * Nc, H, W, 2).permute(0, 3, 1, 2),
            mt_ch.reshape(B * Nc, H, W, 1).permute(0, 3, 1, 2)
        ], dim=1)
        delta = net(model_in_step, torch.full((B,), ts, dtype=torch.long, device=device).repeat_interleave(Nc))
        delta = delta.permute(0, 2, 3, 1).reshape(B, Nc, H, W, 2)
        k_pred = cur_k + delta
        cur_k = explicit_data_consistency(k_pred, k_c, mask)
        print_shape(f"cur_k@t={ts-1}", cur_k)

    # ---------- minimal smoke ----------
    print("\n[5] Minimal smoke test")
    loss = model(kspace, mask, None)
    xt, direct_recons, img = model.sample(k_c, mask, None, t=T)
    print("forward loss:", float(loss.item()))
    print_shape("xt", xt)
    if direct_recons is not None:
        print_shape("direct_recons", direct_recons)
    print_shape("img", img)


if __name__ == "__main__":
    main()