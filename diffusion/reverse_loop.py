import torch
import fastmri

from diffusion.dc import explicit_data_consistency


def run_reverse_loop(
    model,
    k_t,
    k_c,
    acq_mask,
    schedule,
    timesteps: int,
    use_explicit_dc: bool = False,
):
    """
    FilterDiff reverse process:
    for t = T ... 1:
        x0_pred   = phi_theta(M_t, k_t, k_c, t)
        delta_k   = (M_{t-1} - M_t) * FFT(x0_pred)
        k_{t-1}   = k_t + delta_k

    optional explicit DC is kept as an engineering option,
    but strict paper reproduction should set use_explicit_dc=False.
    """
    cur_k = k_t
    direct_recons = None

    for t_scalar in range(timesteps, 0, -1):
        bsz, ncoil, h, w, _ = cur_k.shape

        t = torch.full((bsz,), t_scalar, dtype=torch.long, device=cur_k.device)
        t_prev = torch.clamp(t - 1, min=0)

        m_t = schedule.get_by_t(t, device=cur_k.device, dtype=cur_k.dtype)            # [B,1,H,W,1]
        m_t_minus_1 = schedule.get_by_t(t_prev, device=cur_k.device, dtype=cur_k.dtype)
        delta_mask = m_t_minus_1 - m_t                                                 # [B,1,H,W,1]

        m_t_ch = m_t.expand(-1, ncoil, -1, -1, -1)

        cur_in = cur_k.reshape(bsz * ncoil, h, w, 2).permute(0, 3, 1, 2)              # [B*Nc,2,H,W]
        kc_in = k_c.reshape(bsz * ncoil, h, w, 2).permute(0, 3, 1, 2)                 # [B*Nc,2,H,W]
        mt_in = m_t_ch.reshape(bsz * ncoil, h, w, 1).permute(0, 3, 1, 2)              # [B*Nc,1,H,W]

        model_in = torch.cat([cur_in, kc_in, mt_in], dim=1)                           # [B*Nc,5,H,W]
        t_in = t.repeat_interleave(ncoil)

        # network predicts x0 in image domain
        x0_pred = model(model_in, t_in).permute(0, 2, 3, 1).reshape(bsz, ncoil, h, w, 2)

        # convert to k-space increment exactly as in training / paper
        delta_k = delta_mask * fastmri.fft2c(x0_pred)

        k_pred = cur_k + delta_k

        if direct_recons is None:
            direct_recons = k_pred

        if use_explicit_dc:
            cur_k = explicit_data_consistency(k_pred, k_c, acq_mask)
        else:
            cur_k = k_pred

    return cur_k, direct_recons