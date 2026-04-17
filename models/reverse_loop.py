import torch

from .dc import explicit_data_consistency


def run_reverse_loop(
    model,
    k_t,
    k_c,
    acq_mask,
    schedule,
    timesteps: int,
):
    """
    for t = T ... 1:
      delta_k = model(k_t, k_c, M_t, t)
      k_pred  = k_t + delta_k
      k_{t-1} = explicit_dc(k_pred, k_c, acq_mask)
    """
    cur_k = k_t
    direct_recons = None

    for t_scalar in range(timesteps, 0, -1):
        bsz, ncoil, h, w, _ = cur_k.shape
        t = torch.full((bsz,), t_scalar, dtype=torch.long, device=cur_k.device)
        m_t = schedule.get_by_t(t, device=cur_k.device, dtype=cur_k.dtype)

        m_t_ch = m_t.expand(-1, ncoil, -1, -1, -1)

        cur_in = cur_k.reshape(bsz * ncoil, h, w, 2).permute(0, 3, 1, 2)
        kc_in = k_c.reshape(bsz * ncoil, h, w, 2).permute(0, 3, 1, 2)
        mt_in = m_t_ch.reshape(bsz * ncoil, h, w, 1).permute(0, 3, 1, 2)

        model_in = torch.cat([cur_in, kc_in, mt_in], dim=1)
        t_in = t.repeat_interleave(ncoil)

        delta = model(model_in, t_in).permute(0, 2, 3, 1).reshape(bsz, ncoil, h, w, 2)
        k_pred = cur_k + delta

        if direct_recons is None:
            direct_recons = k_pred

        cur_k = explicit_data_consistency(k_pred, k_c, acq_mask)

    return cur_k, direct_recons
