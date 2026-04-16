import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np

import fastmri
from help_func import print_var_detail


# gaussian diffusion trainer class

def extract(a, t, x_shape):


    b, *_ = t.shape

    out = a.gather(-1, t)

    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):


    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))


    noise = lambda: torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()



def cosine_beta_schedule(timesteps, s=0.008):#余弦噪声调度/较小变化平缓，保留更多结构，中间近似线性，接近时加速的加噪策略。
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """


    steps = timesteps + 1

    x = np.linspace(0, steps, steps)


    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2


    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]


    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])


    return np.clip(betas, a_min=0, a_max=0.999)



def mask_sequence_sample(mask_fold):
    """
    calculate mask sequence given folded mask

    :param mask_fold: input folded mask tensor, [B, 1, H/num_patches_H, W/num_patches_W]
    :return: mask sequence tensor stores all masked patches position idx, [num of masked patches, 2]
    """
    mask_seqs = []


    for i in range(mask_fold.shape[0]):

        mask_idx = torch.where(mask_fold[i][0] == 0)


        mask_seq = torch.cat((mask_idx[0][:, None], mask_idx[1][:, None]), dim=1)


        r = torch.randperm(mask_seq.shape[0])


        mask_seq = mask_seq[r[:]]


        mask_seqs.append(mask_seq)


    return mask_seqs


class KspaceDiffusion(nn.Module):
    """
    diffusion module for kspace diffusing process

    Args:
    ----------
    denoise_fn : model
        actual training model for reverse diffusion process given image and time step t
    image_size : int
        a list containing elements like (file_name, index). A .h5 file contains multi slices.
    device_of_kernel : string
        device for training, usually 'cuda'
    channels : int
        number of channels for input image. default 3
    timesteps: int,
        training steps of reverse diffusion.
    loss_type: string
        type of loss function. default 'l1'
    blur_routine: string
        blur routine from one time step to next, default 'Incremental'
    train_routine : string
        the train routine to set if the training process is throughout to the end or not, default 'Final'
    sampling_routine : string
        the sample routine choose between 'default' or 'x0_step_down'
    discrete: bool,
        if the blur process is discrete or continuous, default 'False'
    """

    def __init__(
            self,
            denoise_fn,
            *,
            image_size,
            device_of_kernel,
            channels=3,
            timesteps=1000,
            loss_type='l1',
            blur_routine='Incremental',
            train_routine='Final',
            sampling_routine='default',
            discrete=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device_of_kernel = device_of_kernel

        self.num_timesteps = timesteps
        self.loss_type = loss_type
        self.blur_routine = blur_routine

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.train_routine = train_routine
        self.sampling_routine = sampling_routine
        self.discrete = discrete

    def q_sample(self, kspace, mask_seqs, mask, mask_fold, t):#前向步骤
        """
        sample a kspace given step t and masks

        :param kspace: fully sampled kspace tensor, [B, Nc, H, W, 2]
        :param mask_seqs: mask sequence tensor stores all masked patches position idx, [num of masked patches, 2]
        :param mask: input unfolded mask tensor, [B, 1, H, W]
        :param mask_fold: input folded mask tensor, [B, 1, H/num_patches_H, W/num_patches_W]
        :param t: time steps tensor, [B,]
        :return: image_part_masked: masked image tensor, [B, Nc, H, W, 2]; kspace_part_masked: masked kspace tensor,
            [B, Nc, H, W, 2]
        """

        # mask at t steps
        patch_size_H = mask.shape[-2] // mask_fold.shape[-2]
        patch_size_W = mask.shape[-1] // mask_fold.shape[-1]
        mask_part = torch.ones(mask_fold.shape) #全1掩码

        for i in range(mask_fold.shape[0]):
            mask_seq = mask_seqs[i]#所有遮挡坐标
            step = mask_seq.shape[0] / self.num_timesteps#每个时间步长的平均遮挡数量
            # 取前 t[i] * step 个 patch 作为当前时间步的遮挡区域
            mask_part[i, ..., mask_seq[0:int(t[i] * step), 0], mask_seq[0:int(
                t[i] * step), 1]] = 0.0
#num_mask=int(t[i] * step);mask_seq[:num_mask, 0]行；mask_seq[:num_mask, 1] 列；[B, 1, H_fold, W_fold]
        #将 patch 级掩码上采样回像素级掩码；nearest：最近邻插值保持二值[0,1]特性。
        mask_expand = torch.nn.functional.interpolate(mask_part, scale_factor=[patch_size_H, patch_size_W],
                                                      mode='nearest').to(self.device_of_kernel)
        #k空间掩
        kspace_part_masked = kspace * mask_expand.unsqueeze(-1).repeat(1, kspace.shape[1], 1, 1, kspace.shape[-1])
       #回到图像域
        image_part_masked = fastmri.ifft2c(kspace_part_masked)  # [B,Nc,H,W,2]

        return image_part_masked, kspace_part_masked  # [B,Nc,H,W,2],  [B,Nc,H,W,2]

    @torch.no_grad()
    def sample(self, kspace, mask, mask_fold, t):#逆向采样
        """
        reverse diffusion sampling given unmasked kspace, masks and time step t

        :param kspace: fully sampled kspace tensor, [B, Nc, H, W, 2]
        :param mask: input unfolded mask tensor, [B, 1, H, W]
        :param mask_fold: input folded mask tensor, [B, 1, H/num_patches_H, W/num_patches_W]
        :param t: time steps tensor, [B,]
        :return: q_sample xt: masked image tensor, [B, Nc, H, W, 2]; direct_recons: direction reconstructed image from
            denoise_fn, [B, Nc, H, W, 2]; img: reverse sampled img [B, Nc, H, W, 2]
        """
        self.denoise_fn.eval()
        batch_size, Nc, H, W, C = kspace.shape

        if t is None:
            t = self.num_timesteps#最后一步往前
        mask_seqs = mask_sequence_sample(mask_fold)#获取坐标
        # sample masked image and kspace given t    # 1. 初始状态：给定时间步 t 的遮挡图像
        with torch.no_grad():
            img, kspace = self.q_sample(kspace, mask_seqs, mask, mask_fold,
                                        torch.full((batch_size,), t, dtype=torch.long).cuda())  # [B,Nc,H,W,2]
        xt = img
        direct_recons = None
       #去噪循环
        while t:
            step = torch.full((batch_size,), t, dtype=torch.long).cuda()
            # 2.1 用去噪网络预测当前步的完整图像 x--zeros
            img_in = img.reshape(batch_size * Nc, H, W, C).permute(0, 3, 1, 2)   # [B*Nc,2,H,W]
            step_in = step.repeat_interleave(Nc)                                   # [B*Nc]
            x = self.denoise_fn(img_in, step_in).permute(0, 2, 3, 1)              # [B*Nc,H,W,2]
            x = x.reshape(batch_size, Nc, H, W, C)                                 # [B,Nc,H,W,2]
            # 2.2 将预测图像变换回 k 空间
            k = fastmri.fft2c(x)
            # 2.3 根据采样策略，计算下一步的输入 img
            if self.train_routine == 'Final':
                if direct_recons is None:
                    direct_recons = x
                if self.sampling_routine == 'default':
                    if self.blur_routine == 'Individual_Incremental':
                        x, _ = self.q_sample(k, mask_seqs, mask, mask_fold, step - 2)
                    else:
                        with torch.no_grad():
                            x, _ = self.q_sample(k, mask_seqs, mask, mask_fold, step)
                elif self.sampling_routine == 'x0_step_down':
                    k_times = k
                    with torch.no_grad():
                        x_times, _ = self.q_sample(k_times, mask_seqs, mask, mask_fold, step)
                    k_times_sub_1 = k
                    with torch.no_grad():
                        x_times_sub_1, _ = self.q_sample(k_times_sub_1, mask_seqs, mask, mask_fold, step - 1)

                    x = img - x_times + x_times_sub_1
            img = x
            t = t - 1
        self.denoise_fn.train()
        return xt, direct_recons, img  # [B,Nc,H,W,2]

    def p_losses(self, kspace, mask_seqs, mask, mask_fold, t):#损失计算
        """
        loss function for denoise module given train_routine and loss_type

        :param kspace: fully sampled kspace tensor, [B, Nc, H, W, 2]
        :param mask_seqs: mask sequence tensor stores all masked patches position idx, [num of masked patches, 2]
        :param mask: input unfolded mask tensor, [B, 1, H, W]
        :param mask_fold: input folded mask tensor, [B, 1, H/num_patches_H, W/num_patches_W]
        :param t: time steps tensor, [B,]
        :return: loss: l1 or l2 loss
        """
        B, Nc, H, W, C = kspace.shape
        x_start = fastmri.ifft2c(kspace)  # [B,Nc,H,W,2]--GT
        if self.train_routine == 'Final':
            # 1. 对完整 k 空间施加前向扩散，得到遮挡图像 x_blur
            x_blur, _ = self.q_sample(kspace, mask_seqs, mask, mask_fold, t)  # [B,Nc,H,W,2]
            # 2. 用去噪网络预测完整图像
            x_in = x_blur.reshape(B * Nc, H, W, C).permute(0, 3, 1, 2)  # [B*Nc,2,H,W]
            t_in = t.repeat_interleave(Nc)  # [B*Nc]
            x_recon = self.denoise_fn(x_in, t_in).permute(0, 2, 3, 1)  # [B*Nc,H,W,2]
            x_recon = x_recon.reshape(B, Nc, H, W,
                                      C)  # [B,Nc,H,W,2]                                                                              1)  # [B,Nc,H,W,2]
            # 3. 计算预测与真值的损失
            if self.loss_type == 'l1':
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        return loss

    def forward(self, kspace, mask, mask_fold, *args, **kwargs):#前向路口
        B, Nc, H, W, C, device, img_size, = *kspace.shape, kspace.device, self.image_size
        assert H == img_size and W == img_size, f'height and width of image must be {img_size}'

        # given random timestep and calculate p_losses
        t = torch.randint(0, self.num_timesteps + 1, (B,), device=device).long()
        mask_seqs = mask_sequence_sample(mask_fold)
        return self.p_losses(kspace, mask_seqs, mask, mask_fold, t, *args, **kwargs)
