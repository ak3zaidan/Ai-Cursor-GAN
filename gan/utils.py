
import torch
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def gradient_penalty(discriminator, real, fake, device):
    bsz = real.size(0)
    alpha = torch.rand(bsz, 1, 1, device=device).expand_as(real)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = discriminator(interpolates)
    grads = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grads = grads.view(bsz, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def resample_sequence(seq_xy_t, fixed_length: int):
    """
    seq_xy_t: ndarray shape [N, 3] -> columns (x, y, t_ms)
    returns: ndarray shape [fixed_length, 3]
    """
    seq = np.asarray(seq_xy_t, dtype=float)
    x, y, t = seq[:,0], seq[:,1], seq[:,2]
    t = np.maximum.accumulate(t)  # ensure monotonic
    t0, t1 = t[0], t[-1]
    if t1 == t0:
        t1 = t0 + 1.0
    t_new = np.linspace(t0, t1, fixed_length)
    x_new = np.interp(t_new, t, x)
    y_new = np.interp(t_new, t, y)
    return np.stack([x_new, y_new, t_new], axis=1)

def per_sequence_normalize(xy, eps=1e-8):
    mean = xy.mean(axis=0, keepdims=True)
    std = xy.std(axis=0, keepdims=True)
    std = np.where(std < eps, eps, std)
    xy_norm = (xy - mean) / std
    return xy_norm, mean, std

def per_sequence_denorm(xy_norm, mean, std):
    return xy_norm * std + mean

def save_checkpoint(path, G, D, optG, optD, step):
    ckpt = {
        "G": G.state_dict(),
        "D": D.state_dict(),
        "optG": optG.state_dict(),
        "optD": optD.state_dict(),
        "step": step,
    }
    torch.save(ckpt, path)

def load_checkpoint(path, G, D=None, optG=None, optD=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    G.load_state_dict(ckpt["G"])
    if D is not None and "D" in ckpt:
        D.load_state_dict(ckpt["D"])
    if optG is not None and "optG" in ckpt:
        optG.load_state_dict(ckpt["optG"])
    if optD is not None and "optD" in ckpt:
        optD.load_state_dict(ckpt["optD"])
    return ckpt.get("step", 0)
