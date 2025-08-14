
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from ts_gan.config import Config
from ts_gan.datasets import CursorDataset
from ts_gan.models import Generator, Discriminator
from ts_gan.losses import wgan_gp_loss
from ts_gan.utils import set_seed, gradient_penalty, save_checkpoint

def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    dataset = CursorDataset(cfg.data_path, fixed_length=cfg.fixed_length, use_time_deltas=cfg.use_time_deltas, per_sequence_norm=cfg.per_sequence_norm)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    in_channels = 3 if cfg.use_time_deltas else 2

    G = Generator(noise_dim=cfg.noise_dim, out_channels=in_channels, base_channels=cfg.gen_channels,
                  num_res_blocks=cfg.num_res_blocks, max_dilation=cfg.max_dilation, self_attn=cfg.self_attn, out_length=cfg.fixed_length).to(device)
    D = Discriminator(in_channels=in_channels, base_channels=cfg.disc_channels, num_res_blocks=cfg.num_res_blocks, max_dilation=cfg.max_dilation).to(device)

    optG = optim.Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas)
    optD = optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)

    step = 0
    while step < cfg.total_steps:
        for real, _meta in loader:
            real = real.to(device)

            # Train D
            for _ in range(cfg.d_steps):
                z = torch.randn(real.size(0), cfg.noise_dim, device=device)
                fake = G(z).detach()
                D_real = D(real)
                D_fake = D(fake)
                gp = gradient_penalty(D, real, fake, device)
                d_loss, _ = wgan_gp_loss(D_real, D_fake, gp, lambda_gp=cfg.lambda_gp)

                optD.zero_grad(set_to_none=True)
                d_loss.backward()
                optD.step()

            # Train G
            z = torch.randn(real.size(0), cfg.noise_dim, device=device)
            fake = G(z)
            D_fake = D(fake)
            _d, g_loss = wgan_gp_loss(D_real=None, D_fake=D_fake, gp=torch.tensor(0.0, device=device), lambda_gp=cfg.lambda_gp)

            optG.zero_grad(set_to_none=True)
            g_loss.backward()
            optG.step()

            if step % cfg.log_every == 0:
                print(f"step {step:6d} | d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f}")

            if step % cfg.ckpt_every == 0 and step > 0:
                os.makedirs(cfg.ckpt_dir, exist_ok=True)
                save_checkpoint(os.path.join(cfg.ckpt_dir, f"tsgan_step{step}.pt"), G, D, optG, optD, step)

            step += 1
            if step >= cfg.total_steps:
                break

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    save_checkpoint(os.path.join(cfg.ckpt_dir, f"tsgan_final.pt"), G, D, optG, optD, step)
    print("Training complete.")

if __name__ == "__main__":
    main()
