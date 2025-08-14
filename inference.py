
import os
import json
import torch
import numpy as np
from ts_gan.config import Config
from ts_gan.models import Generator
from ts_gan.utils import load_checkpoint
import matplotlib.pyplot as plt

def generate_samples(ckpt_path: str, num_samples: int = 8, out_json: str = "samples/generated.json"):
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    in_channels = 3 if cfg.use_time_deltas else 2

    G = Generator(noise_dim=cfg.noise_dim, out_channels=in_channels, base_channels=cfg.gen_channels,
                  num_res_blocks=cfg.num_res_blocks, max_dilation=cfg.max_dilation, self_attn=cfg.self_attn, out_length=cfg.fixed_length).to(device)
    load_checkpoint(ckpt_path, G, map_location=device)
    G.eval()

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    data_out = []

    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, cfg.noise_dim, device=device)
            sample = G(z).cpu().numpy()[0]  # [C, T]
            if cfg.use_time_deltas:
                xy = sample[:2].T  # [T, 2]
            else:
                xy = sample.T      # [T, 2]
            xy_min = xy.min(axis=0, keepdims=True)
            xy_max = xy.max(axis=0, keepdims=True)
            rng = np.maximum(xy_max - xy_min, 1e-6)
            xy_denorm = (xy - xy_min) / rng * 500.0

            T = xy.shape[0]
            t0 = 0.0
            dt = 16.0
            t = t0 + np.arange(T) * dt
            seq = np.concatenate([xy_denorm, t[:, None]], axis=1).tolist()
            data_out.append(seq)

    with open(out_json, "w") as f:
        json.dump({"data": data_out}, f)
    print(f"Saved generated sequences to {out_json}")

    s0 = np.array(data_out[0])
    plt.figure(figsize=(6,4))
    plt.plot(s0[:,0], s0[:,1])
    plt.title("Generated Cursor Trajectory (Sample 0)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(out_json), "sample0.png"))
    print("Saved plot to", os.path.join(os.path.dirname(out_json), "sample0.png"))

if __name__ == "__main__":
    generate_samples(ckpt_path="checkpoints/tsgan_final.pt", num_samples=16, out_json="samples/generated.json")
