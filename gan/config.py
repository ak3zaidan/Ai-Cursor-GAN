
from dataclasses import dataclass

@dataclass
class Config:
    # Data
    data_path: str = "../augmented_data.json"
    fixed_length: int = 128
    use_time_deltas: bool = False
    per_sequence_norm: bool = True

    # Model
    noise_dim: int = 128
    gen_channels: int = 128
    disc_channels: int = 128
    num_res_blocks: int = 6
    max_dilation: int = 16
    self_attn: bool = True

    # Training
    batch_size: int = 64
    lr: float = 2e-4
    betas: tuple = (0.5, 0.9)
    lambda_gp: float = 10.0
    d_steps: int = 5
    total_steps: int = 20000
    log_every: int = 200
    ckpt_every: int = 2000
    device: str = "cuda"

    # Misc
    seed: int = 42
    ckpt_dir: str = "checkpoints"
    sample_dir: str = "samples"
