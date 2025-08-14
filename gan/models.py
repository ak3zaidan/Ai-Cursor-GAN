
import torch
import torch.nn as nn
import torch.nn.functional as F

def spectral_norm(module):
    return nn.utils.spectral_norm(module)

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, causal=True, use_weight_norm=False):
        super().__init__()
        padding = (kernel_size - 1) * dilation if causal else ((kernel_size - 1) // 2) * dilation
        Conv = nn.Conv1d
        self.causal = causal
        self.pad = padding if causal else 0
        conv1 = Conv(channels, channels, kernel_size, padding=0 if causal else padding, dilation=dilation)
        conv2 = Conv(channels, channels, kernel_size, padding=0 if causal else padding, dilation=dilation)
        if use_weight_norm:
            conv1 = nn.utils.weight_norm(conv1)
            conv2 = nn.utils.weight_norm(conv2)
        self.conv1 = conv1
        self.conv2 = conv2
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        if self.causal and self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        x = self.act(self.conv1(x))
        if self.causal and self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        x = self.conv2(x)
        return self.act(x + residual)

class TemporalSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        # x: [B, C, T]
        y = self.norm(x).transpose(1, 2)  # [B, T, C]
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        out = self.proj(attn_out).transpose(1, 2)  # [B, C, T]
        return x + out

class Generator(nn.Module):
    def __init__(self, noise_dim=128, out_channels=2, base_channels=128, num_res_blocks=6, max_dilation=16, self_attn=True, out_length=128):
        super().__init__()
        self.out_length = out_length
        self.fc = nn.Linear(noise_dim, base_channels * out_length // 4)
        self.ups1 = nn.ConvTranspose1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.ups2 = nn.ConvTranspose1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)

        dilations = [1, 2, 4, 8, 16, 1]
        dilations = [d for d in dilations if d <= max_dilation][:num_res_blocks]
        self.resblocks = nn.ModuleList([
            ResidualBlock1D(base_channels, kernel_size=3, dilation=d, causal=True, use_weight_norm=True)
            for d in dilations
        ])
        self.self_attn = TemporalSelfAttention(base_channels) if self_attn else nn.Identity()
        self.out = nn.Conv1d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.fc(z)
        B = x.size(0)
        C = x.size(1) // (self.out_length // 4)
        x = x.view(B, C, self.out_length // 4)
        x = F.leaky_relu(self.ups1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.ups2(x), 0.2, inplace=True)
        for blk in self.resblocks:
            x = blk(x)
        x = self.self_attn(x)
        x = self.out(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=2, base_channels=128, num_res_blocks=6, max_dilation=16):
        super().__init__()
        self.inp = spectral_norm(nn.Conv1d(in_channels, base_channels, kernel_size=5, padding=2))
        dilations = [1, 2, 4, 8, 16, 1]
        dilations = [d for d in dilations if d <= max_dilation][:num_res_blocks]
        self.blocks = nn.ModuleList([
            spectral_norm(nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=d, dilation=d))
            for d in dilations
        ])
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = spectral_norm(nn.Linear(base_channels, 1))

    def forward(self, x):
        x = self.act(self.inp(x))
        for conv in self.blocks:
            x = self.act(conv(x))
        x = self.pool(x).squeeze(-1)
        out = self.fc(x)
        return out.squeeze(-1)
