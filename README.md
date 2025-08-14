
# Cursor Time-Series GAN (TSA-ResGAN)

A PyTorch project to train a **time-series GAN** that generates human-like cursor movement sequences from augmented trajectory data.

## TL;DR
- **Input:** variable-length sequences of `(x, y, t_ms)`
- **Preprocess:** resample to fixed length `T` (default 128), per-sequence normalization
- **Model:** *TSA-ResGAN* — Causal Dilated ResNet (1D) with a **Temporal Self-Attention** (TSA) bottleneck
- **Loss:** WGAN-GP
- **Output:** synthetic `(x, y, t)` sequences

## Architecture (Custom Design)
- **Generator:** two transposed-Conv1d upsamplers -> stack of **causal dilated residual blocks** -> **Temporal Self-Attention** bottleneck -> Conv1d to output channels (2 if `(x,y)`, 3 if `(x,y,dt)`).
- **Discriminator:** spectral-normalized Conv1d front-end -> dilated Conv1d stack -> global pooling -> linear scalar score.
- **Why:** Causality + dilation captures local-to-global motion; TSA adds long-range planning signals; WGAN-GP stabilizes training for time-series.

## Project Layout
```
ts_gan_project/
├── data/augmented_data.json
├── ts_gan/{config.py,datasets.py,losses.py,models.py,utils.py}
├── checkpoints/
├── samples/
├── train.py
├── inference.py
└── requirements.txt
```

## Data
`data/augmented_data.json` format:
```json
{"data": [[[x,y,t], ...], ...]}
```
Sequences may vary in length; code resamples to fixed `T`.

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```
python train.py
```

Adjust hyperparameters in `ts_gan/config.py`.

## Generate
```
python inference.py
```

Outputs `samples/generated.json` and `samples/sample0.png`.
