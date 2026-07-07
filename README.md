# Mask-Net

**Official PyTorch implementation** of  
*Mask-Net: A U-Net Architecture with Spatial and Channel Attention
Mechanisms for Improved Cloud Detection in Sea Surface Temperature
Imagery*  
N'GORAN et al., *Artificial Intelligence in Geosciences* (under review)

---

## Overview

Mask-Net is a U-Net-based deep learning architecture designed for
**cloud masking in infrared Sea Surface Temperature (SST) imagery**.
It introduces an **Inter-Level Fusion Module (ILFM)** that jointly
applies spatial and channel attention to encoder skip connections,
improving discrimination between clouds and cold oceanic structures
such as coastal upwelling fronts — a well-known failure mode of
threshold-based cloud detection methods.

| Configuration | Resolution | Image size | OA (%) | mIoU (%) | Dice (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 9 km  | `[32, 64, 128, 256]` | 288 × 288 | 89.1 | 83.5 | 86.6 |
| 18 km | `[16, 32, 64, 128]`  | 144 × 144 | 90.9 | 82.8 | 77.8 |

---

## Architecture

```
Input SST  (B × 1 × H × W)
      │
 ┌────▼──────────────────────┐
 │  Encoder  (4 stages)      │  →  4 skip connections  s₁…s₄
 │  + Bottleneck             │
 └────┬──────────────────────┘
      │
      ▼
 ┌───────────────────────────┐
 │  SST decoder (ILFM ×4)   │  Joint spatial + channel attention
 │                           │  refines each sᵢ  →  r₁…r₄
 └────┬──────────────────────┘
      │
      ▼
 ┌───────────────────────────┐
 │  Mask decoder (×4)        │  Standard U-Net upsampling using r₁…r₄
 └────┬──────────────────────┘
      │
 Classifier  (1 × 1 conv)
      │
 Logits  (B × 1 × H × W)
```

Apply `torch.sigmoid(logits) > 0.5` to obtain the binary cloud mask.

---

## Repository structure

```
Mask-Net/
├── README.md
├── requirements.txt
├── masknet/
│   ├── __init__.py       # exposes MaskNet, build_masknet
│   ├── model.py          # full architecture
│   ├── ilfm.py           # ILFM + AttentionBlock + _SpatialMLP
│   └── blocks.py         # shared encoder / decoder blocks
├── data.py               # NetCDF loading and train/test split
├── metrics.py            # OA, mIoU, Dice
├── train.py              # training script
└── evaluate.py           # evaluation script
```

---

## Installation

```bash
git clone https://github.com/<your-username>/Mask-Net.git
cd Mask-Net
pip install -r requirements.txt
```

---

## Quick start

```python
import torch
from masknet import build_masknet

# 9 km configuration (288 × 288 inputs) — as used in the paper
model = build_masknet(resolution_km=9)

x      = torch.randn(1, 1, 288, 288)
logits = model(x)
mask   = (torch.sigmoid(logits) > 0.5).float()  # binary cloud mask
```

---

## Training

```bash
python train.py \
    --image-paths  data/sst_2018.nc \
    --mask-paths   data/gt_2018.nc  \
    --resolution-km 9               \
    --batch-size 8                  \
    --epochs 300                    \
    --checkpoint-dir checkpoints/
```

Multi-year training (pass one file per year in matching order):

```bash
python train.py \
    --image-paths data/sst_2014.nc data/sst_2015.nc ... \
    --mask-paths  data/gt_2014.nc  data/gt_2015.nc  ...
```

---

## Evaluation

```bash
python evaluate.py \
    --checkpoint    checkpoints/masknet_9km_final.pth \
    --image-paths   data/sst_2018.nc                  \
    --mask-paths    data/gt_2018.nc
```

---

## Data

The AVHRR Pathfinder SST archive used in the paper is publicly
available from NOAA:
<https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/>

The expert-corrected Ground Truth masks are available from the
corresponding author upon reasonable request (see the paper's
*Data Availability Statement*).

---

## Citation

If you use this code or the dataset, please cite:

```bibtex
@article{ngoran_masknet_2026,
  title   = {Mask-Net: A U-Net Architecture with Spatial and Channel
             Attention Mechanisms for Improved Cloud Detection in
             Sea Surface Temperature Imagery},
  author  = {N'Goran, Kouassi Adelphe Christian and
             Atiampo, Armand Kodjo and
             Demarcq, Herv{\'e} and
             Cauquil, Pascal and
             Loum, Georges Laussane},
  journal = {Artificial Intelligence in Geosciences},
  year    = {2026},
}
```

---

## Acknowledgements

This work was conducted as part of the PhD thesis of Kouassi Adelphe
Christian N'Goran at INPHB (Côte d'Ivoire), in collaboration with
IRD and Ifremer (MARBEC, France).  
Partially funded by the *France Excellence 2025* grant (SCAC,
Embassy of France in Abidjan) and by the Direction de l'Orientation
et des Bourses (DOB) of the Ivorian Ministry of Higher Education and
Scientific Research.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
