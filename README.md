# Mask-Net

**Official PyTorch implementation** of:

> **Mask-Net: A U-Net architecture with spatial and channel attention mechanisms for improved cloud detection in sea surface temperature imagery**

Kouassi Adelphe Christian N'Goran, Armand Kodjo Atiampo, Hervé Demarcq, Pascal Cauquil, and Georges Laussane Loum.

*Artificial Intelligence in Geosciences*, Volume 7, Issue 4, Article 100271, 2026.

**DOI:** [10.1016/j.aiig.2026.100271](https://doi.org/10.1016/j.aiig.2026.100271)
**Article:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666544126000870)

---

## Overview

**Mask-Net** is a U-Net-based deep learning architecture designed for **cloud detection and masking in infrared Sea Surface Temperature (SST) imagery**.

The model introduces an **Inter-Level Fusion Module (ILFM)** that combines **spatial attention** and **channel attention** mechanisms to refine the encoder features used by the decoder.

The objective is to improve the discrimination between **cloud-contaminated pixels** and **cold oceanic structures**, particularly in dynamic coastal regions such as **upwelling zones**, where conventional threshold-based cloud detection methods can lead to systematic over-masking.

Mask-Net was evaluated using one year of daily **AVHRR SST imagery over the Tropical Atlantic** at spatial resolutions of **9 km and 18 km**.

---

## Main results

The performance reported in the published study is summarized below:

| Configuration | Resolution | Image size |   OA (%) | mIoU (%) | Dice (%) |
| :-----------: | :--------: | :--------: | -------: | -------: | -------: |
|    Mask-Net   |    9 km    |  288 × 288 | **89.1** | **83.5** | **86.6** |
|    Mask-Net   |    18 km   |  144 × 144 | **90.9** | **82.8** | **77.8** |

The study also reports a **29% increase in usable SST observations during the cold season**, demonstrating the potential impact of improved cloud masking on subsequent SST analysis.

An ablation study showed that the combination of **spatial and channel attention** contributes to the robustness of the proposed architecture.

> **Note:** Performance values may depend on the dataset split, preprocessing procedure, and evaluation protocol. For the complete experimental setup and detailed results, please refer to the published article.

---

## Architecture

The overall architecture follows a U-Net framework with an **Inter-Level Fusion Module (ILFM)** incorporated into the decoder pathway.

```text
Input SST
(B × 1 × H × W)
       │
       ▼
┌─────────────────────────────┐
│ Encoder                     │
│ 4 convolutional stages      │
│ + bottleneck                │
└──────────────┬──────────────┘
               │
               │ Skip connections
               ▼
┌─────────────────────────────┐
│ Inter-Level Fusion Module   │
│ (ILFM) × 4                  │
│                             │
│ Spatial Attention           │
│          +                  │
│ Channel Attention           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Mask Decoder                │
│ U-Net upsampling stages     │
└──────────────┬──────────────┘
               │
               ▼
        1 × 1 Convolution
               │
               ▼
        Output logits
        (B × 1 × H × W)
```

The model produces a single-channel output containing the **cloud-segmentation logits**.

The binary cloud mask can be obtained using:

```python
mask = (torch.sigmoid(logits) > 0.5).float()
```

---

## Repository structure

```text
Mask-Net/
├── README.md
├── LICENSE
├── requirements.txt
│
├── masknet/
│   ├── __init__.py       # Exposes MaskNet and build_masknet
│   ├── model.py          # Main Mask-Net architecture
│   ├── ilfm.py           # ILFM and attention modules
│   └── blocks.py         # Encoder / decoder building blocks
│
├── data.py               # NetCDF data loading and dataset splitting
├── metrics.py            # OA, mIoU and Dice metrics
├── train.py              # Training script
└── evaluate.py           # Evaluation script
```

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/<your-username>/Mask-Net.git
cd Mask-Net

pip install -r requirements.txt
```

---

## Quick start

The following example initializes Mask-Net for the **9 km configuration**, corresponding to 288 × 288 input images:

```python
import torch
from masknet import build_masknet

# 9 km configuration
model = build_masknet(resolution_km=9)

# Example input
x = torch.randn(1, 1, 288, 288)

# Forward pass
logits = model(x)

# Binary cloud mask
mask = (torch.sigmoid(logits) > 0.5).float()
```

---

## Training

Example training command for the 9 km configuration:

```bash
python train.py \
    --image-paths data/sst_2018.nc \
    --mask-paths data/gt_2018.nc \
    --resolution-km 9 \
    --batch-size 8 \
    --epochs 300 \
    --checkpoint-dir checkpoints/
```

For multi-year datasets, provide one SST file and one corresponding ground-truth mask file per year:

```bash
python train.py \
    --image-paths data/sst_2014.nc data/sst_2015.nc ... \
    --mask-paths data/gt_2014.nc data/gt_2015.nc ...
```

Make sure that the SST files and mask files are provided in the same order.

---

## Evaluation

Example evaluation command:

```bash
python evaluate.py \
    --checkpoint checkpoints/masknet_9km_final.pth \
    --image-paths data/sst_2018.nc \
    --mask-paths data/gt_2018.nc
```

The evaluation script computes the segmentation performance using the metrics implemented in `metrics.py`.

---

## Data

### AVHRR SST data

The AVHRR Pathfinder SST data used in the study are publicly available from NOAA:

[NOAA AVHRR Pathfinder Version 5.3 L3C](https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/)

The study uses daily AVHRR SST observations over the Tropical Atlantic.

### Ground-truth cloud masks

The ground-truth cloud masks used for training and evaluation were generated from threshold-based processing followed by expert correction.

The expert-corrected masks are **not redistributed with this repository**. They are available from the corresponding author upon reasonable request, subject to the conditions described in the article's **Data Availability Statement**.

Please refer to the published article for a detailed description of the dataset, preprocessing procedure, mask generation and validation protocol.

---

## Publication

This repository accompanies the following peer-reviewed publication:

**Kouassi Adelphe Christian N'Goran, Armand Kodjo Atiampo, Hervé Demarcq, Pascal Cauquil, and Georges Laussane Loum.**

*Mask-Net: A U-Net architecture with spatial and channel attention mechanisms for improved cloud detection in sea surface temperature imagery.*

**Artificial Intelligence in Geosciences**, Volume 7, Issue 4, Article 100271, 2026.

**DOI:** [10.1016/j.aiig.2026.100271](https://doi.org/10.1016/j.aiig.2026.100271)

**ScienceDirect:**
https://www.sciencedirect.com/science/article/pii/S2666544126000870

---

## Citation

If you use **Mask-Net**, its implementation, or the associated methodology in your research, please cite:

```bibtex
@article{NGORAN2026100271,
  title   = {Mask-Net: A U-Net architecture with spatial and channel attention mechanisms for improved cloud detection in sea surface temperature imagery},
  author  = {Kouassi Adelphe Christian N'Goran and
             Armand Kodjo Atiampo and
             Hervé Demarcq and
             Pascal Cauquil and
             Georges Laussane Loum},
  journal = {Artificial Intelligence in Geosciences},
  volume  = {7},
  number  = {4},
  pages   = {100271},
  year    = {2026},
  issn    = {2666-5441},
  doi     = {10.1016/j.aiig.2026.100271},
  url     = {https://www.sciencedirect.com/science/article/pii/S2666544126000870}
}
```

---

## Acknowledgements

This work was conducted as part of the PhD thesis of **Kouassi Adelphe Christian N'Goran** at **INPHB (Côte d'Ivoire)**, in collaboration with **IRD** and **Ifremer (MARBEC, France)**.

The research was partially supported by the **France Excellence 2025** grant from the **Service de Coopération et d'Action Culturelle (SCAC), Embassy of France in Côte d'Ivoire**, and by the **Direction de l'Orientation et des Bourses (DOB)** of the Ivorian Ministry of Higher Education and Scientific Research.

---

## License

This project is released under the **MIT License**.

See the [LICENSE](LICENSE) file for details.
