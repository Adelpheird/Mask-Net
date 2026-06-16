# Mask-Net

Mask-Net is a U-Net-based architecture designed for cloud masking in Sea Surface Temperature (SST) imagery.

The model combines:

* Spatial Attention
* Channel Attention
* Inter-Level Fusion Module (ILFM)

to improve cloud-ocean discrimination in dynamic coastal upwelling regions.

## Architecture

Mask-Net extends the classical U-Net architecture by introducing an Inter-Level Fusion Module (ILFM) within the decoder skip connections.

The ILFM combines:

* Spatial attention
* Channel attention

through additive fusion followed by convolutional refinement.

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Test

```bash
python models/masknet.py
```

Expected output:

```text
Input : torch.Size([1, 1, 288, 288])
Output: torch.Size([1, 1, 288, 288])
```

## Citation

If you use this code, please cite:

```bibtex
@article{MASKNET2026,
  title={Mask-Net: Cloud Masking for Sea Surface Temperature Imagery Using Spatial and Channel Attention},
  author={Kouassi Adelphe Christian N’GORAN, XXX},
  year={2026}
}
