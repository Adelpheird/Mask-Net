"""
Mask-Net — full architecture
=============================
Dual-decoder U-Net with Inter-Level Fusion Modules (ILFM).

Architecture summary
--------------------
Input SST image  (B × 1 × H × W)
        │
   ┌────▼────────────────────┐
   │  Encoder  (4 stages)    │  → 4 skip connections  s₁ … s₄
   │  + Bottleneck           │
   └────┬────────────────────┘
        │ bottleneck features
        ▼
   ┌─────────────────────────┐
   │  SST decoder (ILFM)     │  each stage refines sᵢ via joint
   │  4 ILFM blocks          │  spatial + channel attention,
   │                         │  producing refined skips  r₁ … r₄
   └────┬────────────────────┘
        │ refined features
        ▼
   ┌─────────────────────────┐
   │  Mask decoder           │  standard U-Net upsampling path,
   │  4 stages               │  using r₁ … r₄ as skip connections
   └────┬────────────────────┘
        │
   Classifier  (1 × 1 conv)
        │
   Logits  (B × 1 × H × W)   ← apply torch.sigmoid + threshold 0.5
                                 to obtain binary cloud mask

The skip-connection sizes at each decoder level depend on the input
resolution and are computed automatically from ``image_size`` so that
the :class:`~masknet.ilfm.AttentionBlock`'s internal MLP can be
pre-allocated as a proper trainable submodule.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .blocks import DoubleConv, EncoderBlock, MaskDecoderBlock
from .ilfm import ILFM

# Default channel widths: 9 km configuration (288 × 288 inputs).
_DEFAULT_FEATURES: List[int] = [32, 64, 128, 256]


def _skip_sizes(
    image_size: int, features: List[int]
) -> List[Tuple[int, int]]:
    """Compute the skip-connection spatial sizes for each encoder stage.

    Starting from ``image_size × image_size``, each encoder stage
    halves the spatial dimensions via 2 × 2 max-pooling. The skip
    connection at stage *i* (0-indexed from the shallowest) has size
    ``image_size / 2^i``.

    The ILFM decoder iterates the features in *reverse* order (deepest
    first), so the returned list is also reversed to match.

    Parameters
    ----------
    image_size : int
        Square input resolution (e.g. 288 for 9 km, 144 for 18 km).
    features : List[int]
        Channel widths of the encoder stages.

    Returns
    -------
    List[Tuple[int, int]]
        ``(H, W)`` for each decoder level, from the deepest to the
        shallowest (i.e. the order in which the ILFM blocks are applied).
    """
    sizes = []
    size  = image_size
    for _ in features:
        sizes.append((size, size))
        size = size // 2
    return list(reversed(sizes))          # deepest level first


class MaskNet(nn.Module):
    """Mask-Net: U-Net with ILFM-based joint spatial/channel attention.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels (1 for single-band SST imagery).
    out_channels : int, default 1
        Number of output channels (1 for binary cloud/sea masks).
    features : List[int], optional
        Channel widths for the four encoder/decoder stages.
        Default: ``[32, 64, 128, 256]`` (9 km, 288 × 288 inputs).
        Use ``[32, 64, 128, 256]`` for the 18 km / 144 × 144 variant.
    image_size : int, default 288
        Spatial size (height = width) of the input images.
        Used to pre-compute the skip-connection sizes for the ILFM
        internal MLP. Must match the actual input resolution.
    """

    def __init__(
        self,
        in_channels : int = 1,
        out_channels: int = 1,
        features    : Optional[List[int]] = None,
        image_size  : int = 288,
    ) -> None:
        super().__init__()
        features = features or _DEFAULT_FEATURES

        # ── Encoder ─────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        in_c = in_channels
        for width in features:
            self.encoders.append(EncoderBlock(in_c, width))
            in_c = width

        self.bottleneck = EncoderBlock(features[-1], features[-1] * 2)

        # ── SST decoder (ILFM-based attention) ──────────────────────
        # Each ILFM block needs the spatial size of its skip connection
        # so that the internal MLP can be pre-allocated correctly.
        skip_sizes_per_level = _skip_sizes(image_size, features)

        self.sst_decoder = nn.ModuleList()
        for i, width in enumerate(reversed(features)):
            self.sst_decoder.append(
                ILFM(
                    in_channels =[width * 2, width],
                    out_channels=width,
                    skip_size   =skip_sizes_per_level[i],
                )
            )

        # ── Mask decoder (standard upsampling path) ──────────────────
        self.mask_decoder = nn.ModuleList()
        for width in reversed(features):
            self.mask_decoder.append(MaskDecoderBlock(width * 2, width))

        # ── Output head ──────────────────────────────────────────────
        self.normalize  = nn.BatchNorm2d(features[0])
        self.relu       = nn.ReLU(inplace=False)
        self.classifier = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # ----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through Mask-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input SST image, shape ``(B, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Raw logits (pre-sigmoid), shape ``(B, out_channels, H, W)``.
            Apply ``torch.sigmoid(output) > 0.5`` to obtain a binary
            cloud/sea mask.
        """
        # ── Encoder pass ─────────────────────────────────────────────
        skip_connections: List[torch.Tensor] = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        # Bottleneck: deepest feature extraction, no skip stored.
        _, bottleneck = self.bottleneck(x)

        # Reverse skips: decoder processes from deepest to shallowest.
        skip_connections = skip_connections[::-1]

        # ── SST decoder — ILFM attention refinement ──────────────────
        x = bottleneck
        refined_skips: List[torch.Tensor] = []
        for ilfm_block, skip in zip(self.sst_decoder, skip_connections):
            x = self.relu(ilfm_block(x, skip))
            refined_skips.append(x)

        # ── Mask decoder — final segmentation reconstruction ─────────
        x = bottleneck
        for mask_block, refined in zip(self.mask_decoder, refined_skips):
            x = self.relu(mask_block(x, refined))

        x = self.normalize(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def build_masknet(resolution_km: int = 9, **kwargs) -> MaskNet:
    """Build a Mask-Net model for a given spatial resolution.

    Parameters
    ----------
    resolution_km : {9, 18}, default 9
        Selects the channel-width and image-size configuration used
        in the paper.

        - ``9``  → features ``[32, 64, 128, 256]``, image size 288.
        - ``18`` → features ``[32, 64, 128, 256]``, image size 144.

    **kwargs
        Additional keyword arguments forwarded to :class:`MaskNet`
        (e.g. ``in_channels``, ``out_channels``).

    Returns
    -------
    MaskNet
        Randomly initialised Mask-Net model.

    Examples
    --------
    >>> model = build_masknet(resolution_km=9)
    >>> model(torch.randn(1, 1, 288, 288)).shape
    torch.Size([1, 1, 288, 288])
    """
    configs = {
        9 : dict(features=[32, 64, 128, 256], image_size=288),
        18: dict(features=[32, 64, 128, 256],  image_size=144),
    }
    if resolution_km not in configs:
        raise ValueError(
            f"Unsupported resolution_km={resolution_km!r}. "
            "Choose 9 or 18."
        )
    return MaskNet(**configs[resolution_km], **kwargs)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for km, size in [(9, 288), (18, 144)]:
        model  = build_masknet(resolution_km=km).to(device)
        dummy  = torch.randn(1, 1, size, size).to(device)
        output = model(dummy)

        n_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(
            f"[{km} km]  input {tuple(dummy.shape)}"
            f" → output {tuple(output.shape)}"
            f"  |  {n_params / 1e6:.2f} M trainable parameters"
        )
