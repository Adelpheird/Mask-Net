"""
Shared building blocks for Mask-Net's encoder and decoder paths.

All blocks follow the standard U-Net convention: two consecutive
3 × 3 convolutions each followed by Batch Normalisation and ReLU,
with a 2 × 2 max-pooling step for downsampling in the encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BN → ReLU operations.

    This is the standard U-Net convolutional block, reused by both the
    encoder and the mask-prediction decoder.

    Parameters
    ----------
    in_channels, out_channels : int
        Input and output channel counts.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder stage: DoubleConv then 2 × 2 max-pooling.

    Returns both the pre-pooling feature map (skip connection) and the
    downsampled output forwarded to the next stage.

    Parameters
    ----------
    in_channels, out_channels : int
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv       = DoubleConv(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        skip : torch.Tensor
            Pre-pooling features, used as skip connection.
        out : torch.Tensor
            Downsampled features, passed to the next stage.
        """
        skip = self.conv(x)
        return skip, self.downsample(skip)


class MaskDecoderBlock(nn.Module):
    """Decoder stage used in the mask-prediction branch.

    Upsamples ``x`` via transposed convolution, concatenates it with
    the corresponding (ILFM-refined) skip connection, then applies a
    :class:`DoubleConv` to merge the two streams.

    Parameters
    ----------
    in_channels : int
        Channels of the input tensor ``x`` (before upsampling).
    out_channels : int
        Channels of the output tensor.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : torch.Tensor  —  coarser decoder features.
        skip : torch.Tensor  —  ILFM-refined skip connection.
        """
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = TF.resize(x, size=list(skip.shape[2:]), antialias=True)
        return self.conv(torch.cat((x, skip), dim=1))
