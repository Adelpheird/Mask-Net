"""
Inter-Level Fusion Module (ILFM) — Mask-Net
============================================
Joint spatial and channel attention that refines encoder skip
connections before they are merged with the decoder stream.
Described in Section 3.2 of the paper.

Design overview
---------------
For each decoder level the ILFM receives:
  - a *gating* feature map ``g``  (upsampled from the level below),
  - a *skip-connection* map  ``s`` (from the matching encoder level).

It computes two complementary attention-weighted versions of ``s``:

  Spatial attention  — *where* to focus (pixel-level weights),
                       refined by a learned channel-wise MLP.
  Channel attention  — *which* features matter (channel-level weights),
                       derived from pooled statistics of ``s``.

The two outputs are fused by element-wise summation (see Section 3.2
of the paper for the theoretical justification).

"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_size(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Resize ``src`` to the spatial size of ``tgt`` if they differ."""
    if src.shape[2:] != tgt.shape[2:]:
        src = TF.resize(src, size=list(tgt.shape[2:]), antialias=True)
    return src


def _layer_norm(x: torch.Tensor) -> torch.Tensor:
    """LayerNorm over the channel + spatial dimensions (C, H, W)."""
    return F.layer_norm(x, x.shape[1:])


# ---------------------------------------------------------------------------
# Channel-wise bottleneck MLP  (persistent, learned sub-module)
# ---------------------------------------------------------------------------

class _ChannelMLP(nn.Module):
    """Bottleneck MLP that refines channel statistics of a feature map.

    Operates as a Squeeze-and-Excitation-style bottleneck:

      1. Global average pooling  → (B, C, 1, 1)
      2. Flatten                 → (B, C)
      3. Linear(C, C) → ReLU → Linear(C, C//2) → ReLU → Linear(C//2, C)
      4. Reshape back            → (B, C, 1, 1)

    Because the MLP acts on the *channel* dimension only, its parameter
    count is O(C²) rather than O(C · H · W), keeping the total model
    size comparable to the original ~11 M parameter count.

    Parameters
    ----------
    channels : int
        Number of channels of the input feature map.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        bottleneck = channels // 2
        self.gap = nn.AdaptiveAvgPool2d(1)   # global average pool
        self.net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=False),
            nn.Linear(channels, bottleneck),
            nn.ReLU(inplace=False),
            nn.Linear(bottleneck, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Channel-refined tensor of the same shape as ``x``.
            The spatial structure is preserved: the MLP output is
            broadcast back to ``(B, channels, H, W)``.
        """
        B, C, H, W = x.shape
        # Squeeze → MLP → Excite
        squeezed = self.gap(x).reshape(B, C)          # (B, C)
        excited  = self.net(squeezed).reshape(B, C, 1, 1)  # (B, C, 1, 1)
        return x * torch.sigmoid(excited)              # broadcast × x


# ---------------------------------------------------------------------------
# Core attention block  (spatial + channel attention, fused by summation)
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """Joint spatial and channel attention applied to a skip connection.

    Given a gating signal ``g`` and a skip connection ``s``, computes:

    .. math::
        \\text{Output} = (A_{\\text{spatial}} \\odot s)
                       + (A_{\\text{channel}} \\odot s)

    where :math:`A_{\\text{spatial}}` and :math:`A_{\\text{channel}}`
    are the spatial and channel attention maps respectively, and
    :math:`\\odot` denotes element-wise multiplication.

    Parameters
    ----------
    in_channels : List[int]
        ``[gate_channels, skip_channels]``.
        ``gate_channels`` is expected to equal ``2 × skip_channels``
        (standard U-Net bottleneck/decoder convention).
    out_channels : int
        Number of output channels.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
    ) -> None:
        super().__init__()

        gate_c, skip_c = in_channels

        # 1×1 projections into the shared embedding space.
        self.proj_gate = nn.Sequential(
            nn.Conv2d(gate_c // 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.proj_skip = nn.Sequential(
            nn.Conv2d(skip_c, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        # After cat(gate_feat, skip_feat): 2 × out_channels channels.
        # reduce_conv brings it back to out_channels for the spatial map.
        self.reduce_conv = nn.Conv2d(gate_c, out_channels, kernel_size=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.relu     = nn.ReLU(inplace=False)

        # Learned channel-wise MLP — registered persistent sub-module.
        # Input has 2 × out_channels (concatenated gate + skip projections).
        self.channel_mlp = _ChannelMLP(channels=2 * out_channels)

    def forward(self, g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        g : torch.Tensor
            Gating signal (upsampled), shape ``(B, gate_channels, H, W)``.
        s : torch.Tensor
            Skip-connection features, shape ``(B, skip_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Attention-refined tensor, same spatial size as ``s``.
        """
        # Normalise both inputs for training stability.
        g = _layer_norm(g)
        s = _layer_norm(s)

        g_feat = self.proj_gate(g)          # (B, out_c, H, W)
        s_feat = self.proj_skip(s)          # (B, out_c, H, W)

        # ── Channel attention branch ─────────────────────────────────
        # Max-pool + avg-pool of the skip projection → channel map.
        ch_max = self.relu(self.proj_skip(self.max_pool(s)))
        ch_avg = self.relu(self.proj_skip(self.avg_pool(s)))
        A_channel = torch.sigmoid(ch_max + ch_avg)  # (B, out_c, H-1, W-1)

        # ── Spatial attention branch ─────────────────────────────────
        g_feat = _match_size(g_feat, s_feat)
        fused  = torch.cat((g_feat, s_feat), dim=1)   # (B, 2*out_c, H, W)

        # Pool + activate.
        pooled    = self.max_pool(fused) + self.avg_pool(fused)
        activated = self.relu(pooled)

        # Channel-wise MLP refinement (persistent, properly trained).
        mlp_out   = self.channel_mlp(pooled)           # (B, 2*out_c, H-1, W-1)
        refined   = activated * mlp_out
        A_spatial = torch.sigmoid(pooled + refined)
        A_spatial = self.reduce_conv(A_spatial)        # (B, out_c, H-1, W-1)

        # ── Combine with skip connection ─────────────────────────────
        A_channel = _match_size(A_channel, s)
        A_spatial = _match_size(A_spatial, s)

        # Element-wise summation fusion (Section 3.2 of the paper).
        output = (A_spatial * s) + (A_channel * s)
        return _layer_norm(output)


# ---------------------------------------------------------------------------
# Full ILFM block  (upsample → attention → residual conv fusion)
# ---------------------------------------------------------------------------

class ILFM(nn.Module):
    """Inter-Level Fusion Module (full block).

    Wraps :class:`AttentionBlock` with transposed-convolution
    upsampling of the gating signal and a residual double-convolution
    fusion step that produces the enriched feature stream forwarded to
    the next decoder level.

    Parameters
    ----------
    in_channels : List[int]
        ``[gate_channels, skip_channels]``.
    out_channels : int
        Number of output channels.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
    ) -> None:
        super().__init__()
        gate_c, skip_c = in_channels

        self.upsample  = nn.ConvTranspose2d(
            gate_c, skip_c, kernel_size=2, stride=2
        )
        self.attention = AttentionBlock(in_channels, out_channels)

        # Residual double-conv fusion (standard U-Net decoder block).
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(gate_c, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Features from the level below, shape
            ``(B, gate_channels, H/2, W/2)``.
        s : torch.Tensor
            Encoder skip connection, shape ``(B, skip_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Fused output, shape ``(B, out_channels, H, W)``.
        """
        x      = self.upsample(x)                        # (B, skip_c, H, W)
        s_attn = self.attention(x, s)                    # attention-refined skip
        x      = _match_size(x, s_attn)
        fused  = self.relu(
            self.fuse_conv(torch.cat((x, s_attn), dim=1))
        )
        return fused + s_attn                            # residual connection