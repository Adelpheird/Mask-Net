"""
Mask-Net package
================
Import the model directly from the top-level package:

    from masknet import MaskNet, build_masknet
"""

from .model import MaskNet, build_masknet

__all__ = ["MaskNet", "build_masknet"]
