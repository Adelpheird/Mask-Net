"""
Mask-Net

U-Net-based architecture with spatial and channel attention
for cloud masking in Sea Surface Temperature (SST) imagery.

Authors: [Kouassi Adelphe Christian N'GORAN]
Affiliation: [INPHB & IRD]
Date: [16/06/2026]

References:
    [My paper citations here]

License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import List, Tuple, Optional, Union


class ConvBlock(nn.Module):
    """
    Standard convolutional block consisting of two convolutional layers with 
    batch normalization and ReLU activation.
    
    This block is used as a fundamental building component throughout the network,
    providing consistent feature extraction capabilities.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolutional kernel (default: 3)
        stride: Convolution stride (default: 1)
        padding: Padding size (default: 1)
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                     padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, 
                     padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor after convolution and activation
        """
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    Encoder block for the U-Net architecture, combining convolution and downsampling.
    
    This block performs feature extraction through convolution followed by 
    max-pooling for spatial dimension reduction.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        use_maxpool: Whether to use max-pooling (default: True)
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 use_maxpool: bool = True):
        super(EncoderBlock, self).__init__()
        
        self.conv = ConvBlock(in_channels, out_channels)
        
        # Downsampling strategy
        if use_maxpool:
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            # Alternative: strided convolution for learnable downsampling
            self.downsample = nn.Conv2d(out_channels, out_channels, 
                                       kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder block.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Skip connection output (for decoder)
                - Downsampled output (for next encoder layer)
        """
        skip = self.conv(x)      # Features for skip connection
        pooled = self.downsample(skip)  # Downsampled features
        return skip, pooled


class MultiLayerPerceptron(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module for feature transformation.
    
    This module is used within the attention mechanism to project features
    into a higher-dimensional space for improved representation learning.
    
    Args:
        feature_map_size: Number of features in the feature map
        spatial_size: Spatial dimensions of the feature map
        hidden_dim: Dimension of the hidden layer (default: None, uses feature_map_size//2)
    """
    
    def __init__(self, feature_map_size: int, spatial_size: int, 
                 hidden_dim: Optional[int] = None):
        super(MultiLayerPerceptron, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = feature_map_size // 2
            
        input_size = feature_map_size * spatial_size * spatial_size
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, feature_map_size),
            nn.ReLU(inplace=False),
            nn.Linear(feature_map_size, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, input_size),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Transformed tensor with same shape as input
        """
        batch_size, channels, height, width = x.size()
        
        # Flatten spatial dimensions
        x_flat = x.view(batch_size, -1)
        
        # Apply MLP
        x_transformed = self.mlp(x_flat)
        
        # Reshape back to original dimensions
        return x_transformed.view(batch_size, channels, height, width)


class AttentionBlock(nn.Module):
    """
    Attention mechanism combining spatial and channel attention.
    
    This block implements a sophisticated attention mechanism that combines
    both spatial and channel attention for improved feature selection.
    
    Args:
        gate_channels: Number of channels for the gating signal
        skip_channels: Number of channels for the skip connection
        output_channels: Number of output channels
    """
    
    def __init__(self, gate_channels: int, skip_channels: int, 
                 output_channels: int):
        super(AttentionBlock, self).__init__()
        
        # Gating signal processing
        self.Wg = nn.Sequential(
            nn.Conv2d(gate_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels)
        )
        
        # Skip connection processing
        self.Ws = nn.Sequential(
            nn.Conv2d(skip_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels)
        )
        
        # Attention coefficient generation
        self.conv = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.avg = nn.AvgPool2d(kernel_size=2, stride=1)
        self.relu = nn.ReLU(inplace=False)
        
        self.output = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.Sigmoid()
        )
        
        # MLP module for feature transformation
        self.mlp = MultiLayerPerceptron(output_channels, 
                                       spatial_size=1)  # Will be adapted dynamically
        
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention block.
        
        Args:
            gate: Gating signal from decoder path
            skip: Skip connection from encoder path
            
        Returns:
            torch.Tensor: Attended features combining spatial and channel attention
        """
        device = gate.device
        
        # Layer normalization
        gate = nn.LayerNorm(gate.squeeze().shape, device=device)(gate)
        skip = nn.LayerNorm(skip.squeeze().shape, device=device)(skip)
        
        # Process gate and skip signals
        Wg = self.Wg(gate)
        Ws = self.Ws(skip)
        
        # Channel attention branch
        Ws_pool = self.relu(self.Ws(self.pool(skip)))
        Ws_avg = self.relu(self.Ws(self.avg(skip)))
        channel_attention = nn.Sigmoid()(Ws_pool + Ws_avg)
        
        # Spatial attention branch with MLP
        concat = torch.cat((Wg, Ws), dim=1)
        spatial_features = self.pool(concat) + self.avg(concat)
        spatial_features = self.relu(spatial_features)
        
        # Apply MLP for spatial attention
        mlp_output = self.mlp(spatial_features)
        spatial_attention = nn.Sigmoid()(spatial_features + mlp_output)
        
        # Combine spatial and channel attention
        combined_attention = nn.Sigmoid()(concat + spatial_attention)
        combined_attention = self.conv(combined_attention)
        
        # Apply attention to skip connection
        spatial_attention_maps = combined_attention * skip
        channel_attention_maps = channel_attention * skip
        
        # Final attention output
        attention_output = spatial_attention_maps + channel_attention_maps
        attention_output = nn.LayerNorm(attention_output.squeeze().shape, 
                                       device=device)(attention_output)
        
        return attention_output


class LevelFusionModule(nn.Module):
    """
    Multi-level feature fusion module combining decoder and attention features.
    
    This module fuses features from different levels of the network using
    attention mechanisms to enhance feature representation.
    
    Args:
        gate_channels: Number of channels in the gating signal
        skip_channels: Number of channels in the skip connection
        output_channels: Number of output channels
    """
    
    def __init__(self, gate_channels: int, skip_channels: int, 
                 output_channels: int):
        super(LevelFusionModule, self).__init__()
        
        self.up = nn.ConvTranspose2d(gate_channels, skip_channels, 
                                    kernel_size=2, stride=2)
        self.attention = AttentionBlock(skip_channels, skip_channels, output_channels)
        self.conv = ConvBlock(gate_channels + skip_channels, output_channels)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the level fusion module.
        
        Args:
            x: Features from decoder path
            skip: Features from encoder path
            
        Returns:
            torch.Tensor: Fused features
        """
        # Upsample decoder features
        x_up = self.up(x)
        
        # Apply attention to skip connection
        skip_attended = self.attention(x_up, skip)
        
        # Ensure spatial dimensions match
        if x_up.shape != skip_attended.shape:
            x_up = TF.resize(x_up, size=skip_attended.shape[2:], antialias=True)
        
        # Concatenate and process
        x_concat = torch.cat((x_up, skip_attended), dim=1)
        x_fused = self.conv(x_concat)
        x_fused = self.relu(x_fused)
        
        # Residual connection
        return x_fused + skip_attended


class DecoderBlock(nn.Module):
    """
    Standard decoder block for the U-Net architecture.
    
    This block performs upsampling and feature fusion between decoder and
    encoder features.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                    kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder block.
        
        Args:
            x: Features from previous decoder layer
            skip: Features from corresponding encoder layer
            
        Returns:
            torch.Tensor: Decoded features
        """
        # Upsample
        x_up = self.up(x)
        
        # Ensure spatial dimensions match
        if x_up.shape != skip.shape:
            x_up = TF.resize(x_up, size=skip.shape[2:], antialias=True)
        
        # Concatenate and process
        x_cat = torch.cat((x_up, skip), dim=1)
        x_out = self.conv(x_cat)
        
        return x_out


class MaskNet(nn.Module):
    """
    MaskNet: A deep learning architecture for image segmentation with multi-level attention.
    
    This architecture combines U-Net with advanced attention mechanisms and
    multi-level feature fusion for improved segmentation performance. The network
    is designed for medical image segmentation tasks and can handle both single
    and multi-channel inputs.
    
    Architecture Overview:
        - Encoder: Multi-level feature extraction with skip connections
        - Decoder SST: Feature fusion with attention mechanisms
        - Decoder Mask: Final mask generation with multi-level features
        
    Args:
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 1)
        features: Feature dimensions at each encoder level (default: [32, 64, 128, 256])
        device: Device to run the model on (default: None, auto-detects)
        
    Example:
        >>> model = MaskNet(in_channels=1, out_channels=1)
        >>> x = torch.randn(1, 1, 256, 256)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 1, 256, 256])
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: List[int] = [32, 64, 128, 256]):
        super(MaskNet, self).__init__()
        
        self.features = features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize network modules
        self._init_encoders(in_channels)
        self._init_decoders()
        self._init_bottleneck()
        self._init_final_layers()
        
    def _init_encoders(self, in_channels: int) -> None:
        """Initialize encoder blocks."""
        self.encoders = nn.ModuleList()
        for feature in self.features:
            self.encoders.append(EncoderBlock(in_channels, feature))
            in_channels = feature
            
    def _init_decoders(self) -> None:
        """Initialize decoder blocks."""
        self.decoder_sst = nn.ModuleList()
        self.decoder_mask = nn.ModuleList()
        
        for i, feature in enumerate(reversed(self.features)):
            if i < len(self.features) - 1:
                next_feature = self.features[-i-1] if i == 0 else self.features[-i]
                gate_channels = feature * 2
                skip_channels = feature
                output_channels = feature
                
                self.decoder_sst.append(
                    LevelFusionModule(gate_channels, skip_channels, output_channels)
                )
                
                self.decoder_mask.append(
                    DecoderBlock(feature * 2, feature)
                )
            else:
                # Last decoder block
                self.decoder_sst.append(
                    LevelFusionModule(feature * 2, feature, feature)
                )
                self.decoder_mask.append(
                    DecoderBlock(feature * 2, feature)
                )
    
    def _init_bottleneck(self) -> None:
        """Initialize bottleneck layer."""
        last_feature = self.features[-1]
        self.bottleneck = EncoderBlock(last_feature, last_feature * 2)
        
    def _init_final_layers(self) -> None:
        """Initialize final layers."""
        self.normalize = nn.BatchNorm2d(self.features[0])
        self.conv_reduce = nn.Conv2d(self.features[1], self.features[0], kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.output = nn.Conv2d(self.features[0], 1, kernel_size=1)
        
    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass through the MaskNet architecture.
        
        Args:
            x: Input tensor or tuple of tensors. If tuple, processes two inputs
               simultaneously and fuses their features.
        
        Returns:
            torch.Tensor: Segmentation mask output
        """
        if isinstance(x, tuple):
            return self._forward_multi_input(x)
        else:
            return self._forward_single_input(x)
    
    def _forward_single_input(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single input."""
        skip_connections = []
        
        # Encoder path
        for encoder in self.encoders:
            skip, x = encoder(x)
            if x.shape[1] == self.features[0]:
                x = self.normalize(x)
            skip_connections.append(skip)
        
        # Bottleneck
        _, bottleneck = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder path with SST features
        sst_features = self._decode_with_sst(bottleneck, skip_connections)
        
        # Final mask generation
        mask = self._generate_mask(bottleneck, sst_features, skip_connections)
        
        return mask
    
    def _forward_multi_input(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Forward pass for two inputs (e.g., SST and climate data)."""
        x1, x2 = inputs
        skip_connections = []
        skip_end = None
        
        # Encoder path with feature fusion
        for encoder in self.encoders:
            skip1, x1 = encoder(x1)
            skip2, x2 = encoder(x2)
            
            # Fuse features from both inputs
            skip = torch.cat((skip1, skip2), dim=1)
            x = torch.cat((x1, x2), dim=1)
            
            skip_end = skip
            skip = self.conv_reduce(skip)
            
            if x.shape[1] == self.features[0]:
                x = self.normalize(x)
            x = self.conv_reduce(x)
            
            skip_connections.append(skip)
        
        # Bottleneck
        _, bottleneck = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder path with SST features
        sst_features = self._decode_with_sst(bottleneck, skip_connections)
        
        # Final mask generation with skip_end
        mask = self._generate_mask(bottleneck, sst_features, skip_connections, skip_end)
        
        return mask
    
    def _decode_with_sst(self, bottleneck: torch.Tensor, 
                         skip_connections: List[torch.Tensor]) -> List[torch.Tensor]:
        """SST decoder with attention."""
        x = bottleneck
        sst_features = []
        
        for i, decoder in enumerate(self.decoder_sst):
            skip_connection = skip_connections[i]
            x = decoder(x, skip_connection)
            x = self.relu(x)
            sst_features.append(x)
            
        return sst_features
    
    def _generate_mask(self, bottleneck: torch.Tensor, 
                      sst_features: List[torch.Tensor],
                      skip_connections: List[torch.Tensor],
                      skip_end: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate final mask using mask decoder."""
        x = bottleneck
        
        for i, decoder in enumerate(self.decoder_mask):
            skip_connection = skip_connections[i]
            
            # Handle skip_end for multi-input case
            if skip_end is not None and i == 0:
                if skip_end.shape != skip_connection.shape:
                    skip_end = TF.resize(skip_end, size=skip_connection.shape[2:], 
                                       antialias=True)
            
            x = decoder(x, sst_features[i])
            x = self.relu(x)
        
        x = self.normalize(x)
        mask = self.output(x)
        
        return mask


def create_model(in_channels: int = 1, out_channels: int = 1,
                 features: List[int] = [32, 64, 128, 256]) -> MaskNet:
    """
    Factory function to create a MaskNet model.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        features: Feature dimensions at each encoder level
    
    Returns:
        MaskNet: Initialized MaskNet model
    """
    return MaskNet(in_channels=in_channels, out_channels=out_channels, 
                   features=features)


def test_model(device: Optional[torch.device] = None) -> None:
    """
    Test function to verify model architecture and forward pass.
    
    Args:
        device: Device to run the test on (default: None, auto-detects)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Testing MaskNet Architecture")
    print("-" * 50)
    
    # Test single input
    print("Testing single input:")
    x = torch.randn(1, 1, 256, 256).to(device)
    model = MaskNet(in_channels=1, out_channels=1).to(device)
    pred = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test multi-input
    print("\nTesting multi-input:")
    x1 = torch.randn(1, 1, 256, 256).to(device)
    x2 = torch.randn(1, 1, 256, 256).to(device)
    pred = model((x1, x2))
    print(f"Input 1 shape: {x1.shape}")
    print(f"Input 2 shape: {x2.shape}")
    print(f"Output shape: {pred.shape}")
    
    print("-" * 50)
    print("Test completed successfully!")


if __name__ == "__main__":
    test_model()