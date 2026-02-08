"""
Vision Transformer encoders for RGB and Depth images.

Uses timm library for pretrained ViT models with custom adaptations for:
- Multi-camera inputs
- Depth image encoding (single channel)
- Flexible output representations
"""

import torch
import torch.nn as nn
from torch import Tensor
import timm
from typing import Optional
import einops

from kuavo_train.wrapper.policy.vlat.VLATConfigWrapper import CustomVLATConfigWrapper


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer encoder using timm library.
    
    Extracts patch tokens from images and optionally returns:
    - All patch tokens (for cross-attention)
    - CLS token only (for simple pooling)
    - Global average pooled features
    """
    
    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        in_channels: int = 3,
        image_size: tuple[int, int] = (224, 224),
        output_tokens: bool = True,
    ):
        """
        Args:
            model_name: timm model name (e.g., 'vit_small_patch16_224')
            pretrained: whether to load pretrained weights
            in_channels: number of input channels (3 for RGB, 1 for depth)
            image_size: input image size (H, W)
            output_tokens: if True, return all patch tokens; if False, return CLS token only
        """
        super().__init__()
        
        self.output_tokens = output_tokens
        self.image_size = image_size
        self.in_channels = in_channels
        
        # Load ViT model from timm
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=image_size,
            num_classes=0,  # Remove classification head
        )
        
        # Adapt first conv layer for depth (single channel)
        if in_channels != 3:
            old_patch_embed = self.vit.patch_embed.proj
            self.vit.patch_embed.proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_patch_embed.out_channels,
                kernel_size=old_patch_embed.kernel_size,
                stride=old_patch_embed.stride,
                padding=old_patch_embed.padding,
                bias=old_patch_embed.bias is not None,
            )
            
            # Initialize from pretrained RGB weights (average across channels)
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    self.vit.patch_embed.proj.weight = nn.Parameter(
                        old_patch_embed.weight.mean(dim=1, keepdim=True)
                    )
                    if old_patch_embed.bias is not None:
                        self.vit.patch_embed.proj.bias = old_patch_embed.bias
        
        # Get embedding dimension
        self.embed_dim = self.vit.embed_dim
        
        # Calculate number of patches
        patch_size = self.vit.patch_embed.patch_size[0]
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        
        # Feature dimension for output
        if output_tokens:
            self.feature_dim = self.embed_dim * self.num_patches
        else:
            self.feature_dim = self.embed_dim
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through ViT.
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            If output_tokens=True: (B, num_patches, embed_dim) patch tokens
            If output_tokens=False: (B, embed_dim) CLS token
        """
        # Pass through ViT blocks
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        
        if self.output_tokens:
            # Return all patch tokens (excluding CLS token if present)
            if self.vit.global_pool == 'token':
                # CLS token is first, so remove it
                return x[:, 1:, :]  # (B, num_patches, embed_dim)
            else:
                return x  # (B, num_patches, embed_dim)
        else:
            # Return CLS token only or global average pool
            if self.vit.global_pool == 'token':
                return x[:, 0]  # (B, embed_dim)
            else:
                return x.mean(dim=1)  # (B, embed_dim)


class RGBViTEncoder(nn.Module):
    """
    RGB image encoder using Vision Transformer.
    
    Handles multiple camera inputs and outputs patch tokens for each camera.
    """
    
    def __init__(self, config: CustomVLATConfigWrapper):
        super().__init__()
        self.config = config
        
        # Use separate encoder per camera or shared encoder
        self.use_separate_encoders = getattr(
            config, 
            'use_separate_rgb_encoder_per_camera', 
            False
        )
        
        # Number of cameras
        self.num_cameras = len(config.image_features) if config.image_features else 0
        
        if self.use_separate_encoders and self.num_cameras > 0:
            # Create separate ViT encoder for each camera
            self.encoders = nn.ModuleList([
                VisionTransformerEncoder(
                    model_name=config.vision_backbone,
                    pretrained=config.pretrained_backbone_weights is not None,
                    in_channels=3,
                    image_size=tuple(config.image_size),
                    output_tokens=True,
                )
                for _ in range(self.num_cameras)
            ])
            self.embed_dim = self.encoders[0].embed_dim
            self.num_patches = self.encoders[0].num_patches
        elif self.num_cameras > 0:
            # Shared ViT encoder for all cameras
            self.encoder = VisionTransformerEncoder(
                model_name=config.vision_backbone,
                pretrained=config.pretrained_backbone_weights is not None,
                in_channels=3,
                image_size=tuple(config.image_size),
                output_tokens=True,
            )
            self.embed_dim = self.encoder.embed_dim
            self.num_patches = self.encoder.num_patches
        else:
            self.embed_dim = 0
            self.num_patches = 0
    
    def forward(self, images: Tensor) -> Tensor:
        """
        Forward pass for RGB images.
        
        Args:
            images: (B, n_cameras, C, H, W)
        
        Returns:
            (B, n_cameras, num_patches, embed_dim) patch tokens for all cameras
        """
        if self.num_cameras == 0:
            return None
        
        B, n_cam, C, H, W = images.shape
        
        if self.use_separate_encoders:
            # Process each camera with its own encoder
            outputs = []
            for i, encoder in enumerate(self.encoders):
                cam_imgs = images[:, i]  # (B, C, H, W)
                cam_tokens = encoder(cam_imgs)  # (B, num_patches, embed_dim)
                outputs.append(cam_tokens)
            # Stack along camera dimension
            return torch.stack(outputs, dim=1)  # (B, n_cam, num_patches, embed_dim)
        else:
            # Shared encoder: process all cameras in batch
            images_flat = einops.rearrange(images, "b n c h w -> (b n) c h w")
            tokens_flat = self.encoder(images_flat)  # (B*n_cam, num_patches, embed_dim)
            tokens = einops.rearrange(
                tokens_flat, 
                "(b n) p d -> b n p d", 
                b=B, 
                n=n_cam
            )
            return tokens  # (B, n_cam, num_patches, embed_dim)


class DepthViTEncoder(nn.Module):
    """
    Depth image encoder using Vision Transformer.
    
    Handles single-channel depth images with pretrained RGB ViT initialization.
    """
    
    def __init__(self, config: CustomVLATConfigWrapper):
        super().__init__()
        self.config = config
        
        # Check if depth is enabled
        self.use_depth = getattr(config, 'use_depth', False)
        
        if not self.use_depth:
            self.embed_dim = 0
            self.num_patches = 0
            return
        
        # Use separate encoder per camera or shared encoder
        self.use_separate_encoders = getattr(
            config,
            'use_separate_depth_encoder_per_camera',
            False
        )
        
        # Number of depth cameras
        self.num_cameras = len(config.depth_features) if config.depth_features else 0
        
        # Get depth backbone name (may be different from RGB)
        depth_backbone = getattr(config, 'depth_backbone', config.vision_backbone)
        
        if self.use_separate_encoders and self.num_cameras > 0:
            # Create separate ViT encoder for each depth camera
            self.encoders = nn.ModuleList([
                VisionTransformerEncoder(
                    model_name=depth_backbone,
                    pretrained=config.pretrained_backbone_weights is not None,
                    in_channels=1,  # Depth is single channel
                    image_size=tuple(config.image_size),
                    output_tokens=True,
                )
                for _ in range(self.num_cameras)
            ])
            self.embed_dim = self.encoders[0].embed_dim
            self.num_patches = self.encoders[0].num_patches
        elif self.num_cameras > 0:
            # Shared ViT encoder for all depth cameras
            self.encoder = VisionTransformerEncoder(
                model_name=depth_backbone,
                pretrained=config.pretrained_backbone_weights is not None,
                in_channels=1,  # Depth is single channel
                image_size=tuple(config.image_size),
                output_tokens=True,
            )
            self.embed_dim = self.encoder.embed_dim
            self.num_patches = self.encoder.num_patches
        else:
            self.embed_dim = 0
            self.num_patches = 0
    
    def forward(self, depth_images: Tensor) -> Optional[Tensor]:
        """
        Forward pass for depth images.
        
        Args:
            depth_images: (B, n_cameras, 1, H, W) or (B, n_cameras, 3, H, W)
        
        Returns:
            (B, n_cameras, num_patches, embed_dim) patch tokens for all cameras
            or None if depth is not used
        """
        if not self.use_depth or self.num_cameras == 0:
            return None
        
        B, n_cam = depth_images.shape[:2]
        
        # Convert to single channel if needed
        if depth_images.shape[2] == 3:
            # Average RGB channels to get single channel
            depth_images = depth_images.mean(dim=2, keepdim=True)
        
        if self.use_separate_encoders:
            # Process each camera with its own encoder
            outputs = []
            for i, encoder in enumerate(self.encoders):
                cam_depths = depth_images[:, i]  # (B, 1, H, W)
                cam_tokens = encoder(cam_depths)  # (B, num_patches, embed_dim)
                outputs.append(cam_tokens)
            # Stack along camera dimension
            return torch.stack(outputs, dim=1)  # (B, n_cam, num_patches, embed_dim)
        else:
            # Shared encoder: process all cameras in batch
            depth_flat = einops.rearrange(depth_images, "b n c h w -> (b n) c h w")
            tokens_flat = self.encoder(depth_flat)  # (B*n_cam, num_patches, embed_dim)
            tokens = einops.rearrange(
                tokens_flat,
                "(b n) p d -> b n p d",
                b=B,
                n=n_cam
            )
            return tokens  # (B, n_cam, num_patches, embed_dim)
