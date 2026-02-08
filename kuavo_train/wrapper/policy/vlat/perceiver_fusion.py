"""
Perceiver-style multi-modal fusion module.

Uses learnable query tokens and cross-attention to fuse:
- RGB image tokens from multiple cameras
- Depth image tokens from multiple cameras  
- Robot proprioceptive state
- Environment state (optional)

The output is a fixed-size representation regardless of input modalities.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import math


class PerceiverFusionBlock(nn.Module):
    """
    Single Perceiver fusion block with cross-attention and feedforward.
    
    Uses learnable queries to attend to multi-modal inputs.
    """
    
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        feedforward_multiplier: int = 4,
    ):
        super().__init__()
        
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        
        # Cross-attention: queries from learnable tokens, K/V from inputs
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=kv_dim,
            vdim=kv_dim,
            dropout=dropout,
            batch_first=True,
        )
        
        # Self-attention on queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * feedforward_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * feedforward_multiplier, query_dim),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        queries: Tensor, 
        kv: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            queries: (B, N_queries, query_dim)
            kv: (B, N_kv, kv_dim) - concatenated multi-modal features
            key_padding_mask: (B, N_kv) - True for padding positions
        
        Returns:
            (B, N_queries, query_dim) - updated queries
        """
        # Cross-attention
        cross_out, _ = self.cross_attn(
            query=queries,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
        )
        queries = self.norm1(queries + self.dropout(cross_out))
        
        # Self-attention on queries
        self_out, _ = self.self_attn(
            query=queries,
            key=queries,
            value=queries,
        )
        queries = self.norm2(queries + self.dropout(self_out))
        
        # Feedforward
        ffn_out = self.ffn(queries)
        queries = self.norm3(queries + ffn_out)
        
        return queries


class PerceiverMultiModalFusion(nn.Module):
    """
    Multi-modal fusion using Perceiver architecture.
    
    Fuses RGB images, depth images, and robot state into a fixed-size
    representation using learnable query tokens and cross-attention.
    """
    
    def __init__(
        self,
        n_query_tokens: int,
        fusion_dim: int,
        rgb_token_dim: Optional[int] = None,
        depth_token_dim: Optional[int] = None,
        state_dim: Optional[int] = None,
        num_fusion_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_query_tokens: Number of learnable query tokens
            fusion_dim: Hidden dimension for fusion
            rgb_token_dim: Dimension of RGB ViT tokens
            depth_token_dim: Dimension of depth ViT tokens
            state_dim: Dimension of robot state
            num_fusion_layers: Number of Perceiver blocks
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_query_tokens = n_query_tokens
        self.fusion_dim = fusion_dim
        self.rgb_token_dim = rgb_token_dim
        self.depth_token_dim = depth_token_dim
        self.state_dim = state_dim
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, n_query_tokens, fusion_dim) / math.sqrt(fusion_dim)
        )
        
        # Project different modalities to common dimension for K/V
        self.use_rgb = rgb_token_dim is not None
        self.use_depth = depth_token_dim is not None
        self.use_state = state_dim is not None
        
        if self.use_rgb:
            self.rgb_proj = nn.Linear(rgb_token_dim, fusion_dim)
        
        if self.use_depth:
            self.depth_proj = nn.Linear(depth_token_dim, fusion_dim)
        
        if self.use_state:
            self.state_proj = nn.Linear(state_dim, fusion_dim)
        
        # Perceiver fusion blocks
        self.fusion_blocks = nn.ModuleList([
            PerceiverFusionBlock(
                query_dim=fusion_dim,
                kv_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_fusion_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        rgb_tokens: Optional[Tensor] = None,
        depth_tokens: Optional[Tensor] = None,
        state: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Fuse multi-modal inputs into fixed-size representation.
        
        Args:
            rgb_tokens: (B, n_cameras, n_patches, rgb_token_dim) or None
            depth_tokens: (B, n_cameras, n_patches, depth_token_dim) or None
            state: (B, state_dim) or None
        
        Returns:
            (B, n_query_tokens, fusion_dim) - fused multi-modal features
        """
        B = (rgb_tokens.shape[0] if rgb_tokens is not None 
             else depth_tokens.shape[0] if depth_tokens is not None
             else state.shape[0])
        
        # Collect all modalities
        kv_list = []
        
        # Process RGB tokens
        if self.use_rgb and rgb_tokens is not None:
            # Flatten camera and patch dimensions
            B, n_cam, n_patches, rgb_dim = rgb_tokens.shape
            rgb_flat = rgb_tokens.reshape(B, n_cam * n_patches, rgb_dim)
            rgb_proj = self.rgb_proj(rgb_flat)  # (B, n_cam*n_patches, fusion_dim)
            kv_list.append(rgb_proj)
        
        # Process depth tokens
        if self.use_depth and depth_tokens is not None:
            # Flatten camera and patch dimensions
            B, n_cam, n_patches, depth_dim = depth_tokens.shape
            depth_flat = depth_tokens.reshape(B, n_cam * n_patches, depth_dim)
            depth_proj = self.depth_proj(depth_flat)  # (B, n_cam*n_patches, fusion_dim)
            kv_list.append(depth_proj)
        
        # Process state
        if self.use_state and state is not None:
            state_proj = self.state_proj(state).unsqueeze(1)  # (B, 1, fusion_dim)
            kv_list.append(state_proj)
        
        # Concatenate all K/V inputs
        if len(kv_list) == 0:
            raise ValueError("At least one input modality must be provided")
        
        kv = torch.cat(kv_list, dim=1)  # (B, total_tokens, fusion_dim)
        
        # Expand query tokens for batch
        queries = self.query_tokens.expand(B, -1, -1)  # (B, n_query_tokens, fusion_dim)
        
        # Apply Perceiver fusion blocks
        for block in self.fusion_blocks:
            queries = block(queries, kv)
        
        # Output projection
        output = self.output_proj(queries)  # (B, n_query_tokens, fusion_dim)
        
        return output


class RGBDepthCrossAttention(nn.Module):
    """
    Cross-attention between RGB and depth modalities before Perceiver fusion.
    
    This allows RGB and depth to interact directly before being fused with state.
    """
    
    def __init__(
        self,
        rgb_dim: int,
        depth_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.rgb_dim = rgb_dim
        self.depth_dim = depth_dim
        
        # RGB queries depth
        self.rgb_to_depth_attn = nn.MultiheadAttention(
            embed_dim=rgb_dim,
            num_heads=num_heads,
            kdim=depth_dim,
            vdim=depth_dim,
            dropout=dropout,
            batch_first=True,
        )
        
        # Depth queries RGB
        self.depth_to_rgb_attn = nn.MultiheadAttention(
            embed_dim=depth_dim,
            num_heads=num_heads,
            kdim=rgb_dim,
            vdim=rgb_dim,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norms
        self.norm_rgb = nn.LayerNorm(rgb_dim)
        self.norm_depth = nn.LayerNorm(depth_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        rgb_tokens: Tensor,
        depth_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Cross-attend between RGB and depth tokens.
        
        Args:
            rgb_tokens: (B, n_rgb_tokens, rgb_dim)
            depth_tokens: (B, n_depth_tokens, depth_dim)
        
        Returns:
            Tuple of:
                - Enhanced RGB tokens (B, n_rgb_tokens, rgb_dim)
                - Enhanced depth tokens (B, n_depth_tokens, depth_dim)
        """
        # RGB attends to depth
        rgb_enhanced, _ = self.rgb_to_depth_attn(
            query=rgb_tokens,
            key=depth_tokens,
            value=depth_tokens,
        )
        rgb_tokens = self.norm_rgb(rgb_tokens + self.dropout(rgb_enhanced))
        
        # Depth attends to RGB
        depth_enhanced, _ = self.depth_to_rgb_attn(
            query=depth_tokens,
            key=rgb_tokens,
            value=rgb_tokens,
        )
        depth_tokens = self.norm_depth(depth_tokens + self.dropout(depth_enhanced))
        
        return rgb_tokens, depth_tokens
