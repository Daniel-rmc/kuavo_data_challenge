"""
Vision-Language-Action Transformer (VLAT) Model.

Integrates all components:
- ViT encoders for RGB and depth
- Perceiver fusion for multi-modal integration
- Temporal transformer for sequence modeling
- Transformer decoder for action prediction
- Uncertainty estimation head
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict
import math
import einops

from kuavo_train.wrapper.policy.vlat.VLATConfigWrapper import CustomVLATConfigWrapper
from kuavo_train.wrapper.policy.vlat.vision_transformer import RGBViTEncoder, DepthViTEncoder
from kuavo_train.wrapper.policy.vlat.perceiver_fusion import (
    PerceiverMultiModalFusion,
    RGBDepthCrossAttention,
)
from kuavo_train.wrapper.policy.vlat.temporal_transformer import TemporalTransformer
from lerobot.utils.constants import OBS_STATE, OBS_IMAGES, OBS_ENV_STATE, ACTION

OBS_DEPTH = "observation.depth"


class StateEncoder(nn.Module):
    """Simple MLP encoder for robot state."""
    
    def __init__(self, state_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class ActionDecoder(nn.Module):
    """
    Transformer decoder for action prediction.
    
    Autoregressively generates action sequences conditioned on
    temporal context from observations.
    """
    
    def __init__(
        self,
        d_model: int,
        action_dim: int,
        chunk_size: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
        # Learnable action queries (one per action in chunk)
        self.action_queries = nn.Parameter(
            torch.randn(1, chunk_size, d_model) / math.sqrt(d_model)
        )
        
        # Positional encoding for action sequence
        self.register_buffer(
            'pos_enc',
            self._create_positional_encoding(chunk_size, d_model)
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        
        # Output heads
        self.action_head = nn.Linear(d_model, action_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def _create_positional_encoding(self, length: int, d_model: int) -> Tensor:
        """Create sinusoidal positional encoding."""
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, length, d_model)
    
    def forward(self, context: Tensor) -> Tensor:
        """
        Decode actions from temporal context.
        
        Args:
            context: (B, context_dim) or (B, T, context_dim)
        
        Returns:
            (B, chunk_size, action_dim) predicted actions
        """
        B = context.shape[0]
        
        # Expand context to sequence if needed
        if context.dim() == 2:
            context = context.unsqueeze(1)  # (B, 1, D)
        
        # Prepare action queries
        queries = self.action_queries.expand(B, -1, -1)  # (B, chunk_size, d_model)
        queries = queries + self.pos_enc  # Add positional encoding
        
        # Decode actions
        decoded = self.transformer_decoder(
            tgt=queries,
            memory=context,
        )
        
        decoded = self.norm(decoded)
        
        # Project to action space
        actions = self.action_head(decoded)  # (B, chunk_size, action_dim)
        
        return actions


class UncertaintyHead(nn.Module):
    """
    Estimates uncertainty for predicted actions.
    
    Outputs standard deviation for each action dimension.
    """
    
    def __init__(self, d_model: int, action_dim: int, chunk_size: int):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, action_dim),
            nn.Softplus(),  # Ensure positive output
        )
    
    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, chunk_size, d_model)
        
        Returns:
            (B, chunk_size, action_dim) uncertainty (std dev)
        """
        return self.head(features)


class VLATModel(nn.Module):
    """
    Complete VLAT model integrating all components.
    """
    
    def __init__(self, config: CustomVLATConfigWrapper):
        super().__init__()
        
        self.config = config
        
        # 1. Vision encoders
        self.rgb_encoder = RGBViTEncoder(config)
        self.depth_encoder = DepthViTEncoder(config)
        
        # 2. State encoder
        self.state_encoder = None
        if config.robot_state_feature is not None:
            state_dim = config.robot_state_feature.shape[0]
            self.state_encoder = StateEncoder(
                state_dim=state_dim,
                output_dim=config.fusion_dim,
                dropout=config.fusion_dropout,
            )
        
        # 3. RGB-Depth cross-attention (optional)
        self.rgb_depth_cross_attn = None
        if (self.rgb_encoder.num_cameras > 0 and 
            self.depth_encoder.use_depth and 
            self.depth_encoder.num_cameras > 0):
            self.rgb_depth_cross_attn = RGBDepthCrossAttention(
                rgb_dim=self.rgb_encoder.embed_dim,
                depth_dim=self.depth_encoder.embed_dim,
                num_heads=config.fusion_heads,
                dropout=config.fusion_dropout,
            )
        
        # 4. Perceiver fusion
        self.perceiver_fusion = PerceiverMultiModalFusion(
            n_query_tokens=config.n_query_tokens,
            fusion_dim=config.fusion_dim,
            rgb_token_dim=self.rgb_encoder.embed_dim if self.rgb_encoder.num_cameras > 0 else None,
            depth_token_dim=self.depth_encoder.embed_dim if self.depth_encoder.use_depth else None,
            state_dim=config.fusion_dim if self.state_encoder is not None else None,
            num_fusion_layers=config.fusion_layers,
            num_heads=config.fusion_heads,
            dropout=config.fusion_dropout,
        )
        
        # 5. Temporal transformer
        # Input to temporal transformer: fused features from perceiver
        # We flatten query tokens to single vector per timestep
        self.temporal_proj = nn.Linear(
            config.n_query_tokens * config.fusion_dim,
            config.temporal_dim
        )
        
        self.temporal_transformer = TemporalTransformer(
            d_model=config.temporal_dim,
            num_layers=config.temporal_layers,
            num_heads=config.temporal_heads,
            dim_feedforward=config.temporal_dim * 4,
            dropout=config.temporal_dropout,
            causal=config.use_causal_temporal_attention,
            aggregation='attention',  # Aggregate temporal features
        )
        
        # 6. Action decoder
        self.action_decoder = ActionDecoder(
            d_model=config.decoder_dim,
            action_dim=config.action_feature.shape[0],
            chunk_size=config.chunk_size,
            num_layers=config.decoder_layers,
            num_heads=config.decoder_heads,
            dim_feedforward=config.decoder_dim * 4,
            dropout=config.decoder_dropout,
        )
        
        # Project temporal features to decoder dimension
        self.temporal_to_decoder = nn.Linear(config.temporal_dim, config.decoder_dim)
        
        # 7. Uncertainty head (optional)
        self.uncertainty_head = None
        if config.use_uncertainty:
            self.uncertainty_head = UncertaintyHead(
                d_model=config.decoder_dim,
                action_dim=config.action_feature.shape[0],
                chunk_size=config.chunk_size,
            )
    
    def forward(self, batch: Dict[str, Tensor]) -> tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through VLAT model.
        
        Args:
            batch: Dictionary containing:
                - OBS_IMAGES: (B, n_obs_steps, n_cameras, C, H, W)
                - OBS_DEPTH: (B, n_obs_steps, n_cameras, C, H, W) [optional]
                - OBS_STATE: (B, n_obs_steps, state_dim)
                - OBS_ENV_STATE: (B, n_obs_steps, env_state_dim) [optional]
        
        Returns:
            Tuple of:
                - actions: (B, chunk_size, action_dim)
                - uncertainty: (B, chunk_size, action_dim) or None
        """
        B = batch[OBS_STATE].shape[0] if OBS_STATE in batch else batch[OBS_IMAGES].shape[0]
        n_obs_steps = batch[OBS_STATE].shape[1] if OBS_STATE in batch else batch[OBS_IMAGES].shape[1]
        
        # Process each timestep
        fused_features_list = []
        
        for t in range(n_obs_steps):
            # Extract observations for this timestep
            rgb_t = None
            if OBS_IMAGES in batch:
                rgb_t = batch[OBS_IMAGES][:, t]  # (B, n_cameras, C, H, W)
            
            depth_t = None
            if OBS_DEPTH in batch:
                depth_t = batch[OBS_DEPTH][:, t]  # (B, n_cameras, C, H, W)
            
            state_t = None
            if OBS_STATE in batch:
                state_t = batch[OBS_STATE][:, t]  # (B, state_dim)
            
            # 1. Encode visual features
            rgb_tokens = self.rgb_encoder(rgb_t) if rgb_t is not None else None
            # (B, n_cameras, n_patches, rgb_embed_dim)
            
            depth_tokens = self.depth_encoder(depth_t) if depth_t is not None else None
            # (B, n_cameras, n_patches, depth_embed_dim)
            
            # 2. RGB-Depth cross-attention
            if self.rgb_depth_cross_attn is not None and rgb_tokens is not None and depth_tokens is not None:
                # Flatten camera and patch dimensions for cross-attention
                B_t, n_cam, n_patches, _ = rgb_tokens.shape
                rgb_flat = rgb_tokens.reshape(B_t, n_cam * n_patches, -1)
                depth_flat = depth_tokens.reshape(B_t, n_cam * n_patches, -1)
                
                rgb_flat, depth_flat = self.rgb_depth_cross_attn(rgb_flat, depth_flat)
                
                rgb_tokens = rgb_flat.reshape(B_t, n_cam, n_patches, -1)
                depth_tokens = depth_flat.reshape(B_t, n_cam, n_patches, -1)
            
            # 3. Encode state
            state_encoded = None
            if self.state_encoder is not None and state_t is not None:
                state_encoded = self.state_encoder(state_t)  # (B, fusion_dim)
            
            # 4. Fuse modalities with Perceiver
            fused_t = self.perceiver_fusion(
                rgb_tokens=rgb_tokens,
                depth_tokens=depth_tokens,
                state=state_encoded,
            )  # (B, n_query_tokens, fusion_dim)
            
            # Flatten query tokens
            fused_t_flat = fused_t.flatten(start_dim=1)  # (B, n_query_tokens * fusion_dim)
            fused_t_proj = self.temporal_proj(fused_t_flat)  # (B, temporal_dim)
            
            fused_features_list.append(fused_t_proj)
        
        # 5. Stack temporal features
        temporal_features = torch.stack(fused_features_list, dim=1)  # (B, n_obs_steps, temporal_dim)
        
        # 6. Temporal encoding
        context = self.temporal_transformer(temporal_features)  # (B, temporal_dim)
        
        # 7. Project to decoder dimension
        context_decoder = self.temporal_to_decoder(context)  # (B, decoder_dim)
        
        # 8. Decode actions
        actions = self.action_decoder(context_decoder)  # (B, chunk_size, action_dim)
        
        # 9. Estimate uncertainty (optional)
        uncertainty = None
        if self.uncertainty_head is not None:
            # Use decoder features for uncertainty estimation
            # We need to get the decoded features before action head
            # For simplicity, use the context
            decoder_features = self.action_decoder.transformer_decoder(
                tgt=self.action_decoder.action_queries.expand(B, -1, -1) + self.action_decoder.pos_enc,
                memory=context_decoder.unsqueeze(1),
            )
            uncertainty = self.uncertainty_head(decoder_features)  # (B, chunk_size, action_dim)
        
        return actions, uncertainty
