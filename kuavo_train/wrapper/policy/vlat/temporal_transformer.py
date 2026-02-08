"""
Temporal Transformer for modeling sequences of observations.

Processes multiple timesteps of fused multi-modal features to capture
temporal dependencies and dynamics.
"""

import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal positions.
    """
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, D) or (T, B, D)
        
        Returns:
            Positional encoding of same shape as input
        """
        if x.dim() == 3:
            if x.shape[0] > x.shape[1]:  # Assume (B, T, D)
                T = x.shape[1]
                return self.pe[:T].unsqueeze(0)  # (1, T, D)
            else:  # Assume (T, B, D)
                T = x.shape[0]
                return self.pe[:T].unsqueeze(1)  # (T, 1, D)
        else:
            raise ValueError(f"Expected 3D tensor, got shape {x.shape}")


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer encoder for temporal sequence modeling.
    
    Processes a sequence of observation features to capture temporal
    dependencies using self-attention.
    """
    
    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        """
        Args:
            d_model: Feature dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension of feedforward network
            dropout: Dropout rate
            causal: If True, use causal (autoregressive) attention mask
        """
        super().__init__()
        
        self.d_model = d_model
        self.causal = causal
        
        # Sinusoidal positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        
        # Learnable temporal embeddings (optional, added to sinusoidal)
        self.temporal_embed = nn.Parameter(
            torch.randn(1, 100, d_model) / math.sqrt(d_model)
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()
        return mask
    
    def forward(
        self, 
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through temporal transformer.
        
        Args:
            x: (B, T, D) sequence of features
            src_key_padding_mask: (B, T) True for padding positions
        
        Returns:
            (B, T, D) temporally encoded features
        """
        B, T, D = x.shape
        
        # Add positional encoding
        pos_enc = self.pos_encoder(x)  # (1, T, D)
        temporal_emb = self.temporal_embed[:, :T, :]  # (1, T, D)
        x = x + pos_enc + temporal_emb
        
        # Create causal mask if needed
        if self.causal:
            attn_mask = self._generate_causal_mask(T, x.device)
        else:
            attn_mask = None
        
        # Apply transformer encoder
        x = self.transformer_encoder(
            x,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        
        # Final normalization
        x = self.norm(x)
        
        return x


class TemporalFeatureAggregator(nn.Module):
    """
    Aggregates temporal features into a single representation.
    
    Can use different aggregation strategies:
    - 'last': Use only the last timestep
    - 'mean': Average over all timesteps
    - 'attention': Learnable attention-based pooling
    """
    
    def __init__(
        self,
        d_model: int,
        aggregation: str = 'attention',
        num_heads: int = 8,
    ):
        """
        Args:
            d_model: Feature dimension
            aggregation: Aggregation strategy ('last', 'mean', 'attention')
            num_heads: Number of attention heads (for 'attention' mode)
        """
        super().__init__()
        
        self.aggregation = aggregation
        self.d_model = d_model
        
        if aggregation == 'attention':
            # Learnable query for attention pooling
            self.query = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
            
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                batch_first=True,
            )
            
            self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Aggregate temporal features.
        
        Args:
            x: (B, T, D) temporal features
            key_padding_mask: (B, T) True for padding positions
        
        Returns:
            (B, D) aggregated features
        """
        B, T, D = x.shape
        
        if self.aggregation == 'last':
            # Use last timestep
            return x[:, -1, :]  # (B, D)
        
        elif self.aggregation == 'mean':
            # Average over time
            if key_padding_mask is not None:
                # Mask out padding
                mask = ~key_padding_mask  # (B, T)
                x_masked = x * mask.unsqueeze(-1)
                return x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                return x.mean(dim=1)  # (B, D)
        
        elif self.aggregation == 'attention':
            # Attention-based pooling
            query = self.query.expand(B, -1, -1)  # (B, 1, D)
            
            pooled, _ = self.attention(
                query=query,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask,
            )
            
            pooled = self.norm(pooled.squeeze(1))  # (B, D)
            return pooled
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")


class TemporalTransformer(nn.Module):
    """
    Complete temporal transformer module.
    
    Combines temporal encoding and aggregation for processing
    sequences of multi-modal observations.
    """
    
    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        causal: bool = False,
        aggregation: str = 'attention',
    ):
        """
        Args:
            d_model: Feature dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension of feedforward
            dropout: Dropout rate
            causal: Use causal attention
            aggregation: How to aggregate temporal features
        """
        super().__init__()
        
        self.encoder = TemporalTransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            causal=causal,
        )
        
        self.aggregator = TemporalFeatureAggregator(
            d_model=d_model,
            aggregation=aggregation,
            num_heads=num_heads,
        )
    
    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_sequence: bool = False,
    ) -> Tensor:
        """
        Process temporal sequence.
        
        Args:
            x: (B, T, D) temporal features
            key_padding_mask: (B, T) padding mask
            return_sequence: If True, return full sequence; else aggregate
        
        Returns:
            If return_sequence=True: (B, T, D)
            If return_sequence=False: (B, D)
        """
        # Encode temporal dependencies
        x_encoded = self.encoder(x, key_padding_mask)
        
        if return_sequence:
            return x_encoded
        else:
            # Aggregate to single representation
            return self.aggregator(x_encoded, key_padding_mask)
