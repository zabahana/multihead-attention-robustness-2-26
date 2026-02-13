"""
Feature Token Transformer for Cross-Sectional Asset Pricing

This module implements a transformer model where each feature is treated as a token,
with optional head-diversity regularization for improved robustness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict


class ElementWiseLinear(nn.Module):
    """
    Custom layer that performs element-wise linear transformation.
    Matches output_projection.0 structure: weight [d_model] and bias [d_model]
    This is used to match the checkpoint structure where output_projection.0
    has 1D weight and bias tensors instead of a standard 2D Linear layer.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, d_model)
        # Element-wise: x * weight + bias
        return x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention."""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights


class FeatureTokenTransformer(nn.Module):
    """
    Feature Token Transformer for cross-sectional asset pricing.
    
    Each feature is treated as a token, and multi-head attention is applied
    to learn relationships between features.
    """
    
    def __init__(self,
                 num_features: int,
                 d_model: int = 64,  # Changed default to match checkpoint
                 num_heads: int = 4,
                 num_layers: int = 2,
                 d_ff: int = 512,  # Changed default to match checkpoint
                 dropout: float = 0.1,
                 use_head_diversity: bool = False,
                 diversity_weight: Optional[float] = None):
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_head_diversity = use_head_diversity
        self.diversity_weight = diversity_weight if diversity_weight is not None else 0.01
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, num_features, d_model))
        
        # Input projection: project each feature to d_model
        self.input_projection = nn.Linear(1, d_model)
        
        # Multi-head attention layers with layer norms
        # Checkpoint structure: layers.0.w_q, layers.0.w_k, etc. (not nested under 'attention')
        # So we need to flatten the structure to match checkpoint
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'w_q': nn.Linear(d_model, d_model),
                'w_k': nn.Linear(d_model, d_model),
                'w_v': nn.Linear(d_model, d_model),
                'w_o': nn.Linear(d_model, d_model),
                'layer_norm': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(num_layers)
        ])
        
        # Feed-forward layers (separate from attention)
        self.ff_layers = nn.ModuleList([
            nn.ModuleDict({
                'linear1': nn.Linear(d_model, d_ff),
                'linear2': nn.Linear(d_ff, d_model),
                'layer_norm': nn.LayerNorm(d_model)
            })
            for _ in range(num_layers)
        ])
        
        # Output projection (matches saved model structure exactly)
        # Checkpoint structure from inspection:
        # - output_projection.0.weight: [64] - ElementWiseLinear layer
        # - output_projection.0.bias: [64]
        # - output_projection.1.weight: [512, 64] - Linear(64, 512)
        # - output_projection.1.bias: [512]
        # - output_projection.4.weight: [1, 512] - Linear(512, 1)
        # - output_projection.4.bias: [1]
        # Sequential structure: ElementWiseLinear(0), Linear(1), ReLU(2), Dropout(3), Linear(4)
        self.output_projection = nn.Sequential(
            ElementWiseLinear(d_model),  # output_projection.0: element-wise transform
            nn.Linear(d_model, d_ff),    # output_projection.1: (64, 512)
            nn.ReLU(),                   # output_projection.2: (not saved, but exists)
            nn.Dropout(dropout),         # output_projection.3: (not saved, but exists)
            nn.Linear(d_ff, 1)           # output_projection.4: (512, 1)
        )
        
    def compute_diversity_loss(self, attention_weights_list):
        """Compute head diversity loss from attention weights."""
        if not self.use_head_diversity or len(attention_weights_list) == 0:
            return torch.tensor(0.0, device=attention_weights_list[0].device)
        
        # Flatten attention weights for each head
        diversity_loss = 0.0
        count = 0
        
        for attn_weights in attention_weights_list:
            # attn_weights shape: (batch, num_heads, seq_len, seq_len)
            if attn_weights.dim() == 4:
                batch_size, num_heads, seq_len, _ = attn_weights.shape
                # Average over sequence dimension
                head_representations = attn_weights.mean(dim=(2, 3))  # (batch, num_heads)
                
                # Compute pairwise cosine similarity
                for i in range(num_heads):
                    for j in range(i + 1, num_heads):
                        head_i = head_representations[:, i]
                        head_j = head_representations[:, j]
                        # Cosine similarity
                        cos_sim = F.cosine_similarity(head_i.unsqueeze(0), head_j.unsqueeze(0))
                        diversity_loss += cos_sim.mean()
                        count += 1
        
        if count > 0:
            diversity_loss = -self.diversity_weight * (diversity_loss / count)
        
        return diversity_loss
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
        
        Returns:
            Tuple of (predictions, attention_weights_dict)
        """
        batch_size = x.size(0)
        
        # Input embedding
        # x shape: (batch_size, num_features)
        # Treat each feature as a token: (batch_size, num_features, 1) -> (batch_size, num_features, d_model)
        x = x.unsqueeze(-1)  # (batch_size, num_features, 1)
        x = self.input_projection(x)  # (batch_size, num_features, d_model)
        
        # Add positional encoding - handle dimension mismatch
        # If pos_encoding has different num_features, adjust it
        if x.size(1) != self.pos_encoding.size(1):
            # Interpolate or truncate positional encoding to match actual input size
            actual_num_features = x.size(1)
            if actual_num_features < self.pos_encoding.size(1):
                pos_enc = self.pos_encoding[:, :actual_num_features, :]
            else:
                # Pad with zeros if needed
                padding = torch.zeros(1, actual_num_features - self.pos_encoding.size(1), 
                                    self.pos_encoding.size(2), device=x.device)
                pos_enc = torch.cat([self.pos_encoding, padding], dim=1)
            x = x + pos_enc
        else:
            x = x + self.pos_encoding  # (batch_size, num_features, d_model)
        
        # Store attention weights for diversity loss
        attention_weights_list = []
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            # Self-attention with residual connection
            # Manual attention computation to match checkpoint structure
            batch_size = x.size(0)
            num_heads = self.num_heads
            d_k = self.d_k
            
            # Linear transformations
            Q = layer['w_q'](x).view(batch_size, -1, num_heads, d_k).transpose(1, 2)
            K = layer['w_k'](x).view(batch_size, -1, num_heads, d_k).transpose(1, 2)
            V = layer['w_v'](x).view(batch_size, -1, num_heads, d_k).transpose(1, 2)
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = layer['dropout'](attn_weights)
            attn_output = torch.matmul(attn_weights, V)
            
            # Concatenate heads
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, -1, self.d_model
            )
            
            # Final linear transformation
            attn_output = layer['w_o'](attn_output)
            
            # Residual connection and layer norm
            x = layer['layer_norm'](x + attn_output)
            attention_weights_list.append(attn_weights)
            
            # Feed-forward with residual connection
            ff_layer = self.ff_layers[i]
            ff_output = F.relu(ff_layer['linear1'](x))
            ff_output = ff_layer['linear2'](ff_output)
            x = ff_layer['layer_norm'](x + ff_output)
        
        # Global average pooling over feature dimension
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Check for NaN/Inf before output projection
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            # Return zeros if we have NaN/Inf
            predictions = torch.zeros(batch_size, 1, device=x.device)
            attn_dict = {}
            for i, attn_weights in enumerate(attention_weights_list):
                attn_dict[f'layer_{i}'] = attn_weights
            return predictions, attn_dict
        
        # Output projection (matches checkpoint structure)
        # Sequential handles: ElementWiseLinear(0), Linear(1), ReLU(2), Dropout(3), Linear(4)
        predictions = self.output_projection(x)  # (batch_size, 1)
        
        # Check for NaN/Inf in predictions
        if torch.any(torch.isnan(predictions)) or torch.any(torch.isinf(predictions)):
            # Replace NaN with zeros
            predictions = torch.where(
                torch.isnan(predictions) | torch.isinf(predictions),
                torch.zeros_like(predictions),
                predictions
            )
        
        # Prepare attention weights dictionary
        attn_dict = {}
        for i, attn_weights in enumerate(attention_weights_list):
            attn_dict[f'layer_{i}'] = attn_weights
        
        return predictions, attn_dict


class SingleHeadTransformer(nn.Module):
    """Single-head transformer (baseline for comparison)."""
    
    def __init__(self, num_features: int, d_model: int = 128, num_layers: int = 2):
        super().__init__()
        # Use FeatureTokenTransformer with num_heads=1
        self.model = FeatureTokenTransformer(
            num_features=num_features,
            d_model=d_model,
            num_heads=1,
            num_layers=num_layers,
            use_head_diversity=False
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.model(x)
