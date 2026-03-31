"""Transformer encoder used after prompt-prefix concatenation."""

import torch
import torch.nn as nn

from utils.init import init_linear_xavier


class TransformerBlock(nn.Module):
    """A clear Transformer encoder block with post-residual LayerNorm.

    Input/Output shape:
        [B, L, d_model]
    """

    def __init__(self, d_model: int = 64, n_heads: int = 8, ffn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize MHSA and FFN parameters."""
        # Q/K/V are projected by the in_proj matrix inside nn.MultiheadAttention.
        nn.init.xavier_uniform_(self.mhsa.in_proj_weight)
        if self.mhsa.in_proj_bias is not None:
            nn.init.zeros_(self.mhsa.in_proj_bias)

        init_linear_xavier(self.mhsa.out_proj)

        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                init_linear_xavier(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Sequence tokens, shape [B, L, d_model].

        Returns:
            Updated sequence tokens, shape [B, L, d_model].
        """
        # Self-attention with Q=K=V=x.
        attn_out, _ = self.mhsa(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder blocks.

    Input:
        z: [B, L, d_model]
    Output:
        z_out: [B, L, d_model]
    """

    def __init__(self, n_layers: int = 4, d_model: int = 64, n_heads: int = 8, ffn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Input tokens (prompt + EEG tokens), shape [B, L, d_model].

        Returns:
            Encoded tokens, shape [B, L, d_model].
        """
        for layer in self.layers:
            z = layer(z)
        return z
