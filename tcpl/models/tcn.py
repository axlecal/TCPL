"""Temporal Convolutional Network (TCN) backbone for raw EEG."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init import init_conv_kaiming


class CausalConv1d(nn.Module):
    """1D causal convolution with left padding only.

    For kernel size K and dilation D, left padding is (K - 1) * D,
    ensuring output at time t depends only on inputs <= t.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
        )
        init_conv_kaiming(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape [B, C_in, T].

        Returns:
            Output tensor, shape [B, C_out, T].
        """
        # Left-only padding keeps causality and preserves temporal length.
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class ResidualTCNBlock(nn.Module):
    """Residual TCN block with two causal Conv1d layers."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.res_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            init_conv_kaiming(self.res_proj)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape [B, C_in, T].

        Returns:
            Output tensor, shape [B, C_out, T].
        """
        residual = self.res_proj(x)

        # [B, C_in, T] -> [B, C_out, T]
        out = self.conv1(x)
        out = self.act(out)
        out = self.dropout(out)

        # [B, C_out, T] -> [B, C_out, T]
        out = self.conv2(out)
        out = self.act(out)
        out = self.dropout(out)

        # Residual connection keeps same temporal length T.
        out = out + residual
        out = self.act(out)
        return out


class TemporalConvNet(nn.Module):
    """4-block residual TCN for EEG temporal feature extraction.

    Input:
        x: [B, C, T]
    Output:
        h: [B, T, d_model]
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        dilations: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        d_model: int = 64,
    ):
        super().__init__()
        if len(channels) != 4 or len(dilations) != 4:
            raise ValueError("TCN must use exactly 4 blocks to match the required setup.")

        blocks = []
        prev_ch = in_channels
        for ch, dil in zip(channels, dilations):
            blocks.append(
                ResidualTCNBlock(
                    in_channels=prev_ch,
                    out_channels=ch,
                    kernel_size=kernel_size,
                    dilation=dil,
                    dropout=dropout,
                )
            )
            prev_ch = ch

        self.blocks = nn.ModuleList(blocks)
        self.out_proj = nn.Conv1d(prev_ch, d_model, kernel_size=1)
        init_conv_kaiming(self.out_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Raw EEG trials, shape [B, C, T].

        Returns:
            Temporal features, shape [B, T', d_model]. Here T' = T.
        """
        # [B, C, T] -> [B, ch, T]
        h = x
        for block in self.blocks:
            h = block(h)

        # Project channel dim to d_model: [B, ch_last, T] -> [B, d_model, T]
        h = self.out_proj(h)
        # Transformer expects [B, L, d]. Here L = T.
        h = h.transpose(1, 2).contiguous()  # [B, T, d_model]
        return h
