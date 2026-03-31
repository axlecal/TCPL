"""Support encoder for dynamic subject prompt generation."""

import torch
import torch.nn as nn

from utils.init import init_conv_kaiming, init_linear_xavier


class SupportEncoder(nn.Module):
    """Encode support EEG trials into d-dimensional embeddings.

    Architecture:
    - Conv1d(C -> hidden, k=3, s=1, padding=same) + ReLU
    - Conv1d(hidden -> hidden, k=3, s=1, padding=same) + ReLU
    - Global average pooling over time
    - Linear(hidden -> d_model)
    """

    def __init__(self, in_channels: int, d_model: int = 64, hidden_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Linear(hidden_channels, d_model)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize layer weights."""
        init_conv_kaiming(self.conv1)
        init_conv_kaiming(self.conv2)
        init_linear_xavier(self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Support EEG batch, shape [B, C, T].

        Returns:
            Support embeddings, shape [B, d_model].
        """
        # [B, C, T] -> [B, hidden, T]
        h = self.act(self.conv1(x))
        # [B, hidden, T] -> [B, hidden, T]
        h = self.act(self.conv2(h))
        # Global average over time: [B, hidden]
        h = h.mean(dim=-1)
        # [B, hidden] -> [B, d_model]
        e = self.proj(h)
        return e
