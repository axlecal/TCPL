"""Classification head for TCPL.

Fixed rule:
1) drop prompt tokens
2) mean-pool EEG tokens
3) linear classifier
"""

import torch
import torch.nn as nn

from utils.init import init_linear_xavier


class EEGTokenClassifier(nn.Module):
    """Classifier operating only on EEG tokens after Transformer."""

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize classifier weights."""
        init_linear_xavier(self.fc)

    def forward(self, z_out: torch.Tensor, n_prompt_tokens: int) -> torch.Tensor:
        """Forward pass.

        Args:
            z_out: Transformer output, shape [B, k + T', d_model].
            n_prompt_tokens: Number of prompt tokens k.

        Returns:
            logits: Class logits, shape [B, n_classes].
        """
        # Remove prefix prompt tokens: [B, T', d_model]
        z_eeg = z_out[:, n_prompt_tokens:, :]
        # Mean pool EEG tokens only: [B, d_model]
        pooled = z_eeg.mean(dim=1)
        # [B, d_model] -> [B, n_classes]
        logits = self.fc(pooled)
        return logits
