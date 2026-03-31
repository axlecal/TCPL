"""Task-conditioned prompt generator from subject support summary."""

import torch
import torch.nn as nn

from utils.init import init_linear_xavier, init_prompt_output_small


class PromptGenerator(nn.Module):
    """Generate prompt tokens from one subject summary embedding.

    Input:
        h_s: [B, d_model] or [d_model]
    Output:
        prompts: [B, n_prompts, d_model]
    """

    def __init__(self, d_model: int = 64, n_prompts: int = 10):
        super().__init__()
        self.d_model = d_model
        self.n_prompts = n_prompts

        self.fc1 = nn.Linear(d_model, d_model)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(d_model, n_prompts * d_model)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize MLP weights.

        Final layer uses smaller initialization scale to keep prompt output small at start.
        """
        init_linear_xavier(self.fc1)
        init_prompt_output_small(self.fc2, scale=0.1)

    def forward(self, h_s: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            h_s: Subject summary embedding, shape [B, d_model] or [d_model].

        Returns:
            Prompt tokens, shape [B, n_prompts, d_model].
        """
        if h_s.ndim == 1:
            h_s = h_s.unsqueeze(0)

        # [B, d] -> [B, d]
        x = self.act(self.fc1(h_s))
        # [B, d] -> [B, n_prompts * d]
        x = self.fc2(x)
        # [B, n_prompts * d] -> [B, n_prompts, d]
        prompts = x.view(x.size(0), self.n_prompts, self.d_model)
        return prompts
