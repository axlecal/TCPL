"""Full TCPL model: support encoder + prompt generator + TCN + Transformer + classifier."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classifier import EEGTokenClassifier
from models.prompt_generator import PromptGenerator
from models.support_encoder import SupportEncoder
from models.tcn import TemporalConvNet
from models.transformer import TransformerEncoder


class TCPLModel(nn.Module):
    """TCPL for few-shot cross-subject motor imagery EEG decoding.

    Pipeline per episode:
    1) support -> support encoder -> mean pooling -> prompt generator -> prompt tokens
    2) query -> TCN -> EEG tokens
    3) concat [prompt; eeg_tokens] -> Transformer
    4) drop prompt tokens -> mean over EEG tokens -> linear classifier
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        d_model: int = 64,
        n_prompts: int = 10,
        tcn_channels=None,
        tcn_dilations=None,
        tcn_kernel_size: int = 3,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [64, 64, 128, 128]
        if tcn_dilations is None:
            tcn_dilations = [1, 2, 4, 8]

        self.d_model = d_model
        self.n_prompts = n_prompts

        self.support_encoder = SupportEncoder(in_channels=n_channels, d_model=d_model, hidden_channels=64)
        self.prompt_generator = PromptGenerator(d_model=d_model, n_prompts=n_prompts)

        self.tcn = TemporalConvNet(
            in_channels=n_channels,
            channels=tcn_channels,
            dilations=tcn_dilations,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
            d_model=d_model,
        )
        self.transformer = TransformerEncoder(
            n_layers=transformer_layers,
            d_model=d_model,
            n_heads=transformer_heads,
            ffn_dim=transformer_ffn_dim,
            dropout=dropout,
        )
        self.classifier = EEGTokenClassifier(d_model=d_model, n_classes=n_classes)

    def build_subject_prompt(self, support_x: torch.Tensor, support_y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Build subject-conditioned prompt tokens from support set.

        Args:
            support_x: Support EEG trials, shape [N_support, C, T].
            support_y: Optional support labels, shape [N_support]. Unused currently.

        Returns:
            prompt: Subject prompt tokens, shape [1, k, d_model].
        """
        del support_y  # Reserved for future extensions.

        # Encode each support trial: [N_support, C, T] -> [N_support, d_model]
        support_emb = self.support_encoder(support_x)
        # Subject summary pooling over support samples: [1, d_model]
        h_s = support_emb.mean(dim=0, keepdim=True)
        # Generate k prompt tokens: [1, k, d_model]
        prompt = self.prompt_generator(h_s)
        return prompt

    def predict_with_prompt(self, query_x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """Predict query labels using a provided subject prompt.

        Args:
            query_x: Query EEG trials, shape [B, C, T].
            prompt: Prompt tokens, shape [1, k, d_model] or [B, k, d_model].

        Returns:
            logits: Class logits, shape [B, n_classes].
        """
        # Raw EEG -> TCN temporal features: [B, C, T] -> [B, T', d_model]
        eeg_tokens = self.tcn(query_x)

        if prompt.ndim != 3:
            raise ValueError(f"Prompt must be [B or 1, k, d], got shape {tuple(prompt.shape)}")

        if prompt.size(0) == 1:
            # Broadcast one subject prompt to all query trials in this episode.
            prompt = prompt.expand(query_x.size(0), -1, -1)

        # Prefix-token injection before Transformer: Z = [P_s; H]
        # [B, k, d] + [B, T', d] -> [B, k + T', d]
        z = torch.cat([prompt, eeg_tokens], dim=1)

        # Global token interaction via self-attention.
        z_out = self.transformer(z)

        # Fixed classifier rule: drop prompt tokens, mean pool EEG tokens, linear head.
        logits = self.classifier(z_out, n_prompt_tokens=self.n_prompts)
        return logits

    def infer_subject(self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        """Few-shot inference for an unseen subject (no model fine-tuning).

        Args:
            support_x: [N_support, C, T]
            support_y: [N_support]
            query_x: [N_query, C, T]

        Returns:
            logits: [N_query, n_classes]
        """
        prompt = self.build_subject_prompt(support_x, support_y)
        logits = self.predict_with_prompt(query_x, prompt)
        return logits

    def forward_episode(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        prompt_l2_lambda: float = 1e-4,
    ) -> Dict[str, torch.Tensor]:
        """Forward one meta-learning episode and compute loss.

        Args:
            support_x: [N_support, C, T]
            support_y: [N_support]
            query_x: [N_query, C, T]
            query_y: [N_query]
            prompt_l2_lambda: Prompt regularization coefficient.

        Returns:
            Dictionary with total loss, ce loss, prompt regularization, logits.
        """
        prompt = self.build_subject_prompt(support_x, support_y)
        logits = self.predict_with_prompt(query_x, prompt)

        ce_loss = F.cross_entropy(logits, query_y)
        # ||P_s||_2^2 regularization
        prompt_reg = torch.norm(prompt, p=2) ** 2
        total_loss = ce_loss + prompt_l2_lambda * prompt_reg

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "prompt_reg": prompt_reg,
            "logits": logits,
        }
