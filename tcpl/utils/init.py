"""Weight initialization utilities."""

import torch.nn as nn


def init_linear_xavier(linear: nn.Linear, gain: float = 1.0) -> None:
    """Initialize a linear layer with Xavier uniform.

    Args:
        linear: Target linear layer.
        gain: Xavier gain.
    """
    nn.init.xavier_uniform_(linear.weight, gain=gain)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def init_conv_kaiming(conv: nn.Conv1d) -> None:
    """Initialize Conv1d using Kaiming normal for ReLU blocks."""
    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def init_prompt_output_small(linear: nn.Linear, scale: float = 0.1) -> None:
    """Initialize prompt output layer with a smaller Xavier scale.

    Args:
        linear: Final output layer of prompt generator.
        scale: Multiplicative scale on Xavier gain.
    """
    nn.init.xavier_uniform_(linear.weight, gain=scale)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def initialize_module(module: nn.Module) -> None:
    """Apply default project-wide initialization recursively.

    Rules:
    - Linear: Xavier uniform
    - Conv1d: Kaiming normal
    - LayerNorm: keep PyTorch defaults
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init_linear_xavier(m)
        elif isinstance(m, nn.Conv1d):
            init_conv_kaiming(m)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a module."""
    params = module.parameters()
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return sum(p.numel() for p in params)
