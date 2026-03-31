"""Checkpoint IO helpers."""

import os
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], save_dir: str, filename: str = "last.pt", is_best: bool = False) -> str:
    """Save a training checkpoint.

    Args:
        state: Serialized state dict.
        save_dir: Directory for checkpoints.
        filename: Checkpoint filename.
        is_best: Whether this checkpoint is best on validation.

    Returns:
        Path to saved checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    torch.save(state, ckpt_path)

    if is_best:
        best_path = os.path.join(save_dir, "best.pt")
        torch.save(state, best_path)

    return ckpt_path


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load checkpoint from path.

    Args:
        path: Checkpoint file path.
        map_location: Torch map_location argument.

    Returns:
        Loaded checkpoint dictionary.
    """
    return torch.load(path, map_location=map_location)
