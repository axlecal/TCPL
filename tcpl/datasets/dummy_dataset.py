"""Dummy EEG dataset generator for minimal runnable TCPL demo."""

from typing import Dict, Hashable, List, Tuple

import numpy as np
import torch

from datasets.eeg_dataset import SubjectWiseEEGDataset


def _generate_class_pattern(seq_len: int, class_id: int) -> np.ndarray:
    """Create a class-dependent temporal base pattern, shape [T]."""
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    freq = float(class_id + 1)
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def generate_dummy_subject_data(
    subject_ids: List[Hashable],
    n_channels: int,
    seq_len: int,
    n_classes: int,
    trials_per_class_per_subject: int,
    seed: int = 42,
) -> Dict[Hashable, List[Dict[str, object]]]:
    """Generate synthetic subject-wise EEG trials.

    Each trial has shape [C, T] and class label in [0, n_classes-1].
    Data contains class and subject structure so metrics are meaningful.
    """
    rng = np.random.default_rng(seed)
    data: Dict[Hashable, List[Dict[str, object]]] = {}

    for sid in subject_ids:
        trials: List[Dict[str, object]] = []

        # Subject-specific scale/offset introduce cross-subject shift.
        subject_scale = rng.normal(loc=1.0, scale=0.1, size=(n_channels, 1)).astype(np.float32)
        subject_bias = rng.normal(loc=0.0, scale=0.05, size=(n_channels, 1)).astype(np.float32)

        for cls in range(n_classes):
            base = _generate_class_pattern(seq_len, cls)[None, :]  # [1, T]
            channel_profile = rng.normal(loc=1.0, scale=0.15, size=(n_channels, 1)).astype(np.float32)

            for _ in range(trials_per_class_per_subject):
                noise = rng.normal(loc=0.0, scale=0.25, size=(n_channels, seq_len)).astype(np.float32)
                x = subject_scale * (channel_profile * base) + subject_bias + noise
                trials.append({"x": torch.from_numpy(x), "y": int(cls)})

        rng.shuffle(trials)
        data[sid] = trials

    return data


def build_dummy_datasets_from_config(config: Dict) -> Tuple[SubjectWiseEEGDataset, SubjectWiseEEGDataset, SubjectWiseEEGDataset]:
    """Build train/val/test subject-split datasets from config."""
    data_cfg = config["data"]
    train_subjects = data_cfg["train_subjects"]
    val_subjects = data_cfg["val_subjects"]
    test_subjects = data_cfg["test_subjects"]

    all_subjects = sorted(set(train_subjects + val_subjects + test_subjects))
    all_data = generate_dummy_subject_data(
        subject_ids=all_subjects,
        n_channels=int(data_cfg["n_channels"]),
        seq_len=int(data_cfg["seq_len"]),
        n_classes=int(data_cfg["n_classes"]),
        trials_per_class_per_subject=int(data_cfg["trials_per_class_per_subject"]),
        seed=int(config["train"]["seed"]),
    )

    full_dataset = SubjectWiseEEGDataset(all_data)
    train_dataset = SubjectWiseEEGDataset.subset(full_dataset, train_subjects)
    val_dataset = SubjectWiseEEGDataset.subset(full_dataset, val_subjects)
    test_dataset = SubjectWiseEEGDataset.subset(full_dataset, test_subjects)

    return train_dataset, val_dataset, test_dataset
