"""Evaluation script for a trained TCPL checkpoint."""

import argparse
import os
import sys
from typing import Dict

import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.dummy_dataset import build_dummy_datasets_from_config
from datasets.episode_sampler import EpisodeSampler
from models.tcpl_model import TCPLModel
from trainers.meta_trainer import MetaTrainer
from utils.checkpoint import load_checkpoint


def load_config(path: str) -> Dict:
    """Load YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_name: str) -> torch.device:
    """Resolve device with CUDA fallback."""
    if "cuda" in device_name and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_model(config: Dict) -> TCPLModel:
    """Build model from config."""
    return TCPLModel(
        n_channels=int(config["data"]["n_channels"]),
        n_classes=int(config["data"]["n_classes"]),
        d_model=int(config["model"]["d_model"]),
        n_prompts=int(config["model"]["n_prompts"]),
        tcn_channels=list(config["model"]["tcn_channels"]),
        tcn_dilations=list(config["model"]["tcn_dilations"]),
        tcn_kernel_size=int(config["model"]["tcn_kernel_size"]),
        transformer_layers=int(config["model"]["transformer_layers"]),
        transformer_heads=int(config["model"]["transformer_heads"]),
        transformer_ffn_dim=int(config["model"]["transformer_ffn_dim"]),
        dropout=float(config["model"]["dropout"]),
    )


def main() -> None:
    """CLI entry."""
    parser = argparse.ArgumentParser(description="Test TCPL checkpoint.")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["train"]
    device = resolve_device(train_cfg["device"])

    _, val_ds, test_ds = build_dummy_datasets_from_config(config)
    target_ds = val_ds if args.split == "val" else test_ds

    ep_cfg = config["episode"]
    sampler = EpisodeSampler(
        dataset=target_ds,
        subject_ids=target_ds.get_subject_ids(),
        n_way=int(ep_cfg["n_way"]),
        k_shot=int(ep_cfg["k_shot"]),
        q_query=int(ep_cfg["q_query"]),
        seed=int(train_cfg["seed"]),
    )

    model = build_model(config)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    trainer = MetaTrainer(
        model=model,
        optimizer=None,
        device=device,
        prompt_l2=float(train_cfg["prompt_l2"]),
        grad_clip=float(train_cfg["grad_clip"]),
        logger=None,
        save_dir=None,
    )

    num_episodes = args.episodes if args.episodes is not None else int(train_cfg["test_episodes"])
    metrics = trainer.evaluate(sampler, num_episodes=num_episodes)

    print(f"[{args.split}] episodes={num_episodes}")
    print(f"accuracy : {metrics['accuracy']:.4f}")
    print(f"macro_f1 : {metrics['macro_f1']:.4f}")
    print(f"kappa    : {metrics['kappa']:.4f}")
    print(f"loss     : {metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
