"""Meta-training entry script for TCPL."""

import argparse
import copy
import os
import sys
from statistics import mean, stdev
from typing import Dict, List

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
from utils.logger import get_logger
from utils.seed import set_seed


def load_config(path: str) -> Dict:
    """Load yaml config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_name: str) -> torch.device:
    """Resolve runtime device with graceful CUDA fallback."""
    if "cuda" in device_name and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_model_from_config(config: Dict) -> TCPLModel:
    """Instantiate TCPL model from config dict."""
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


def run_single_seed(config: Dict, seed: int, run_dir: str) -> Dict[str, float]:
    """Run full training/validation/testing for one seed."""
    os.makedirs(run_dir, exist_ok=True)
    config = copy.deepcopy(config)
    config["train"]["seed"] = int(seed)

    set_seed(seed)

    logger = get_logger(name=f"tcpl_seed_{seed}", save_dir=run_dir)
    logger.info("Running seed=%d", seed)

    with open(os.path.join(run_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    train_ds, val_ds, test_ds = build_dummy_datasets_from_config(config)

    ep_cfg = config["episode"]
    train_sampler = EpisodeSampler(
        dataset=train_ds,
        subject_ids=train_ds.get_subject_ids(),
        n_way=int(ep_cfg["n_way"]),
        k_shot=int(ep_cfg["k_shot"]),
        q_query=int(ep_cfg["q_query"]),
        seed=seed,
    )
    val_sampler = EpisodeSampler(
        dataset=val_ds,
        subject_ids=val_ds.get_subject_ids(),
        n_way=int(ep_cfg["n_way"]),
        k_shot=int(ep_cfg["k_shot"]),
        q_query=int(ep_cfg["q_query"]),
        seed=seed + 1,
    )
    test_sampler = EpisodeSampler(
        dataset=test_ds,
        subject_ids=test_ds.get_subject_ids(),
        n_way=int(ep_cfg["n_way"]),
        k_shot=int(ep_cfg["k_shot"]),
        q_query=int(ep_cfg["q_query"]),
        seed=seed + 2,
    )

    model = build_model_from_config(config)
    train_cfg = config["train"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    device = resolve_device(train_cfg["device"])

    trainer = MetaTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        prompt_l2=float(train_cfg["prompt_l2"]),
        grad_clip=float(train_cfg["grad_clip"]),
        logger=logger,
        save_dir=run_dir,
    )

    trainer.train(
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        epochs=int(train_cfg["epochs"]),
        meta_batch_size=int(ep_cfg["meta_batch_size"]),
        episodes_per_epoch=int(train_cfg["episodes_per_epoch"]),
        val_episodes=int(train_cfg["val_episodes"]),
        early_stopping_patience=train_cfg.get("early_stopping_patience", None),
        run_name=f"seed_{seed}",
    )

    best_path = os.path.join(run_dir, "best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Best checkpoint not found: {best_path}")

    checkpoint = load_checkpoint(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_trainer = MetaTrainer(
        model=model,
        optimizer=None,
        device=device,
        prompt_l2=float(train_cfg["prompt_l2"]),
        grad_clip=float(train_cfg["grad_clip"]),
        logger=logger,
        save_dir=run_dir,
    )
    test_metrics = test_trainer.evaluate(test_sampler, num_episodes=int(train_cfg["test_episodes"]))

    logger.info(
        "Test | acc=%.4f | f1=%.4f | kappa=%.4f | loss=%.4f",
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
        test_metrics["kappa"],
        test_metrics["loss"],
    )

    return test_metrics


def summarize_multi_seed(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate mean/std results over seeds."""
    keys = ["accuracy", "macro_f1", "kappa", "loss"]
    summary = {}
    for k in keys:
        values = [m[k] for m in metrics_list]
        summary[f"{k}_mean"] = mean(values)
        summary[f"{k}_std"] = stdev(values) if len(values) > 1 else 0.0
    return summary


def main() -> None:
    """CLI entry."""
    parser = argparse.ArgumentParser(description="Train TCPL with episodic meta-learning.")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    save_root = config["train"]["save_dir"]
    os.makedirs(save_root, exist_ok=True)

    seeds = config["train"].get("seeds", [config["train"]["seed"]])
    all_metrics = []

    for seed in seeds:
        run_dir = os.path.join(save_root, f"seed_{seed}")
        metrics = run_single_seed(config=config, seed=int(seed), run_dir=run_dir)
        all_metrics.append(metrics)

    summary = summarize_multi_seed(all_metrics)
    print("Multi-seed summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
