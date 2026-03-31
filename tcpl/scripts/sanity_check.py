"""Minimal runnable sanity check for the full TCPL pipeline."""

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
from utils.metrics import compute_metrics
from utils.seed import set_seed


def load_config(path: str) -> Dict:
    """Load YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_name: str) -> torch.device:
    """Resolve runtime device."""
    if "cuda" in device_name and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_model(config: Dict) -> TCPLModel:
    """Construct TCPL model from config."""
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
    """Run one train step + one val step + one unseen subject inference."""
    parser = argparse.ArgumentParser(description="Sanity check TCPL end-to-end pipeline.")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["train"]["seed"]))
    device = resolve_device(config["train"]["device"])

    train_ds, val_ds, test_ds = build_dummy_datasets_from_config(config)

    ep_cfg = config["episode"]
    train_sampler = EpisodeSampler(
        dataset=train_ds,
        subject_ids=train_ds.get_subject_ids(),
        n_way=int(ep_cfg["n_way"]),
        k_shot=int(ep_cfg["k_shot"]),
        q_query=int(ep_cfg["q_query"]),
        seed=int(config["train"]["seed"]),
    )
    val_sampler = EpisodeSampler(
        dataset=val_ds,
        subject_ids=val_ds.get_subject_ids(),
        n_way=int(ep_cfg["n_way"]),
        k_shot=int(ep_cfg["k_shot"]),
        q_query=int(ep_cfg["q_query"]),
        seed=int(config["train"]["seed"]) + 1,
    )
    test_sampler = EpisodeSampler(
        dataset=test_ds,
        subject_ids=test_ds.get_subject_ids(),
        n_way=int(ep_cfg["n_way"]),
        k_shot=int(ep_cfg["k_shot"]),
        q_query=int(ep_cfg["q_query"]),
        seed=int(config["train"]["seed"]) + 2,
    )

    model = build_model(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    trainer = MetaTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        prompt_l2=float(config["train"]["prompt_l2"]),
        grad_clip=float(config["train"]["grad_clip"]),
        logger=None,
        save_dir=None,
    )

    # 1) One meta-train step.
    train_step_stats = trainer.train_step(train_sampler, meta_batch_size=1)
    print("[Sanity] one meta-train step")
    print(f"  loss      : {train_step_stats['loss']:.4f}")
    print(f"  ce_loss   : {train_step_stats['ce_loss']:.4f}")
    print(f"  prompt_reg: {train_step_stats['prompt_reg']:.4f}")

    # 2) One validation episode.
    val_metrics = trainer.evaluate(val_sampler, num_episodes=1)
    print("[Sanity] one validation episode")
    print(f"  accuracy  : {val_metrics['accuracy']:.4f}")
    print(f"  macro_f1  : {val_metrics['macro_f1']:.4f}")
    print(f"  kappa     : {val_metrics['kappa']:.4f}")
    print(f"  loss      : {val_metrics['loss']:.4f}")

    # 3) One unseen subject inference episode.
    model.eval()
    unseen_ep = test_sampler.sample_episode()
    support_x = unseen_ep["support_x"].to(device)
    support_y = unseen_ep["support_y"].to(device)
    query_x = unseen_ep["query_x"].to(device)
    query_y = unseen_ep["query_y"].to(device)

    with torch.no_grad():
        prompt = model.build_subject_prompt(support_x, support_y)
        logits = model.predict_with_prompt(query_x, prompt)
        preds = logits.argmax(dim=1)

    infer_metrics = compute_metrics(
        query_y.detach().cpu().numpy(),
        preds.detach().cpu().numpy(),
    )

    print("[Sanity] unseen subject inference")
    print(f"  subject_id: {unseen_ep['subject_id']}")
    print(f"  support_x : {tuple(unseen_ep['support_x'].shape)}")
    print(f"  query_x   : {tuple(unseen_ep['query_x'].shape)}")
    print(f"  prompt    : {tuple(prompt.shape)}")
    print(f"  accuracy  : {infer_metrics['accuracy']:.4f}")
    print(f"  macro_f1  : {infer_metrics['macro_f1']:.4f}")
    print(f"  kappa     : {infer_metrics['kappa']:.4f}")

    print("\nSanity check passed: full TCPL pipeline runs end-to-end.")


if __name__ == "__main__":
    main()
