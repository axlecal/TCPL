"""Few-shot inference script for one unseen subject."""

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
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_metrics


def load_config(path: str) -> Dict:
    """Load YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_name: str) -> torch.device:
    """Resolve runtime device with fallback to CPU."""
    if "cuda" in device_name and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_model(config: Dict) -> TCPLModel:
    """Instantiate TCPL from config."""
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
    """CLI entry for unseen-subject few-shot inference."""
    parser = argparse.ArgumentParser(description="Infer one unseen subject with support-conditioned prompt.")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--subject_id", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["train"]["device"])

    _, _, test_ds = build_dummy_datasets_from_config(config)
    test_subject_ids = test_ds.get_subject_ids()
    if len(test_subject_ids) == 0:
        raise ValueError("No test subjects available.")

    target_subject = args.subject_id if args.subject_id is not None else int(test_subject_ids[0])

    ep_cfg = config["episode"]
    sampler = EpisodeSampler(
        dataset=test_ds,
        subject_ids=test_subject_ids,
        n_way=int(ep_cfg["n_way"]),
        k_shot=int(ep_cfg["k_shot"]),
        q_query=int(ep_cfg["q_query"]),
        seed=int(config["train"]["seed"]),
    )

    episode = sampler.sample_episode(subject_id=target_subject)

    model = build_model(config)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    support_x = episode["support_x"].to(device)
    support_y = episode["support_y"].to(device)
    query_x = episode["query_x"].to(device)
    query_y = episode["query_y"].to(device)

    with torch.no_grad():
        # 1) Build prompt from support set only.
        prompt = model.build_subject_prompt(support_x, support_y)
        # 2) Predict query with fixed prompt (no fine-tuning).
        logits = model.predict_with_prompt(query_x, prompt)
        preds = logits.argmax(dim=1)

    metrics = compute_metrics(
        query_y.detach().cpu().numpy(),
        preds.detach().cpu().numpy(),
    )

    print(f"Unseen subject inference | subject_id={target_subject}")
    print(f"support_x: {tuple(episode['support_x'].shape)}")
    print(f"query_x  : {tuple(episode['query_x'].shape)}")
    print(f"prompt   : {tuple(prompt.shape)}")
    print(f"accuracy : {metrics['accuracy']:.4f}")
    print(f"macro_f1 : {metrics['macro_f1']:.4f}")
    print(f"kappa    : {metrics['kappa']:.4f}")


if __name__ == "__main__":
    main()
