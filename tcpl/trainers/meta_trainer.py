"""Episodic meta-training loop for TCPL."""

import os
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from datasets.episode_sampler import EpisodeSampler
from utils.checkpoint import save_checkpoint
from utils.metrics import compute_metrics


class MetaTrainer:
    """Meta-trainer for episode-based joint optimization (non-MAML)."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        device: torch.device,
        prompt_l2: float = 1e-4,
        grad_clip: float = 1.0,
        logger=None,
        save_dir: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.prompt_l2 = prompt_l2
        self.grad_clip = grad_clip
        self.logger = logger
        self.save_dir = save_dir

        self.model.to(self.device)

    def _to_device(self, episode: Dict[str, object]) -> Dict[str, object]:
        """Move episode tensors to target device."""
        return {
            "support_x": episode["support_x"].to(self.device),
            "support_y": episode["support_y"].to(self.device),
            "query_x": episode["query_x"].to(self.device),
            "query_y": episode["query_y"].to(self.device),
            "subject_id": episode["subject_id"],
        }

    def train_step(self, train_sampler: EpisodeSampler, meta_batch_size: int) -> Dict[str, float]:
        """Run one meta-training optimization step.

        Args:
            train_sampler: Episode sampler over training subjects.
            meta_batch_size: Number of episodes per optimizer step.

        Returns:
            Dict with averaged losses.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer is required for train_step.")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        episodes = train_sampler.sample_meta_batch(meta_batch_size)

        loss_sum = 0.0
        ce_sum = 0.0
        reg_sum = 0.0

        for episode in episodes:
            ep = self._to_device(episode)
            out = self.model.forward_episode(
                support_x=ep["support_x"],
                support_y=ep["support_y"],
                query_x=ep["query_x"],
                query_y=ep["query_y"],
                prompt_l2_lambda=self.prompt_l2,
            )
            loss_sum = loss_sum + out["loss"]
            ce_sum += float(out["ce_loss"].item())
            reg_sum += float(out["prompt_reg"].item())

        loss_avg = loss_sum / meta_batch_size
        loss_avg.backward()

        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        return {
            "loss": float(loss_avg.item()),
            "ce_loss": ce_sum / meta_batch_size,
            "prompt_reg": reg_sum / meta_batch_size,
        }

    def train_epoch(self, train_sampler: EpisodeSampler, meta_batch_size: int, episodes_per_epoch: int) -> Dict[str, float]:
        """Run one full training epoch with episodic steps."""
        stats = {"loss": [], "ce_loss": [], "prompt_reg": []}

        pbar = tqdm(range(episodes_per_epoch), desc="Meta-train", leave=False)
        for _ in pbar:
            step_stats = self.train_step(train_sampler, meta_batch_size=meta_batch_size)
            for k in stats:
                stats[k].append(step_stats[k])
            pbar.set_postfix(loss=f"{step_stats['loss']:.4f}")

        return {k: float(np.mean(v)) for k, v in stats.items()}

    @torch.no_grad()
    def evaluate(self, sampler: EpisodeSampler, num_episodes: int) -> Dict[str, float]:
        """Evaluate model using episodic protocol.

        Args:
            sampler: Episode sampler from val/test subjects.
            num_episodes: Number of episodes.

        Returns:
            Dict with loss and metrics.
        """
        self.model.eval()

        all_true = []
        all_pred = []
        losses = []

        for _ in tqdm(range(num_episodes), desc="Meta-eval", leave=False):
            episode = self._to_device(sampler.sample_episode())
            out = self.model.forward_episode(
                support_x=episode["support_x"],
                support_y=episode["support_y"],
                query_x=episode["query_x"],
                query_y=episode["query_y"],
                prompt_l2_lambda=self.prompt_l2,
            )

            losses.append(float(out["loss"].item()))
            preds = out["logits"].argmax(dim=1)

            all_true.append(episode["query_y"].detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())

        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)
        metrics = compute_metrics(y_true, y_pred)
        metrics["loss"] = float(np.mean(losses))
        return metrics

    def train(
        self,
        train_sampler: EpisodeSampler,
        val_sampler: EpisodeSampler,
        epochs: int,
        meta_batch_size: int,
        episodes_per_epoch: int,
        val_episodes: int,
        early_stopping_patience: Optional[int] = None,
        run_name: str = "default",
    ) -> Dict[str, float]:
        """Full training loop with validation and best checkpoint saving."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer is required for train.")

        best_val_acc = -float("inf")
        best_metrics = {}
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(train_sampler, meta_batch_size, episodes_per_epoch)
            val_stats = self.evaluate(val_sampler, num_episodes=val_episodes)

            if self.logger is not None:
                self.logger.info(
                    "[%s] Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f | val_kappa=%.4f",
                    run_name,
                    epoch,
                    epochs,
                    train_stats["loss"],
                    val_stats["loss"],
                    val_stats["accuracy"],
                    val_stats["macro_f1"],
                    val_stats["kappa"],
                )

            is_best = val_stats["accuracy"] > best_val_acc
            if is_best:
                best_val_acc = val_stats["accuracy"]
                best_metrics = dict(val_stats)
                patience_counter = 0
            else:
                patience_counter += 1

            if self.save_dir is not None:
                state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_stats,
                    "train_stats": train_stats,
                }
                save_checkpoint(state, save_dir=self.save_dir, filename="last.pt", is_best=is_best)

            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                if self.logger is not None:
                    self.logger.info("Early stopping triggered at epoch %d", epoch)
                break

        if self.logger is not None and best_metrics:
            self.logger.info(
                "Best val | acc=%.4f | f1=%.4f | kappa=%.4f",
                best_metrics["accuracy"],
                best_metrics["macro_f1"],
                best_metrics["kappa"],
            )

        return best_metrics
