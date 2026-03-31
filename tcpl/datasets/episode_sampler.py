"""Episode sampler for subject-conditioned few-shot meta-learning."""

from typing import Dict, Hashable, List, Optional

import numpy as np
import torch

from datasets.eeg_dataset import SubjectWiseEEGDataset


class EpisodeSampler:
    """Sample episodes from a subject-wise EEG dataset.

    One episode is built from exactly one subject:
    - Select `n_way` classes from this subject
    - For each class sample `k_shot` support and `q_query` query trials

    Output tensors:
    - support_x: [n_way * k_shot, C, T]
    - support_y: [n_way * k_shot]
    - query_x: [n_way * q_query, C, T]
    - query_y: [n_way * q_query]
    """

    def __init__(
        self,
        dataset: SubjectWiseEEGDataset,
        subject_ids: List[Hashable],
        n_way: int,
        k_shot: int,
        q_query: int,
        seed: Optional[int] = None,
    ):
        """Initialize sampler.

        Args:
            dataset: Subject-wise EEG dataset.
            subject_ids: Candidate subject IDs to sample episodes from.
            n_way: Number of classes per episode.
            k_shot: Number of support samples per class.
            q_query: Number of query samples per class.
            seed: Optional RNG seed.
        """
        self.dataset = dataset
        self.subject_ids = list(subject_ids)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.rng = np.random.default_rng(seed)

        self.required_per_class = k_shot + q_query
        self.eligible_subjects = [sid for sid in self.subject_ids if self._is_subject_eligible(sid)]

        if not self.eligible_subjects:
            raise ValueError("No eligible subjects for the requested n_way/k_shot/q_query setting.")

    def _is_subject_eligible(self, subject_id: Hashable) -> bool:
        """Check whether a subject can provide one complete balanced episode."""
        class_indices = self.dataset.get_class_indices(subject_id)
        eligible_classes = [c for c, idxs in class_indices.items() if len(idxs) >= self.required_per_class]
        return len(eligible_classes) >= self.n_way

    def _eligible_classes(self, subject_id: Hashable) -> List[int]:
        """Return class labels that have enough trials for support+query."""
        class_indices = self.dataset.get_class_indices(subject_id)
        return [c for c, idxs in class_indices.items() if len(idxs) >= self.required_per_class]

    def sample_episode(self, subject_id: Optional[Hashable] = None) -> Dict[str, object]:
        """Sample one class-balanced episode from one subject.

        Args:
            subject_id: Optional fixed subject id. If None, sample from eligible subjects.

        Returns:
            Dict with support/query tensors and subject id.
        """
        if subject_id is None:
            subject_id = self.rng.choice(self.eligible_subjects).item()
        elif not self._is_subject_eligible(subject_id):
            raise ValueError(f"Subject {subject_id} is not eligible for current episode settings.")

        classes = self._eligible_classes(subject_id)
        chosen_classes = self.rng.choice(classes, size=self.n_way, replace=False).tolist()

        support_x, support_y = [], []
        query_x, query_y = [], []

        trials = self.dataset.get_trials(subject_id)
        class_indices = self.dataset.get_class_indices(subject_id)

        for cls in chosen_classes:
            idxs = class_indices[cls]
            picked = self.rng.choice(idxs, size=self.required_per_class, replace=False)
            support_idxs = picked[: self.k_shot]
            query_idxs = picked[self.k_shot :]

            for idx in support_idxs:
                support_x.append(trials[idx]["x"])
                support_y.append(trials[idx]["y"])

            for idx in query_idxs:
                query_x.append(trials[idx]["x"])
                query_y.append(trials[idx]["y"])

        # [N, C, T], [N]
        support_x_tensor = torch.stack(support_x, dim=0).float()
        support_y_tensor = torch.tensor(support_y, dtype=torch.long)
        query_x_tensor = torch.stack(query_x, dim=0).float()
        query_y_tensor = torch.tensor(query_y, dtype=torch.long)

        return {
            "support_x": support_x_tensor,
            "support_y": support_y_tensor,
            "query_x": query_x_tensor,
            "query_y": query_y_tensor,
            "subject_id": subject_id,
        }

    def sample_meta_batch(self, meta_batch_size: int) -> List[Dict[str, object]]:
        """Sample multiple episodes for one meta-update step."""
        return [self.sample_episode() for _ in range(meta_batch_size)]
