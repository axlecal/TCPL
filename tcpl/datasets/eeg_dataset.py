"""Dataset interfaces for subject-wise EEG trials."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Hashable, List, Sequence

import numpy as np
import torch


Trial = Dict[str, object]


class BaseEEGDataset(ABC):
    """Abstract EEG dataset interface organized by subject."""

    @abstractmethod
    def get_subject_ids(self) -> List[Hashable]:
        """Return all subject IDs."""

    @abstractmethod
    def get_trials(self, subject_id: Hashable) -> List[Trial]:
        """Return trial list for a subject.

        Each trial is a dict: `{"x": Tensor[C, T], "y": int}`.
        """

    @abstractmethod
    def get_classes(self, subject_id: Hashable) -> List[int]:
        """Return class labels available for a subject."""


class SubjectWiseEEGDataset(BaseEEGDataset):
    """In-memory subject-wise EEG dataset.

    The canonical storage is:
        dataset[subject_id] -> List[trial]
        trial = {"x": Tensor[C, T], "y": int}
    """

    def __init__(self, subject_to_trials: Dict[Hashable, List[Trial]]):
        """Build dataset from a subject -> trials dictionary.

        Args:
            subject_to_trials: Mapping from subject id to list of trials.
        """
        self.subject_to_trials: Dict[Hashable, List[Trial]] = {}
        self.subject_to_class_indices: Dict[Hashable, Dict[int, List[int]]] = {}

        for subject_id, trials in subject_to_trials.items():
            normalized_trials: List[Trial] = []
            class_indices = defaultdict(list)

            for idx, trial in enumerate(trials):
                x = trial["x"]
                y = int(trial["y"])

                if isinstance(x, np.ndarray):
                    x_tensor = torch.from_numpy(x).float()
                elif torch.is_tensor(x):
                    x_tensor = x.float()
                else:
                    raise TypeError(f"Unsupported trial tensor type: {type(x)}")

                if x_tensor.ndim != 2:
                    raise ValueError(f"Each trial must be shape [C, T], got {tuple(x_tensor.shape)}")

                normalized_trials.append({"x": x_tensor, "y": y})
                class_indices[y].append(idx)

            self.subject_to_trials[subject_id] = normalized_trials
            self.subject_to_class_indices[subject_id] = dict(class_indices)

    def __len__(self) -> int:
        """Return number of subjects."""
        return len(self.subject_to_trials)

    def get_subject_ids(self) -> List[Hashable]:
        """Return all subject IDs."""
        return list(self.subject_to_trials.keys())

    def get_trials(self, subject_id: Hashable) -> List[Trial]:
        """Return all trials for one subject."""
        return self.subject_to_trials[subject_id]

    def get_classes(self, subject_id: Hashable) -> List[int]:
        """Return class labels available in one subject."""
        return sorted(self.subject_to_class_indices[subject_id].keys())

    def get_class_indices(self, subject_id: Hashable) -> Dict[int, List[int]]:
        """Return mapping `class_id -> trial_indices` for one subject."""
        return self.subject_to_class_indices[subject_id]

    @classmethod
    def subset(cls, dataset: "SubjectWiseEEGDataset", subject_ids: Sequence[Hashable]) -> "SubjectWiseEEGDataset":
        """Create a new dataset containing only selected subjects."""
        data = {sid: dataset.get_trials(sid) for sid in subject_ids}
        return cls(data)
