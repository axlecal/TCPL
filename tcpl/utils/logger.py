"""Logging helper for reproducible training outputs."""

import logging
import os
from typing import Optional


def get_logger(name: str, save_dir: Optional[str] = None) -> logging.Logger:
    """Create a configured logger.

    Args:
        name: Logger name.
        save_dir: Optional directory for `train.log` file.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_dir, "train.log"), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
