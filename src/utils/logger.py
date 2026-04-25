"""
Experiment logging.

Integrations:
- Console + rotating file handler (always on)
- CSV metrics log (one row per scalar call)
- TensorBoard SummaryWriter (optional, activated when available)
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Logs scalars to console, a log file, and a CSV metrics file."""

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        self._log_dir = Path(log_dir) / experiment_name
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._logger = self._build_logger(experiment_name)
        self._csv_path = self._log_dir / "metrics.csv"
        self._csv_file = self._csv_path.open("a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._tb_writer: Any = None

        self._try_init_tensorboard()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """No-op kept for API compatibility; init happens in __init__."""

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Write one metric to CSV (and TensorBoard if available)."""
        self._csv_writer.writerow([step, tag, value])
        self._csv_file.flush()
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, step)
        self._logger.info("step=%d  %s=%.6f", step, tag, value)

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        """Convenience wrapper — logs every key/value pair in *metrics*."""
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def info(self, msg: str, *args: Any) -> None:
        self._logger.info(msg, *args)

    def debug(self, msg: str, *args: Any) -> None:
        self._logger.debug(msg, *args)

    def close(self) -> None:
        self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # avoid duplicate lines via root logger (Colab installs one)
        if logger.handlers:
            return logger  # already configured (e.g. called twice)

        fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(fmt)

        file_handler = logging.FileHandler(self._log_dir / "experiment.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)

        logger.addHandler(console)
        logger.addHandler(file_handler)
        return logger

    def _try_init_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
            self._tb_writer = SummaryWriter(log_dir=str(self._log_dir / "tensorboard"))
        except ImportError:
            pass
