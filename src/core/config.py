"""
Experiment configuration.

Module goals:
- define config structures,
- validate required fields,
- expose one shared config object for all stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentMeta:
    name: str
    seed: int


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    data_root: str
    image_size: int
    num_workers: int
    batch_size_ssl: int
    batch_size_sup: int
    val_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    patch_size: int
    embed_dim: int
    projection_dim: int


@dataclass(frozen=True)
class SimCLRConfig:
    epochs: int
    temperature: float
    learning_rate: float
    log_every_n_steps: int


@dataclass(frozen=True)
class LinearProbeConfig:
    epochs: int
    learning_rate: float


@dataclass(frozen=True)
class FineTuneConfig:
    epochs: int
    learning_rate: float


@dataclass(frozen=True)
class TTTConfig:
    enabled: bool
    steps: int
    learning_rate: float
    adapt_scope: str


@dataclass(frozen=True)
class LoggingConfig:
    use_wandb: bool
    log_dir: str


@dataclass(frozen=True)
class CheckpointConfig:
    dir: str
    save_best_only: bool


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config composed of all section configs."""
    experiment: ExperimentMeta
    data: DataConfig
    model: ModelConfig
    simclr: SimCLRConfig
    linear_probe: LinearProbeConfig
    finetune: FineTuneConfig
    ttt: TTTConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig


class ConfigLoader:
    """Load and validate a YAML config file into an ExperimentConfig."""

    _REQUIRED_SECTIONS = (
        "experiment", "data", "model", "simclr",
        "linear_probe", "finetune", "ttt", "logging", "checkpoint",
    )

    @classmethod
    def load(cls, path: str) -> ExperimentConfig:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with config_path.open("r") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        cls._validate_sections(raw)

        config = ExperimentConfig(
            experiment=ExperimentMeta(**raw["experiment"]),
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            simclr=SimCLRConfig(**raw["simclr"]),
            linear_probe=LinearProbeConfig(**raw["linear_probe"]),
            finetune=FineTuneConfig(**raw["finetune"]),
            ttt=TTTConfig(**raw["ttt"]),
            logging=LoggingConfig(**raw["logging"]),
            checkpoint=CheckpointConfig(**raw["checkpoint"]),
        )
        cls._validate_values(config)
        return config

    @classmethod
    def _validate_sections(cls, raw: dict[str, Any]) -> None:
        missing = [s for s in cls._REQUIRED_SECTIONS if s not in raw]
        if missing:
            raise ValueError(f"Config is missing required sections: {missing}")

    @classmethod
    def _validate_values(cls, config: ExperimentConfig) -> None:
        if not 0.0 < config.data.val_fraction < 1.0:
            raise ValueError("data.val_fraction must be between 0 and 1.")
        if config.data.batch_size_ssl <= 0 or config.data.batch_size_sup <= 0:
            raise ValueError("Batch sizes must be positive.")
        if config.simclr.epochs <= 0:
            raise ValueError("simclr.epochs must be positive.")
        if config.simclr.log_every_n_steps <= 0:
            raise ValueError("simclr.log_every_n_steps must be positive.")
