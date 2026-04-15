"""
Experiment orchestrator.

ExperimentPipeline should:
- wire all dependencies (data/model/trainers/evaluator),
- run stages in the correct order,
- control artifact persistence.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.core.config import ExperimentConfig
from src.data.dataset import CIFARDataModule
from src.models.backbone import ViTBackboneBuilder
from src.models.simclr import SimCLRModel
from src.training.simclr_trainer import SimCLRTrainer
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed


class ExperimentPipeline:
    def __init__(self, config: ExperimentConfig, config_path: str | None = None) -> None:
        self.config = config
        self.config_path = config_path or "<in-memory>"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = ExperimentLogger(
            log_dir=self.config.logging.log_dir,
            experiment_name=self.config.experiment.name,
        )
        checkpoint_dir = Path(self.config.checkpoint.dir) / self.config.experiment.name
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            save_best_only=self.config.checkpoint.save_best_only,
        )

    def run(self) -> dict[str, object]:
        try:
            self.logger.info("Loaded config from %s", self.config_path)
            self.logger.info("Experiment name: %s", self.config.experiment.name)
            self.logger.info("Using seed: %d", self.config.experiment.seed)
            self.logger.info("Selected device: %s", self.device)

            set_seed(self.config.experiment.seed)

            data_module = CIFARDataModule(
                data_root=self.config.data.data_root,
                image_size=self.config.data.image_size,
                batch_size_ssl=self.config.data.batch_size_ssl,
                batch_size_sup=self.config.data.batch_size_sup,
                num_workers=self.config.data.num_workers,
                val_fraction=self.config.data.val_fraction,
                seed=self.config.experiment.seed,
            )
            self.logger.info("Preparing dataset '%s' from %s", self.config.data.dataset, self.config.data.data_root)
            data_module.prepare_data()
            data_module.setup()
            split_sizes = data_module.split_sizes()
            self.logger.info(
                "Data splits prepared - ssl_train=%d ssl_val=%d supervised_train=%d supervised_val=%d test=%d",
                split_sizes["ssl_train"],
                split_sizes["ssl_val"],
                split_sizes["supervised_train"],
                split_sizes["supervised_val"],
                split_sizes["test"],
            )

            train_loader, val_loader = data_module.ssl_loaders()
            self.logger.debug(
                "SSL loaders ready - train_batches=%d val_batches=%d batch_size_ssl=%d",
                len(train_loader),
                len(val_loader),
                self.config.data.batch_size_ssl,
            )

            encoder = ViTBackboneBuilder(
                variant=self.config.model.backbone,
                patch_size=self.config.model.patch_size,
                image_size=self.config.data.image_size,
                embed_dim=self.config.model.embed_dim,
            ).build()
            simclr_model = SimCLRModel(
                encoder=encoder,
                embed_dim=self.config.model.embed_dim,
                projection_dim=self.config.model.projection_dim,
            )
            self.logger.info(
                "Built SimCLR model - backbone=%s patch_size=%d embed_dim=%d projection_dim=%d",
                self.config.model.backbone,
                self.config.model.patch_size,
                self.config.model.embed_dim,
                self.config.model.projection_dim,
            )

            optimizer = torch.optim.Adam(simclr_model.parameters(), lr=self.config.simclr.learning_rate)
            trainer = SimCLRTrainer(
                model=simclr_model,
                optimizer=optimizer,
                scheduler=None,
                device=self.device,
                logger=self.logger,
                checkpoint_mgr=self.checkpoint_mgr,
                epochs=self.config.simclr.epochs,
                checkpoint_filename="simclr_best.pt",
                temperature=self.config.simclr.temperature,
                log_every_n_steps=self.config.simclr.log_every_n_steps,
            )

            self.logger.info("Starting Stage A: SimCLR pretraining")
            history = trainer.fit(train_loader, val_loader)
            self.logger.info("Completed Stage A: SimCLR pretraining")

            best_epoch = self.checkpoint_mgr.load_model(
                simclr_model,
                filename="simclr_best.pt",
                map_location=self.device,
            )
            self.logger.info("Reloaded best SimCLR checkpoint from epoch %d", best_epoch)

            encoder_path = self.checkpoint_mgr.directory / "encoder_pretrained.pt"
            torch.save(simclr_model.encoder.state_dict(), encoder_path)
            self.logger.info("Exported pretrained encoder to %s", encoder_path)

            return {
                "history": history,
                "artifacts": {
                    "simclr_checkpoint": str(self.checkpoint_mgr.directory / "simclr_best.pt"),
                    "encoder_checkpoint": str(encoder_path),
                },
            }
        finally:
            self.logger.close()
