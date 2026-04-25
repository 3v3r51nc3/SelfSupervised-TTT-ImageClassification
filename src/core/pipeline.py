"""
Experiment orchestrator.

ExperimentPipeline wires data, models, trainers and evaluator together
and runs the stages in order:

- Stage A   - SimCLR self-supervised pretraining of the ViT encoder.
- Stage B.1 - Linear probe with the frozen encoder (representation quality baseline).
- Stage B.2 - Full fine-tuning of encoder + classifier.
- Stage C   - Evaluation of the fine-tuned model on CIFAR-10-C
              across all corruptions and severities.

Stage D (test-time training) will plug into the same pipeline once
the TTT adapter is implemented.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.core.config import ExperimentConfig
from src.data.dataset import CIFARDataModule
from src.evaluation.evaluator import Evaluator
from src.models.backbone import ViTBackboneBuilder
from src.models.classifier import FineTuneModel
from src.models.simclr import SimCLRModel
from src.training.finetune_trainer import FineTuneTrainer
from src.training.linear_probe_trainer import LinearProbeTrainer
from src.training.simclr_trainer import SimCLRTrainer
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed


CIFAR10_NUM_CLASSES = 10


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

            ssl_train_loader, ssl_val_loader = data_module.ssl_loaders()
            self.logger.debug(
                "SSL loaders ready - train_batches=%d val_batches=%d batch_size_ssl=%d",
                len(ssl_train_loader),
                len(ssl_val_loader),
                self.config.data.batch_size_ssl,
            )

            # Stage A - SimCLR pretraining
            encoder_path = self._run_stage_a(ssl_train_loader, ssl_val_loader)

            # Supervised loaders are shared between Stage B.1 and Stage B.2.
            sup_train_loader, sup_val_loader = data_module.supervised_loaders()

            # Stage B.1 - Linear probe
            lp_history = self._run_stage_b1(encoder_path, sup_train_loader, sup_val_loader)

            # Stage B.2 - Full fine-tuning
            ft_model, ft_history = self._run_stage_b2(encoder_path, sup_train_loader, sup_val_loader)

            # Stage C - CIFAR-10-C evaluation on the fine-tuned model
            cifar10c_results = self._run_stage_c(ft_model, data_module)

            return {
                "history": {
                    "linear_probe": lp_history,
                    "finetune": ft_history,
                },
                "cifar10c_results": cifar10c_results,
                "artifacts": {
                    "simclr_checkpoint": str(self.checkpoint_mgr.directory / "simclr_best.pt"),
                    "encoder_checkpoint": str(encoder_path),
                    "linear_probe_checkpoint": str(self.checkpoint_mgr.directory / "linear_probe_best.pt"),
                    "finetune_checkpoint": str(self.checkpoint_mgr.directory / "finetune_best.pt"),
                },
            }
        finally:
            self.logger.close()

    # ------------------------------------------------------------------
    # Stage A
    # ------------------------------------------------------------------

    def _run_stage_a(self, train_loader, val_loader) -> Path:
        encoder = self._build_encoder()
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
        trainer.fit(train_loader, val_loader)
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
        return encoder_path

    # ------------------------------------------------------------------
    # Stage B.1
    # ------------------------------------------------------------------

    def _run_stage_b1(self, encoder_path: Path, train_loader, val_loader) -> dict[str, list[float]]:
        self.logger.info("Starting Stage B.1: Linear Probe")

        encoder = self._build_encoder()
        encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        for p in encoder.parameters():
            p.requires_grad = False

        lp_model = FineTuneModel(
            encoder=encoder,
            embed_dim=self.config.model.embed_dim,
            num_classes=CIFAR10_NUM_CLASSES,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            lp_model.classifier.parameters(),
            lr=self.config.linear_probe.learning_rate,
        )
        trainer = LinearProbeTrainer(
            model=lp_model,
            optimizer=optimizer,
            scheduler=None,
            device=self.device,
            logger=self.logger,
            checkpoint_mgr=self.checkpoint_mgr,
            epochs=self.config.linear_probe.epochs,
            checkpoint_filename="linear_probe_best.pt",
        )

        history = trainer.fit(train_loader, val_loader)
        self.logger.info("Completed Stage B.1: Linear Probe")
        return history

    # ------------------------------------------------------------------
    # Stage B.2
    # ------------------------------------------------------------------

    def _run_stage_b2(
        self, encoder_path: Path, train_loader, val_loader
    ) -> tuple[FineTuneModel, dict[str, list[float]]]:
        self.logger.info("Starting Stage B.2: Fine-tuning")

        encoder = self._build_encoder()
        encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))

        ft_model = FineTuneModel(
            encoder=encoder,
            embed_dim=self.config.model.embed_dim,
            num_classes=CIFAR10_NUM_CLASSES,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            ft_model.parameters(),
            lr=self.config.finetune.learning_rate,
        )
        trainer = FineTuneTrainer(
            model=ft_model,
            optimizer=optimizer,
            scheduler=None,
            device=self.device,
            logger=self.logger,
            checkpoint_mgr=self.checkpoint_mgr,
            epochs=self.config.finetune.epochs,
            checkpoint_filename="finetune_best.pt",
        )

        history = trainer.fit(train_loader, val_loader)
        self.logger.info("Completed Stage B.2: Fine-tuning")
        return ft_model, history

    # ------------------------------------------------------------------
    # Stage C
    # ------------------------------------------------------------------

    def _run_stage_c(
        self, model: FineTuneModel, data_module: CIFARDataModule
    ) -> dict[str, dict[int, dict[str, float]]]:
        self.logger.info("Starting Stage C: CIFAR-10-C evaluation")

        evaluator = Evaluator(model=model, device=self.device)
        results: dict[str, dict[int, dict[str, float]]] = {}

        for corruption in CIFARDataModule.cifar10c_corruptions():
            results[corruption] = {}
            for severity in range(1, 6):
                loader = data_module.cifar10c_loader(corruption, severity)
                metrics = evaluator.evaluate(loader)
                results[corruption][severity] = metrics
                self.logger.info(
                    "CIFAR-10-C %s severity %d - acc=%.4f loss=%.4f",
                    corruption,
                    severity,
                    metrics["accuracy"],
                    metrics["loss"],
                )

        self.logger.info("Completed Stage C: CIFAR-10-C evaluation")
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_encoder(self):
        return ViTBackboneBuilder(
            variant=self.config.model.backbone,
            patch_size=self.config.model.patch_size,
            image_size=self.config.data.image_size,
            embed_dim=self.config.model.embed_dim,
        ).build()
