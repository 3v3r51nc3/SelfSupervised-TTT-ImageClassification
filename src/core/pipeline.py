"""
Experiment orchestrator.

ExperimentPipeline wires data, models, trainers and evaluator together
and runs the stages in order:

- Stage A   - SimCLR self-supervised pretraining of the ViT encoder.
- Stage B.1 - Linear probe with the frozen encoder (representation quality baseline).
- Stage B.2 - Full fine-tuning of encoder + classifier.
- Stage C   - Robustness eval on CIFAR-10-C (baseline).
- Stage D   - Test-time training (TENT) on each (corruption, severity) stream.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.core.config import ExperimentConfig
from src.data.dataset import CIFARDataModule
from src.evaluation.evaluator import Evaluator
from src.models.backbone import ViTBackboneBuilder
from src.models.classifier import FineTuneModel
from src.models.simclr import SimCLRModel
from src.training.finetune_trainer import FineTuneTrainer
from src.training.linear_probe_trainer import LinearProbeTrainer
from src.training.simclr_trainer import SimCLRTrainer
from src.ttt.adapter import TestTimeAdapter
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed


CIFAR10_NUM_CLASSES = 10


def _build_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float = 0.01,
):
    """Linear warmup followed by cosine annealing — both step per epoch."""
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    base_lr = optimizer.param_groups[0]["lr"]
    eta_min = base_lr * min_lr_ratio
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=eta_min)

    if warmup_epochs <= 0:
        return cosine

    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


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

    def setup_data(self) -> CIFARDataModule:
        """Build, prepare and set up the CIFAR-10 data module."""
        set_seed(self.config.experiment.seed)

        data_module = CIFARDataModule(
            data_root=self.config.data.data_root,
            image_size=self.config.data.image_size,
            batch_size_ssl=self.config.data.batch_size_ssl,
            batch_size_sup=self.config.data.batch_size_sup,
            num_workers=self.config.data.num_workers,
            val_fraction=self.config.data.val_fraction,
            seed=self.config.experiment.seed,
            augment_supervised=self.config.data.augment_supervised,
            randaugment_n=self.config.data.randaugment_n,
            randaugment_m=self.config.data.randaugment_m,
        )
        self.logger.info(
            "Preparing dataset '%s' from %s",
            self.config.data.dataset,
            self.config.data.data_root,
        )
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
        return data_module

    def load_finetuned_model(self, checkpoint_filename: str = "finetune_best.pt") -> FineTuneModel:
        """Rebuild a FineTuneModel and load weights from a saved fine-tune checkpoint."""
        encoder = self._build_encoder()
        ft_model = FineTuneModel(
            encoder=encoder,
            embed_dim=self.config.model.embed_dim,
            num_classes=CIFAR10_NUM_CLASSES,
        ).to(self.device)
        best_epoch = self.checkpoint_mgr.load_model(
            ft_model,
            filename=checkpoint_filename,
            map_location=self.device,
        )
        self.logger.info("Reloaded fine-tune checkpoint '%s' from epoch %d", checkpoint_filename, best_epoch)
        return ft_model

    def run(self) -> dict[str, object]:
        try:
            self.logger.info("Loaded config from %s", self.config_path)
            self.logger.info("Experiment name: %s", self.config.experiment.name)
            self.logger.info("Using seed: %d", self.config.experiment.seed)
            self.logger.info("Selected device: %s", self.device)

            data_module = self.setup_data()

            ssl_train_loader, ssl_val_loader = data_module.ssl_loaders()
            self.logger.debug(
                "SSL loaders ready - train_batches=%d val_batches=%d batch_size_ssl=%d",
                len(ssl_train_loader),
                len(ssl_val_loader),
                self.config.data.batch_size_ssl,
            )

            encoder_path = self.run_stage_a(ssl_train_loader, ssl_val_loader)

            sup_train_loader, sup_val_loader = data_module.supervised_loaders()

            lp_history = self.run_stage_b1(encoder_path, sup_train_loader, sup_val_loader)

            ft_model, ft_history = self.run_stage_b2(encoder_path, sup_train_loader, sup_val_loader)

            cifar10c_results = self.run_stage_c(ft_model, data_module)

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

    def run_stage_a(self, train_loader, val_loader) -> Path:
        self.checkpoint_mgr.reset_best()

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

        optimizer = torch.optim.AdamW(
            simclr_model.parameters(),
            lr=self.config.simclr.learning_rate,
            weight_decay=self.config.simclr.weight_decay,
        )
        scheduler = _build_warmup_cosine(
            optimizer,
            total_epochs=self.config.simclr.epochs,
            warmup_epochs=self.config.simclr.warmup_epochs,
        )
        trainer = SimCLRTrainer(
            model=simclr_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            logger=self.logger,
            checkpoint_mgr=self.checkpoint_mgr,
            epochs=self.config.simclr.epochs,
            checkpoint_filename="simclr_best.pt",
            temperature=self.config.simclr.temperature,
            log_every_n_steps=self.config.simclr.log_every_n_steps,
            use_amp=self.config.simclr.use_amp,
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

    def run_stage_b1(self, encoder_path: Path, train_loader, val_loader) -> dict[str, list[float]]:
        self.logger.info("Starting Stage B.1: Linear Probe")
        self.checkpoint_mgr.reset_best()

        encoder = self._build_encoder()
        encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        for p in encoder.parameters():
            p.requires_grad = False

        lp_model = FineTuneModel(
            encoder=encoder,
            embed_dim=self.config.model.embed_dim,
            num_classes=CIFAR10_NUM_CLASSES,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            lp_model.classifier.parameters(),
            lr=self.config.linear_probe.learning_rate,
            weight_decay=self.config.linear_probe.weight_decay,
        )
        scheduler = _build_warmup_cosine(
            optimizer,
            total_epochs=self.config.linear_probe.epochs,
            warmup_epochs=self.config.linear_probe.warmup_epochs,
        )
        trainer = LinearProbeTrainer(
            model=lp_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            logger=self.logger,
            checkpoint_mgr=self.checkpoint_mgr,
            epochs=self.config.linear_probe.epochs,
            checkpoint_filename="linear_probe_best.pt",
            use_amp=self.config.linear_probe.use_amp,
        )

        history = trainer.fit(train_loader, val_loader)
        self.logger.info("Completed Stage B.1: Linear Probe")
        return history

    # ------------------------------------------------------------------
    # Stage B.2
    # ------------------------------------------------------------------

    def run_stage_b2(
        self, encoder_path: Path, train_loader, val_loader
    ) -> tuple[FineTuneModel, dict[str, list[float]]]:
        self.logger.info("Starting Stage B.2: Fine-tuning")
        self.checkpoint_mgr.reset_best()

        encoder = self._build_encoder()
        encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))

        ft_model = FineTuneModel(
            encoder=encoder,
            embed_dim=self.config.model.embed_dim,
            num_classes=CIFAR10_NUM_CLASSES,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            ft_model.parameters(),
            lr=self.config.finetune.learning_rate,
            weight_decay=self.config.finetune.weight_decay,
        )
        scheduler = _build_warmup_cosine(
            optimizer,
            total_epochs=self.config.finetune.epochs,
            warmup_epochs=self.config.finetune.warmup_epochs,
        )
        trainer = FineTuneTrainer(
            model=ft_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            logger=self.logger,
            checkpoint_mgr=self.checkpoint_mgr,
            epochs=self.config.finetune.epochs,
            checkpoint_filename="finetune_best.pt",
            use_amp=self.config.finetune.use_amp,
            label_smoothing=self.config.finetune.label_smoothing,
            early_stopping_patience=self.config.finetune.early_stopping_patience,
        )

        history = trainer.fit(train_loader, val_loader)

        best_epoch = self.checkpoint_mgr.load_model(
            ft_model, filename="finetune_best.pt", map_location=self.device,
        )
        self.logger.info("Reloaded best fine-tune checkpoint from epoch %d", best_epoch)
        self.logger.info("Completed Stage B.2: Fine-tuning")
        return ft_model, history

    # ------------------------------------------------------------------
    # Stage C — robustness eval (baseline + TTT)
    # ------------------------------------------------------------------

    def run_stage_c(
        self, model: FineTuneModel, data_module: CIFARDataModule
    ) -> dict[str, dict[int, dict[str, float]]]:
        self.logger.info("Starting Stage C: CIFAR-10-C evaluation (baseline + TTT)")

        evaluator = Evaluator(model=model, device=self.device, num_classes=CIFAR10_NUM_CLASSES)
        ttt_enabled = self.config.ttt.enabled

        # Build the adapter once so its snapshot captures the clean fine-tuned weights.
        # `reset()` restores those weights before each new (corruption, severity) eval.
        adapter = self._make_ttt_adapter(model) if ttt_enabled else None

        clean_loader = data_module.test_loader()
        clean_baseline = evaluator.evaluate(clean_loader)
        self.logger.info(
            "Clean test - acc=%.4f loss=%.4f",
            clean_baseline["accuracy"],
            clean_baseline["loss"],
        )

        clean_ttt: dict[str, object] | None = None
        if adapter is not None:
            clean_ttt = evaluator.evaluate_with_ttt(clean_loader, adapter)
            self.logger.info(
                "Clean test (TTT) - acc=%.4f loss=%.4f delta=%+.4f",
                clean_ttt["accuracy"],
                clean_ttt["loss"],
                clean_ttt["accuracy"] - clean_baseline["accuracy"],
            )

        results: dict[str, dict[int, dict[str, float]]] = {
            "_clean": {0: {**{f"baseline_{k}": v for k, v in clean_baseline.items() if k != "per_class_accuracy"},
                           **({f"ttt_{k}": v for k, v in clean_ttt.items() if k != "per_class_accuracy"} if clean_ttt else {})}},
        }

        per_class_rows: list[dict[str, object]] = []
        per_class_rows.append(self._per_class_row("clean", 0, "baseline", clean_baseline))
        if clean_ttt is not None:
            per_class_rows.append(self._per_class_row("clean", 0, "ttt", clean_ttt))

        rows: list[dict[str, object]] = []
        for corruption in CIFARDataModule.cifar10c_corruptions():
            results[corruption] = {}
            for severity in range(1, 6):
                loader = data_module.cifar10c_loader(corruption, severity)

                if adapter is not None:
                    adapter.reset()
                baseline = evaluator.evaluate(loader)
                row: dict[str, object] = {
                    "corruption": corruption,
                    "severity": severity,
                    "baseline_accuracy": baseline["accuracy"],
                    "baseline_loss": baseline["loss"],
                }
                per_class_rows.append(self._per_class_row(corruption, severity, "baseline", baseline))

                if adapter is not None:
                    ttt_metrics = evaluator.evaluate_with_ttt(loader, adapter)
                    row["ttt_accuracy"] = ttt_metrics["accuracy"]
                    row["ttt_loss"] = ttt_metrics["loss"]
                    row["delta_accuracy"] = ttt_metrics["accuracy"] - baseline["accuracy"]
                    self.logger.info(
                        "CIFAR-10-C %s sev=%d - base_acc=%.4f ttt_acc=%.4f Δ=%+.4f",
                        corruption,
                        severity,
                        baseline["accuracy"],
                        ttt_metrics["accuracy"],
                        row["delta_accuracy"],
                    )
                    results[corruption][severity] = {
                        "baseline_accuracy": baseline["accuracy"],
                        "baseline_loss": baseline["loss"],
                        "ttt_accuracy": ttt_metrics["accuracy"],
                        "ttt_loss": ttt_metrics["loss"],
                        "delta_accuracy": row["delta_accuracy"],
                    }
                    per_class_rows.append(self._per_class_row(corruption, severity, "ttt", ttt_metrics))
                else:
                    self.logger.info(
                        "CIFAR-10-C %s sev=%d - acc=%.4f loss=%.4f",
                        corruption,
                        severity,
                        baseline["accuracy"],
                        baseline["loss"],
                    )
                    results[corruption][severity] = {
                        "baseline_accuracy": baseline["accuracy"],
                        "baseline_loss": baseline["loss"],
                    }

                rows.append(row)

        self._dump_stage_c_csv(rows, clean_baseline, clean_ttt)
        self._dump_stage_c_per_class_csv(per_class_rows)
        self.logger.info("Completed Stage C: CIFAR-10-C evaluation")
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_ttt_adapter(self, model: FineTuneModel) -> TestTimeAdapter:
        return TestTimeAdapter(
            model=model,
            steps=self.config.ttt.steps,
            learning_rate=self.config.ttt.learning_rate,
            adapt_scope=self.config.ttt.adapt_scope,
        )

    def _dump_stage_c_csv(
        self,
        rows: list[dict[str, object]],
        clean_baseline: dict[str, object],
        clean_ttt: dict[str, object] | None,
    ) -> None:
        log_dir = Path(self.config.logging.log_dir) / self.config.experiment.name
        log_dir.mkdir(parents=True, exist_ok=True)
        out_path = log_dir / "cifar10c_results.csv"

        fieldnames = [
            "corruption", "severity",
            "baseline_accuracy", "baseline_loss",
            "ttt_accuracy", "ttt_loss", "delta_accuracy",
        ]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            clean_row = {
                "corruption": "clean",
                "severity": 0,
                "baseline_accuracy": clean_baseline["accuracy"],
                "baseline_loss": clean_baseline["loss"],
            }
            if clean_ttt is not None:
                clean_row["ttt_accuracy"] = clean_ttt["accuracy"]
                clean_row["ttt_loss"] = clean_ttt["loss"]
                clean_row["delta_accuracy"] = clean_ttt["accuracy"] - clean_baseline["accuracy"]
            writer.writerow(clean_row)
            for row in rows:
                writer.writerow(row)
        self.logger.info("Stage C report saved to %s", out_path)

    @staticmethod
    def _per_class_row(
        corruption: str, severity: int, branch: str, metrics: dict[str, object]
    ) -> dict[str, object]:
        per_class = metrics["per_class_accuracy"]
        row: dict[str, object] = {
            "corruption": corruption,
            "severity": severity,
            "branch": branch,
        }
        for cls_idx, acc in enumerate(per_class):
            row[f"class_{cls_idx}"] = acc
        return row

    def _dump_stage_c_per_class_csv(self, per_class_rows: list[dict[str, object]]) -> None:
        log_dir = Path(self.config.logging.log_dir) / self.config.experiment.name
        log_dir.mkdir(parents=True, exist_ok=True)
        out_path = log_dir / "cifar10c_per_class.csv"

        fieldnames = ["corruption", "severity", "branch"] + [
            f"class_{i}" for i in range(CIFAR10_NUM_CLASSES)
        ]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_class_rows:
                writer.writerow(row)
        self.logger.info("Stage C per-class report saved to %s", out_path)

    def _build_encoder(self):
        return ViTBackboneBuilder(
            variant=self.config.model.backbone,
            patch_size=self.config.model.patch_size,
            image_size=self.config.data.image_size,
            embed_dim=self.config.model.embed_dim,
            drop_path_rate=self.config.model.drop_path_rate,
        ).build()
