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
from src.models.classifier import LinearClassifier, FineTuneModel
from src.training.linear_probe_trainer import LinearProbeTrainer
from src.training.finetune_trainer import FineTuneTrainer
from src.evaluation.evaluator import Evaluator
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed


class ExperimentPipeline:
    """
    Pipeline d’orchestration de l’expérience (Stages A → B → C).

    Ce module coordonne l’ensemble du workflow expérimental :
    - préparation des données (CIFAR‑10 et CIFAR‑10‑C),
    - construction des modèles (SimCLR, Linear Probe, Fine‑tuning),
    - exécution séquentielle des différents stages,
    - gestion des checkpoints et des logs,
    - évaluation finale sur données propres et corrompues.

    Le pipeline suit la structure suivante :

    Stage A — Pré‑entraînement auto‑supervisé (SimCLR)
        - Entraîne un encodeur ViT sur CIFAR‑10 via contraste NT‑Xent.
        - Sauvegarde l’encodeur pré‑entraîné pour les stages suivants.

    Stage B.1 — Linear Probe
        - Gèle l’encodeur SSL.
        - Entraîne uniquement un classifieur linéaire pour évaluer la qualité
        des représentations apprises en auto‑supervisé.

    Stage B.2 — Fine‑tuning complet
        - Dégèle l’encodeur.
        - Entraîne l’ensemble du modèle (encoder + classifier) en supervision
        pour obtenir les meilleures performances sur CIFAR‑10.

    Stage C — Évaluation sur CIFAR‑10‑C
        - Mesure la robustesse du modèle fine‑tuned face aux corruptions
        (bruit, blur, weather, digital) et aux 5 niveaux de sévérité.
        - Utilise l’Evaluator générique pour calculer loss et accuracy.

    Ce pipeline centralise la logique d’exécution, garantit l’ordre correct
    des stages, et assure la persistance des artefacts (checkpoints, logs,
    résultats d’évaluation).
    """

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
            
            # -----------------------------
            # Stage B.1 — Linear Probe
            # -----------------------------
            
            self.logger.info("Starting Stage B.1: Linear Probe")

            # 1) Recharger l’encoder pré-entraîné
            encoder_lp = ViTBackboneBuilder(
                variant=self.config.model.backbone,
                patch_size=self.config.model.patch_size,
                image_size=self.config.data.image_size,
                embed_dim=self.config.model.embed_dim,
            ).build()
            encoder_lp.load_state_dict(torch.load(encoder_path, map_location=self.device))

            # 2) Geler l’encoder
            for p in encoder_lp.parameters():
                p.requires_grad = False

            # 3) modèle Linear Probe
            linear_head = LinearClassifier(
                embed_dim=self.config.model.embed_dim,
                num_classes=10,
            )
            lp_model = FineTuneModel(
                encoder=encoder_lp,
                classifier=linear_head,
            ).to(self.device)

            # 4) Optimizer
            optimizer_lp = torch.optim.Adam(
                lp_model.parameters(),
                lr=self.config.linear_probe.learning_rate,
            )

            # 5) Loaders supervisés
            sup_train_loader, sup_val_loader = data_module.supervised_loaders()

            # 6) Trainer
            lp_trainer = LinearProbeTrainer(
                model=lp_model,
                optimizer=optimizer_lp,
                scheduler=None,
                device=self.device,
                logger=self.logger,
                checkpoint_mgr=self.checkpoint_mgr,
                epochs=self.config.linear_probe.epochs,
                checkpoint_filename="linear_probe_best.pt",
            )

            # 7) Entraînement
            lp_history = lp_trainer.fit(sup_train_loader, sup_val_loader)
            self.logger.info("Completed Stage B.1: Linear Probe")
            
            # -----------------------------
            # Stage B.2 — Full Fine-tuning
            # -----------------------------
            self.logger.info("Starting Stage B.2: Fine-tuning")

            # 1) l’encoder pré-entraîné
            encoder_ft = ViTBackboneBuilder(
                variant=self.config.model.backbone,
                patch_size=self.config.model.patch_size,
                image_size=self.config.data.image_size,
                embed_dim=self.config.model.embed_dim,
            ).build()
            encoder_ft.load_state_dict(torch.load(encoder_path, map_location=self.device))

            # 2) modèle complet (encoder + classifier)
            classifier_ft = LinearClassifier(
                embed_dim=self.config.model.embed_dim,
                num_classes=10,
            )
            ft_model = FineTuneModel(
                encoder=encoder_ft,
                classifier=classifier_ft,
            ).to(self.device)

            # 3) Optimizer (LR plus petit)
            optimizer_ft = torch.optim.Adam(
                ft_model.parameters(),
                lr=self.config.finetune.learning_rate,
            )

            # 4) Trainer
            ft_trainer = FineTuneTrainer(
                model=ft_model,
                optimizer=optimizer_ft,
                scheduler=None,
                device=self.device,
                logger=self.logger,
                checkpoint_mgr=self.checkpoint_mgr,
                epochs=self.config.finetune.epochs,
                checkpoint_filename="finetune_best.pt",
            )

            # 5) Entraînement
            ft_history = ft_trainer.fit(sup_train_loader, sup_val_loader)
            self.logger.info("Completed Stage B.2: Fine-tuning")
            
            # -----------------------------
            # Stage C — CIFAR-10-C Evaluation
            # -----------------------------
            self.logger.info("Starting Stage C: CIFAR-10-C evaluation")
            
            evaluator = Evaluator(
                model=ft_model,   # modèle fine-tuned
                device=self.device,
                logger=self.logger,
            )

            corruptions = CIFARDataModule.cifar10c_corruptions()
            cifar10c_results = {}

            for corruption in corruptions:
                cifar10c_results[corruption] = {}
                for severity in range(1, 6):
                    loader = data_module.cifar10c_loader(corruption, severity)
                    metrics = evaluator.evaluate(loader)
                    cifar10c_results[corruption][severity] = metrics

                    self.logger.info(
                        "CIFAR-10-C %s severity %d — acc=%.4f loss=%.4f",
                        corruption, severity, metrics["accuracy"], metrics["loss"]
                    )

            self.logger.info("Completed Stage C: CIFAR-10-C evaluation")


            return {
                    "history": history,
                    "artifacts": {
                        "simclr_checkpoint": str(self.checkpoint_mgr.directory / "simclr_best.pt"),
                        "encoder_checkpoint": str(encoder_path),
                    },
                }
        finally:
            self.logger.close()
