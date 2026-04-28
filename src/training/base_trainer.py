def fit(self, train_loader, val_loader):
    history: dict[str, list[float]] = {}
    epochs_since_improvement = 0

    # --- NEW: try to resume ---
    start_epoch = 1
    resume_state = self.checkpoint_mgr.load_if_exists(self.checkpoint_filename)
    if resume_state is not None:
        self.logger.info(
            "Resuming training from checkpoint '%s' (epoch %d)",
            self.checkpoint_filename,
            resume_state["epoch"],
        )
        self.model.load_state_dict(resume_state["model"])
        self.optimizer.load_state_dict(resume_state["optimizer"])
        start_epoch = resume_state["epoch"] + 1

    # --- Training loop ---
    for epoch in range(start_epoch, self.epochs + 1):
        self.logger.info("Epoch %d/%d started", epoch, self.epochs)
        train_metrics = self._train_one_epoch(train_loader)
        val_metrics = self._validate(val_loader)

        all_metrics = {
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }
        self.logger.log_dict(all_metrics, step=epoch)

        for key, value in all_metrics.items():
            history.setdefault(key, []).append(value)

        primary_metric = val_metrics.get("loss", val_metrics.get("accuracy", 0.0))
        higher_is_better = "accuracy" in val_metrics and "loss" not in val_metrics

        saved_best = self.checkpoint_mgr.save_best(
            self.model,
            self.optimizer,
            epoch,
            primary_metric,
            filename=self.checkpoint_filename,
            higher_is_better=higher_is_better,
        )

        self.logger.info(
            "Epoch %d/%d completed - train: %s | val: %s",
            epoch,
            self.epochs,
            self._format_metrics(train_metrics),
            self._format_metrics(val_metrics),
        )

        if saved_best:
            self.logger.info(
                "Saved best checkpoint '%s' at epoch %d (metric=%.6f)",
                self.checkpoint_filename,
                epoch,
                primary_metric,
            )
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if self.scheduler is not None:
            self.scheduler.step()

        if (
            self.early_stopping_patience is not None
            and epochs_since_improvement >= self.early_stopping_patience
        ):
            self.logger.info(
                "Early stopping triggered after %d epochs without improvement (epoch %d).",
                self.early_stopping_patience,
                epoch,
            )
            break

    return history
