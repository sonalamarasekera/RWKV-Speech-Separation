"""High-level Trainer orchestrator that uses engine, optimizer factory and callbacks."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from rwkv_ss.training import engine
from rwkv_ss.training.optim.factory import build_optimizer, build_scheduler
from rwkv_ss.training.callbacks.checkpoint import CheckpointCallback
from rwkv_ss.training.callbacks.early_stopping import EarlyStopping


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        datamodule: Any,
        stft_processor: Any,
        cfg: Dict[str, Any],
        device: Optional[torch.device] = None,
        work_dir: str = ".",
    ):
        self.model = model
        self.datamodule = datamodule
        self.stft_processor = stft_processor
        self.cfg = cfg
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.work_dir = work_dir

        self.optimizer = build_optimizer(self.model, cfg.get("optimizer", {}))
        self.scheduler = build_scheduler(self.optimizer, cfg.get("optimizer", {}))

        ckpt_dir = os.path.join(self.work_dir, cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_cb = CheckpointCallback(ckpt_dir)
        es_cfg = cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(patience=int(es_cfg.get("patience", 10)))

        self.model.to(self.device)

    def fit(self, max_epochs: int = 100):
        # Prepare datasets and dataloaders
        self.datamodule.setup()
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        # Support resume-from-checkpoint if provided in cfg
        start_epoch = 1
        resume_path = None
        training_cfg = (
            self.cfg.get("training", {}) if isinstance(self.cfg, dict) else {}
        )
        if training_cfg:
            resume_path = training_cfg.get("resume_from") or self.cfg.get(
                "resume_from_checkpoint"
            )

        if resume_path:
            try:
                ckpt = torch.load(resume_path, map_location=self.device)
                model_state = ckpt.get("model_state_dict")
                if model_state is not None:
                    self.model.load_state_dict(model_state)
                opt_state = ckpt.get("optimizer_state_dict")
                if opt_state is not None:
                    try:
                        self.optimizer.load_state_dict(opt_state)
                    except Exception:
                        pass
                sched_state = ckpt.get("scheduler_state_dict")
                if sched_state is not None and self.scheduler is not None:
                    try:
                        self.scheduler.load_state_dict(sched_state)
                    except Exception:
                        pass
                ckpt_epoch = ckpt.get("epoch", None)
                if ckpt_epoch is not None:
                    start_epoch = int(ckpt_epoch) + 1
            except Exception as exc:
                print(f"Warning: failed to load resume checkpoint {resume_path}: {exc}")

        for epoch in range(start_epoch, max_epochs + 1):
            train_loss = engine.train_one_epoch(
                epoch,
                self.model,
                self.stft_processor,
                train_loader,
                self.optimizer,
                self.device,
                grad_clip=self.cfg.get("grad_clip", 5.0),
            )

            val_loss = engine.validate(
                epoch,
                self.model,
                self.stft_processor,
                val_loader,
                self.device,
            )

            if self.scheduler is not None:
                # Support ReduceLROnPlateau separately
                try:
                    if isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                except Exception:
                    # Best-effort: ignore scheduler errors here
                    pass

            # Save checkpoint with recommended schema
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "resolved_config": self.cfg,
            }
            ckpt_path = self.checkpoint_cb.save(epoch, state)

            if self.early_stopping.step(val_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        return ckpt_path
