"""Simple checkpoint callback to persist training state."""

from __future__ import annotations

import os
from typing import Any, Dict

import torch


class CheckpointCallback:
    def __init__(self, directory: str, prefix: str = "ckpt"):
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.prefix = prefix

    def _path_for_epoch(self, epoch: int) -> str:
        return os.path.join(self.directory, f"{self.prefix}-epoch{epoch:03d}.pt")

    def save(self, epoch: int, state: Dict[str, Any]):
        # Enhance state with metadata
        state_out = dict(state)
        state_out.setdefault("schema_version", "1.0")
        state_out.setdefault("global_step", None)

        # Add simple random states best-effort
        try:
            import random
            import numpy as _np
            import torch as _torch

            state_out.setdefault("random_states", {})
            state_out["random_states"]["python"] = random.getstate()
            state_out["random_states"]["numpy"] = _np.random.get_state()
            try:
                state_out["random_states"]["torch"] = (
                    _torch.random.get_rng_state().tolist()
                )
            except Exception:
                state_out["random_states"]["torch"] = None
        except Exception:
            pass

        # Atomic save: write to tmp then replace
        tmp_path = self._path_for_epoch(epoch) + ".tmp"
        final_path = self._path_for_epoch(epoch)
        torch.save(state_out, tmp_path)
        os.replace(tmp_path, final_path)
        return final_path
