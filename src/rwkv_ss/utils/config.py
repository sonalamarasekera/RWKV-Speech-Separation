"""Config helpers: resolve derived values and validate model config."""
from __future__ import annotations

from typing import Any, Dict


def resolve_and_validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Populate derived values and perform sanity checks in-place.

    Modifies and returns cfg dict. Expected keys: `stft`, `model`, `data`, `training`.
    """
    stft = cfg.get("stft", {})
    model = cfg.get("model", {})

    # Derive codec_dim from STFT if not provided
    n_fft = int(stft.get("n_fft", 512))
    codec_dim = model.get("codec_dim")
    if codec_dim is None:
        model["codec_dim"] = n_fft // 2 + 1

    # Validate divisibility
    n_embd = int(model.get("n_embd", 512))
    n_groups = int(model.get("n_groups", 1))
    if n_embd % max(1, n_groups) != 0:
        raise ValueError(f"Invalid model config: n_embd ({n_embd}) must be divisible by n_groups ({n_groups})")

    cfg["model"] = model
    return cfg
