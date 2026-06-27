"""Device resolution helper."""

from __future__ import annotations

import torch


def resolve_device(preferred: str | None = "auto") -> torch.device:
    if preferred is None or preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            # MPS availability check
            if (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")
    if preferred == "cuda":
        return torch.device("cuda")
    if preferred == "mps":
        return torch.device("mps")
    return torch.device("cpu")
