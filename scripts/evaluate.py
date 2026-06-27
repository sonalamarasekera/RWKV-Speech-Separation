"""Evaluation script: load checkpoint, run separation pipeline, compute SI-SDR."""

from __future__ import annotations

import argparse
import yaml
import torch

from rwkv_ss.utils.config import resolve_and_validate
from rwkv_ss.transforms.stft import STFTProcessor
from rwkv_ss.models.registry import build_model
from rwkv_ss.data.datamodule import Libri2MixDataModule, DataConfig
from rwkv_ss.training.pipeline import run_separation
from rwkv_ss.training.losses.pit_si_sdr import pit_si_sdr_with_perm


def main(cfg_path: str, checkpoint: str):
    with open(cfg_path, "r", encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    cfg = resolve_and_validate(cfg)

    data_cfg = cfg.get("data", {})
    dm = Libri2MixDataModule(DataConfig(**data_cfg))
    dm.setup()
    loader = dm.test_dataloader()

    stft = STFTProcessor(**cfg.get("stft", {}))
    model = build_model(cfg.get("model", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    total_sdr = 0.0
    n = 0
    with torch.no_grad():
        for mix, sources, _rows in loader:
            mix = mix.to(device)
            sources = sources.to(device).squeeze(2)
            est = run_separation(mix, model, stft)
            est = est.to(device)
            # compute average SI-SDR per batch
            sdrs, _perm = pit_si_sdr_with_perm(est, sources)
            total_sdr += sdrs.sum().item()
            n += sdrs.numel()

    print(f"Average SI-SDR: {total_sdr / max(1, n):.4f} dB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    main(args.config, args.checkpoint)
