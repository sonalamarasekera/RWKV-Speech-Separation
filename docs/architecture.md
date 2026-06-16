# Architecture

Pipeline:

1. Waveform → STFT (`STFTProcessor`)
2. Magnitude features → `RWKVv7Separator` (TF-domain model)
3. Model outputs per-source magnitudes → ReLU
4. iSTFT with mixture phase → time-domain separated waveforms
5. PIT SI-SDR evaluation / loss

Module boundaries:

- `src/rwkv_ss/transforms/stft.py`: STFT/iSTFT single source of truth
- `src/rwkv_ss/data`: Libri2Mix dataset + DataModule
- `src/rwkv_ss/models`: model registry + RWKVv7 separator
- `src/rwkv_ss/training`: engine, trainer, pipeline, losses, callbacks
- `scripts/`: user-facing CLI entrypoints (train, evaluate, make_csv)
