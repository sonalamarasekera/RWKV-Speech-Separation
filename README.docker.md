# Docker Usage

Build the Docker image:

```powershell
docker build -t rwkv-ss:latest .
```

Run the default training entrypoint:

```powershell
docker run --rm -v ${PWD}:/app rwkv-ss:latest
```

Run with a custom config file from the repo:

```powershell
docker run --rm -v ${PWD}:/app rwkv-ss:latest --config configs/experiment/dry_run.yaml
```

If the repo includes `poetry.lock`, Docker will use it for deterministic dependency installs.
