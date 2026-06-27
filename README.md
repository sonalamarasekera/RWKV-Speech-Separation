# RWKV-SS — TF-Domain Speech Separation

A migrated pipeline for Time-Frequency (TF) domain speech separation utilizing the RWKV architecture.

---

## ⚡ Quickstart (Local Native Setup)

Follow these steps to run the pipeline directly on your host machine using a standard Python virtual environment.

### 1. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -U pip
# If managing dependencies with Poetry:
poetry install --with dev --with eval

# Alternative: Standard pip install
pip install -r requirements.txt
```

### 3. Run a Training Dry-Run
This project uses Hydra for configuration management. Run a baseline training experiment with:
```powershell
python -m scripts.train --config configs/experiment/baseline_tf.yaml
```

---

## 🐳 Docker & Docker Compose Usage

The recommended way to run training (with full GPU acceleration) or inference isolated from your host system is via Docker Compose.

### 1. First-Time Setup
Ensure your local directories exist before launching containers:
```powershell
mkdir -p data checkpoints
```

### 2. Run Training (GPU Required)
Launches the training pipeline using the optimized development stage of the unified Dockerfile:
```powershell
docker compose up train
```
*Note: Your local `src/`, `configs/`, and `scripts/` directories are live-mounted into the container so code changes reflect immediately without rebuilding.*

### 3. Run Inference (CPU-Safe)
Runs evaluation and inference using an ultra-slim, multi-stage production image:
```powershell
docker compose up infer
```

### 4. Run Inference (GPU-Accelerated)
To leverage dedicated hardware acceleration during evaluation, invoke the GPU compose profile:
```powershell
docker compose --profile gpu up infer-gpu
```

### Manual Docker Builds (Advanced)
If you need to build or run individual raw images without Docker Compose:
```powershell
# Build production image
docker build -t rwkv-ss:prod .

# Build development stage image
docker build --target dev -t rwkv-ss:dev .
```
