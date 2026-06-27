# ==============================================================================
# STAGE 1: Base Environment & Dependency Installer
# ==============================================================================
FROM python:3.10-slim AS builder

WORKDIR /build

# build-essential compiles source-only wheels (like pesq)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv to pass down to future stages
RUN python -m venv /build/.venv
ENV PATH="/build/.venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ==============================================================================
# STAGE 2: Development / Training Image
# ==============================================================================
# Build with: docker build --target dev -t rwkv-ss:dev .
FROM python:3.10-slim AS dev

WORKDIR /app

# Dev environment needs compiler (build-essential) and runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pull the compiled virtual environment
COPY --from=builder /build/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code into the dev layer
COPY src/     ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Default development command
CMD ["python", "-m", "scripts.train", "--config", "configs/experiment/baseline_tf.yaml"]


# ==============================================================================
# STAGE 3: Production Inference Image (Final/Default)
# ==============================================================================
# Build with: docker build -t rwkv-ss:prod .
FROM python:3.10-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# No compilers here—only runtime-specific C libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pull the compiled virtual environment
COPY --from=builder /build/.venv /app/.venv

# Bake code permanently into production
COPY src/     ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Run as non-root user for security
RUN groupadd --system app && useradd --system --gid app --create-home app
USER app

# Default production command
CMD ["python", "-m", "scripts.evaluate"]