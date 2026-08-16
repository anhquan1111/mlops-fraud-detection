# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────
# Fraud Detection FastAPI — Production Dockerfile
# Port: 10000 (Render default for web services)
# Model: loaded at runtime via HF_REPO_ID env var (Hugging Face Hub)
# ─────────────────────────────────────────────────────────────

FROM python:3.12-slim

# LightGBM requires libgomp (OpenMP) — not included in slim image
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ ./src/

# Model loading strategy (set via env vars at runtime, NOT baked in):
#   LOCAL:  MODEL_PATH=models/my_model.pkl  (bind-mount or copy manually)
#   RENDER: HF_REPO_ID=username/repo-name   (downloads from Hugging Face Hub on startup)
# Do NOT COPY model files here — .pkl files are gitignored
# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# Expose Render's default port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/health')" || exit 1

# Start uvicorn
CMD ["uv", "run", "uvicorn", "src.api:app", \
     "--host", "0.0.0.0", \
     "--port", "10000", \
     "--workers", "1", \
     "--log-level", "info"]
