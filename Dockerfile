# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────
# Fraud Detection FastAPI — Production Dockerfile
# Port: 10000 (Render default for web services)
# Model: bundled at build time from models/baseline_lr.pkl
# ─────────────────────────────────────────────────────────────

FROM python:3.12-slim

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ ./src/

# Copy pre-exported model artifact (gitignored, must exist locally before docker build)
# Run `uv run python scripts/export_model.py` first to generate this file.
COPY models/baseline_lr.pkl ./models/baseline_lr.pkl

# Environment variables
ENV MODEL_PATH=models/baseline_lr.pkl \
    PYTHONUNBUFFERED=1 \
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
