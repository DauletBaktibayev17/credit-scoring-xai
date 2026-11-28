FROM python:3.11-slim

# LightGBM requires OpenMP (libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast reproducible dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/
COPY app.py api.py ./
COPY reports/figures/ ./reports/figures/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8501