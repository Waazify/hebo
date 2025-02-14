FROM python:3.13-rc-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY proxy/pyproject.toml ./pyproject.toml

# Set environment variables for UV
ENV UV_SYSTEM_PYTHON=1
ENV UV_CACHE_DIR=/root/.cache/uv

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r pyproject.toml

# Copy the project
COPY proxy .

# Runtime stage
FROM python:3.13-rc-slim

# Create a non-root user
RUN useradd -m -u 1000 app

# Copy project files from builder
COPY --from=builder --chown=app:app /app /app
COPY --from=builder --chown=app:app /usr/local /usr/local

# Set environment variables
ENV PYTHONPATH="/app:$PYTHONPATH"

# Switch to non-root user
USER app
WORKDIR /app

# Run the FastAPI application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"] 