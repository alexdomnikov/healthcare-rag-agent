# Dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

# Copy dependency files AND the README
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (excluding dev and eval groups)
RUN uv sync --frozen --no-dev --no-group eval

# Copy application code
COPY src/ ./src/
ENV PYTHONPATH=/app/src

# Run the app
CMD ["uv", "run", "uvicorn", "healthcare_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]