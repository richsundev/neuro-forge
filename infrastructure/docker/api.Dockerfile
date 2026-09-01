FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src
COPY apps/api ./apps/api
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini

RUN uv pip install --system --no-cache . && \
    uv pip install --system --no-cache ./apps/api

EXPOSE 8000
CMD ["uvicorn", "neuroforge_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
