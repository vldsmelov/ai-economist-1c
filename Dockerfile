FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for scientific Python stacks
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions first to leverage Docker layer caching
COPY pyproject.toml README.md ./

# Copy application source
COPY ai_economist_service ./ai_economist_service
COPY scripts ./scripts

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "ai_economist_service.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
