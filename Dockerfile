# Multi-stage Dockerfile for UITBOT
# Stage 1: Build virtual environment and compile dependencies
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt


# Stage 2: Minimal runtime image with non-root user and Tesseract OCR
FROM python:3.12-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/uitbot/.cache/huggingface \
    PATH="/opt/venv/bin:$PATH"

# Install Tesseract OCR with Vietnamese language pack and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-vie \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user
RUN groupadd -g 1000 uitbot && \
    useradd -u 1000 -g uitbot -m -s /bin/bash uitbot

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Pre-create model cache and directory structures
RUN mkdir -p /home/uitbot/.cache/huggingface /app/data/processed && \
    chown -R uitbot:uitbot /home/uitbot /app

# Copy project files
COPY --chown=uitbot:uitbot . .

USER uitbot

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
