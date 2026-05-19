# Stage 1: build React/TypeScript frontend
FROM node:20-slim AS frontend
WORKDIR /build
COPY ui/frontend/package*.json ./
RUN npm ci
COPY ui/frontend/ ./
RUN npm run build

# Stage 2: Python runtime — full project dependencies + built frontend
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl bash \
    && rm -rf /var/lib/apt/lists/*

# Main pipeline Python deps (needed so run.sh works when triggered from the UI)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# UI-specific Python deps
COPY ui/requirements.txt ./ui/requirements.txt
RUN pip install --no-cache-dir -r ui/requirements.txt

# Copy built frontend into expected location
COPY --from=frontend /build/dist /app/ui/frontend/dist

# Copy project source
COPY . .

# Suppress caffeinate re-exec (not available in Linux containers)
ENV CAFFEINATED=1
EXPOSE 8000

ENTRYPOINT ["bash", "ui/docker-entrypoint.sh"]
