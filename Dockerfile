FROM python:3.11-slim-bookworm

ARG NODE_VERSION=20.19.5
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz" \
        | tar -xz -C /usr/local --strip-components=1 \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY tools/imessage/package.json tools/imessage/package-lock.json tools/imessage/
RUN npm ci --omit=dev --prefix tools/imessage

COPY backend ./backend
COPY ai ./ai
COPY web ./web
COPY data/patient.json ./data/patient.json
COPY data/audio/fallback_escalate.mp3 data/audio/fallback_monitor.mp3 \
     data/audio/fallback_normal.mp3 data/audio/fallback_notify.mp3 ./data/audio/
COPY tools/imessage/send.mjs ./tools/imessage/send.mjs

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
