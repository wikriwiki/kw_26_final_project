# syntax=docker/dockerfile:1
FROM node:22-alpine AS ui-build

WORKDIR /build/web/ui
COPY web/ui/package.json web/ui/package-lock.json ./
RUN npm ci
COPY web/ui/ ./
COPY web/fixtures/ /build/web/fixtures/

ARG VITE_READ_ONLY=true
ENV VITE_READ_ONLY=${VITE_READ_ONLY}
RUN npm run build


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WEB_READ_ONLY=true \
    SIM_DATA_ROOT=/srv/sim-data \
    SIM_VIZ_FILE=/srv/viz/sim_standalone.html

WORKDIR /app

COPY web/api/requirements.txt /tmp/web-requirements.txt
RUN pip install --no-cache-dir -r /tmp/web-requirements.txt \
    && groupadd --gid 10001 console \
    && useradd --uid 10001 --gid console --no-create-home --shell /usr/sbin/nologin console

COPY --chown=console:console . /app
COPY --from=ui-build --chown=console:console /build/web/ui/dist /app/web/ui/dist

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"]

CMD ["python", "-m", "uvicorn", "web.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1"]
