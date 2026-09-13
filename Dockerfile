# ---------------------------------------------------------------------------
# Frontend build stage: produces apps/web/dist, which the API serves directly.
# apps/web/vite.config.ts sets base: './' specifically so the built bundle can
# be served from the FastAPI process itself -- no separate web server or CDN.
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend

WORKDIR /app/apps/web
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci
COPY apps/web/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Runtime stage: the API, the fitted-model code, and the built dashboard in
# one process. src/tyremind/serve.py is the single entrypoint for both.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# lightgbm's wheel links libgomp at runtime; nothing else here needs a compiler.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Everything the live API reads straight off disk: cached demo sessions,
# reference artefacts (pit calibration, track geometry), physics priors, the
# recorded experiment results the dashboard's evidence panels display, and the
# docs/research corpus the /api/ask retrieval endpoint indexes.
COPY data/ ./data/
COPY experiments/ ./experiments/
COPY configs/ ./configs/
COPY docs/ ./docs/
COPY research/ ./research/

COPY --from=frontend /app/apps/web/dist ./apps/web/dist

ENV PYTHONUNBUFFERED=1
EXPOSE 8077

# --no-warm: platform health checks expect the port open quickly, not after
# every cached session has been fit. Sessions fit lazily on first request
# instead -- slower for that one click, never for the boot.
CMD ["python", "-m", "tyremind.serve", "--host", "0.0.0.0", "--no-browser", "--no-warm"]
