# syntax=docker/dockerfile:1

# Crystal Cache (CRYS) — self-hosted memory server image (WS E, E.1).
#
# One image, two shapes (WS E, D2):
#   * docker run     -> API + in-process background workers + a SQLite volume
#                       (the default CMD). Zero-config: no LLM key is needed
#                       to store/retrieve memories.
#   * docker compose -> this SAME image run twice: an API container with
#                       workers OFF (CC_RUN_WORKERS=false) and a separate
#                       worker container, both pointed at Postgres. See
#                       docker-compose.yml (E.2).
#
# Baked in (D4): the gtr-t5-base sentence-transformer (~440 MB) so the first
# boot needs no network, and CPU-only torch (no CUDA). Honest size ~= 2 GB.
# Excluded (D6): the coding agent ([agent]), the vec2text decoder
# ([decoder]), and mem0 ([mem0]). The admin frontend is not built in v1 —
# /admin returns 503 until a later WS-E step wires it.

FROM python:3.13-slim

# libgomp1: the OpenMP runtime scikit-learn (and some torch ops) dlopen at
# import time. The slim base omits it and the import crashes without it.
# ca-certificates already ships in the slim base.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Environment ---------------------------------------------------------
# HF_HOME / SENTENCE_TRANSFORMERS_HOME: pin the HuggingFace cache so the model
#   baked below is found at runtime (same path -> no re-download).
# CC_DATABASE_URL: default the single-container shape to a SQLite file on the
#   /data volume (absolute path -> four slashes). Compose overrides this to
#   Postgres. Alembic and the app both read settings.database_url from it.
# CC_VECTOR_BACKEND: the single-container default is the in-process sqlite-vec
#   backend (2c) — the vec0 fact index + the live routing scan in the same
#   /data SQLite file, no extra service. It REQUIRES a SQLite store, so the
#   Postgres compose shape overrides this to "memory" (see docker-compose.yml);
#   a Postgres deployment that wants a real vector index points CC_QDRANT_URL
#   at a Qdrant server and sets CC_VECTOR_BACKEND=qdrant instead.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    CC_DATABASE_URL=sqlite+aiosqlite:////data/crystal_cache.db \
    CC_VECTOR_BACKEND=sqlite_vec

WORKDIR /app

# --- Heavy, source-independent layers (cached across source edits) -------
# 1) CPU-only torch FIRST so the sentence-transformers install below sees it
#    already satisfied and never pulls a multi-GB CUDA build. Multi-arch (2d):
#    BuildKit/buildx sets TARGETARCH per platform. On amd64 we use the PyTorch
#    CPU wheel index (PyPI's amd64 torch is the CUDA default — huge). On arm64
#    there is no CUDA, so PyPI's aarch64 torch is already CPU-only — install it
#    straight from PyPI (the CPU index's arm64 coverage is not relied on).
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    else \
      pip install --no-cache-dir torch; \
    fi

# 2) sentence-transformers (the [embeddings] dependency) — heavy, so install
#    it before the app source for layer caching. Kept in sync with the
#    [embeddings] extra in pyproject.toml (sentence-transformers>=3.0).
RUN pip install --no-cache-dir "sentence-transformers>=3.0"

# 3) Bake gtr-t5-base into the image's HF cache so the first boot is offline.
#    The id MUST match crystal_cache.encoding.semantic.DEFAULT_MODEL_NAME.
#    (A CC_SEMANTIC_MODEL override at runtime downloads that model on first
#    use instead — only the default is baked.)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/gtr-t5-base')"

# Lock the HuggingFace cache to offline now that the model is baked: the
# running container never calls the Hub — no freshness check, no token nag,
# deterministic offline boot. This MUST come AFTER the bake above (with these
# set, the bake's download would refuse). A CC_SEMANTIC_MODEL override to an
# un-baked model now fails rather than silently downloading at runtime — the
# correct locked-down behavior for a shipped image (rebuild with it baked).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# --- Application ---------------------------------------------------------
# Copy only what the wheel build + migrations need. This keeps the coding
# agent, docs, tests, and local data out of the image regardless of
# .dockerignore.
COPY pyproject.toml README.md ./
COPY memory/src ./memory/src
COPY memory/migrations ./memory/migrations
COPY alembic.ini ./alembic.ini
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Install crystal-cache with the embeddings + sqlite-vec extras. torch +
# sentence-transformers are already satisfied, so this resolves only the light
# core deps (incl. sqlite-vec, the default self-host vector backend set above)
# and builds the wheel from ./src.
RUN pip install --no-cache-dir ".[embeddings,sqlite-vec,gcp]"

# --- Runtime user + writable dirs ----------------------------------------
# Non-root. /data holds the SQLite DB (the persisted volume); /opt/hf holds
# the baked model. Both must be writable by the runtime user. The entrypoint
# lives in /usr/local/bin (root-owned, world-executable).
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data \
    && chmod +x /usr/local/bin/docker-entrypoint.sh \
    && chown -R app:app /data /opt/hf /app
USER app

VOLUME ["/data"]
EXPOSE 8000

# Healthcheck targets the API's /health. The worker container (compose) has
# no HTTP server, so E.2 disables this healthcheck for that service.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# The entrypoint runs `alembic upgrade head` (unless CC_RUN_MIGRATIONS=false)
# then execs the CMD. Default CMD = the API with in-process workers.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "crystal_cache.app:app", "--host", "0.0.0.0", "--port", "8000"]
