# CRYSTAL Inspector — hosted frontend image (2026-07-05).
#
# A dedicated, tiny service: nginx serves the built SPA and reverse-
# proxies the SPA's two RELATIVE API path families (/admin/api, /v1) to
# the CRYSTAL API. Relative paths are the load-bearing choice — the
# frontend needs no build-time API URL and no CORS anywhere; pointing
# this image at a different deployment is one env var (API_UPSTREAM).
#
# Build FROM THE REPO ROOT (the frontend dir is the context's child):
#   docker build -f deploy/inspector/Dockerfile -t crystal-inspector .
#
# Run:
#   docker run -p 8080:8080 -e API_UPSTREAM=https://your-api.example crystal-inspector
#
# Self-hosters: this image is yours too — point API_UPSTREAM at your own
# CRYSTAL and put it wherever you like. Or skip it entirely and use
# `npm run dev` in frontend/ (VITE_API_TARGET does the same job there).

# --- Stage 1: build the SPA -------------------------------------------------
FROM node:22-slim AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# --- Stage 2: serve + proxy -------------------------------------------------
FROM nginx:1.27-alpine
# The SPA is built with base=/admin/, so it lives under /admin in the doc root.
COPY --from=build /app/dist /usr/share/nginx/html/admin
COPY deploy/inspector/nginx.conf.template /etc/nginx/templates/default.conf.template
# nginx's official image runs envsubst over /etc/nginx/templates/*.template
# at startup, substituting ${PORT} and ${API_UPSTREAM}.
ENV PORT=8080
ENV API_UPSTREAM=http://localhost:8000
EXPOSE 8080
