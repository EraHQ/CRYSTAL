# SearXNG for the CRYSTAL hosted platform (Phase D follow-on, 2026-07-07).
#
# The self-hosted web-search backend behind CC_WEB_SEARCH_PROVIDER=searxng —
# the zero-paid-services end state (BACKLOG §13): SearXNG snippets + our own
# guarded fetch/extraction replace hosted search APIs.
#
# Deployed on Cloud Run with INGRESS=INTERNAL: SearXNG has no auth of its
# own, so reachability control IS the auth — only same-project VPC traffic
# (crystal-api / crystal-worker via Direct VPC egress) can reach it. Never
# deploy this with public ingress.
#
# The settings file enables JSON output (required by search/web.py) and
# disables the rate limiter (single internal client; the app enforces its
# own caps). SEARXNG_SECRET should be overridden via env at deploy time.
FROM searxng/searxng:latest

COPY settings.yml /etc/searxng/settings.yml
