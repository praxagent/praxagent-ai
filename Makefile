HUGO ?= hugo
PYTHON ?= python3

BLOG_SOURCE := blog-source
BLOG_OUTPUT := blog
HUGO_CONFIG := $(BLOG_SOURCE)/hugo.yaml

.DEFAULT_GOAL := help

.PHONY: help blog blog-serve blog-serve-tailscale check ci

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

blog: ## Compile blog-source/content/posts into blog/
	$(HUGO) \
		--source "$(BLOG_SOURCE)" \
		--config "$(abspath $(HUGO_CONFIG))" \
		--destination "$(abspath $(BLOG_OUTPUT))"

blog-serve: ## Start Hugo with live reload at http://127.0.0.1:1313/
	$(HUGO) server \
		--source "$(BLOG_SOURCE)" \
		--config "$(abspath $(HUGO_CONFIG))" \
		--bind 127.0.0.1 \
		--port 1313 \
		--baseURL http://127.0.0.1:1313/ \
		--buildDrafts \
		--disableFastRender \
		--noHTTPCache

# Live reload works on the phone too: the page's reload websocket connects
# back to the same MagicDNS host. Requires Tailscale running on both devices.
blog-serve-tailscale: TS_HOST = $(shell tailscale status --peers=false --json | $(PYTHON) -c "import json,sys; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))")
blog-serve-tailscale: ## Serve with live reload on your tailnet (open http://<magicdns>:1313/ on your phone)
	@test -n "$(TS_HOST)" || { echo "Tailscale not running or CLI not found"; exit 1; }
	@echo "Serving on tailnet: http://$(TS_HOST):1313/ (also works on this machine)"
	$(HUGO) server \
		--source "$(BLOG_SOURCE)" \
		--config "$(abspath $(HUGO_CONFIG))" \
		--bind 0.0.0.0 \
		--port 1313 \
		--baseURL http://$(TS_HOST):1313/ \
		--buildDrafts \
		--disableFastRender \
		--noHTTPCache

check: ## Validate Hugo, local links, anchors, JSON, SVG, Python, and branding
	$(HUGO) \
		--source "$(BLOG_SOURCE)" \
		--config "$(abspath $(HUGO_CONFIG))" \
		--renderToMemory \
		--panicOnWarning
	$(PYTHON) -m compileall -q "$(BLOG_SOURCE)"
	$(PYTHON) scripts/check_site.py

ci: blog check ## Run the same build and validation used by GitHub Actions
