HUGO ?= hugo
PYTHON ?= python3
RSYNC ?= rsync

BLOG_SOURCE := blog-source
BLOG_OUTPUT := blog
HUGO_CONFIG := $(BLOG_SOURCE)/hugo.yaml

# Port used to serve the full static site locally.
SITE_PORT ?= 8000
# Port used by the blog-only Hugo live-reload server.
BLOG_PORT ?= 1313
# Override the blog's baseURL (empty = use the value in hugo.yaml).
BLOG_BASEURL ?=
# MagicDNS name for previews available to other devices on this tailnet.
TS_HOST = $(shell tailscale status --peers=false --json | $(PYTHON) -c "import json,sys; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))")

.DEFAULT_GOAL := help

.PHONY: help blog blog-serve blog-serve-tailscale run-site-local run-site-tailscale check ci

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

blog: ## Compile blog-source/content/posts into blog/
	$(HUGO) \
		--source "$(BLOG_SOURCE)" \
		--config "$(abspath $(HUGO_CONFIG))" \
		--destination "$(abspath $(BLOG_OUTPUT))" \
		$(if $(BLOG_BASEURL),--baseURL "$(BLOG_BASEURL)",) \
		--noBuildLock \
		--cleanDestinationDir
	$(RSYNC) -a \
		--include='*/' \
		--include='README.md' \
		--include='WEB.md' \
		--exclude='*' \
		"$(BLOG_SOURCE)/content/" \
		"$(BLOG_OUTPUT)/"
	$(PYTHON) scripts/copy_blog_docs.py \
		"$(BLOG_SOURCE)/content/posts" \
		"$(BLOG_OUTPUT)/posts"

blog-serve: ## Serve only the blog with live reload at http://127.0.0.1:1313/blog/
	$(HUGO) server \
		--source "$(BLOG_SOURCE)" \
		--config "$(abspath $(HUGO_CONFIG))" \
		--bind 127.0.0.1 \
		--port $(BLOG_PORT) \
		--baseURL http://127.0.0.1:$(BLOG_PORT)/blog/ \
		--buildDrafts \
		--disableFastRender \
		--noHTTPCache \
		--noBuildLock \
		--renderToMemory

# Live reload works on the phone too: the page's reload websocket connects
# back to the same MagicDNS host. Requires Tailscale running on both devices.
blog-serve-tailscale: ## Serve only the blog with live reload at http://<magicdns>:1313/blog/
	@test -n "$(TS_HOST)" || { echo "Tailscale not running or CLI not found"; exit 1; }
	@echo "Serving blog only: http://$(TS_HOST):$(BLOG_PORT)/blog/"
	$(HUGO) server \
		--source "$(BLOG_SOURCE)" \
		--config "$(abspath $(HUGO_CONFIG))" \
		--bind 0.0.0.0 \
		--port $(BLOG_PORT) \
		--baseURL http://$(TS_HOST):$(BLOG_PORT)/blog/ \
		--buildDrafts \
		--disableFastRender \
		--noHTTPCache \
		--navigateToChanged \
		--noBuildLock \
		--renderToMemory

run-site-local: ## Build the blog and serve the whole site (default http://127.0.0.1:8000/)
	$(MAKE) blog BLOG_BASEURL="http://127.0.0.1:$(SITE_PORT)/blog/"
	@echo "Serving full site at http://127.0.0.1:$(SITE_PORT)/ (use 127.0.0.1 exactly, not localhost)"
	$(PYTHON) -m http.server $(SITE_PORT) --bind 127.0.0.1

run-site-tailscale: ## Build and serve the whole site on your tailnet at http://<magicdns>:8000/
	@test -n "$(TS_HOST)" || { echo "Tailscale not running or CLI not found"; exit 1; }
	$(MAKE) blog BLOG_BASEURL="http://$(TS_HOST):$(SITE_PORT)/blog/"
	@echo "Serving full site on your tailnet: http://$(TS_HOST):$(SITE_PORT)/"
	$(PYTHON) -m http.server $(SITE_PORT) --bind 0.0.0.0

check: ## Validate Hugo, local links, anchors, JSON, SVG, Python, and branding
	$(HUGO) \
		--source "$(BLOG_SOURCE)" \
		--config "$(abspath $(HUGO_CONFIG))" \
		--renderToMemory \
		--noBuildLock \
		--panicOnWarning
	$(PYTHON) -m compileall -q "$(BLOG_SOURCE)"
	$(PYTHON) scripts/check_site.py

ci: blog check ## Run the same build and validation used by GitHub Actions
