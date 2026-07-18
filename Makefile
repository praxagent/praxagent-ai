HUGO ?= hugo
PYTHON ?= python3
RSYNC ?= rsync

BLOG_SOURCE := blog-source
BLOG_OUTPUT := blog
HUGO_CONFIG := $(BLOG_SOURCE)/hugo.yaml

# Prax docs are imported at build time; set PRAX_DOCS_SOURCE to use an
# existing checkout instead of updating the shallow cache.
PRAX_DOCS_REPO ?= https://github.com/praxagent/prax.git
PRAX_DOCS_REF ?= main
PRAX_DOCS_CACHE ?= .cache/prax-docs/repo
PRAX_DOCS_SOURCE ?=
PRAX_DOCS_CONTENT_OUTPUT := $(BLOG_SOURCE)/content/knowledge-base/prax
PRAX_DOCS_STATIC_OUTPUT := $(BLOG_SOURCE)/static/prax-docs
LATE_CHUNKING_BUNDLE := $(BLOG_SOURCE)/content/knowledge-base/deep-dives/late-chunking
LATE_CHUNKING_PUBLIC := $(BLOG_OUTPUT)/knowledge-base/deep-dives/late-chunking

# Port used to serve the full static site locally.
SITE_PORT ?= 8000
# Port used by the blog-only Hugo live-reload server.
BLOG_PORT ?= 1313
# Override the blog's baseURL (empty = use the value in hugo.yaml).
BLOG_BASEURL ?=
# Extra Hugo build flags (used by blog-drafts).
BLOG_BUILD_FLAGS ?=
# Set FORCE=1 to restart an already-running background serve.
FORCE ?= 0
# MagicDNS name for previews available to other devices on this tailnet.
TS_HOST = $(shell tailscale status --peers=false --json | $(PYTHON) -c "import json,sys; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))")

# Background serve state (make <target> up|down).
PID_DIR := .pids
BLOG_PID_FILE := $(PID_DIR)/blog-serve.pid
BLOG_LOG_FILE := $(PID_DIR)/blog-serve.log
BLOG_MODE_FILE := $(PID_DIR)/blog-serve.mode
SITE_PID_FILE := $(PID_DIR)/run-site.pid
SITE_LOG_FILE := $(PID_DIR)/run-site.log
SITE_MODE_FILE := $(PID_DIR)/run-site.mode
SERVE_ACTION := $(firstword $(filter up down,$(MAKECMDGOALS)))
SUPERVISE := $(abspath scripts/supervise_serve.sh)
DETACH := $(abspath scripts/detach_serve.sh)

.DEFAULT_GOAL := help

.PHONY: help sync-prax-docs verify-late-chunking blog blog-drafts blog-serve blog-serve-tailscale run-site-local run-site-tailscale check ci up down

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nServe lifecycle (any terminal):\n"
	@printf "  make blog-serve up|down\n"
	@printf "  make blog-serve-tailscale up|down\n"
	@printf "  make run-site-local up|down\n"
	@printf "  make run-site-tailscale up|down\n"
	@printf "  (bare = foreground; up = background; down = stop)\n"
	@printf "  up is idempotent — already-running servers are left alone.\n"
	@printf "  Restart with: make <target> up FORCE=1\n"

# Allow: make blog-serve-tailscale up
up down:
	@if [ -z "$(filter-out up down,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make <target> up|down"; \
		echo "  targets: blog-serve blog-serve-tailscale run-site-local run-site-tailscale"; \
		exit 1; \
	fi

sync-prax-docs: ## Fetch Prax and generate Hugo-ready documentation
	$(PYTHON) scripts/import_prax_docs.py \
		$(if $(PRAX_DOCS_SOURCE),--source "$(PRAX_DOCS_SOURCE)",--repo "$(PRAX_DOCS_REPO)" --ref "$(PRAX_DOCS_REF)" --cache-dir "$(PRAX_DOCS_CACHE)") \
		--content-output "$(PRAX_DOCS_CONTENT_OUTPUT)" \
		--static-output "$(PRAX_DOCS_STATIC_OUTPUT)"

verify-late-chunking: ## Verify the committed Late Chunking benchmark artifacts offline
	$(PYTHON) "$(LATE_CHUNKING_BUNDLE)/reproduce.py" --verify

blog: sync-prax-docs verify-late-chunking ## Compile blog-source/content/posts into blog/
	$(HUGO) \
		--source "$(BLOG_SOURCE)" \
		--config "$(abspath $(HUGO_CONFIG))" \
		--destination "$(abspath $(BLOG_OUTPUT))" \
		$(if $(BLOG_BASEURL),--baseURL "$(BLOG_BASEURL)",) \
		$(BLOG_BUILD_FLAGS) \
		--noBuildLock \
		--cleanDestinationDir
	$(RSYNC) -a \
		--include='/reproduce.py' \
		--include='/reproduce.py.lock' \
		--include='/ATTRIBUTION.md' \
		--include='/provenance.json' \
		--include='/fig-scifact-retrieval.svg' \
		--include='/fig-query-deltas.svg' \
		--include='/fig-scifact-retrieval.receipt.json' \
		--include='/fig-query-deltas.receipt.json' \
		--include='/receipts/' \
		--include='/receipts/aggregate.json' \
		--include='/receipts/per-query.csv' \
		--include='/receipts/scifact-test-qrels.tsv' \
		--include='/receipts/top-10-rankings.jsonl' \
		--include='/receipts/run.receipt.json' \
		--exclude='*' \
		"$(LATE_CHUNKING_BUNDLE)/" \
		"$(LATE_CHUNKING_PUBLIC)/"
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

blog-drafts: ## Compile blog output including draft Research Notes
	$(MAKE) blog BLOG_BUILD_FLAGS=--buildDrafts

ifneq ($(SERVE_ACTION),down)
blog-serve blog-serve-tailscale: sync-prax-docs
endif

blog-serve: ## Blog live reload at http://127.0.0.1:1313/blog/ (up|down supported)
ifeq ($(SERVE_ACTION),down)
	@$(call stop-serve,blog,$(BLOG_PORT),$(BLOG_PID_FILE),1)
	@rm -f "$(BLOG_MODE_FILE)"
else ifeq ($(SERVE_ACTION),up)
	@$(call ensure-serve-up,blog,$(BLOG_PORT),$(BLOG_PID_FILE),$(BLOG_LOG_FILE),$(BLOG_MODE_FILE),local,http://127.0.0.1:$(BLOG_PORT)/blog/,$(HUGO) server --source $(BLOG_SOURCE) --config $(abspath $(HUGO_CONFIG)) --bind 127.0.0.1 --port $(BLOG_PORT) --baseURL http://127.0.0.1:$(BLOG_PORT)/blog/ --buildDrafts --disableFastRender --noHTTPCache --noBuildLock --renderToMemory)
else
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
endif

# Live reload works on the phone too: the page's reload websocket connects
# back to the same MagicDNS host. Requires Tailscale running on both devices.
blog-serve-tailscale: ## Blog live reload at http://<magicdns>:1313/blog/ (up|down supported)
ifeq ($(SERVE_ACTION),down)
	@$(call stop-serve,blog,$(BLOG_PORT),$(BLOG_PID_FILE),1)
	@rm -f "$(BLOG_MODE_FILE)"
else ifeq ($(SERVE_ACTION),up)
	@test -n "$(TS_HOST)" || { echo "Tailscale not running or CLI not found"; exit 1; }
	@$(call ensure-serve-up,blog,$(BLOG_PORT),$(BLOG_PID_FILE),$(BLOG_LOG_FILE),$(BLOG_MODE_FILE),tailscale,http://$(TS_HOST):$(BLOG_PORT)/blog/,$(HUGO) server --source $(BLOG_SOURCE) --config $(abspath $(HUGO_CONFIG)) --bind 0.0.0.0 --port $(BLOG_PORT) --baseURL http://$(TS_HOST):$(BLOG_PORT)/blog/ --buildDrafts --disableFastRender --noHTTPCache --navigateToChanged --noBuildLock --renderToMemory)
else
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
endif

run-site-local: ## Full site at http://127.0.0.1:8000/ (up|down supported)
ifeq ($(SERVE_ACTION),down)
	@$(call stop-serve,site,$(SITE_PORT),$(SITE_PID_FILE),1)
	@rm -f "$(SITE_MODE_FILE)"
else ifeq ($(SERVE_ACTION),up)
	$(MAKE) blog BLOG_BASEURL="http://127.0.0.1:$(SITE_PORT)/blog/"
	@$(call ensure-serve-up,site,$(SITE_PORT),$(SITE_PID_FILE),$(SITE_LOG_FILE),$(SITE_MODE_FILE),local,http://127.0.0.1:$(SITE_PORT)/,$(PYTHON) -m http.server $(SITE_PORT) --bind 127.0.0.1)
else
	$(MAKE) blog BLOG_BASEURL="http://127.0.0.1:$(SITE_PORT)/blog/"
	@echo "Serving full site at http://127.0.0.1:$(SITE_PORT)/ (use 127.0.0.1 exactly, not localhost)"
	$(PYTHON) -m http.server $(SITE_PORT) --bind 127.0.0.1
endif

run-site-tailscale: ## Full site on tailnet at http://<magicdns>:8000/ (up|down supported)
ifeq ($(SERVE_ACTION),down)
	@$(call stop-serve,site,$(SITE_PORT),$(SITE_PID_FILE),1)
	@rm -f "$(SITE_MODE_FILE)"
else ifeq ($(SERVE_ACTION),up)
	@test -n "$(TS_HOST)" || { echo "Tailscale not running or CLI not found"; exit 1; }
	$(MAKE) blog BLOG_BASEURL="http://$(TS_HOST):$(SITE_PORT)/blog/"
	@$(call ensure-serve-up,site,$(SITE_PORT),$(SITE_PID_FILE),$(SITE_LOG_FILE),$(SITE_MODE_FILE),tailscale,http://$(TS_HOST):$(SITE_PORT)/,$(PYTHON) -m http.server $(SITE_PORT) --bind 0.0.0.0)
else
	@test -n "$(TS_HOST)" || { echo "Tailscale not running or CLI not found"; exit 1; }
	$(MAKE) blog BLOG_BASEURL="http://$(TS_HOST):$(SITE_PORT)/blog/"
	@echo "Serving full site on your tailnet: http://$(TS_HOST):$(SITE_PORT)/"
	$(PYTHON) -m http.server $(SITE_PORT) --bind 0.0.0.0
endif

check: sync-prax-docs verify-late-chunking ## Validate Hugo, local links, anchors, JSON, SVG, Python, branding, and data provenance
	$(HUGO) \
		--source "$(BLOG_SOURCE)" \
		--config "$(abspath $(HUGO_CONFIG))" \
		--renderToMemory \
		--noBuildLock \
		--panicOnWarning
	$(PYTHON) -m compileall -q "$(BLOG_SOURCE)"
	$(PYTHON) scripts/check_site.py
	$(PYTHON) scripts/check_provenance.py

ci: blog check ## Run the same build and validation used by GitHub Actions

# $(1)=name $(2)=port $(3)=pidfile $(4)=logfile $(5)=modefile $(6)=mode $(7)=url $(8...)=command
define ensure-serve-up
	mkdir -p "$(PID_DIR)"; \
	if [ "$(FORCE)" != "1" ] && lsof -nP -iTCP:$(2) -sTCP:LISTEN >/dev/null 2>&1; then \
		mode="unknown"; \
		[ -f "$(5)" ] && mode=$$(cat "$(5)"); \
		echo "$(1) already up on port $(2) (mode=$$mode) — leaving it alone"; \
		echo "  $(7)"; \
		echo "  restart with: FORCE=1 make <same-target> up"; \
		exit 0; \
	fi; \
	$(call stop-serve,$(1),$(2),$(3),0); \
	: > "$(4)"; \
	echo "$(6)" > "$(5)"; \
	echo "Serving $(1): $(7)"; \
	echo "Log: $(4)"; \
	chmod +x "$(SUPERVISE)" "$(DETACH)"; \
	"$(DETACH)" "$(4)" "$(3)" $(8); \
	$(call wait-listen,$(2),$(1),$(3),$(4))
endef

# Stop by PID file and by whatever is still listening on the port (cross-terminal).
# $(4)=1 prints "not running"; $(4)=0 stays quiet when already stopped (for pre-up cleanup).
define stop-serve
	stopped=0; \
	if [ -f "$(3)" ]; then \
		pid=$$(cat "$(3)"); \
		rm -f "$(3)" "$(3).child"; \
		if kill -0 "$$pid" 2>/dev/null; then \
			kill "$$pid" 2>/dev/null || true; \
			i=0; while kill -0 "$$pid" 2>/dev/null && [ $$i -lt 25 ]; do sleep 0.1; i=$$((i+1)); done; \
			kill -9 "$$pid" 2>/dev/null || true; \
			echo "Stopped $(1) (pid $$pid)"; \
			stopped=1; \
		fi; \
	fi; \
	i=0; \
	while [ $$i -lt 20 ]; do \
		pids=$$(lsof -nP -iTCP:$(2) -sTCP:LISTEN -t 2>/dev/null || true); \
		[ -z "$$pids" ] && break; \
		echo "$$pids" | xargs kill 2>/dev/null || true; \
		stopped=1; \
		sleep 0.1; \
		i=$$((i+1)); \
	done; \
	pids=$$(lsof -nP -iTCP:$(2) -sTCP:LISTEN -t 2>/dev/null || true); \
	if [ -n "$$pids" ]; then \
		echo "$$pids" | xargs kill -9 2>/dev/null || true; \
		echo "Stopped $(1) on port $(2)"; \
		stopped=1; \
	elif [ "$$stopped" -eq 1 ] && [ $$i -gt 0 ]; then \
		echo "Stopped $(1) on port $(2)"; \
	elif [ "$$stopped" -eq 0 ] && [ "$(4)" = "1" ]; then \
		echo "$(1) is not running"; \
	fi
endef

# Confirm background serve came up; otherwise dump the log and fail.
define wait-listen
	i=0; \
	while [ $$i -lt 50 ]; do \
		if lsof -nP -iTCP:$(1) -sTCP:LISTEN >/dev/null 2>&1; then \
			echo "$(2) is up (pid $$(cat "$(3)" 2>/dev/null || echo '?'))"; \
			exit 0; \
		fi; \
		if [ -f "$(3)" ] && ! kill -0 "$$(cat "$(3)")" 2>/dev/null; then \
			break; \
		fi; \
		sleep 0.1; \
		i=$$((i+1)); \
	done; \
	echo "$(2) failed to start; last log lines:"; \
	tail -n 40 "$(4)" 2>/dev/null || true; \
	rm -f "$(3)" "$(3).child"; \
	exit 1
endef
