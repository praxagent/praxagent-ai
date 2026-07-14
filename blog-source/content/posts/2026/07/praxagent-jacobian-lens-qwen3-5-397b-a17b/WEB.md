# WEB.md — how the Jacobian-lens post visuals were built

This note documents the **website-facing** artifacts for
`blog-source/content/posts/2026/07/praxagent-jacobian-lens-qwen3-5-397b-a17b/`.

The heavy experiment code and full pod receipts live in the sibling research repo
(`../jacobian-lens-research-202607a/.../experiments/lens_demo/`). That tree is **not**
deployed with praxagent.com. Anything the static site must serve is copied under
this post directory (see **Local mirrors** below).

## What the site serves (runtime)

| File | Role |
|------|------|
| `jspace-layer-clouds.json` | Explorer data: band layers, per-condition top-40 clouds, gloss map |
| `jspace-layer-explorer.js` | Interactive tabs + layer slider + approximate English gloss |
| `jspace-*-topk40*.svg` | Static still frames (raw / glossed) |
| `receipts/*.json` | Downloadable audit receipts + prompt specs (web mirrors) |
| `tools/export_jspace_clouds.py` | Copy of the exporter used to build `jspace-layer-clouds.json` |
| `gloss-review.json` | Pretty-printed gloss map for peer review |
| Shortcode | `blog-source/layouts/shortcodes/jspace_layer_explorer.html` |
| CSS | `.jspace-explorer*` rules in `blog-source/static/blog.css` |

Hugo copies post-folder assets into `blog/posts/2026/07/praxagent-jacobian-lens-qwen3-5-397b-a17b/`
on build (`hugo --destination ../blog` from `blog-source/`).

## Local mirrors (`receipts/`)

Copied from `jacobian-lens-research-202607a/.../lens_demo/` so readers can open receipts
without that repo being public or deployed:

| Web path | Source | Notes |
|----------|--------|-------|
| `receipts/demo2_consciousness_qwen35-397b_n24.json` | same name | Self-ref / denial / Japan clouds (last-token readout) |
| `receipts/demo2_canada_addon.json` | same name | Maple-leaf / Canada condition |
| `receipts/demo2_probes_qwen35-397b_n24.json` | same name | Pre-span probe batch (shows `?` position artifact) |
| `receipts/demo2_probes_span_qwen35-397b_n24.json` | same name | **Slimmed**: `per_position_cloud` stripped (~20 MB → ~2 MB). Full span file stays in the research tree |
| `receipts/demo_qwen35-397b.json` | same name | Act-1/2/3 verification receipt |
| `receipts/prompts_consciousness.json` | same name | Consciousness + Canada prompts |
| `receipts/prompts_probes.json` | same name | Deception / statue / digit / meristem prompts |

GitHub links to *code* (`demo2.py`, `fit_at_scale.py`, jspace-audit) still point at
`github.com/praxagent/jacobian-lens-research-202607a` — those are not required for the page
to render. Receipt / prompt links in the post body point at **`receipts/...`**.

## Pipeline (how to regenerate)

### 0. Prerequisites

- Pod with Qwen3.5-397B-A17B + published lens
  `praxagent-org/jacobian-lens-qwen3.5-397b-a17b` (sha256 `668c3bf1…99e97`)
- Research checkout with `experiments/lens_demo/demo2.py`

### 1. Run readouts on the pod

```bash
cd experiments/lens_demo

# Consciousness set (last-token readout) — already have n24 receipt
python demo2.py \
  --big-model Qwen/Qwen3.5-397B-A17B:model.language_model \
  --lens-hf praxagent-org/jacobian-lens-qwen3.5-397b-a17b:jlens/wikitext/qwen35_397b.pt \
  --expected-sha256 668c3bf17305b0d52495cb7ba589a1c1173301b1d13c3c6ad84e58245dc99e97 \
  --lens-fit-n 24 \
  --out demo2_consciousness_qwen35-397b_n24.json

# Canada add-on only
python demo2.py ... --conditions neutral_factual_canada --out demo2_canada_addon.json

# Span probes (deception / statue / digit / meristem) — MUST use --span for "?" prompts
python demo2.py ... --span --prompts-file prompts_probes.json \
  --out demo2_probes_span_qwen35-397b_n24.json
```

Do **not** pass `--skip-per-layer-topk` if you need the explorer.

### 2. Export explorer JSON

From `lens_demo/` (or use the copy in `tools/`):

```bash
python export_jspace_clouds.py \
  --receipt demo2_consciousness_qwen35-397b_n24.json \
  --receipt demo2_canada_addon.json \
  --receipt demo2_probes_span_qwen35-397b_n24.json \
  --keep-gloss-from /path/to/praxagent/.../jspace-layer-clouds.json \
  --out /path/to/praxagent/blog-source/content/posts/YYYY/MM/SLUG/jspace-layer-clouds.json,\
/path/to/praxagent/blog/posts/YYYY/MM/SLUG/jspace-layer-clouds.json
```

Behavior:

- Merges conditions by id (later receipts override).
- Trivia / span conditions pick a **content-anchor layer** from keyword hits in top-40.
- `meristem` uses span showcase **prompt position 4** (botanical neighborhood); other
  span tabs use `per_layer_topk`.
- Preserves / extends the `gloss` object from `--keep-gloss-from`.

### 3. Static SVG stills

Generated with an ad-hoc packing script (same spiral layout idea as the explorer JS):
read `jspace-layer-clouds.json`, take each condition’s `anchor_layer` top-40, pack into
SVG, write `jspace-<condition>-topk40.svg` (+ `-glossed` where useful).

Outputs land in:

- `blog-source/content/posts/YYYY/MM/SLUG/` (source of truth for Hugo)
- `blog/posts/YYYY/MM/SLUG/` (built site)
- optionally `assets/` for reuse

Re-run after changing anchors or glosses. There is no checked-in one-liner script yet;
recreate from this note or ask the agent that last regenerated them (session that added
span probes).

### 4. Interactive explorer

- Markup: Hugo shortcode `jspace_layer_explorer` → `#jspace-explorer`
- Logic: `jspace-layer-explorer.js` (fetch JSON, pack cloud, gloss toggle
  `original → English`, per-tab probe-lexicon median-rank sparkline)
- Cache-bust: `?v=N` on `data-src` and `<script src>` in the shortcode when shipping
  JSON/JS changes

### 5. Act-2 statistics and paired-rank figure

Both are generated from the web-mirrored `receipts/demo_qwen35-397b.json`:

```bash
cd blog-source/content/posts/2026/07/praxagent-jacobian-lens-qwen3-5-397b-a17b
uv run --with scipy python tools/recompute_act2_statistics.py \
  --out receipts/act2_statistics.json
uv run python tools/export_act2_paired_ranks.py
```

Outputs:

- `receipts/act2_statistics.json`: Wilson intervals; exact two-sided sign, Wilcoxon,
  and McNemar tests; one- and two-sided Fisher sensitivity checks; item ranks.
- `act2-paired-ranks.svg`: log-rank plot generated from that statistics receipt.

### 6. Hugo build

```bash
cd blog-source
hugo --destination ../blog
# ensure post assets synced if Hugo didn’t copy a newly added file:
cp content/posts/2026/07/praxagent-jacobian-lens-qwen3-5-397b-a17b/jspace-layer-*.{js,json} \
   ../blog/posts/2026/07/praxagent-jacobian-lens-qwen3-5-397b-a17b/
cp -R content/posts/2026/07/praxagent-jacobian-lens-qwen3-5-397b-a17b/receipts \
   ../blog/posts/2026/07/praxagent-jacobian-lens-qwen3-5-397b-a17b/
```

Local preview from repo root: `python3 -m http.server 8000` then open
`/blog/posts/2026/07/praxagent-jacobian-lens-qwen3-5-397b-a17b/index.html`.

## Gloss map

- Source of truth: `jspace-layer-clouds.json` → `"gloss"`
- Peer-review dump: `gloss-review.json`
- UI copy: **approximate English glosses** (not exact translations); fragments marked;
  language tags are hints (`RU`, `JA`, `ZH-Hant`, …)

## Method note (why `--span` exists)

Default `demo2` reads the lens at the **last prompt token**. Prompts ending in `?` often
show multilingual junk there — not a clean null. Span mode mins over every prompt
position × band layer. Deception / digit-meta / meristems need span; Statue of Liberty
also re-run under span for consistency.

## Checklist when adding a new condition

1. Add prompt to the right `prompts_*.json` and run `demo2.py` (with `--span` if needed).
2. `scp` receipt → research `lens_demo/` **and** slim-copy into `receipts/` if linked.
3. Re-run `export_jspace_clouds.py` with `--keep-gloss-from`.
4. Regenerate SVG still(s); add to `index.md`.
5. Bump `?v=` on the shortcode; `hugo` build; smoke-test slider + gloss.
6. Update this `WEB.md` if the pipeline changed.
