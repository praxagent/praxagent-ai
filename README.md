# PraxAgent Website

Professional website for [praxagent.ai](https://praxagent.ai) — AI research, apps, and a Hugo-powered blog with LaTeX math support and syntax highlighting.

## Project Structure

```
praxagent/
├── index.html                 # Homepage (animated mountain hero)
├── apps.html                  # Apps showcase (Prax, TeamWork, TippyTip, Chaos Lab)
├── stochastic-mountain.js     # Animated mountain canvas (see below)
├── styles-v2.css              # Site-wide CSS (light/dark, mobile-first)
├── script.js                  # Theme toggle, mobile menu, scroll effects
├── blog-source/               # Hugo source files (edit here)
│   ├── content/posts/         # Blog posts (Markdown)
│   ├── layouts/               # Hugo templates
│   ├── static/                # Blog CSS, JS assets
│   └── hugo.yaml              # Hugo configuration
├── blog/                      # Generated blog output (do not edit directly)
└── assets/                    # Static images, logos
```

## Getting Started

### Prerequisites

- [Hugo](https://gohugo.io/) (`brew install hugo`)

### Development

```bash
# Blog live reload (edit blog-source/; browser refreshes on save)
cd blog-source
hugo server --bind 127.0.0.1 --port 1313 --baseURL http://127.0.0.1:1313/ --disableFastRender

# Post URL example:
# http://127.0.0.1:1313/posts/praxagent-jacobian-lens-qwen3-5-397b-a17b/

# Full site (homepage + built blog/) — no auto-rebuild
python3 -m http.server 8000   # from repo root → http://localhost:8000
```

### Adding Blog Posts

```bash
cd blog-source
hugo new content/posts/my-new-post/index.md

# While developing, use `hugo server` above (no manual rebuild).
# For a production-style write into blog/:
hugo --destination ../blog
```

### Post Format

```markdown
---
title: "Post Title"
date: 2024-01-30
author: "Author Name"
description: "Brief description"
tags: ["ai", "machine-learning"]
---

Your content here. Supports full Markdown, LaTeX math ($E = mc^2$),
code blocks with syntax highlighting, and custom shortcodes like
{{< panel "info" >}}content{{< /panel >}}.
```

## Stochastic Mountain Animation

The homepage hero features an animated stochastic mountain rendered on a `<canvas>` element (`stochastic-mountain.js`). The walk lines continuously drift and reshape using multi-frequency sine offsets, creating an organic, AI-like living landscape. The ridgeline itself also breathes.

All parameters live in `stochastic-mountain.js`. No build step needed — edit and reload.

### Architecture

| Layer | What it does |
|-------|-------------|
| **Ridgeline** | Jagged mountain silhouette built from anchor points with Hermite interpolation + random perturbation. Subtly drifts over time. |
| **Regular walks** | 70-150 random-walk lines below the ridge. Each has unique drift parameters so they reshape independently. |
| **Hero walks** | 10-22 brighter lines near the ridgeline with stronger drift and pulsing brightness. |
| **Fades** | Top ~48% fades to page background (keeps text readable). Bottom ~10% fades out for seamless section transition. |
| **Theme** | Reads `data-theme` attribute + listens for changes. Dark and light palettes with matching background gradients. |

### Tuning the Animation

The drift for each walk point is computed as:

```
offset = (sin(t * speed0 + x * freq0 + phase0) * 0.40
        + sin(t * speed1 + x * freq1 + phase1) * 0.35
        + sin(t * speed2 + x * freq2 + phase2) * 0.25) * amplitude
```

Three overlapping sine waves at different temporal speeds and spatial frequencies create complex, non-repeating motion.

#### Drift Parameters (`makeDriftParams`)

| Parameter | Range | Effect |
|-----------|-------|--------|
| `sp[0]` (slow wave) | 0.50 - 0.90 | Base drift speed. Lower = more languid, higher = more lively. |
| `sp[1]` (mid wave) | 0.90 - 1.40 | Secondary motion. Adds complexity to the drift pattern. |
| `sp[2]` (fast wave) | 1.40 - 2.10 | Fine detail. Creates quick local reshaping on top of broader sway. |
| `fr[0]` (broad spatial) | 0.003 - 0.008 | How wide the wave pattern is along each walk. Lower = big sweeping curves. |
| `fr[1]` (mid spatial) | 0.008 - 0.018 | Medium-scale ripples along the walk. |
| `fr[2]` (fine spatial) | 0.015 - 0.030 | Tight ripples. Higher values = more local, jittery reshaping. |

#### Amplitude

| Walk type | Amplitude range | Notes |
|-----------|----------------|-------|
| Regular (surface) | 4 - 11 px | `(1 - depth) * (4.0 + rand * 7.0)`. Near-ridge walks drift most. |
| Regular (deep) | 0 - 4 px | Deep walks barely move, providing a stable base. |
| Hero walks | 7 - 17 px | Strongest drift. These are the bright lines that catch the eye. |
| Ridgeline | ±5.5 px | Subtle but visible. Makes the mountain itself feel alive. |

#### Brightness Modulation

Walks fade in and out independently, creating a "thinking" effect:

| Parameter | Value | Effect |
|-----------|-------|--------|
| Walk brightness range | 0.55 - 1.0 | How much walks dim at their lowest. Lower min = more dramatic fade. |
| Walk modulation speed | 0.45 / 0.70 | Two sine speeds (dual-frequency avoids obvious pulsing). |
| Hero brightness range | 0.50 - 1.0 | Heroes pulse more dramatically. |
| Hero modulation speed | 0.55 / 0.85 | Slightly faster than regular walks. |
| Global breathe | ±8% at 0.5 Hz | Subtle overall contrast shift across the entire mountain. |

#### Mountain Shape (Ridgeline Anchors)

The ridgeline is defined by anchor points as `[x_fraction, y_fraction]` of the canvas:

```javascript
var anchors = [
    [0.00, 0.92], [0.06, 0.86], [0.14, 0.78], [0.22, 0.82],
    [0.30, 0.72], [0.38, 0.74], [0.44, 0.62], [0.48, 0.50],
    [0.50, 0.44],  // <-- summit
    [0.52, 0.50], [0.56, 0.62], [0.62, 0.68],
    [0.70, 0.74], [0.78, 0.80], [0.86, 0.86], [0.93, 0.91],
    [1.00, 0.94]
];
```

- **Peak height**: `0.44` means the summit is at 44% from the top of the canvas. Lower = taller peak.
- **Peak sharpness**: The gap between the summit and its neighbors (0.48→0.44→0.52) controls steepness. Closer x-values with bigger y-jumps = sharper peak.
- **Jaggedness**: Controlled by the perturbation loop after interpolation. The `peakProx` multiplier adds more jitter near the summit.

#### Responsiveness

| Viewport | Walks | Heroes | Steps per walk |
|----------|-------|--------|----------------|
| Desktop (600px+) | 150 | 22 | 250 |
| Mobile (<600px) | 70 | 10 | 140 |

### Quick Recipes

**More energetic** — increase `sp` values in `makeDriftParams` and `driftAmp` multipliers.

**Calmer / meditative** — halve the `sp` values, reduce `driftAmp` to `(1-depth) * (1.5 + rand * 3)`.

**More walks (denser)** — increase `n` in `buildWalks` / `buildHeroes`. Watch mobile performance past ~200.

**Taller peak** — lower the summit y-value in anchors (e.g., `0.44` → `0.38`).

**Wider peak** — spread the shoulder anchors further from the summit x-position.

## Deployment

```bash
cd blog-source
hugo --destination ../blog
```

Upload the entire repo to GitHub Pages (or any static host). The `CNAME` file points to `praxagent.ai`.

## License

See LICENSE file for details.
