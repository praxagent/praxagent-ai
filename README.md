# PraxAgent Website

Professional website for PraxAgent with a Hugo-powered blog featuring LaTeX math support and syntax highlighting.

## Project Structure

```
praxagent/
├── blog-source/           # Hugo source files (edit here)
│   ├── content/posts/     # Blog posts (Markdown)
│   ├── layouts/           # Hugo templates
│   ├── static/            # CSS, JS assets
│   └── hugo.yaml          # Hugo configuration
├── blog/                  # Generated blog (do not edit)
├── index.html             # Main homepage
├── styles-v2.css          # Site CSS
└── script.js              # JavaScript
```

## Getting Started

### Prerequisites

- Hugo (`brew install hugo`)

### Development

```bash
# Start local server
python3 -m http.server 8000

# View at http://localhost:8000
```

### Adding Blog Posts

```bash
cd blog-source
hugo new content/posts/my-new-post.md

# After editing, rebuild:
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

Your content here. Supports full Markdown, LaTeX math ($E = mc^2$), and code blocks.
```

## Deployment

Build and deploy the static files:

```bash
cd blog-source
hugo --destination ../blog
```

Upload the `blog/` directory to any static hosting service (Netlify, Vercel, GitHub Pages, S3, etc.).

## License

See LICENSE file for details.
