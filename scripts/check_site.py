#!/usr/bin/env python3
"""Dependency-free validation for the generated static site."""

from __future__ import annotations

import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".cursor", ".cache", "node_modules", "__pycache__"}
EXTERNAL_SCHEMES = {"http", "https", "mailto", "tel", "data", "javascript"}


class DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[tuple[str, str]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if element_id := values.get("id"):
            self.ids.add(element_id)
        for attr in ("href", "src"):
            if value := values.get(attr):
                self.links.append((attr, value))


def walk(pattern: str) -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob(pattern)
        if not any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)
    )


def parse_html(path: Path) -> DocumentParser:
    parser = DocumentParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def resolve_local(source: Path, raw_url: str) -> tuple[Path, str] | None:
    parsed = urlsplit(raw_url)
    if parsed.scheme in EXTERNAL_SCHEMES or raw_url.startswith("//"):
        return None
    if not parsed.path:
        return source, unquote(parsed.fragment)

    decoded = unquote(parsed.path)
    if decoded.startswith("/"):
        target = ROOT / decoded.lstrip("/")
    else:
        target = source.parent / decoded
    target = target.resolve()

    try:
        target.relative_to(ROOT)
    except ValueError:
        return None

    if target.is_dir() or decoded.endswith("/"):
        target /= "index.html"
    return target, unquote(parsed.fragment)


def check_brand(errors: list[str]) -> None:
    for path in walk("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".html",
            ".md",
            ".xml",
            ".yaml",
            ".yml",
            ".json",
            ".css",
            ".js",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "PraxAgent" in text:
            errors.append(
                f"{path.relative_to(ROOT)}: brand must be lowercase 'praxagent'"
            )


def check_html(errors: list[str]) -> None:
    # Validate deployable HTML, not Hugo templates containing {{ ... }} expressions.
    html_files = [
        path
        for path in walk("*.html")
        if "blog-source" not in path.relative_to(ROOT).parts
    ]
    parsed = {path.resolve(): parse_html(path) for path in html_files}

    for source in html_files:
        document = parsed[source.resolve()]
        for attr, raw_url in document.links:
            resolved = resolve_local(source.resolve(), raw_url)
            if resolved is None:
                continue
            target, fragment = resolved
            if not target.exists():
                errors.append(
                    f"{source.relative_to(ROOT)}: broken {attr}={raw_url!r} "
                    f"(missing {target.relative_to(ROOT)})"
                )
                continue
            if fragment and target.suffix.lower() == ".html":
                target_document = parsed.get(target.resolve())
                if target_document is None:
                    target_document = parse_html(target)
                    parsed[target.resolve()] = target_document
                if fragment not in target_document.ids:
                    errors.append(
                        f"{source.relative_to(ROOT)}: broken fragment {raw_url!r}"
                    )


def check_json(errors: list[str]) -> None:
    for path in walk("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: invalid JSON: {exc}")


def check_svg(errors: list[str]) -> None:
    for path in walk("*.svg"):
        try:
            ElementTree.parse(path)
        except (ElementTree.ParseError, OSError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: invalid SVG/XML: {exc}")


def main() -> int:
    errors: list[str] = []
    check_brand(errors)
    check_html(errors)
    check_json(errors)
    check_svg(errors)

    if errors:
        print("Site validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "Site validation passed: lowercase brand, local links, anchors, JSON, and SVG."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
