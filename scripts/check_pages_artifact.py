#!/usr/bin/env python3
"""Validate the allowlisted artifact uploaded to GitHub Pages."""

from __future__ import annotations

import argparse
import stat
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from stage_pages import PUBLIC_DIRECTORIES, PUBLIC_FILES, ROOT


EXTERNAL_SCHEMES = {"data", "http", "https", "javascript", "mailto", "tel"}
EXPECTED_TOP_LEVEL = frozenset((*PUBLIC_FILES, *PUBLIC_DIRECTORIES))
REQUIRED_PATHS = (
    ".nojekyll",
    "CNAME",
    "index.html",
    "work/index.html",
    "research/index.html",
    "blog/index.html",
    "blog/knowledge-base/index.html",
    "blog/knowledge-base/prax/index.html",
    "blog/prax-docs/prax-docs-manifest.json",
)
FORBIDDEN_NAMES = {
    ".DS_Store",
    ".env",
    ".git",
    ".github",
    ".pids",
    "Makefile",
    "__pycache__",
    "blog-source",
    "pages-artifact",
    "scripts",
}


class DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[tuple[str, str]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = {key.casefold(): value for key, value in attrs}
        if element_id := values.get("id"):
            self.ids.add(element_id)
        for attribute in ("href", "src"):
            if value := values.get(attribute):
                self.links.append((attribute, value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=ROOT / "pages-artifact",
    )
    return parser.parse_args()


def _parse_html(path: Path) -> DocumentParser:
    parser = DocumentParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def _resolve_local(artifact: Path, source: Path, raw_url: str) -> tuple[Path, str] | None:
    parsed = urlsplit(raw_url)
    if parsed.scheme.casefold() in EXTERNAL_SCHEMES or raw_url.startswith("//"):
        return None
    if not parsed.path:
        return source, unquote(parsed.fragment)

    source_directory = source.parent.relative_to(artifact).as_posix().strip("/")
    base = f"https://local.invalid/{source_directory}/"
    resolved = urlsplit(urljoin(base, raw_url))
    target = artifact / unquote(resolved.path).lstrip("/")
    try:
        target.resolve(strict=False).relative_to(artifact)
    except ValueError:
        return None
    if target.is_dir() or resolved.path.endswith("/"):
        target /= "index.html"
    return target, unquote(resolved.fragment)


def check_artifact(artifact: Path) -> list[str]:
    errors: list[str] = []
    if not artifact.is_dir():
        return [f"missing Pages artifact directory: {artifact}"]

    actual_top_level = {path.name for path in artifact.iterdir()}
    missing = sorted(EXPECTED_TOP_LEVEL - actual_top_level)
    unexpected = sorted(actual_top_level - EXPECTED_TOP_LEVEL)
    if missing:
        errors.append(f"artifact is missing top-level entries: {', '.join(missing)}")
    if unexpected:
        errors.append(
            f"artifact has unexpected top-level entries: {', '.join(unexpected)}"
        )

    for relative in REQUIRED_PATHS:
        if not (artifact / relative).is_file():
            errors.append(f"artifact is missing required file: {relative}")

    cname = artifact / "CNAME"
    if cname.is_file() and cname.read_text(encoding="utf-8").strip() != "praxagent.ai":
        errors.append("CNAME must contain exactly 'praxagent.ai'")

    for path in (artifact, *artifact.rglob("*")):
        relative = path.relative_to(artifact)
        if path.is_symlink():
            errors.append(f"artifact contains a symlink: {relative}")
            continue
        if relative != Path("."):
            hidden_parts = [part for part in relative.parts if part.startswith(".")]
            if hidden_parts and relative != Path(".nojekyll"):
                errors.append(f"artifact contains unexpected hidden path: {relative}")
            if any(part in FORBIDDEN_NAMES for part in relative.parts):
                errors.append(f"artifact contains forbidden path: {relative}")

        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir() and mode & 0o005 != 0o005:
            errors.append(f"artifact directory is not world-readable/traversable: {relative}")
        elif path.is_file() and mode & 0o004 != 0o004:
            errors.append(f"artifact file is not world-readable: {relative}")

    html_files = sorted(artifact.rglob("*.html"))
    documents: dict[Path, DocumentParser] = {}
    for path in html_files:
        try:
            documents[path.resolve()] = _parse_html(path)
        except UnicodeDecodeError:
            errors.append(f"HTML is not UTF-8: {path.relative_to(artifact)}")

    for source in html_files:
        document = documents.get(source.resolve())
        if document is None:
            continue
        for attribute, raw_url in document.links:
            resolved = _resolve_local(artifact, source, raw_url)
            if resolved is None:
                continue
            target, fragment = resolved
            if not target.exists():
                errors.append(
                    f"{source.relative_to(artifact)}: broken {attribute}={raw_url!r}"
                )
                continue
            if fragment and target.suffix.casefold() == ".html":
                target_document = documents.get(target.resolve())
                if target_document is None:
                    try:
                        target_document = _parse_html(target)
                    except UnicodeDecodeError:
                        continue
                    documents[target.resolve()] = target_document
                if fragment not in target_document.ids:
                    errors.append(
                        f"{source.relative_to(artifact)}: broken fragment {raw_url!r}"
                    )

    return errors


def main() -> None:
    raw_artifact = parse_args().artifact
    artifact = raw_artifact if raw_artifact.is_absolute() else ROOT / raw_artifact
    artifact = artifact.resolve(strict=False)
    errors = check_artifact(artifact)
    if errors:
        print("Pages artifact validation failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    file_count = sum(path.is_file() for path in artifact.rglob("*"))
    print(f"Pages artifact validation passed ({file_count} public files).")


if __name__ == "__main__":
    main()
