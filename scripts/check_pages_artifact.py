#!/usr/bin/env python3
"""Validate the allowlisted artifact uploaded to GitHub Pages."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    "tippytip/privacy/index.html",
    "research/index.html",
    "blog/index.html",
    "blog/search/index.html",
    "blog/pagefind/pagefind.js",
    "blog/js/semantic-search.js",
    "blog/js/semantic-search-worker.js",
    "blog/search-assets/THIRD_PARTY_NOTICES.txt",
    "blog/search-assets/index/semantic-index.json",
    "blog/search-assets/index/embeddings.f32",
    "blog/knowledge-base/index.html",
    "blog/knowledge-base/prax/index.html",
    "blog/prax-docs/prax-docs-manifest.json",
)
FORBIDDEN_NAMES = {
    ".DS_Store",
    ".cache",
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

SEARCH_RUNTIME_FILES = (
    "transformers.min.js",
    "ort-wasm-simd-threaded.jsep.mjs",
    "ort-wasm-simd-threaded.jsep.wasm",
)
MODEL_WEIGHT_NAMES = ("model_int8.onnx", "model_quantized.onnx")
MODEL_WEIGHT_SIZE = 22_972_370
MODEL_WEIGHT_SHA256 = (
    "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1"
)
MODEL_SUPPORT_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
EMBEDDING_DIMENSION = 384
FLOAT32_BYTES = 4


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_search_assets(artifact: Path) -> list[str]:
    """Check assets loaded dynamically by the search worker.

    These files are not discoverable from ordinary HTML ``src`` attributes,
    so the general link checker below cannot prove that they reached the Pages
    artifact. Keep this validation independent of the generated directory
    layout while requiring the exact pinned model bytes.
    """

    errors: list[str] = []
    search_assets = artifact / "blog/search-assets"
    if not search_assets.is_dir():
        return ["artifact is missing search asset directory: blog/search-assets"]

    for filename in SEARCH_RUNTIME_FILES:
        matches = sorted(search_assets.rglob(filename))
        if len(matches) != 1:
            errors.append(
                f"search assets require exactly one {filename!r}; found {len(matches)}"
            )
        elif matches[0].stat().st_size == 0:
            errors.append(f"search runtime is empty: {matches[0].relative_to(artifact)}")

    weight_matches = sorted(
        path
        for filename in MODEL_WEIGHT_NAMES
        for path in search_assets.rglob(filename)
    )
    if len(weight_matches) != 1:
        errors.append(
            "search assets require exactly one pinned int8 MiniLM weight; "
            f"found {len(weight_matches)}"
        )
    else:
        weight = weight_matches[0]
        if weight.stat().st_size != MODEL_WEIGHT_SIZE:
            errors.append(
                f"{weight.relative_to(artifact)}: expected {MODEL_WEIGHT_SIZE:,} "
                f"bytes, found {weight.stat().st_size:,}"
            )
        if _sha256(weight) != MODEL_WEIGHT_SHA256:
            errors.append(
                f"{weight.relative_to(artifact)}: SHA-256 does not match the "
                "pinned MiniLM model"
            )

        model_root = weight.parent.parent
        for filename in MODEL_SUPPORT_FILES:
            support_file = model_root / filename
            if not support_file.is_file() or support_file.stat().st_size == 0:
                errors.append(
                    f"search model is missing support file: "
                    f"{support_file.relative_to(artifact)}"
                )

    semantic_index = search_assets / "index/semantic-index.json"
    if semantic_index.is_file():
        try:
            payload = json.loads(semantic_index.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(
                f"{semantic_index.relative_to(artifact)} is not valid UTF-8 JSON: {exc}"
            )
        else:
            if not isinstance(payload, (dict, list)) or not payload:
                errors.append(
                    f"{semantic_index.relative_to(artifact)} must contain search records"
                )

    embeddings = search_assets / "index/embeddings.f32"
    if embeddings.is_file():
        vector_bytes = EMBEDDING_DIMENSION * FLOAT32_BYTES
        size = embeddings.stat().st_size
        if size == 0 or size % vector_bytes:
            errors.append(
                f"{embeddings.relative_to(artifact)}: expected raw "
                f"{EMBEDDING_DIMENSION}-dimensional float32 vectors, found "
                f"{size:,} bytes"
            )

    return errors


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

    errors.extend(_check_search_assets(artifact))

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
