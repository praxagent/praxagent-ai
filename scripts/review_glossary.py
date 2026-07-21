#!/usr/bin/env python3
"""Submit Knowledge Base content to GPT-5.6 Sol Pro for technical review.

The preferred interface is ``--entry PATH``, where PATH is any Knowledge Base
Markdown page or leaf-bundle directory. The script discovers that entry's local
text evidence and SVG sources; the historical no-argument mode still reviews
the complete glossary collection.

The script uses only the Python standard library. It reads OPENAI_API_KEY (or
the project's OPENAI_KEY alias) from the process environment first, then from
the repository-root .env file. The key is sent only in the Authorization header
and is never written to an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOT = ROOT / "blog-source" / "content"
KNOWLEDGE_BASE_DIR = CONTENT_ROOT / "knowledge-base"
GLOSSARY_DIR = KNOWLEDGE_BASE_DIR / "glossary"
STATIC_ROOT = ROOT / "blog-source" / "static"
DIAGRAM_DIR = STATIC_ROOT / "knowledge-base" / "glossary"
DOTENV_PATH = ROOT / ".env"
PRIVATE_OUTPUT_DIR = ROOT / ".cache" / "glossary-review"
DEFAULT_OUTPUT = PRIVATE_OUTPUT_DIR / "gpt-5.6-sol-pro.json"
DEFAULT_CONTINUATION_OUTPUT = PRIVATE_OUTPUT_DIR / "gpt-5.6-sol-pro-continued.json"
API_KEY_NAMES = ("OPENAI_API_KEY", "OPENAI_KEY")
REVIEW_METADATA_KEY = "_prax_review"
DATA_TEXT_SUFFIXES = frozenset({".csv", ".jsonl", ".tsv", ".txt"})
CORE_TEXT_SUFFIXES = frozenset(
    {
        ".adoc",
        ".c",
        ".cc",
        ".cff",
        ".cfg",
        ".cpp",
        ".css",
        ".dot",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".ini",
        ".ipynb",
        ".java",
        ".jl",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".m",
        ".md",
        ".mmd",
        ".py",
        ".r",
        ".rb",
        ".rs",
        ".rst",
        ".scala",
        ".sh",
        ".sql",
        ".svg",
        ".swift",
        ".tex",
        ".toml",
        ".ts",
        ".tsx",
        ".xml",
        ".yaml",
        ".yml",
    }
)
HASH_ONLY_SUFFIXES = frozenset({".lock", ".lockb"})
REVIEWABLE_TEXT_SUFFIXES = DATA_TEXT_SUFFIXES | CORE_TEXT_SUFFIXES | HASH_ONLY_SUFFIXES
REVIEWABLE_TEXT_NAMES = frozenset({"Dockerfile", "LICENSE", "Makefile", "NOTICE", "README"})
LOCKFILE_NAMES = frozenset(
    {
        "bun.lock",
        "bun.lockb",
        "cargo.lock",
        "composer.lock",
        "gemfile.lock",
        "go.sum",
        "npm-shrinkwrap.json",
        "package-lock.json",
        "pipfile.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "yarn.lock",
    }
)
BINARY_ASSET_SUFFIXES = frozenset(
    {".avif", ".gif", ".jpeg", ".jpg", ".pdf", ".png", ".webp"}
)
INTERNAL_GUIDANCE_NAMES = frozenset({"AGENTS.md", "SKILL.md"})
FORBIDDEN_PARTS = frozenset({".cache", ".git"})
DEFAULT_MAX_ARTIFACT_BYTES = 256 * 1024
DEFAULT_MAX_BUNDLE_BYTES = 1024 * 1024
MAX_NOTEBOOK_RAW_BYTES = 32 * 1024 * 1024

SRC_ATTRIBUTE_RE = re.compile(r'''\bsrc\s*=\s*["']([^"']+)["']''')
HREF_ATTRIBUTE_RE = re.compile(r'''\bhref\s*=\s*["']([^"']+)["']''')
MARKDOWN_LINK_RE = re.compile(r"\]\(([^)\s]+)(?:\s+[^)]*)?\)")
OG_IMAGE_RE = re.compile(r'''(?m)^og_image\s*[:=]\s*["']?([^"'\s]+)''')
TITLE_RE = re.compile(r'''(?m)^title\s*[:=]\s*["']?([^"'\n]+)''')
PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
HIGH_CONFIDENCE_TOKEN_RES = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b"),
)
SECRET_ASSIGNMENT_RE = re.compile(
    r'''(?i)(?<![A-Za-z0-9])["']?(?P<name>[A-Za-z][A-Za-z0-9_.-]{2,})["']?\s*[:=]\s*(?P<quote>["']?)(?P<value>[A-Za-z0-9_./+=-]{16,})'''
)
INLINE_RASTER_DATA_RE = re.compile(
    r"data:image/(?:avif|gif|jpe?g|png|webp)(?:;[^,\s]*)?;base64,",
    re.IGNORECASE,
)
PLACEHOLDER_MARKERS = (
    "CHANGEME",
    "EXAMPLE",
    "PLACEHOLDER",
    "REDACTED",
    "REPLACE_",
    "YOUR_",
)

KNOWLEDGE_BASE_REVIEW_INSTRUCTIONS = """\
You are the final scientific, technical, and educational reviewer for a
Knowledge Base entry. Review every supplied artifact as one connected work.
Apply each check only where relevant; an entry does not need code, experiments,
citations, or figures merely because the review system can inspect them.

Check, with exceptional care:
1. factual, conceptual, and mathematical accuracy, including assumptions,
   notation, dimensions, units, sign and normalization conventions, numeric
   examples, boundary cases, and the distinction between definitions and
   approximations;
2. dependency order for a careful newcomer: expand and define every acronym,
   symbol, specialized term, and overloaded word at first substantive use in
   prose, equations, code, tables, captions, alt text, and diagrams;
3. consistency among prose, equations, code, pseudocode, notebook cells, tables,
   links, Mermaid, SVGs, captions, alt text, receipts, and provenance;
4. whether supplied code and notebook cells implement the stated method,
   inputs, preprocessing, configuration, randomness, comparisons, and claimed
   outputs without leakage, hidden assumptions, or misleading edge behavior;
5. whether every numeric or empirical claim is supported by the supplied data,
   code, receipt, or provenance, and whether inputs, versions, hashes, analysis
   units, comparisons, uncertainty, limitations, and reproducibility boundaries
   are recorded to the degree the claim requires;
6. whether the entry clearly separates toy or synthetic examples, measured
   results, cited external findings, and author inference, and avoids extending
   sample-, dataset-, model-, implementation-, or environment-specific evidence
   beyond its demonstrated scope;
7. statistical and experimental validity where applicable, including
   independence, replication, grouping, pairing, splitting, selection, missing
   data, uncertainty, multiplicity, effect size, generalization, and the
   distinctions among description, association, prediction, causation, and
   mechanism;
8. every supplied SVG and Mermaid diagram as substantive technical content:
   inspect labels, arrows, grouping, geometry, scales, legends, counts, and
   encoded relationships, plus title, description, alt text, and caption.
   Require a complete nonvisual explanation and meaning that does not depend on
   color alone;
9. raster figures listed as metadata-only through the supplied surrounding
   claim, alt text, caption, generator, receipt, data, and provenance. Do not
   claim to have inspected their rendered pixels, clipping, contrast, or layout;
10. whether citations are attached to the claims they are meant to support,
    sufficiently precise, and no broader than the supplied source evidence.
    Do not claim to have inspected an external source unless its contents are
    in the bundle, and do not invent sources or citations;
11. whether the entry teaches an accurate mental model, states material limits
    and non-goals, and avoids ambiguity that would mislead its intended reader.

Treat filenames as authoritative and report cross-artifact conflicts explicitly.
Do not reward length, demand decorative diagrams, request optional sections, or
rewrite correct passages. A missing artifact is a finding only when a claim
depends on it or the entry represents it as available. Recommend the smallest
change that resolves each substantive defect.

Return valid JSON only, with this shape:
{
  "verdict": "pass" | "revise",
  "summary": "short overall assessment",
  "findings": [
    {
      "severity": "error" | "important" | "minor",
      "file": "repository-relative path",
      "section": "heading, code region, table, or figure element",
      "claim": "short exact excerpt or identifier",
      "issue": "specific explanation",
      "recommended_replacement": "ready-to-apply wording or code",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "cross_artifact_conflicts": ["specific conflict, if any"],
  "checks_that_passed": ["important point explicitly verified"]
}
Sort findings by severity, then file. Use verdict "pass" when no error or
important finding remains; optional minor polish should normally be omitted.
"""

GLOSSARY_REVIEW_ADDENDUM = """\
This entry is a glossary article. Require a compact, usable mental model, a
just-in-time definition, and enough example or visual support to prevent a
common misconception. Do not demand an experiment, long derivation, or extra
sections when the concept is already taught accurately and accessibly.
"""

DEEP_DIVE_REVIEW_ADDENDUM = """\
This entry is a Deep Dive. Where it makes empirical or reproducibility claims,
check frozen inputs, analysis units, comparisons, code paths, receipts,
provenance, figures, uncertainty, and stated scope as one evidence chain. Apply
those checks conditionally; do not demand an experiment from a conceptual or
source-synthesis article that makes no original empirical claim.
"""

FOLLOWUP_REVIEW_INSTRUCTIONS = """\
This is review round {round_number}, not an initial review. Concentrate on
whether the current bundle has any remaining factual, mathematical,
implementation, accessibility, or materially misleading teaching defects.
Do not introduce new stylistic preferences, optional elaborations, or
terminology polish merely because further refinement is possible. Do not
rephrase correct text. A specialized term warrants a finding only when leaving
it as written would likely give the intended careful newcomer a materially
wrong mental model or prevent them from understanding a central claim.

Use verdict "pass" when no error or important finding remains. Minor,
non-blocking polish does not prevent a pass and should normally be omitted.
"""

CONTINUATION_INSTRUCTION = """\
The preceding GPT-5.6 Sol Pro response exhausted its token allowance during
reasoning before emitting a visible answer. Reuse that encrypted reasoning;
do not restart or repeat the review. Synthesize the final review now. Return
only the requested JSON object, keep every finding concrete and actionable,
and omit internal deliberation.
"""


def glossary_source_paths() -> list[Path]:
    bundled_entries = sorted(GLOSSARY_DIR.glob("*/index.md"))
    bundled_diagrams = sorted(GLOSSARY_DIR.glob("*/*.svg"))
    shared_diagrams = sorted(DIAGRAM_DIR.glob("*.svg"))
    return bundled_entries + bundled_diagrams + shared_diagrams


@dataclass(frozen=True)
class ArtifactNote:
    path: Path
    reason: str
    referrer: Path | None = None

    def packet_line(self) -> str:
        relative = self.path.relative_to(ROOT).as_posix()
        media_type = mimetypes.guess_type(self.path.name)[0] or "application/octet-stream"
        referrer = (
            f"; referenced by {self.referrer.relative_to(ROOT).as_posix()}"
            if self.referrer is not None
            else ""
        )
        return (
            f"- {relative}: {self.reason}; media={media_type}; "
            f"bytes={self.path.stat().st_size}; sha256={sha256_file(self.path)}"
            f"{referrer}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_reviewable_text_path(path: Path) -> bool:
    return (
        path.suffix.lower() in REVIEWABLE_TEXT_SUFFIXES
        or path.name in REVIEWABLE_TEXT_NAMES
        or is_lockfile(path)
    )


def is_core_text_path(path: Path) -> bool:
    return path.suffix.lower() in CORE_TEXT_SUFFIXES or path.name in REVIEWABLE_TEXT_NAMES


def is_lockfile(path: Path) -> bool:
    return path.suffix.lower() in HASH_ONLY_SUFFIXES or path.name.lower() in LOCKFILE_NAMES


def review_payload_size(path: Path) -> int:
    """Measure what will be sent, not discarded notebook outputs or metadata."""
    if path.suffix.lower() != ".ipynb":
        return path.stat().st_size
    raw_size = path.stat().st_size
    if raw_size > MAX_NOTEBOOK_RAW_BYTES:
        raise ValueError(
            f"notebook raw file exceeds the {MAX_NOTEBOOK_RAW_BYTES:,}-byte safety limit: "
            f"{path} ({raw_size:,} bytes)"
        )
    return len(sanitized_notebook_source(path).encode("utf-8"))


def validate_repository_file(path: Path, *, allow_binary: bool = False) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if candidate.is_symlink():
        raise ValueError(f"review input must not be a symbolic link: {path}")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"review input must stay inside {ROOT}: {path}") from exc
    if any(part in FORBIDDEN_PARTS or part.startswith(".") for part in relative.parts):
        raise ValueError(f"review input is inside a private repository area: {path}")
    if resolved.name in INTERNAL_GUIDANCE_NAMES:
        raise ValueError(f"internal guidance is never a review artifact: {path}")
    if not resolved.is_file():
        raise ValueError(f"review input is not a file: {path}")
    is_binary = allow_binary and resolved.suffix.lower() in BINARY_ASSET_SUFFIXES
    if not is_reviewable_text_path(resolved) and not is_binary:
        raise ValueError(f"unsupported review input type: {path}")
    return resolved


def selected_source_paths(candidates: list[Path]) -> list[Path]:
    """Resolve explicit review inputs while keeping them inside this repository."""
    selected: list[Path] = []
    for candidate in candidates:
        resolved = validate_repository_file(candidate)
        if resolved not in selected:
            selected.append(resolved)
    return selected


def resolve_entry_path(candidate: Path) -> Path:
    path = candidate.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(KNOWLEDGE_BASE_DIR.resolve())
    except ValueError as exc:
        raise ValueError(
            f"--entry must be under {KNOWLEDGE_BASE_DIR}: {candidate}"
        ) from exc
    if path.is_dir():
        choices = [path / name for name in ("index.md", "_index.md") if (path / name).is_file()]
        if len(choices) != 1:
            raise ValueError(
                f"entry directory must contain exactly one index.md or _index.md: {path}"
            )
        path = choices[0]
    if path.suffix.lower() != ".md":
        raise ValueError(f"--entry must select a Markdown page: {candidate}")
    return validate_repository_file(path)


def infer_entry_kind(entry: Path | None, legacy_profile: str) -> str:
    if entry is not None:
        relative = entry.relative_to(KNOWLEDGE_BASE_DIR)
        if relative.parts and relative.parts[0] == "glossary":
            inferred = "glossary"
        elif relative.parts and relative.parts[0] == "deep-dives":
            inferred = "deep-dive"
        else:
            inferred = "knowledge-base"
        legacy_kind = (
            "deep-dive" if legacy_profile in {"deep-dive", "pca-deep-dive"} else legacy_profile
        )
        if legacy_kind != "knowledge-base" and legacy_kind != inferred:
            raise ValueError(
                f"--review-profile {legacy_profile} conflicts with entry kind {inferred}"
            )
        return inferred
    if legacy_profile in {"deep-dive", "pca-deep-dive"}:
        return "deep-dive"
    if legacy_profile == "glossary":
        return "glossary"
    return "knowledge-base"


def entry_title(entry: Path | None) -> str:
    if entry is None:
        return "explicit Knowledge Base artifact bundle"
    source = entry.read_text(encoding="utf-8")
    match = TITLE_RE.search(source)
    return match.group(1).strip() if match else entry.stem


def local_asset_references(markdown_path: Path) -> list[str]:
    source = markdown_path.read_text(encoding="utf-8")
    references: list[str] = []
    for pattern in (
        SRC_ATTRIBUTE_RE,
        HREF_ATTRIBUTE_RE,
        MARKDOWN_LINK_RE,
        OG_IMAGE_RE,
    ):
        for match in pattern.finditer(source):
            raw = match.group(1).strip().strip("<>")
            parsed = urlsplit(raw)
            if parsed.scheme or parsed.netloc or raw.startswith(("#", "data:")):
                continue
            referenced_path = Path(unquote(parsed.path))
            suffix = referenced_path.suffix.lower()
            if (
                suffix not in REVIEWABLE_TEXT_SUFFIXES | BINARY_ASSET_SUFFIXES
                and referenced_path.name not in REVIEWABLE_TEXT_NAMES
            ):
                continue
            if raw not in references:
                references.append(raw)
    return references


def resolve_local_asset(markdown_path: Path, raw_reference: str) -> Path:
    parsed = urlsplit(raw_reference)
    raw_path = unquote(parsed.path)
    normalized = raw_path.removeprefix("/blog/").lstrip("/")
    candidates: list[Path] = []
    if not raw_path.startswith("/") and not raw_path.startswith("knowledge-base/"):
        candidates.append(markdown_path.parent / raw_path)
    if normalized:
        candidates.extend((CONTENT_ROOT / normalized, STATIC_ROOT / normalized))

    matches: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(
                f"local asset reference escapes the repository: {raw_reference}"
            ) from exc
        if resolved.is_file() and resolved not in matches:
            matches.append(resolved)
    if not matches:
        raise ValueError(
            f"unresolved local asset in {markdown_path.relative_to(ROOT)}: "
            f"{raw_reference}"
        )
    if len(matches) > 1:
        choices = ", ".join(path.relative_to(ROOT).as_posix() for path in matches)
        raise ValueError(
            f"ambiguous local asset in {markdown_path.relative_to(ROOT)}: "
            f"{raw_reference} -> {choices}"
        )
    return validate_repository_file(matches[0], allow_binary=True)


def belongs_to_child_bundle(path: Path, bundle_root: Path) -> bool:
    parent = path.parent
    while parent != bundle_root:
        if (parent / "index.md").is_file() or (parent / "_index.md").is_file():
            return True
        if parent == parent.parent:
            break
        parent = parent.parent
    return False


def artifact_note(
    path: Path,
    reason: str,
    referrer: Path | None = None,
) -> ArtifactNote:
    return ArtifactNote(path=path, reason=reason, referrer=referrer)


def discover_entry_artifacts(
    entry: Path,
    max_artifact_bytes: int,
) -> tuple[list[Path], list[ArtifactNote]]:
    paths: list[Path] = [entry]
    notes: list[ArtifactNote] = []
    note_paths: set[Path] = set()

    def consider(path: Path, referrer: Path | None = None) -> None:
        if path.name in INTERNAL_GUIDANCE_NAMES or any(
            part.startswith(".") for part in path.relative_to(ROOT).parts
        ):
            return
        suffix = path.suffix.lower()
        if not is_reviewable_text_path(path) and suffix not in BINARY_ASSET_SUFFIXES:
            return
        path = validate_repository_file(path, allow_binary=True)
        if suffix in BINARY_ASSET_SUFFIXES:
            if path not in note_paths:
                notes.append(
                    artifact_note(
                        path,
                        "binary asset listed for claim/provenance review; pixels not supplied",
                        referrer,
                    )
                )
                note_paths.add(path)
            return
        if is_lockfile(path):
            if path not in note_paths:
                notes.append(artifact_note(path, "lockfile represented by hash only", referrer))
                note_paths.add(path)
            return
        size = review_payload_size(path)
        if size > max_artifact_bytes:
            if is_core_text_path(path):
                raise ValueError(
                    f"core review artifact exceeds --max-artifact-bytes: "
                    f"{path.relative_to(ROOT)} ({size} bytes)"
                )
            if path not in note_paths:
                notes.append(
                    artifact_note(
                        path,
                        "oversized data artifact represented by hash only",
                        referrer,
                    )
                )
                note_paths.add(path)
            return
        if path not in paths:
            paths.append(path)

    if entry.name == "index.md":
        for candidate in sorted(entry.parent.rglob("*")):
            if not candidate.is_file() or candidate == entry:
                continue
            if belongs_to_child_bundle(candidate, entry.parent):
                continue
            consider(candidate)

    scanned_markdown: set[Path] = set()
    while True:
        markdown_paths = [
            path for path in paths if path.suffix.lower() == ".md" and path not in scanned_markdown
        ]
        if not markdown_paths:
            break
        for markdown_path in markdown_paths:
            scanned_markdown.add(markdown_path)
            for raw_reference in local_asset_references(markdown_path):
                consider(resolve_local_asset(markdown_path, raw_reference), markdown_path)

    ordered = [entry] + sorted(
        (path for path in paths if path != entry),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    notes.sort(key=lambda note: note.path.relative_to(ROOT).as_posix())
    return ordered, notes


def augment_explicit_artifacts(
    paths: list[Path],
    max_artifact_bytes: int,
) -> tuple[list[Path], list[ArtifactNote]]:
    augmented: list[Path] = []
    notes: list[ArtifactNote] = []
    noted: set[Path] = set()
    scanned_markdown: set[Path] = set()

    def consider(asset: Path, referrer: Path | None = None) -> None:
        asset = validate_repository_file(asset, allow_binary=True)
        suffix = asset.suffix.lower()
        if suffix in BINARY_ASSET_SUFFIXES:
            if asset not in noted:
                notes.append(
                    artifact_note(
                        asset,
                        "binary asset listed for claim/provenance review; pixels not supplied",
                        referrer,
                    )
                )
                noted.add(asset)
            return
        if is_lockfile(asset):
            if asset not in noted:
                notes.append(
                    artifact_note(asset, "lockfile represented by hash only", referrer)
                )
                noted.add(asset)
            return
        size = review_payload_size(asset)
        if size > max_artifact_bytes:
            if is_core_text_path(asset):
                raise ValueError(
                    f"core review artifact exceeds --max-artifact-bytes: "
                    f"{asset.relative_to(ROOT)} ({size} bytes)"
                )
            if asset not in noted:
                notes.append(
                    artifact_note(
                        asset,
                        "oversized data artifact represented by hash only",
                        referrer,
                    )
                )
                noted.add(asset)
            return
        if asset not in augmented:
            augmented.append(asset)

    for path in paths:
        consider(path)

    while True:
        markdown_paths = [
            path
            for path in augmented
            if path.suffix.lower() == ".md" and path not in scanned_markdown
        ]
        if not markdown_paths:
            break
        for markdown_path in markdown_paths:
            scanned_markdown.add(markdown_path)
            for raw_reference in local_asset_references(markdown_path):
                consider(resolve_local_asset(markdown_path, raw_reference), markdown_path)

    return augmented, sorted(
        notes,
        key=lambda note: note.path.relative_to(ROOT).as_posix(),
    )


def sanitized_notebook_source(path: Path) -> str:
    """Return reviewable cell source without notebook outputs or metadata."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4:
        raise ValueError(f"notebook must use nbformat 4: {path}")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"notebook cells must be a list: {path}")

    output_count = 0
    attachment_count = 0
    chunks = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValueError(f"notebook cell {index} must be an object: {path}")
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            raise ValueError(
                f"unsupported notebook cell type {cell_type!r} at {index}: {path}"
            )
        outputs = cell.get("outputs", [])
        attachments = cell.get("attachments", {})
        output_count += len(outputs) if isinstance(outputs, list) else 0
        attachment_count += len(attachments) if isinstance(attachments, dict) else 0
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source_text = "".join(source)
        elif isinstance(source, str):
            source_text = source
        else:
            raise ValueError(f"invalid source for notebook cell {index}: {path}")
        chunks.extend(
            (
                f"\n===== NOTEBOOK CELL {index} ({cell_type}) =====",
                source_text,
            )
        )
    preamble = (
        "Notebook cell source only. Metadata, execution state, "
        f"{output_count} output item(s), and {attachment_count} attachment(s) "
        "were excluded from this review packet. "
        f"Original sha256={sha256_file(path)}."
    )
    return "\n".join((preamble, *chunks))


def review_source(path: Path) -> str:
    if path.suffix.lower() == ".ipynb":
        source = sanitized_notebook_source(path)
    else:
        source = path.read_text(encoding="utf-8")
    if contains_inline_raster(source):
        raise ValueError(
            f"inline raster bytes are not allowed in a text review artifact: "
            f"{path.relative_to(ROOT)}"
        )
    if contains_possible_credential(source):
        raise ValueError(
            f"possible credential material in review artifact: "
            f"{path.relative_to(ROOT)}"
        )
    return source


def contains_inline_raster(source: str) -> bool:
    return INLINE_RASTER_DATA_RE.search(source) is not None


def contains_possible_credential(source: str) -> bool:
    if PRIVATE_KEY_RE.search(source):
        return True
    for match in OPENAI_KEY_RE.finditer(source):
        normalized = match.group(0).upper()
        if not any(marker in normalized for marker in PLACEHOLDER_MARKERS):
            return True
    if any(pattern.search(source) for pattern in HIGH_CONFIDENCE_TOKEN_RES):
        return True
    for match in SECRET_ASSIGNMENT_RE.finditer(source):
        normalized_name = re.sub(r"[^a-z0-9]", "", match.group("name").lower())
        if not normalized_name.endswith(
            (
                "apikey",
                "accesstoken",
                "authkey",
                "authtoken",
                "clientsecret",
                "githubtoken",
                "password",
                "passwd",
                "privatekey",
                "secretaccesskey",
                "token",
            )
        ):
            continue
        normalized = match.group("value").upper()
        if any(marker in normalized for marker in PLACEHOLDER_MARKERS):
            continue
        value = match.group("value")
        character_classes = sum(
            (
                any(character.islower() for character in value),
                any(character.isupper() for character in value),
                any(character.isdigit() for character in value),
                any(not character.isalnum() for character in value),
            )
        )
        if character_classes >= 3:
            return True
        if (
            len(value) >= 24
            and any(character.isalpha() for character in value)
            and any(character.isdigit() for character in value)
            and len(set(value)) / len(value) >= 0.35
        ):
            return True
    return False


def build_bundle(
    paths: list[Path],
    notes: list[ArtifactNote],
    *,
    kind: str,
    title: str,
) -> str:
    chunks = [
        f"Selected entry kind: {kind}",
        f"Selected entry title: {title}",
        "Review all artifacts below. Repository-relative filenames are authoritative.",
    ]
    if notes:
        chunks.append(
            "\n===== METADATA-ONLY OR OMITTED ARTIFACTS =====\n"
            + "\n".join(note.packet_line() for note in notes)
            + "\nDo not claim to have inspected omitted bytes or raster pixels."
        )
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        chunks.extend(
            (
                f"\n===== BEGIN FILE: {relative} =====",
                review_source(path),
                f"===== END FILE: {relative} =====",
            )
        )
    return "\n".join(chunks)


def merge_artifact_notes(*groups: list[ArtifactNote]) -> list[ArtifactNote]:
    """Deduplicate metadata-only artifacts while preserving a useful referrer."""
    merged: dict[Path, ArtifactNote] = {}
    for group in groups:
        for note in group:
            existing = merged.get(note.path)
            if existing is None or (existing.referrer is None and note.referrer is not None):
                merged[note.path] = note
    return sorted(
        merged.values(),
        key=lambda note: note.path.relative_to(ROOT).as_posix(),
    )


def review_instructions_for_kind(kind: str, review_round: int) -> str:
    instructions = KNOWLEDGE_BASE_REVIEW_INSTRUCTIONS
    if kind == "glossary":
        instructions += "\n\n" + GLOSSARY_REVIEW_ADDENDUM
    elif kind == "deep-dive":
        instructions += "\n\n" + DEEP_DIVE_REVIEW_ADDENDUM
    if review_round > 1:
        instructions += "\n\n" + FOLLOWUP_REVIEW_INSTRUCTIONS.format(
            round_number=review_round
        )
    return instructions


def default_entry_output_path(
    entry: Path,
    kind: str,
    review_round: int,
    *,
    continuation: bool,
) -> Path:
    entry_name = entry.parent.name if entry.name in {"index.md", "_index.md"} else entry.stem
    safe_name = re.sub(r"[^a-z0-9]+", "-", entry_name.lower()).strip("-") or "entry"
    safe_kind = re.sub(r"[^a-z0-9]+", "-", kind.lower()).strip("-")
    suffix = "-continued" if continuation else ""
    return PRIVATE_OUTPUT_DIR / f"{safe_kind}-{safe_name}-round{review_round}{suffix}.json"


def validate_review_json(review_text: str) -> None:
    """Reject a visible response that does not satisfy the promised basic schema."""
    payload = json.loads(review_text)
    if not isinstance(payload, dict):
        raise ValueError("review output must be a JSON object")
    if payload.get("verdict") not in {"pass", "revise"}:
        raise ValueError("review output verdict must be 'pass' or 'revise'")
    if not isinstance(payload.get("summary"), str):
        raise ValueError("review output summary must be a string")
    findings = payload.get("findings")
    if not isinstance(findings, list):
        raise ValueError("review output findings must be an array")
    if not isinstance(payload.get("cross_artifact_conflicts"), list):
        raise ValueError("review output cross_artifact_conflicts must be an array")
    if not isinstance(payload.get("checks_that_passed"), list):
        raise ValueError("review output checks_that_passed must be an array")


def extract_output_text(response: dict[str, object]) -> str:
    texts: list[str] = []
    output = response.get("output", [])
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                texts.append(part["text"])
    return "\n".join(texts)


def read_dotenv_api_key(path: Path) -> str | None:
    """Read only the accepted OpenAI key names from a simple dotenv file."""
    if not path.exists():
        return None
    if path.stat().st_mode & 0o077:
        raise ValueError(f"{path} must not be readable by group or others; run chmod 600")

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        key_name = name.strip()
        if separator and key_name in API_KEY_NAMES:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if not value:
                raise ValueError(f"{key_name} is empty in {path}")
            values[key_name] = value
    return next((values[name] for name in API_KEY_NAMES if name in values), None)


def resolve_private_path(path: Path) -> Path:
    """Confine API response artifacts to the ignored private cache."""
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    private_root = PRIVATE_OUTPUT_DIR.resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise ValueError(
            f"review artifacts must stay under ignored directory {private_root}"
        ) from exc
    return resolved


def write_private_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.chmod(path, 0o600)


def review_context_sha256(bundle: str, instructions: str) -> str:
    digest = hashlib.sha256()
    digest.update(bundle.encode("utf-8"))
    digest.update(b"\0")
    digest.update(instructions.encode("utf-8"))
    return digest.hexdigest()


def validate_prior_binding(
    prior: dict[str, object], expected_context_sha256: str
) -> None:
    metadata = prior.get(REVIEW_METADATA_KEY)
    if not isinstance(metadata, dict):
        raise ValueError(
            "prior response predates review-bundle binding; start a fresh review "
            "instead of replaying it against an unverified entry"
        )
    actual = metadata.get("context_sha256")
    if actual != expected_context_sha256:
        raise ValueError(
            "prior response was created for different entry content or review "
            "instructions; repeat the original selection without changes"
        )


def load_prior_response(
    path: Path, expected_context_sha256: str
) -> dict[str, object]:
    prior_path = resolve_private_path(path)
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if not isinstance(prior, dict):
        raise ValueError(f"{prior_path} must contain a response object")
    validate_prior_binding(prior, expected_context_sha256)
    output = prior.get("output")
    if not isinstance(output, list) or not output:
        raise ValueError(f"{prior_path} contains no response output items")
    if not any(
        isinstance(item, dict)
        and item.get("type") == "reasoning"
        and item.get("encrypted_content")
        for item in output
    ):
        raise ValueError(f"{prior_path} contains no replayable encrypted reasoning")
    return prior


def build_request_input(
    bundle: str, prior: dict[str, object] | None
) -> str | list[object]:
    if prior is None:
        return bundle
    return [
        {"role": "user", "content": bundle},
        *prior["output"],
        {"role": "user", "content": CONTINUATION_INSTRUCTION},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-profile",
        choices=("knowledge-base", "glossary", "deep-dive", "pca-deep-dive"),
        default="knowledge-base",
        help=(
            "Generic rubric and optional entry-kind override; pca-deep-dive is "
            "a deprecated compatibility alias (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--entry",
        type=Path,
        help=(
            "Knowledge Base Markdown page or leaf-bundle directory; discovers "
            "its safe review artifacts automatically"
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        action="append",
        default=[],
        help="Repository-local artifact to review; repeat for a connected bundle",
    )
    parser.add_argument(
        "--max-artifact-bytes",
        type=int,
        default=DEFAULT_MAX_ARTIFACT_BYTES,
        help=(
            "Maximum auto-discovered text artifact size before hash-only "
            "treatment or a core-artifact error (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-bundle-bytes",
        type=int,
        default=DEFAULT_MAX_BUNDLE_BYTES,
        help="Maximum assembled text packet size (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Private-cache path for raw Responses API JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Replay encrypted reasoning from an earlier stateless response",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="high",
        help="Reasoning effort inside Pro mode (default: %(default)s)",
    )
    parser.add_argument(
        "--review-round",
        type=int,
        default=1,
        help="Review round number; rounds after 1 use a regression-focused rubric",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=48_000,
        help="Shared allowance for reasoning and visible output (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and summarize the review bundle without making an API call",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.review_round < 1:
        print("--review-round must be at least 1.", file=sys.stderr)
        return 1
    if args.max_artifact_bytes < 1:
        print("--max-artifact-bytes must be at least 1.", file=sys.stderr)
        return 1
    if args.max_bundle_bytes < 1:
        print("--max-bundle-bytes must be at least 1.", file=sys.stderr)
        return 1
    if args.max_output_tokens < 25_000:
        print(
            "--max-output-tokens must be at least 25000 for a reasoning review.",
            file=sys.stderr,
        )
        return 1

    if args.resume_from and not (args.entry or args.path):
        print(
            "--resume-from requires the original --entry or --path selection so "
            "the continuation cannot review a different default bundle.",
            file=sys.stderr,
        )
        return 1

    if args.review_profile == "pca-deep-dive":
        print(
            "Note: --review-profile pca-deep-dive is deprecated; it now uses the "
            "generic Deep Dive review.",
            file=sys.stderr,
        )

    try:
        entry = resolve_entry_path(args.entry) if args.entry else None
        kind = infer_entry_kind(entry, args.review_profile)
        if entry is None and not args.path and kind == "deep-dive":
            raise ValueError(
                f"--review-profile {args.review_profile} requires --entry or --path"
            )

        if entry is not None:
            paths, notes = discover_entry_artifacts(entry, args.max_artifact_bytes)
            if args.path:
                explicit_paths = selected_source_paths(args.path)
                combined = list(dict.fromkeys([*paths, *explicit_paths]))
                paths, explicit_notes = augment_explicit_artifacts(
                    combined, args.max_artifact_bytes
                )
                notes = merge_artifact_notes(notes, explicit_notes)
            title = entry_title(entry)
        elif args.path:
            explicit_paths = selected_source_paths(args.path)
            paths, notes = augment_explicit_artifacts(
                explicit_paths, args.max_artifact_bytes
            )
            title = "explicit Knowledge Base artifact bundle"
        else:
            # Preserve the original no-argument all-glossary review workflow.
            kind = "glossary"
            title = "Knowledge Base glossary collection"
            paths, notes = augment_explicit_artifacts(
                glossary_source_paths(), args.max_artifact_bytes
            )

        if not paths:
            raise ValueError("no review artifacts found")
        bundle = build_bundle(paths, notes, kind=kind, title=title)
        bundle_bytes = len(bundle.encode("utf-8"))
        if bundle_bytes > args.max_bundle_bytes:
            raise ValueError(
                f"assembled review bundle is {bundle_bytes:,} bytes, above "
                f"--max-bundle-bytes={args.max_bundle_bytes:,}; narrow the explicit "
                "packet or raise the reviewed limit intentionally"
            )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Invalid review input: {exc}", file=sys.stderr)
        return 1

    review_instructions = review_instructions_for_kind(kind, args.review_round)
    bundle_sha256 = hashlib.sha256(bundle.encode("utf-8")).hexdigest()
    context_sha256 = review_context_sha256(bundle, review_instructions)

    try:
        prior = (
            load_prior_response(args.resume_from, context_sha256)
            if args.resume_from
            else None
        )
        output_candidate = args.output
        if output_candidate == DEFAULT_OUTPUT and entry is not None:
            output_candidate = default_entry_output_path(
                entry,
                kind,
                args.review_round,
                continuation=prior is not None,
            )
        elif output_candidate == DEFAULT_OUTPUT and prior is not None:
            output_candidate = DEFAULT_CONTINUATION_OUTPUT
        output_path = resolve_private_path(output_candidate)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Invalid private review artifact: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        counts = {
            suffix: sum(path.suffix == suffix for path in paths)
            for suffix in (
                ".md",
                ".svg",
                ".py",
                ".ipynb",
                ".json",
                ".lock",
                ".csv",
                ".jsonl",
                ".tsv",
                ".txt",
            )
        }
        print(
            f"Review bundle ready for {kind} entry {title!r}: "
            f"{counts['.md']} Markdown, {counts['.svg']} SVG, "
            f"{counts['.py']} Python, {counts['.ipynb']} notebook, "
            f"{counts['.json']} JSON, "
            f"{counts['.lock']} lock, {counts['.csv']} CSV, "
            f"{counts['.jsonl']} JSONL, {counts['.tsv']} TSV, "
            f"{counts['.txt']} text, {bundle_bytes:,} bytes."
        )
        print("Artifacts:")
        for path in paths:
            print(f"  {path.relative_to(ROOT).as_posix()}")
        if notes:
            print("Metadata-only artifacts:")
            for note in notes:
                print(f"  {note.packet_line()[2:]}")
        continuation = " with encrypted-reasoning continuation" if prior else ""
        print(
            "Target: gpt-5.6-sol, reasoning mode pro, "
            f"effort {args.reasoning_effort}, max output {args.max_output_tokens}, "
            f"store false, review round {args.review_round}{continuation}."
        )
        print(f"Private output: {output_path}")
        return 0

    try:
        api_key = next(
            (os.environ[name] for name in API_KEY_NAMES if os.environ.get(name)),
            None,
        ) or read_dotenv_api_key(DOTENV_PATH)
    except (OSError, ValueError) as exc:
        print(f"Cannot load API credential safely: {exc}", file=sys.stderr)
        return 2
    if not api_key:
        print(
            "OPENAI_API_KEY/OPENAI_KEY is not set in the environment or root .env file; "
            "do not paste the key into chat or commit it.",
            file=sys.stderr,
        )
        return 2

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    request_body = {
        "model": "gpt-5.6-sol",
        "reasoning": {
            "mode": "pro",
            "effort": args.reasoning_effort,
            "context": "all_turns" if prior is not None else "current_turn",
        },
        "instructions": review_instructions,
        "input": build_request_input(bundle, prior),
        "max_output_tokens": args.max_output_tokens,
        "store": False,
    }
    request = Request(
        f"{base_url.rstrip('/')}/responses",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=1800) as result:
            response = json.loads(result.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"OpenAI API returned HTTP {exc.code}: {detail[:4000]}", file=sys.stderr)
        return 3
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Review request failed: {exc}", file=sys.stderr)
        return 3

    if not isinstance(response, dict):
        print("Review request returned a non-object JSON response.", file=sys.stderr)
        return 3

    response[REVIEW_METADATA_KEY] = {
        "schema_version": 1,
        "model": "gpt-5.6-sol",
        "kind": kind,
        "entry": entry.relative_to(ROOT).as_posix() if entry is not None else None,
        "review_round": args.review_round,
        "bundle_sha256": bundle_sha256,
        "instructions_sha256": hashlib.sha256(
            review_instructions.encode("utf-8")
        ).hexdigest(),
        "context_sha256": context_sha256,
        "artifacts": [path.relative_to(ROOT).as_posix() for path in paths],
        "metadata_only_artifacts": [
            note.path.relative_to(ROOT).as_posix() for note in notes
        ],
    }

    try:
        write_private_json(output_path, response)
    except OSError as exc:
        print(f"Could not save the private review response: {exc}", file=sys.stderr)
        return 4

    review_text = extract_output_text(response)
    if response.get("status") == "incomplete":
        details = response.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else "unknown"
        print(f"The review response is incomplete ({reason}).", file=sys.stderr)
        if review_text:
            print(review_text)
        print(f"Raw response saved privately to {output_path}", file=sys.stderr)
        return 4
    if review_text:
        try:
            validate_review_json(review_text)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"The visible review is not valid review JSON: {exc}", file=sys.stderr)
            print(f"Raw response saved privately to {output_path}", file=sys.stderr)
            return 4
        print(review_text)
    else:
        print("The response contained no output_text item.", file=sys.stderr)
        print(f"Raw response saved privately to {output_path}", file=sys.stderr)
        return 4
    print(f"\nRaw response saved privately to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
