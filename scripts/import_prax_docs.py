#!/usr/bin/env python3
"""Import a curated, commit-pinned snapshot of the Prax documentation into Hugo.

The importer deliberately treats the upstream checkout as data.  It never imports
Python modules, invokes project commands, or asks Hugo to interpret upstream front
matter.  Only a fixed set of Markdown paths and README-referenced root assets are
published.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit, urlunsplit


DEFAULT_REPOSITORY = "https://github.com/praxagent/prax.git"
DEFAULT_REF = "main"
DEFAULT_CACHE = Path(".cache/prax-docs/repo")
PUBLIC_ROOT = "/blog/knowledge-base/prax/"
LEGACY_ALIAS_ROOT = "/references/prax/"
STATIC_PUBLIC_ROOT = "/blog/prax-docs/"
MANIFEST_NAME = "prax-docs-manifest.json"

# docs/research is intentionally absent: it is embryonic material and is not
# part of the public Prax documentation collection.
CURATED_SECTIONS = (
    "agents",
    "architecture",
    "guides",
    "infrastructure",
    "security",
)

ROOT_CARD_METADATA: tuple[tuple[str, object], ...] = (
    ("weight", 30),
    ("card_index", "03"),
    ("card_label", "Documentation"),
    ("card_title", "Prax Harness"),
    (
        "card_summary",
        "Documentation and practical guides for praxagent's flagship open-source agent harness.",
    ),
    ("card_action", "Read the docs"),
    ("title_lead", "Prax"),
    ("title_accent", "Harness"),
)

H1_RE = re.compile(r"(?m)^ {0,3}#(?!#)[ \t]+(.+?)[ \t]*#*[ \t]*$")
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
INLINE_LINK_RE = re.compile(
    r"(?P<open>!?\[(?:\\.|[^\]])*\]\()"
    r"(?P<destination><[^>\n]+>|[^\s)\n]+)"
    r"(?P<title>[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^\)\n]*\)))?"
    r"(?P<close>\))"
)
REFERENCE_LINK_RE = re.compile(
    r"(?P<open>^ {0,3}\[[^\]\n]+\]:[ \t]*)"
    r"(?P<destination><[^>\n]+>|\S+)"
    r"(?P<rest>.*)$"
)
HTML_TAG_RE = re.compile(
    r"<\s*/?\s*(?P<name>[A-Za-z][A-Za-z0-9:-]*)\b(?P<attrs>[^<>]*)>", re.DOTALL
)
EVENT_ATTRIBUTE_RE = re.compile(r"(?i)(?:^|\s)on[a-z0-9_-]+\s*=")
DANGEROUS_PROTOCOL_RE = re.compile(
    r"(?i)^\s*(?:(?:javascript|vbscript)\s*:|data\s*:\s*text/html(?:[;,]|$))"
)
SHORTCODE_RE = re.compile(r"{{\s*[<%]")
# The public brand is always lowercase. This qualified Python class name is an
# executable API identifier, not a spelling of the brand, so preserve it.
MIXED_CASE_BRAND_RE = re.compile(r"(?<!prax\.eval\.tb_agent:)PraxAgent")

# Goldmark is configured with unsafe HTML enabled.  Reject real HTML tags after
# sanitizing the one known README wrapper/image; angle-bracket placeholders such
# as <hostname> and Markdown autolinks remain ordinary documentation.
KNOWN_HTML_TAGS = frozenset(
    {
        "a",
        "abbr",
        "address",
        "area",
        "article",
        "aside",
        "audio",
        "b",
        "base",
        "bdi",
        "bdo",
        "blockquote",
        "body",
        "br",
        "button",
        "canvas",
        "caption",
        "cite",
        "code",
        "col",
        "colgroup",
        "data",
        "datalist",
        "dd",
        "del",
        "details",
        "dfn",
        "dialog",
        "div",
        "dl",
        "dt",
        "em",
        "embed",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "head",
        "header",
        "hgroup",
        "hr",
        "html",
        "i",
        "iframe",
        "img",
        "input",
        "ins",
        "kbd",
        "label",
        "legend",
        "li",
        "link",
        "main",
        "map",
        "mark",
        "menu",
        "meta",
        "meter",
        "nav",
        "noscript",
        "object",
        "ol",
        "optgroup",
        "option",
        "output",
        "p",
        "picture",
        "pre",
        "progress",
        "q",
        "rp",
        "rt",
        "ruby",
        "s",
        "samp",
        "script",
        "search",
        "section",
        "select",
        "slot",
        "small",
        "source",
        "span",
        "strong",
        "style",
        "sub",
        "summary",
        "sup",
        "table",
        "tbody",
        "td",
        "template",
        "textarea",
        "tfoot",
        "th",
        "thead",
        "time",
        "title",
        "tr",
        "track",
        "u",
        "ul",
        "var",
        "video",
        "wbr",
    }
)


class ImportFailure(RuntimeError):
    """Raised when an upstream snapshot cannot be imported safely."""


@dataclass(frozen=True)
class SourceDocument:
    source_path: PurePosixPath
    output_path: PurePosixPath
    is_section: bool


@dataclass(frozen=True)
class Provenance:
    repository: str
    ref: str
    commit: str


def _run_git(checkout: Path, *arguments: str) -> str:
    command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "submodule.recurse=false",
        "-C",
        str(checkout),
        *arguments,
    ]
    environment = os.environ.copy()
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise ImportFailure(f"git command failed: {' '.join(command)}\n{detail.strip()}") from error
    return result.stdout.strip()


def acquire_checkout(repository: str, ref: str, cache_dir: Path) -> Path:
    """Fetch *ref* into a shallow, blobless, sparse, hook-disabled checkout."""

    cache_dir = cache_dir.expanduser().resolve()
    if cache_dir.exists() and not (cache_dir / ".git").is_dir():
        raise ImportFailure(f"cache exists but is not a Git checkout: {cache_dir}")

    if not cache_dir.exists():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "clone",
            "--filter=blob:none",
            "--depth=1",
            "--sparse",
            "--no-checkout",
            "--no-tags",
            repository,
            str(cache_dir),
        ]
        environment = os.environ.copy()
        environment["GIT_LFS_SKIP_SMUDGE"] = "1"
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=environment,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as error:
            # A failed clone may leave a partial directory.  It is exclusively
            # importer-owned because it did not exist on entry.
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            detail = getattr(error, "stderr", "") or str(error)
            raise ImportFailure(f"unable to clone {repository}: {detail.strip()}") from error
    else:
        _run_git(cache_dir, "remote", "set-url", "origin", repository)

    _run_git(cache_dir, "sparse-checkout", "init", "--cone")
    # Cone mode includes root files automatically.  These two directories are
    # the only non-root trees the importer may inspect.
    _run_git(cache_dir, "sparse-checkout", "set", "docs", "assets")
    _run_git(cache_dir, "fetch", "--depth=1", "--filter=blob:none", "--no-tags", "origin", ref)
    _run_git(cache_dir, "checkout", "--detach", "--no-recurse-submodules", "FETCH_HEAD")
    return cache_dir


def _repository_web_url(repository: str) -> str:
    repository = repository.strip()
    if repository.startswith("git@github.com:"):
        repository = "https://github.com/" + repository.removeprefix("git@github.com:")
    elif repository.startswith("ssh://git@github.com/"):
        repository = "https://github.com/" + repository.removeprefix("ssh://git@github.com/")
    return repository.removesuffix(".git").rstrip("/")


def _checkout_provenance(source_root: Path, repository_fallback: str, ref_fallback: str) -> Provenance:
    if not (source_root / ".git").exists():
        # Worktrees use a .git file, regular repositories use a directory.
        try:
            _run_git(source_root, "rev-parse", "--is-inside-work-tree")
        except ImportFailure as error:
            raise ImportFailure(f"local source must be a committed Git checkout: {source_root}") from error

    commit = _run_git(source_root, "rev-parse", "HEAD")
    try:
        repository = _run_git(source_root, "remote", "get-url", "origin")
    except ImportFailure:
        repository = repository_fallback
    try:
        ref = _run_git(source_root, "symbolic-ref", "--quiet", "--short", "HEAD")
    except ImportFailure:
        ref = ref_fallback or commit

    relevant_paths = ["README.md", "docs/library.md", "assets", *(f"docs/{name}" for name in CURATED_SECTIONS)]
    dirty = _run_git(source_root, "status", "--porcelain", "--untracked-files=all", "--", *relevant_paths)
    if dirty:
        raise ImportFailure(
            "refusing to attach commit provenance to modified or untracked imported files:\n" + dirty
        )
    return Provenance(_repository_web_url(repository), ref, commit)


def discover_documents(source_root: Path) -> list[SourceDocument]:
    documents = [SourceDocument(PurePosixPath("README.md"), PurePosixPath("_index.md"), True)]

    library = source_root / "docs" / "library.md"
    if library.is_file():
        documents.append(
            SourceDocument(PurePosixPath("docs/library.md"), PurePosixPath("library.md"), False)
        )

    for section in CURATED_SECTIONS:
        section_root = source_root / "docs" / section
        if not section_root.is_dir():
            continue
        for path in sorted(section_root.rglob("*.md")):
            if path.is_symlink():
                raise ImportFailure(f"refusing to import a symbolic link: {path}")
            relative = path.relative_to(source_root).as_posix()
            within_section = path.relative_to(section_root)
            if within_section.name.casefold() == "readme.md":
                output = PurePosixPath(section, *within_section.parent.parts, "_index.md")
                is_section = True
            else:
                output = PurePosixPath(section, *within_section.parts)
                is_section = False
            documents.append(SourceDocument(PurePosixPath(relative), output, is_section))

    missing = [str(document.source_path) for document in documents if not (source_root / document.source_path).is_file()]
    if missing:
        raise ImportFailure("required Prax documentation is missing: " + ", ".join(missing))

    outputs: dict[PurePosixPath, PurePosixPath] = {}
    for document in documents:
        previous = outputs.setdefault(document.output_path, document.source_path)
        if previous != document.source_path:
            raise ImportFailure(
                f"output collision: {previous} and {document.source_path} both map to {document.output_path}"
            )
    return documents


def _without_code(text: str, *, preserve_inline: bool = False) -> str:
    """Remove fenced code; optionally retain inline terms for prose summaries."""

    result: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    for line in text.splitlines(keepends=True):
        fence = FENCE_RE.match(line)
        if fence_character is not None:
            result.append("\n" if line.endswith("\n") else "")
            if fence and fence.group(1)[0] == fence_character and len(fence.group(1)) >= fence_length:
                fence_character = None
            continue
        if fence:
            fence_character = fence.group(1)[0]
            fence_length = len(fence.group(1))
            result.append("\n" if line.endswith("\n") else "")
            continue

        # Markdown code spans cannot cross a line in the upstream corpus.  Keep
        # line length unimportant but preserve newlines for useful diagnostics.
        if not preserve_inline:
            line = re.sub(r"(`+)(?:[^`]|`(?!\1))*?\1", lambda match: " " * len(match.group(0)), line)
        result.append(line)
    return "".join(result)


ROOT_DIV_OPEN_RE = re.compile(r"(?im)^[ \t]*<div\s+align=(['\"]?)center\1\s*>[ \t]*$")
ROOT_DIV_CLOSE_RE = re.compile(r"(?im)^[ \t]*</div\s*>[ \t]*$")
ROOT_IMAGE_RE = re.compile(
    r"(?im)^[ \t]*<img\s+src=(?P<sq>['\"])(?P<src>assets/[A-Za-z0-9._/ -]+)(?P=sq)"
    r"\s+alt=(?P<aq>['\"])(?P<alt>[^'\"<>]*)(?P=aq)\s*/?>[ \t]*$"
)
EXCLUDED_ROOT_SECTION_RE = re.compile(
    r"(?ims)^###\s+[^\n]*\(docs/research/README\.md\)[^\n]*\n"
    r".*?(?=^#{1,3}\s|\Z)"
)


def _sanitize_root_markup(text: str) -> str:
    text = ROOT_DIV_OPEN_RE.sub("", text)
    text = ROOT_DIV_CLOSE_RE.sub("", text)
    text = EXCLUDED_ROOT_SECTION_RE.sub("", text)

    def image_to_markdown(match: re.Match[str]) -> str:
        alt = match.group("alt").replace("[", "\\[").replace("]", "\\]")
        encoded_source = quote(match.group("src"), safe="/-._~")
        return f"![{alt}]({encoded_source})"

    return ROOT_IMAGE_RE.sub(image_to_markdown, text)


def normalize_brand_case(text: str) -> str:
    """Normalize the public brand without changing the Terminal-Bench class path."""
    return MIXED_CASE_BRAND_RE.sub("praxagent", text)


def validate_safe_markdown(text: str, source_path: PurePosixPath) -> None:
    visible = _without_code(text)
    if SHORTCODE_RE.search(visible):
        raise ImportFailure(f"{source_path}: Hugo shortcode syntax is not allowed in imported content")

    for match in HTML_TAG_RE.finditer(visible):
        name = match.group("name").casefold()
        attrs = html.unescape(match.group("attrs"))
        if EVENT_ATTRIBUTE_RE.search(attrs):
            raise ImportFailure(f"{source_path}: event-handler HTML attributes are not allowed")
        if re.search(r"(?i)(?:href|src)\s*=\s*(['\"]?)\s*(?:javascript|vbscript|data\s*:\s*text/html)\s*:", attrs):
            raise ImportFailure(f"{source_path}: executable HTML URLs are not allowed")
        if name in KNOWN_HTML_TAGS:
            raise ImportFailure(f"{source_path}: raw HTML <{name}> is not allowed")


def _clean_inline_markdown(value: str) -> str:
    code_spans: list[str] = []

    def preserve_code(match: re.Match[str]) -> str:
        code_spans.append(match.group(0)[len(match.group(1)):-len(match.group(1))].strip())
        return f"\x00{len(code_spans) - 1}\x00"

    # Remove formatting without destroying underscores or literal Markdown
    # characters in filenames, settings, and commands.
    value = re.sub(r"(`+)(?:[^`]|`(?!\1))*?\1", preserve_code, value)
    value = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"[`*_~]", "", value)
    value = html.unescape(value)
    for index, code in enumerate(code_spans):
        value = value.replace(f"\x00{index}\x00", code)
    return " ".join(value.split()).strip()


def extract_title_and_body(text: str, source_path: PurePosixPath) -> tuple[str, str]:
    match = H1_RE.search(text)
    if match is None:
        raise ImportFailure(f"{source_path}: expected a level-one Markdown heading")
    title = _clean_inline_markdown(match.group(1))
    if not title:
        raise ImportFailure(f"{source_path}: level-one heading has no usable title")
    body = text[: match.start()] + text[match.end() :]
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    return title, body


def derive_summary(body: str, title: str) -> str:
    visible = _without_code(body, preserve_inline=True)
    paragraphs = re.split(r"\n\s*\n", visible)
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        if any(line.startswith(("#", "|", "- ", "* ", "+ ", ">", "![")) for line in lines):
            continue
        candidate = _clean_inline_markdown(" ".join(lines))
        if candidate and not candidate.startswith(("←", "[←")):
            if len(candidate) > 240:
                candidate = candidate[:237].rsplit(" ", 1)[0] + "…"
            return candidate
    return f"Prax documentation: {title}."


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _route_suffix_for_output(output: PurePosixPath) -> str:
    if output == PurePosixPath("_index.md"):
        return ""
    if output.name == "_index.md":
        relative = output.parent.as_posix().strip("/")
    else:
        relative = output.with_suffix("").as_posix().strip("/")
    return quote(relative, safe="/-._~") + "/"


def _legacy_alias_for_output(output: PurePosixPath) -> str:
    return LEGACY_ALIAS_ROOT + _route_suffix_for_output(output)


def render_front_matter(
    title: str,
    summary: str,
    document: SourceDocument,
    provenance: Provenance,
) -> str:
    fields: list[tuple[str, object]] = [
        ("title", title),
        ("summary", summary),
        ("layout", "prax-docs-section" if document.is_section else "prax-doc"),
        ("aliases", [_legacy_alias_for_output(document.output_path)]),
    ]
    if document.source_path == PurePosixPath("README.md"):
        fields.extend(ROOT_CARD_METADATA)
    source_path = document.source_path.as_posix()
    encoded_path = quote(source_path, safe="/-._~")
    fields.extend(
        (
            ("source_repo", provenance.repository),
            ("source_ref", provenance.ref),
            ("source_commit", provenance.commit),
            ("source_path", source_path),
            ("edit_url", f"{provenance.repository}/edit/{provenance.commit}/{encoded_path}"),
        )
    )
    rendered = ["+++"]
    for key, value in fields:
        if isinstance(value, int):
            rendered.append(f"{key} = {value}")
        elif isinstance(value, (list, tuple)):
            items = ", ".join(_toml_string(str(item)) for item in value)
            rendered.append(f"{key} = [{items}]")
        else:
            rendered.append(f"{key} = {_toml_string(str(value))}")
    rendered.extend(("+++", ""))
    return "\n".join(rendered)


def _route_for_output(output: PurePosixPath) -> str:
    return PUBLIC_ROOT + _route_suffix_for_output(output)


def _split_destination(destination: str) -> tuple[str, bool]:
    bracketed = destination.startswith("<") and destination.endswith(">")
    return (destination[1:-1] if bracketed else destination), bracketed


def _join_destination(destination: str, bracketed: bool) -> str:
    return f"<{destination}>" if bracketed else destination


def _repo_link(provenance: Provenance, repo_path: str, directory: bool) -> str:
    link_kind = "tree" if directory else "blob"
    return (
        f"{provenance.repository}/{link_kind}/{provenance.commit}/"
        f"{quote(repo_path, safe='/-._~')}"
    )


KNOWN_LINK_OVERRIDES = {
    ("docs/guides/README.md", "docs/guides/channels.md"): PUBLIC_ROOT
    + "security/configuration/#channel-setup",
    ("docs/guides/extending.md", "docs/guides/SELF_MODIFY_PLAN.md"): "docs/SELF_MODIFY_PLAN.md",
}

KNOWN_RAW_LINK_OVERRIDES = {
    # The upstream docs are commonly checked out beside prax-sandbox.  On the
    # public site there is no sibling checkout, so send readers to that repo.
    (
        "docs/guides/cloud-gpu.md",
        "../../../prax-sandbox/docs/remote.md",
    ): "https://github.com/praxagent/prax-sandbox/blob/main/docs/remote.md",
}

# Upstream has a handful of stale or ambiguous fragments.  Resolve them to the
# public page/heading they describe so Hugo's fragment validator can remain
# strict without requiring changes in the Prax repository.
KNOWN_DESTINATION_OVERRIDES = {
    ("docs/guides/setup.md", "#quick-start"): PUBLIC_ROOT + "#quick-start",
    ("docs/guides/extending.md", "#plugin-security"): PUBLIC_ROOT
    + "security/plugin-trust/#plugin-security",
    (
        "docs/infrastructure/memory.md",
        "#memory-decay-ebbinghaus-forgetting-curve",
    ): PUBLIC_ROOT + "infrastructure/memory/#memory-decay-dual-time--interaction",
}


class LinkRewriter:
    def __init__(
        self,
        source_root: Path,
        documents: list[SourceDocument],
        provenance: Provenance,
        static_staging: Path,
    ) -> None:
        self.source_root = source_root
        self.provenance = provenance
        self.static_staging = static_staging
        self.routes = {document.source_path.as_posix(): _route_for_output(document.output_path) for document in documents}
        self.directory_routes = {
            posixpath.dirname(path): route
            for path, route in self.routes.items()
            if posixpath.basename(path).casefold() == "readme.md"
        }
        self.asset_paths: set[str] = set()

    def _resolved_repo_path(self, source_path: str, link_path: str) -> str:
        decoded = unquote(link_path).replace("\\", "/")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), decoded))
        if resolved == ".." or resolved.startswith("../") or resolved.startswith("/"):
            raise ImportFailure(f"{source_path}: relative link escapes the repository: {link_path}")
        return resolved.removeprefix("./")

    def _copy_readme_asset(self, repo_path: str, source_path: str) -> str:
        asset = self.source_root / PurePosixPath(repo_path)
        if not asset.is_file() or asset.is_symlink():
            raise ImportFailure(f"{source_path}: README asset is missing or unsafe: {repo_path}")
        try:
            _run_git(self.source_root, "ls-files", "--error-unmatch", "--", repo_path)
        except ImportFailure as error:
            raise ImportFailure(f"{source_path}: README asset is not tracked by Git: {repo_path}") from error
        destination = self.static_staging / PurePosixPath(repo_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset, destination)
        self.asset_paths.add(repo_path)
        return STATIC_PUBLIC_ROOT + quote(repo_path, safe="/-._~")

    def rewrite_destination(self, destination: str, source_path: str) -> str:
        raw, bracketed = _split_destination(destination)
        decoded_for_safety = html.unescape(unquote(raw)).lstrip()
        if DANGEROUS_PROTOCOL_RE.match(decoded_for_safety):
            raise ImportFailure(f"{source_path}: executable Markdown URL is not allowed: {raw}")

        destination_override = KNOWN_DESTINATION_OVERRIDES.get((source_path, raw))
        if destination_override is not None:
            return _join_destination(destination_override, bracketed)

        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc or raw.startswith("//"):
            return destination
        if parsed.path.startswith("/"):
            return destination

        if not parsed.path:
            current_route = self.routes[source_path]
            rewritten = urlunsplit(("", "", current_route, parsed.query, parsed.fragment))
            return _join_destination(rewritten, bracketed)

        raw_override = KNOWN_RAW_LINK_OVERRIDES.get((source_path, unquote(parsed.path)))
        if raw_override is not None:
            rewritten = urlunsplit(("", "", raw_override, parsed.query, parsed.fragment))
            return _join_destination(rewritten, bracketed)

        repo_path = self._resolved_repo_path(source_path, parsed.path)
        override = KNOWN_LINK_OVERRIDES.get((source_path, repo_path))
        if override is not None:
            if override.startswith("/"):
                rewritten = override
            else:
                rewritten = _repo_link(self.provenance, override, directory=False)
            if parsed.query:
                rewritten += "?" + parsed.query
            if parsed.fragment:
                rewritten += "#" + parsed.fragment
            return _join_destination(rewritten, bracketed)

        if source_path == "README.md" and repo_path.startswith("assets/"):
            rewritten = self._copy_readme_asset(repo_path, source_path)
        elif repo_path in self.asset_paths:
            rewritten = STATIC_PUBLIC_ROOT + quote(repo_path, safe="/-._~")
        elif repo_path in self.routes:
            rewritten = self.routes[repo_path]
        elif repo_path in self.directory_routes:
            rewritten = self.directory_routes[repo_path]
        else:
            candidate = self.source_root / PurePosixPath(repo_path)
            rewritten = _repo_link(self.provenance, repo_path, directory=candidate.is_dir())

        rewritten = urlunsplit(("", "", rewritten, parsed.query, parsed.fragment))
        return _join_destination(rewritten, bracketed)

    def _rewrite_visible_segment(self, segment: str, source_path: str) -> str:
        reference = REFERENCE_LINK_RE.match(segment)
        if reference:
            segment = (
                reference.group("open")
                + self.rewrite_destination(reference.group("destination"), source_path)
                + reference.group("rest")
            )
        return segment

    @staticmethod
    def _inline_code_spans(line: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(line):
            tick = line.find("`", cursor)
            if tick < 0:
                break
            end_of_run = tick
            while end_of_run < len(line) and line[end_of_run] == "`":
                end_of_run += 1
            marker = line[tick:end_of_run]
            closing = line.find(marker, end_of_run)
            if closing < 0:
                break
            closing += len(marker)
            spans.append((tick, closing))
            cursor = closing
        return spans

    def _rewrite_prose(self, prose: str, source_path: str) -> str:
        code_spans = self._inline_code_spans(prose)
        result: list[str] = []
        cursor = 0
        for match in INLINE_LINK_RE.finditer(prose):
            # A code span can contain Markdown-looking text, while a real link
            # label may itself contain code (the dominant style in Prax docs).
            # Suppress only links wholly enclosed by a code span.
            if any(match.start() >= start and match.end() <= end for start, end in code_spans):
                continue
            result.append(prose[cursor : match.start()])
            result.extend(
                (
                    match.group("open"),
                    self.rewrite_destination(match.group("destination"), source_path),
                    match.group("title") or "",
                    match.group("close"),
                )
            )
            cursor = match.end()
        result.append(prose[cursor:])
        rewritten = "".join(result)
        return "".join(
            self._rewrite_visible_segment(line, source_path)
            for line in rewritten.splitlines(keepends=True)
        )

    def rewrite(self, text: str, source_path: PurePosixPath) -> str:
        rendered: list[str] = []
        prose: list[str] = []
        fence_character: str | None = None
        fence_length = 0
        source = source_path.as_posix()
        for line in text.splitlines(keepends=True):
            fence = FENCE_RE.match(line)
            if fence_character is not None:
                rendered.append(line)
                if fence and fence.group(1)[0] == fence_character and len(fence.group(1)) >= fence_length:
                    fence_character = None
                continue
            if fence:
                if prose:
                    rendered.append(self._rewrite_prose("".join(prose), source))
                    prose.clear()
                fence_character = fence.group(1)[0]
                fence_length = len(fence.group(1))
                rendered.append(line)
                continue
            prose.append(line)
        if prose:
            rendered.append(self._rewrite_prose("".join(prose), source))
        return "".join(rendered)


def _make_staging_directory(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    # tempfile.mkdtemp defaults to 0o700. GitHub Pages (and the post-Hugo
    # rsync that mirrors content directories into blog/) cannot publish a
    # non-world-executable tree, so open the staging root immediately.
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    staging.chmod(0o755)
    return staging


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _ensure_world_traversable(path: Path) -> None:
    """Ensure directories are world-executable/readable for static hosting."""

    if not path.is_dir() or path.is_symlink():
        return
    mode = path.stat().st_mode
    path.chmod(mode | 0o755)


def _atomic_replace_directories(replacements: list[tuple[Path, Path]]) -> None:
    """Transactionally swap staged directories into place, removing stale files."""

    backups: list[tuple[Path, Path]] = []
    installed: list[tuple[Path, Path]] = []
    try:
        for staging, target in replacements:
            if target.exists() or target.is_symlink():
                backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
                os.replace(target, backup)
                backups.append((backup, target))
        for staging, target in replacements:
            os.replace(staging, target)
            _ensure_world_traversable(target)
            installed.append((target, staging))
    except BaseException:
        for target, _staging in reversed(installed):
            _remove_path(target)
        for backup, target in reversed(backups):
            os.replace(backup, target)
        raise
    else:
        for backup, _target in backups:
            _remove_path(backup)


def _validate_output_locations(source_root: Path, content_output: Path, static_output: Path) -> None:
    source = source_root.resolve()
    content = content_output.resolve()
    static = static_output.resolve()
    if content == static:
        raise ImportFailure("content and static outputs must be different directories")
    if content in static.parents or static in content.parents:
        raise ImportFailure("content and static outputs must not contain one another")
    for output in (content, static):
        if output == source or source in output.parents or output in source.parents:
            raise ImportFailure("generated outputs must be outside the upstream checkout")


def import_documentation(
    source_root: Path,
    content_output: Path,
    static_output: Path,
    *,
    repository: str = DEFAULT_REPOSITORY,
    ref: str = DEFAULT_REF,
) -> dict[str, object]:
    source_root = source_root.expanduser().resolve()
    content_output = content_output.expanduser().resolve()
    static_output = static_output.expanduser().resolve()
    _validate_output_locations(source_root, content_output, static_output)

    provenance = _checkout_provenance(source_root, repository, ref)
    documents = discover_documents(source_root)
    content_staging = _make_staging_directory(content_output)
    static_staging = _make_staging_directory(static_output)

    try:
        rewriter = LinkRewriter(source_root, documents, provenance, static_staging)
        generated_files: list[dict[str, str]] = []
        for document in documents:
            source_file = source_root / document.source_path
            try:
                original = source_file.read_text(encoding="utf-8")
            except UnicodeError as error:
                raise ImportFailure(f"{document.source_path}: documentation must be UTF-8") from error
            original = normalize_brand_case(original)
            if document.source_path == PurePosixPath("README.md"):
                original = _sanitize_root_markup(original)
            validate_safe_markdown(original, document.source_path)
            title, body = extract_title_and_body(original, document.source_path)
            summary = derive_summary(body, title)
            body = rewriter.rewrite(body, document.source_path)
            output_file = content_staging / document.output_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(
                render_front_matter(title, summary, document, provenance) + body,
                encoding="utf-8",
                newline="\n",
            )
            generated_files.append(
                {
                    "source_path": document.source_path.as_posix(),
                    "content_path": document.output_path.as_posix(),
                }
            )

        asset_files = sorted(rewriter.asset_paths)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "source_repo": provenance.repository,
            "source_ref": provenance.ref,
            "source_commit": provenance.commit,
            "counts": {
                "source_markdown": len(documents),
                "content_files": len(generated_files),
                "asset_files": len(asset_files),
                "generated_files": len(generated_files) + len(asset_files) + 1,
            },
            "content_files": generated_files,
            "asset_files": asset_files,
        }
        (static_staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _atomic_replace_directories(
            [(content_staging, content_output), (static_staging, static_output)]
        )
    except BaseException:
        _remove_path(content_staging)
        _remove_path(static_staging)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import a safe, curated, commit-pinned snapshot of the Prax docs into Hugo."
    )
    parser.add_argument("--source", type=Path, help="existing local Prax Git checkout")
    parser.add_argument("--repo", default=DEFAULT_REPOSITORY, help="Prax Git repository URL")
    parser.add_argument("--ref", default=DEFAULT_REF, help="branch, tag, or commit to import")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="sparse checkout cache used when --source is omitted",
    )
    parser.add_argument("--content-output", required=True, type=Path, help="dedicated Hugo content output")
    parser.add_argument("--static-output", required=True, type=Path, help="dedicated Hugo static output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source = args.source if args.source is not None else acquire_checkout(args.repo, args.ref, args.cache_dir)
        manifest = import_documentation(
            source,
            args.content_output,
            args.static_output,
            repository=args.repo,
            ref=args.ref,
        )
    except ImportFailure as error:
        print(f"Prax documentation import failed: {error}", file=sys.stderr)
        return 1
    counts = manifest["counts"]
    assert isinstance(counts, dict)
    print(
        "Imported "
        f"{counts['content_files']} Prax document(s) and {counts['asset_files']} asset(s) "
        f"at {manifest['source_commit']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
