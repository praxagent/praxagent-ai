#!/usr/bin/env python3
"""Dependency-free validation for the generated static site."""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".cursor", ".cache", "node_modules", "__pycache__"}
EXTERNAL_SCHEMES = {"http", "https", "mailto", "tel", "data", "javascript"}
KNOWLEDGE_BASE_PATH = "/blog/knowledge-base/"
LATE_CHUNKING_BUNDLE = Path(
    "blog-source/content/knowledge-base/deep-dives/late-chunking"
)
LATE_CHUNKING_ARTIFACTS = (
    "ATTRIBUTION.md",
    "reproduce.py",
    "reproduce.py.lock",
    "receipts/aggregate.json",
    "receipts/per-query.csv",
    "receipts/scifact-test-qrels.tsv",
    "receipts/top-10-rankings.jsonl",
    "receipts/run.receipt.json",
    "fig-scifact-retrieval.svg",
    "fig-query-deltas.svg",
    "fig-scifact-retrieval.receipt.json",
    "fig-query-deltas.receipt.json",
    "provenance.json",
)
REFERENCE_FIGURE_RE = re.compile(
    r"{{<\s*reference-figure\s+(.*?)\s*>}}", re.DOTALL
)
SHORTCODE_ATTRIBUTE_RE = re.compile(r'([A-Za-z][\w-]*)="([^"]*)"')


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


def check_prax_docs(errors: list[str]) -> None:
    entry = ROOT / "blog/knowledge-base/prax/index.html"
    manifest_path = ROOT / "blog/prax-docs/prax-docs-manifest.json"

    for path in (entry, manifest_path):
        if not path.is_file():
            errors.append(
                f"{path.relative_to(ROOT)}: missing generated Prax documentation"
            )

    if not manifest_path.is_file():
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return  # check_json reports the detailed parse error.

    if not isinstance(manifest, dict):
        errors.append(
            "blog/prax-docs/prax-docs-manifest.json: root must be an object"
        )
        return

    content_files = manifest.get("content_files")
    if not isinstance(content_files, list):
        errors.append(
            "blog/prax-docs/prax-docs-manifest.json: content_files must be a list"
        )
        return

    imported = {
        item.get("source_path"): item.get("content_path")
        for item in content_files
        if isinstance(item, dict)
    }
    if imported.get("README.md") != "_index.md":
        errors.append(
            "blog/prax-docs/prax-docs-manifest.json: Prax README must be the "
            "documentation entry page"
        )

    research_sources = sorted(
        source
        for source in imported
        if isinstance(source, str)
        and (source == "docs/research" or source.startswith("docs/research/"))
    )
    if research_sources:
        errors.append(
            "blog/prax-docs/prax-docs-manifest.json: excluded Prax research "
            f"documentation was imported: {', '.join(research_sources)}"
        )

    research_output = ROOT / "blog/knowledge-base/prax/research"
    if research_output.exists():
        errors.append(
            f"{research_output.relative_to(ROOT)}: excluded Prax research "
            "documentation was generated"
        )


def _route_parts_for_generated_content(content_path: str) -> tuple[str, ...]:
    path = PurePosixPath(content_path)
    if path.name == "_index.md":
        return tuple(part for part in path.parent.parts if part != ".")
    return path.with_suffix("").parts


def _check_legacy_redirect(
    errors: list[str],
    route_parts: tuple[str, ...],
    *,
    require_legacy: bool = True,
) -> None:
    canonical = ROOT / "blog" / "knowledge-base" / Path(*route_parts) / "index.html"
    legacy = ROOT / "blog" / "references" / Path(*route_parts) / "index.html"
    encoded_route = "/".join(quote(part, safe="-._~") for part in route_parts)
    expected_path = KNOWLEDGE_BASE_PATH + (encoded_route + "/" if encoded_route else "")

    if not canonical.is_file():
        errors.append(
            f"{canonical.relative_to(ROOT)}: missing canonical Knowledge Base page"
        )
    else:
        public_page = canonical.read_text(encoding="utf-8").casefold()
        if "pro-reviewed" in public_page or "pro_reviewed" in public_page:
            errors.append(
                f"{canonical.relative_to(ROOT)}: private Pro review metadata "
                "must not be rendered publicly"
            )
    if not require_legacy:
        return
    if not legacy.is_file():
        errors.append(
            f"{legacy.relative_to(ROOT)}: missing legacy References redirect"
        )
        return

    try:
        redirect = legacy.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append(f"{legacy.relative_to(ROOT)}: redirect must be UTF-8")
        return
    canonical_link = re.search(
        r'<link rel="canonical" href="([^"]+)">', redirect
    )
    if canonical_link is None or urlsplit(canonical_link.group(1)).path != expected_path:
        errors.append(
            f"{legacy.relative_to(ROOT)}: legacy redirect must canonicalize to "
            f"{expected_path}"
        )
    refresh = re.search(
        r'<meta http-equiv="refresh" content="[^"]*url=([^"]+)">', redirect
    )
    if refresh is None or urlsplit(refresh.group(1)).path != expected_path:
        errors.append(
            f"{legacy.relative_to(ROOT)}: legacy redirect must refresh to {expected_path}"
        )


def _reference_figure_attributes(source: str) -> list[dict[str, str]]:
    return [
        dict(SHORTCODE_ATTRIBUTE_RE.findall(match.group(1)))
        for match in REFERENCE_FIGURE_RE.finditer(source)
    ]


def _check_late_chunking_figure_alt_receipts(
    errors: list[str], source: Path, source_text: str
) -> None:
    shortcodes = _reference_figure_attributes(source_text)
    figure_receipts = {
        "fig-scifact-retrieval.svg": "fig-scifact-retrieval.receipt.json",
        "fig-query-deltas.svg": "fig-query-deltas.receipt.json",
    }
    for figure_name, receipt_name in figure_receipts.items():
        receipt_path = source.parent / receipt_name
        if not receipt_path.is_file():
            continue  # The required-artifact check reports the missing receipt.
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue  # check_json reports the detailed parse error.
        if not isinstance(receipt, dict) or not isinstance(
            receipt.get("alt_text"), str
        ):
            errors.append(
                f"{receipt_path.relative_to(ROOT)}: figure receipt requires "
                "a string alt_text"
            )
            continue

        matches = [
            attributes
            for attributes in shortcodes
            if Path(urlsplit(attributes.get("src", "")).path).name == figure_name
        ]
        if len(matches) != 1:
            errors.append(
                f"{source.relative_to(ROOT)}: expected exactly one reference-figure "
                f"shortcode for {figure_name}, found {len(matches)}"
            )
            continue
        shortcode_alt = matches[0].get("alt")
        receipt_alt = receipt["alt_text"]
        if shortcode_alt != receipt_alt:
            errors.append(
                f"{source.relative_to(ROOT)}: {figure_name} shortcode alt must "
                f"exactly match {receipt_name} alt_text"
            )


def check_knowledge_base_routes(errors: list[str]) -> None:
    route_parts: dict[tuple[str, ...], bool] = {
        (): True,
        ("deep-dives",): True,
        ("glossary",): True,
    }

    content_dir = ROOT / "blog-source/content/knowledge-base"
    for path in sorted(content_dir.glob("*.md")):
        if path.name in {"_index.md", "AGENTS.md", "SKILL.md"}:
            continue
        text = path.read_text(encoding="utf-8")
        frontmatter_parts = text.split("---", 2)
        frontmatter = frontmatter_parts[1] if len(frontmatter_parts) == 3 else ""
        if re.search(r"(?mi)^draft:\s*true\s*$", frontmatter):
            continue
        slug = re.search(
            r'(?m)^slug:\s*["\']?([^"\'\s]+)["\']?\s*$', frontmatter
        )
        if slug is None:
            errors.append(f"{path.relative_to(ROOT)}: glossary entry requires a slug")
            continue
        has_legacy_alias = re.search(
            r"(?m)^\s*-\s*/references/", frontmatter
        ) is not None
        route_parts[(slug.group(1),)] = has_legacy_alias

    manifest_path = ROOT / "blog/prax-docs/prax-docs-manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            manifest = None  # Detailed parse errors are reported elsewhere.
        if isinstance(manifest, dict):
            content_files = manifest.get("content_files")
            if isinstance(content_files, list):
                for item in content_files:
                    if not isinstance(item, dict):
                        continue
                    content_path = item.get("content_path")
                    if isinstance(content_path, str):
                        route_parts[
                            ("prax",) + _route_parts_for_generated_content(content_path)
                        ] = True

    for parts, require_legacy in sorted(route_parts.items()):
        _check_legacy_redirect(
            errors, parts, require_legacy=require_legacy
        )

    sitemap = ROOT / "blog/sitemap.xml"
    if sitemap.is_file():
        sitemap_text = sitemap.read_text(encoding="utf-8")
        if "/blog/references/" in sitemap_text:
            errors.append(
                "blog/sitemap.xml: legacy References redirects must not be indexed"
            )
        if "/blog/knowledge-base/" not in sitemap_text:
            errors.append(
                "blog/sitemap.xml: canonical Knowledge Base routes are missing"
            )


def check_late_chunking_deep_dive(errors: list[str]) -> None:
    expected_path = "/blog/knowledge-base/deep-dives/late-chunking/"
    source_bundle = ROOT / LATE_CHUNKING_BUNDLE
    source = source_bundle / "index.md"
    if not source.is_file():
        errors.append(f"{source.relative_to(ROOT)}: missing Deep Dive source")
    else:
        source_text = source.read_text(encoding="utf-8")
        if re.search(
            r"(?m)^pro_reviewed:\s*(?:true|false)\s*$", source_text
        ) is None:
            errors.append(
                f"{source.relative_to(ROOT)}: missing private pro_reviewed metadata"
            )
        if re.search(r"(?m)^date:\s*2025-08-02\s*$", source_text) is None:
            errors.append(
                f"{source.relative_to(ROOT)}: original date must remain 2025-08-02"
            )
        if re.search(r"(?m)^lastmod\s*:", source_text):
            errors.append(
                f"{source.relative_to(ROOT)}: Deep Dive must not set an updated date"
            )
        if re.search(r"(?mi)^correction\s*:", source_text) or re.search(
            r"editorial correction", source_text, re.IGNORECASE
        ):
            errors.append(
                f"{source.relative_to(ROOT)}: public editorial correction must be removed"
            )
        _check_late_chunking_figure_alt_receipts(errors, source, source_text)

    for relative in LATE_CHUNKING_ARTIFACTS:
        artifact = source_bundle / relative
        if not artifact.is_file():
            errors.append(
                f"{artifact.relative_to(ROOT)}: missing reproduction artifact"
            )

    canonical = ROOT / "blog/knowledge-base/deep-dives/late-chunking/index.html"
    if not canonical.is_file():
        errors.append(
            f"{canonical.relative_to(ROOT)}: missing Late Chunking Deep Dive"
        )
    else:
        page = canonical.read_text(encoding="utf-8")
        folded = page.casefold()
        canonical_link = re.search(
            r'<link rel="canonical" href="([^"]+)">', page
        )
        if (
            canonical_link is None
            or urlsplit(canonical_link.group(1)).path != expected_path
        ):
            errors.append(
                f"{canonical.relative_to(ROOT)}: page must canonicalize to "
                f"{expected_path}"
            )
        if "editorial correction" in folded:
            errors.append(
                f"{canonical.relative_to(ROOT)}: public editorial correction "
                "must be removed"
            )
        if "august 2, 2025" not in folded:
            errors.append(
                f"{canonical.relative_to(ROOT)}: original August 2, 2025 date "
                "must be displayed"
            )
        if "updated july 17, 2026" in folded or "post-updated" in folded:
            errors.append(
                f"{canonical.relative_to(ROOT)}: updated date must not be displayed"
            )
        disclosure_position = folded.find("ai-use disclosure")
        contents_position = folded.find('class="table-of-contents"')
        if disclosure_position < 0:
            errors.append(
                f"{canonical.relative_to(ROOT)}: missing AI-use disclosure"
            )
        if contents_position < 0:
            errors.append(
                f"{canonical.relative_to(ROOT)}: missing Deep Dive table of contents"
            )
        elif disclosure_position < 0 or disclosure_position > contents_position:
            errors.append(
                f"{canonical.relative_to(ROOT)}: AI-use disclosure must precede "
                "the table of contents"
            )
        if "pro-reviewed" in folded or "pro_reviewed" in folded:
            errors.append(
                f"{canonical.relative_to(ROOT)}: private Pro review metadata "
                "must not be rendered publicly"
            )

    public_bundle = canonical.parent
    for relative in LATE_CHUNKING_ARTIFACTS:
        source_artifact = source_bundle / relative
        public_artifact = public_bundle / relative
        if not public_artifact.is_file():
            errors.append(
                f"{public_artifact.relative_to(ROOT)}: reproduction artifact "
                "was not published"
            )
        elif (
            source_artifact.is_file()
            and source_artifact.read_bytes() != public_artifact.read_bytes()
        ):
            errors.append(
                f"{public_artifact.relative_to(ROOT)}: published reproduction "
                "artifact differs from its source"
            )

    aliases = (
        ROOT / "blog/posts/2025/08/late-chunking/index.html",
        ROOT / "blog/posts/late-chunking/index.html",
    )
    for alias in aliases:
        if not alias.is_file():
            errors.append(f"{alias.relative_to(ROOT)}: missing historical redirect")
            continue
        redirect = alias.read_text(encoding="utf-8")
        canonical_link = re.search(
            r'<link rel="canonical" href="([^"]+)">', redirect
        )
        refresh = re.search(
            r'<meta http-equiv="refresh" content="[^"]*url=([^"]+)">', redirect
        )
        if (
            canonical_link is None
            or urlsplit(canonical_link.group(1)).path != expected_path
        ):
            errors.append(
                f"{alias.relative_to(ROOT)}: historical redirect must canonicalize "
                f"to {expected_path}"
            )
        if refresh is None or urlsplit(refresh.group(1)).path != expected_path:
            errors.append(
                f"{alias.relative_to(ROOT)}: historical redirect must refresh to "
                f"{expected_path}"
            )

    collection = ROOT / "blog/knowledge-base/deep-dives/index.html"
    if collection.is_file() and expected_path not in collection.read_text(
        encoding="utf-8"
    ):
        errors.append(
            f"{collection.relative_to(ROOT)}: Late Chunking is missing from Deep Dives"
        )

    for index in (ROOT / "blog/index.html", ROOT / "blog/posts/index.html"):
        if not index.is_file():
            continue
        index_text = index.read_text(encoding="utf-8")
        if (
            expected_path in index_text
            or "late chunking: context before pooling" in index_text.casefold()
        ):
            errors.append(
                f"{index.relative_to(ROOT)}: Deep Dive must not appear as a Research Note"
            )

    sitemap = ROOT / "blog/sitemap.xml"
    if sitemap.is_file():
        sitemap_text = sitemap.read_text(encoding="utf-8")
        if expected_path not in sitemap_text:
            errors.append("blog/sitemap.xml: Late Chunking canonical route is missing")
        for old_path in (
            "/blog/posts/2025/08/late-chunking/",
            "/blog/posts/late-chunking/",
        ):
            if old_path in sitemap_text:
                errors.append(
                    f"blog/sitemap.xml: historical Late Chunking route {old_path} "
                    "must not be indexed"
                )


def check_svg(errors: list[str]) -> None:
    for path in walk("*.svg"):
        try:
            ElementTree.parse(path)
        except (ElementTree.ParseError, OSError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: invalid SVG/XML: {exc}")


def check_knowledge_base_svg_accessibility(errors: list[str]) -> None:
    roots = (
        ROOT / "blog-source/static/knowledge-base",
        ROOT / "blog-source/content/knowledge-base/deep-dives",
    )
    svg_paths = sorted(
        {path for directory in roots for path in directory.rglob("*.svg")}
    )
    for path in svg_paths:
        try:
            root = ElementTree.parse(path).getroot()
        except (ElementTree.ParseError, OSError):
            continue  # check_svg reports the detailed parse error.

        relative = path.relative_to(ROOT)
        for attribute in ("width", "height", "viewBox"):
            if not root.get(attribute):
                errors.append(f"{relative}: accessible SVG requires {attribute}")

        if root.get("role") != "img":
            errors.append(f"{relative}: accessible SVG requires role='img'")

        children = {
            child.tag.rsplit("}", 1)[-1]: child
            for child in root
            if child.tag.rsplit("}", 1)[-1] in {"title", "desc"}
        }
        labelled_by = set(root.get("aria-labelledby", "").split())
        for element_name in ("title", "desc"):
            element = children.get(element_name)
            if element is None:
                errors.append(
                    f"{relative}: accessible SVG requires a direct <{element_name}>"
                )
                continue

            element_id = element.get("id")
            element_text = "".join(element.itertext()).strip()
            if not element_id:
                errors.append(f"{relative}: <{element_name}> requires an id")
            elif element_id not in labelled_by:
                errors.append(
                    f"{relative}: aria-labelledby must include {element_name} id "
                    f"{element_id!r}"
                )
            if not element_text:
                errors.append(f"{relative}: <{element_name}> must not be empty")

        if "—" in path.read_text(encoding="utf-8"):
            errors.append(f"{relative}: SVG text must not contain an em dash")


def main() -> int:
    errors: list[str] = []
    check_brand(errors)
    check_html(errors)
    check_json(errors)
    check_prax_docs(errors)
    check_knowledge_base_routes(errors)
    check_late_chunking_deep_dive(errors)
    check_svg(errors)
    check_knowledge_base_svg_accessibility(errors)

    if errors:
        print("Site validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "Site validation passed: lowercase brand, local links, anchors, JSON, SVG "
        "accessibility, and Prax docs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
