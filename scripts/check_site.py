#!/usr/bin/env python3
"""Dependency-free validation for the generated static site."""

from __future__ import annotations

import json
import re
import struct
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urljoin, urlsplit
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
SOCIAL_IMAGE_FIELDS = (
    "og:image",
    "twitter:image",
)
SOCIAL_IMAGE_ALT_FIELDS = (
    "og:image:alt",
    "twitter:image:alt",
)
SOCIAL_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SOCIAL_IMAGE_WIDTH = 1200
SOCIAL_IMAGE_HEIGHT = 630
SOCIAL_IMAGE_BUNDLE_ROOTS = (
    (
        Path("blog-source/content/posts"),
        Path("blog/posts"),
        "Research Note",
    ),
    (
        Path("blog-source/content/knowledge-base/deep-dives"),
        Path("blog/knowledge-base/deep-dives"),
        "Deep Dive",
    ),
)


class DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[tuple[str, str]] = []
        self.meta: dict[str, list[str]] = {}
        self.is_redirect = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = {key.casefold(): value for key, value in attrs}
        if element_id := values.get("id"):
            self.ids.add(element_id)
        for attr in ("href", "src"):
            if value := values.get(attr):
                self.links.append((attr, value))
        if tag.casefold() == "meta":
            if (values.get("http-equiv") or "").casefold() == "refresh":
                self.is_redirect = True
            meta_name = values.get("property") or values.get("name")
            content = values.get("content")
            if meta_name and content is not None:
                self.meta.setdefault(meta_name.casefold(), []).append(content)


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


def _frontmatter_scalars(path: Path) -> dict[str, str]:
    """Return simple, top-level scalar values from YAML front matter.

    The social-card convention deliberately uses one-line scalar fields, so a
    full YAML dependency would add complexity without improving this check.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = re.match(r"^([A-Za-z_][\w-]*):\s*(.*?)\s*$", line)
        if match is None:
            continue
        key, raw_value = match.groups()
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = value[1:-1]
            if isinstance(decoded, str):
                value = decoded
        elif len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1].replace("''", "'")
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        fields[key] = value
    return fields


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG signature")
    if data[12:16] != b"IHDR" or struct.unpack(">I", data[8:12])[0] != 13:
        raise ValueError("PNG does not begin with a valid IHDR chunk")
    width, height = struct.unpack(">II", data[16:24])
    if width < 1 or height < 1:
        raise ValueError("PNG has invalid zero dimensions")
    return width, height


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError("invalid JPEG signature")

    # Start-of-frame markers that carry sample precision, height, and width.
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    standalone_markers = {0x01, *range(0xD0, 0xDA)}
    offset = 2
    while offset < len(data):
        while offset < len(data) and data[offset] != 0xFF:
            offset += 1
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break

        marker = data[offset]
        offset += 1
        if marker == 0x00:
            continue
        if marker in standalone_markers:
            continue
        if marker in {0xD8, 0xD9}:
            if marker == 0xD9:
                break
            continue
        if offset + 2 > len(data):
            raise ValueError("truncated JPEG segment length")

        segment_length = struct.unpack(">H", data[offset : offset + 2])[0]
        if segment_length < 2 or offset + segment_length > len(data):
            raise ValueError("invalid or truncated JPEG segment")
        if marker in sof_markers:
            if segment_length < 7:
                raise ValueError("truncated JPEG start-of-frame segment")
            height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
            if width < 1 or height < 1:
                raise ValueError("JPEG has invalid zero dimensions")
            return width, height
        if marker == 0xDA:
            break
        offset += segment_length

    raise ValueError("JPEG has no supported start-of-frame marker")


def _raster_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if path.suffix.casefold() == ".png":
        return _png_dimensions(data)
    return _jpeg_dimensions(data)


def _resolved_public_path(page: Path, raw_url: str) -> str:
    page_directory = "/" + page.parent.relative_to(ROOT).as_posix().strip("/") + "/"
    base = f"https://local.invalid{page_directory}"
    return unquote(urlsplit(urljoin(base, raw_url)).path)


def _generated_social_bundle(
    *,
    source_root: Path,
    output_root: Path,
    bundle: Path,
    fields: dict[str, str],
    content_label: str,
) -> Path:
    explicit_url = fields.get("url", "").strip()
    if explicit_url:
        public_path = unquote(urlsplit(explicit_url).path).strip("/")
        return ROOT / public_path

    bundle_relative = bundle.relative_to(source_root)
    slug = fields.get("slug", "").strip() or bundle_relative.name
    if content_label == "Research Note":
        date_match = re.match(r"^(\d{4})-(\d{2})", fields.get("date", ""))
        if date_match is not None:
            year, month = date_match.groups()
            return output_root / year / month / slug
    return output_root / bundle_relative.parent / slug


def _check_social_meta_value(
    errors: list[str],
    *,
    source: Path,
    document: DocumentParser,
    field: str,
    expected: str,
    image_field: bool = False,
    generated_page: Path,
) -> None:
    values = document.meta.get(field, [])
    if len(values) != 1:
        errors.append(
            f"{generated_page.relative_to(ROOT)}: expected exactly one {field} "
            f"meta tag for {source.relative_to(ROOT)}, found {len(values)}"
        )
        return

    actual = values[0]
    if image_field:
        actual = _resolved_public_path(generated_page, actual)
    if actual != expected:
        errors.append(
            f"{generated_page.relative_to(ROOT)}: {field} must be {expected!r} "
            f"for {source.relative_to(ROOT)}, found {actual!r}"
        )


def check_social_images(errors: list[str]) -> None:
    """Validate source cards and their generated Open Graph/Twitter metadata."""

    for source_root_relative, output_root_relative, content_label in (
        SOCIAL_IMAGE_BUNDLE_ROOTS
    ):
        source_root = ROOT / source_root_relative
        output_root = ROOT / output_root_relative
        for source in sorted(source_root.rglob("index.md")):
            bundle = source.parent
            fields = _frontmatter_scalars(source)
            image_name = fields.get("og_image", "").strip()
            image_alt = fields.get("og_image_alt", "").strip()

            if not image_name:
                errors.append(
                    f"{source.relative_to(ROOT)}: {content_label} requires "
                    "og_image"
                )
            if not image_alt:
                errors.append(
                    f"{source.relative_to(ROOT)}: {content_label} requires "
                    "og_image_alt"
                )

            valid_image_name = bool(image_name)
            parsed_image = urlsplit(image_name)
            if image_name and (
                parsed_image.scheme
                or parsed_image.netloc
                or image_name.startswith("/")
                or parsed_image.query
                or parsed_image.fragment
                or "/" in image_name
                or "\\" in image_name
                or PurePosixPath(image_name).name != image_name
            ):
                valid_image_name = False
                errors.append(
                    f"{source.relative_to(ROOT)}: og_image must be a local "
                    f"page-bundle filename, found {image_name!r}"
                )

            extension = Path(image_name).suffix.casefold() if image_name else ""
            if image_name and extension not in SOCIAL_IMAGE_EXTENSIONS:
                valid_image_name = False
                errors.append(
                    f"{source.relative_to(ROOT)}: og_image must use PNG or JPEG, "
                    f"found {extension or 'no extension'}"
                )

            source_image = bundle / image_name if valid_image_name else None
            if source_image is not None and not source_image.is_file():
                errors.append(
                    f"{source_image.relative_to(ROOT)}: og_image page-bundle "
                    "file is missing"
                )
            elif source_image is not None:
                try:
                    width, height = _raster_dimensions(source_image)
                except (OSError, ValueError) as exc:
                    errors.append(
                        f"{source_image.relative_to(ROOT)}: invalid social image: {exc}"
                    )
                else:
                    if (width, height) != (
                        SOCIAL_IMAGE_WIDTH,
                        SOCIAL_IMAGE_HEIGHT,
                    ):
                        errors.append(
                            f"{source_image.relative_to(ROOT)}: social image must be "
                            f"{SOCIAL_IMAGE_WIDTH}x{SOCIAL_IMAGE_HEIGHT}, found "
                            f"{width}x{height}"
                        )

            generated_bundle = _generated_social_bundle(
                source_root=source_root,
                output_root=output_root,
                bundle=bundle,
                fields=fields,
                content_label=content_label,
            )
            generated_page = generated_bundle / "index.html"
            if not generated_page.is_file() or not valid_image_name:
                continue

            try:
                document = parse_html(generated_page)
            except UnicodeDecodeError:
                errors.append(
                    f"{generated_page.relative_to(ROOT)}: generated page must be UTF-8"
                )
                continue
            if document.is_redirect:
                continue

            generated_image = generated_bundle / image_name
            if not generated_image.is_file():
                errors.append(
                    f"{generated_image.relative_to(ROOT)}: generated social image "
                    "page resource is missing"
                )
            expected_public_path = "/" + generated_image.relative_to(ROOT).as_posix()
            for field in SOCIAL_IMAGE_FIELDS:
                _check_social_meta_value(
                    errors,
                    source=source,
                    document=document,
                    field=field,
                    expected=expected_public_path,
                    image_field=True,
                    generated_page=generated_page,
                )
            for field in SOCIAL_IMAGE_ALT_FIELDS:
                _check_social_meta_value(
                    errors,
                    source=source,
                    document=document,
                    field=field,
                    expected=image_alt,
                    generated_page=generated_page,
                )


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


def _is_world_traversable_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    mode = path.stat().st_mode
    return bool(mode & 0o005) and bool(mode & 0o004)


def check_prax_docs(errors: list[str]) -> None:
    entry = ROOT / "blog/knowledge-base/prax/index.html"
    manifest_path = ROOT / "blog/prax-docs/prax-docs-manifest.json"
    publish_dirs = (
        ROOT / "blog/knowledge-base/prax",
        ROOT / "blog/references/prax",
        ROOT / "blog/prax-docs",
    )

    for path in (entry, manifest_path):
        if not path.is_file():
            errors.append(
                f"{path.relative_to(ROOT)}: missing generated Prax documentation"
            )

    for directory in publish_dirs:
        if not directory.is_dir():
            errors.append(
                f"{directory.relative_to(ROOT)}: missing generated Prax directory"
            )
        elif not _is_world_traversable_dir(directory):
            errors.append(
                f"{directory.relative_to(ROOT)}: directory mode "
                f"{oct(directory.stat().st_mode & 0o777)} is not world-traversable; "
                "GitHub Pages will 404"
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
    check_social_images(errors)
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
        "accessibility, social images, and Prax docs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
