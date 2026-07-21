#!/usr/bin/env python3
"""Extract canonical, anchored article sections from the rendered Hugo site."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path


WHITESPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:!?%\)\]])")
SPACE_AFTER_OPENING_RE = re.compile(r"([\(\[])\s+")
DISPLAY_MATH_RE = re.compile(r"\\\[.*?\\\]|\$\$.*?\$\$", re.DOTALL)
INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
LATEX_NAMED_RE = re.compile(
    r"\\(?:operatorname|mathrm|mathbf|text)\s*\{([^{}]*)\}"
)
LATEX_FRACTION_RE = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
BLOCK_BOUNDARIES = {
    "blockquote",
    "br",
    "dd",
    "div",
    "dt",
    "figcaption",
    "h4",
    "h5",
    "h6",
    "li",
    "p",
    "td",
    "th",
    "tr",
}
FULLY_EXCLUDED = {"button", "math", "nav", "script", "style", "svg"}
CODE_ELEMENTS = {"code", "pre"}
PRE_ELEMENTS = {"pre"}


def normalize_text(parts: list[str]) -> str:
    text = WHITESPACE_RE.sub(" ", " ".join(parts)).strip()
    text = SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", text)
    return SPACE_AFTER_OPENING_RE.sub(r"\1", text)


def excerpt(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    shortened = text[: limit + 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{shortened}…"


def readable_latex(fragment: str) -> str:
    """Turn short inline TeX into useful plain text for a search snippet."""

    replacements = {
        r"\alpha": "alpha",
        r"\beta": "beta",
        r"\lambda": "lambda",
        r"\mu": "mu",
        r"\sigma": "sigma",
        r"\ge": ">=",
        r"\le": "<=",
        r"\times": "x",
        r"\approx": "about",
        r"\in": "in",
    }
    text = fragment
    for _ in range(3):
        updated = LATEX_NAMED_RE.sub(r"\1", text)
        updated = LATEX_FRACTION_RE.sub(r"\1 / \2", updated)
        if updated == text:
            break
        text = updated
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    text = re.sub(r"\\(?:sqrt|sum|lVert|rVert|left|right|qquad|cdot)\b", " ", text)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[_^]\s*\{?([^{}\s]+)\}?", r" \1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
    return normalize_text([text])


def display_excerpt(text: str) -> str:
    """Create a compact snippet without raw display equations or TeX syntax."""

    text = DISPLAY_MATH_RE.sub(" ", text)
    text = re.sub(r"\\\[[\s\S]*$|\$\$[\s\S]*$", " ", text)
    text = INLINE_MATH_RE.sub(lambda match: readable_latex(match.group(1)), text)
    text = re.sub(
        r"\\\(([\s\S]*)$",
        lambda match: readable_latex(match.group(1)),
        text,
    )
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\bis\s*\.", "is defined on the page.", text)
    text = re.sub(r"\bas\s*\.", "as shown on the page.", text)
    text = re.sub(r"\bis\s+(?=[A-Z])", "is defined on the page. ", text)
    text = re.sub(r"\bas\s+(?=[A-Z])", "as shown on the page. ", text)
    text = re.sub(r"\bThen\s*\.", "The equation is shown on the page.", text)
    text = re.sub(r"\bwhere\s*\.\s*", "", text)
    return excerpt(normalize_text([text]))


def content_kind(path: str) -> str:
    if "/knowledge-base/glossary/" in path:
        return "Glossary"
    if "/knowledge-base/deep-dives/" in path:
        return "Deep Dive"
    if "/knowledge-base/prax/" in path:
        return "Prax documentation"
    if "/posts/" in path:
        return "Research Note"
    return "Knowledge Base"


class RenderedPageParser(HTMLParser):
    """Collect text from the reader-facing article body without layout chrome."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.description = ""
        self.page_title = ""
        self._h1_depth: int | None = None
        self._h1_parts: list[str] = []
        self._content_depth: int | None = None
        self._excluded_depth: int | None = None
        self._code_depth: int | None = None
        self._pre_depth: int | None = None
        self._heading_depth: int | None = None
        self._heading_id = ""
        self._heading_parts: list[str] = []
        self._current_heading = "Overview"
        self._current_anchor = ""
        self._lexical_parts: list[str] = []
        self._semantic_parts: list[str] = []
        self._display_parts: list[str] = []
        self.sections: list[dict[str, str]] = []

    @staticmethod
    def _attrs(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key.casefold(): value or "" for key, value in attrs}

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = tag.casefold()
        self.depth += 1
        values = self._attrs(attrs)
        classes = set(values.get("class", "").split())

        if tag == "meta" and values.get("name") == "description":
            self.description = values.get("content", "")

        if tag == "h1" and not self.page_title:
            self._h1_depth = self.depth
            self._h1_parts = []

        if self._content_depth is None:
            if tag in {"article", "div", "section"} and "post-body" in classes:
                if "post-preface" not in classes:
                    self._content_depth = self.depth
            self._close_void(tag)
            return

        if self._excluded_depth is None and (
            tag in FULLY_EXCLUDED
            or "table-of-contents" in classes
            or "code-block-toolbar" in classes
        ):
            self._excluded_depth = self.depth

        if self._excluded_depth is None:
            if tag in CODE_ELEMENTS and self._code_depth is None:
                self._code_depth = self.depth
            if tag in PRE_ELEMENTS and self._pre_depth is None:
                self._pre_depth = self.depth

            if tag in {"h2", "h3"}:
                self._flush_section()
                self._heading_depth = self.depth
                self._heading_id = values.get("id", "")
                self._heading_parts = []
            elif tag == "img" and values.get("alt"):
                self._append(values["alt"])
            elif tag in BLOCK_BOUNDARIES:
                self._append(" ")

        self._close_void(tag)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        if self._h1_depth is not None:
            self._h1_parts.append(data)

        if self._content_depth is None or self._excluded_depth is not None:
            return
        if self._heading_depth is not None:
            self._heading_parts.append(data)
            return
        self._append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()

        if self._h1_depth == self.depth:
            self.page_title = normalize_text(self._h1_parts)
            self._h1_depth = None

        if self._heading_depth == self.depth:
            self._current_heading = normalize_text(self._heading_parts) or "Overview"
            self._current_anchor = self._heading_id
            self._heading_depth = None

        if self._code_depth == self.depth:
            self._code_depth = None
        if self._pre_depth == self.depth:
            self._pre_depth = None
        if self._excluded_depth == self.depth:
            self._excluded_depth = None
        if self._content_depth == self.depth:
            self._flush_section()
            self._content_depth = None

        self.depth = max(0, self.depth - 1)

    def _append(self, data: str) -> None:
        self._lexical_parts.append(data)
        if self._code_depth is None:
            self._semantic_parts.append(data)
        if self._pre_depth is None:
            self._display_parts.append(data)

    def _close_void(self, tag: str) -> None:
        if tag not in VOID_ELEMENTS:
            return
        if self._code_depth == self.depth:
            self._code_depth = None
        if self._excluded_depth == self.depth:
            self._excluded_depth = None
        self.depth = max(0, self.depth - 1)

    def _flush_section(self) -> None:
        lexical = normalize_text(self._lexical_parts)
        semantic = normalize_text(self._semantic_parts)
        display = normalize_text(self._display_parts)
        self._lexical_parts = []
        self._semantic_parts = []
        self._display_parts = []
        if not lexical and not semantic:
            return
        self.sections.append(
            {
                "heading": self._current_heading,
                "anchor": self._current_anchor,
                "lexical_text": lexical,
                "semantic_text": semantic or lexical,
                "display_text": display or semantic or lexical,
            }
        )


def rendered_url(path: Path, site: Path) -> str:
    """Return the public URL represented by a file in Hugo's output tree.

    The rendered path is authoritative for site search. A page's canonical
    metadata may intentionally point elsewhere, and a mistaken canonical must
    not create a search result whose local target does not exist.
    """

    relative = path.relative_to(site)
    if relative.name == "index.html":
        directory = "" if relative.parent == Path(".") else relative.parent.as_posix()
        suffix = f"{directory}/" if directory else ""
        return f"/blog/{suffix}"
    return f"/blog/{relative.as_posix()}"


def extract_page(path: Path, site: Path) -> list[dict[str, str]]:
    parser = RenderedPageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    if not parser.sections or not parser.page_title:
        return []

    canonical_path = rendered_url(path, site)
    if not canonical_path.startswith("/blog/") or canonical_path == "/blog/search/":
        return []

    records: list[dict[str, str]] = []
    for order, section in enumerate(parser.sections):
        url = canonical_path
        if section["anchor"]:
            url = f"{url}#{section['anchor']}"
        identity = f"{url}\0{order}"
        section_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        searchable = normalize_text(
            [parser.page_title, section["heading"], section["lexical_text"]]
        )
        semantic = normalize_text(
            [parser.page_title, section["heading"], section["semantic_text"]]
        )
        records.append(
            {
                "id": section_id,
                "url": url,
                "title": parser.page_title,
                "heading": section["heading"],
                "kind": content_kind(canonical_path),
                "excerpt": display_excerpt(section["display_text"] or parser.description),
                "lexical_text": searchable,
                "semantic_text": semantic,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    site = args.site.resolve()
    records: list[dict[str, str]] = []
    for page in sorted(site.rglob("*.html")):
        if any(part in {"pagefind", "search-assets"} for part in page.parts):
            continue
        records.extend(extract_page(page, site))

    records.sort(key=lambda item: (item["url"], item["id"]))
    if not records:
        raise SystemExit(f"no searchable article sections found beneath {site}")

    payload = {"version": 1, "sections": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Extracted {len(records)} searchable section(s) from {site}.")


if __name__ == "__main__":
    main()
