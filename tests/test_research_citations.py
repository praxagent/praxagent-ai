from __future__ import annotations

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOG_SOURCE = ROOT / "blog-source"
HUGO_CONFIG = BLOG_SOURCE / "hugo.yaml"
ORCID = "https://orcid.org/0009-0007-5992-0652"
RIGHTS_URL = (
    "https://github.com/praxagent/praxagent-ai/blob/main/CONTENT-LICENSE.md"
)


def front_matter(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        return {}

    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line == "---":
            break
        match = re.match(r"^([A-Za-z_][\w-]*):\s*(.*?)\s*$", line)
        if match is None:
            continue
        key, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        fields[key] = value
    return fields


def build_site(destination: Path, content_dir: Path | None = None) -> None:
    command = [
        "hugo",
        "--source",
        str(BLOG_SOURCE),
        "--config",
        str(HUGO_CONFIG),
        "--destination",
        str(destination),
        "--baseURL",
        "https://praxagent.ai/blog/",
        "--noBuildLock",
        "--panicOnWarning",
    ]
    if content_dir is not None:
        command.extend(["--contentDir", str(content_dir)])
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)


class ResearchCitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary_directory.name) / "site"
        cls.sources = sorted(
            (BLOG_SOURCE / "content/posts").glob("**/index.md")
        )
        build_site(cls.output)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def generated_page(self, source: Path, fields: dict[str, str]) -> Path:
        year, month = fields["date"].split("-")[:2]
        return (
            self.output
            / "posts"
            / year
            / month
            / fields["slug"]
            / "index.html"
        )

    def test_all_current_research_notes_opt_in(self) -> None:
        self.assertEqual(len(self.sources), 5)

        for source in self.sources:
            with self.subTest(source=source):
                fields = front_matter(source)
                self.assertEqual(fields.get("citation_enabled"), "true")
                self.assertEqual(fields.get("author_id"), "timothy-jones")
                self.assertRegex(fields.get("lastmod", ""), r"^\d{4}-\d{2}-\d{2}$")
                self.assertRegex(
                    fields.get("citation_version", ""), r"^\d{4}\.\d{2}\.\d{2}$"
                )

                page = self.generated_page(source, fields)
                self.assertTrue(page.is_file(), page)
                rendered = page.read_text(encoding="utf-8")
                canonical = (
                    "https://praxagent.ai/blog/posts/"
                    f"{fields['date'][:4]}/{fields['date'][5:7]}/{fields['slug']}/"
                )

                self.assertEqual(rendered.count('class="research-citation"'), 1)
                self.assertIn(f'<link rel="canonical" href="{canonical}">', rendered)
                self.assertIn(ORCID, rendered)
                self.assertIn(RIGHTS_URL, rendered)
                self.assertIn('data-copy-target="#research-note-citation-text"', rendered)
                self.assertIn('data-copy-target="#research-note-bibtex"', rendered)
                self.assertIn(
                    f'<meta name="citation_version" content="{fields["citation_version"]}">',
                    rendered,
                )
                self.assertIn(
                    f'<time class="post-date" datetime="{fields["date"]}">',
                    rendered,
                )

                json_ld_match = re.search(
                    r'<script type="application/ld\+json">(.*?)</script>',
                    rendered,
                    flags=re.DOTALL,
                )
                self.assertIsNotNone(json_ld_match)
                schema = json.loads(json_ld_match.group(1))
                self.assertEqual(schema["@type"], "ScholarlyArticle")
                self.assertEqual(schema["author"]["sameAs"], ORCID)
                self.assertEqual(schema["datePublished"], fields["date"])
                self.assertEqual(schema["dateModified"], fields["lastmod"])
                self.assertEqual(schema["version"], fields["citation_version"])
                self.assertEqual(schema["url"], canonical)
                self.assertEqual(schema["usageInfo"], RIGHTS_URL)

                markdown_images = len(
                    re.findall(r"!\[[^\]]*\]\([^)]+\)", source.read_text(encoding="utf-8"))
                )
                self.assertGreater(markdown_images, 0)
                self.assertEqual(
                    rendered.count('class="research-figure__credit"'),
                    markdown_images,
                )

    def test_guest_post_can_disable_registered_citation_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = root / "content"
            guest_bundle = content / "posts/2026/07/guest-note"
            guest_bundle.mkdir(parents=True)
            (guest_bundle / "index.md").write_text(
                """---
title: "Guest Note"
slug: "guest-note"
date: 2026-07-25
citation_enabled: false
author: "Guest Writer"
summary: "A guest note used to verify that scholarly attribution is opt-in."
---

A guest-authored note.

![Guest figure](guest-figure.svg)
""",
                encoding="utf-8",
            )
            output = root / "site"
            build_site(output, content)
            rendered = (
                output / "posts/2026/07/guest-note/index.html"
            ).read_text(encoding="utf-8")

        self.assertIn("by Guest Writer", rendered)
        self.assertNotIn(ORCID, rendered)
        self.assertNotIn('class="research-citation"', rendered)
        self.assertNotIn('"@type":"ScholarlyArticle"', rendered)
        self.assertNotIn('name="citation_title"', rendered)
        self.assertNotIn('class="research-figure__credit"', rendered)


if __name__ == "__main__":
    unittest.main()
