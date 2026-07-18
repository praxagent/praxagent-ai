#!/usr/bin/env python3
"""Tests for the dependency-free Prax documentation importer."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import import_prax_docs as importer


def run(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


class FixtureRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._write(
            "README.md",
            """<div align="center">

# Prax

**A teaching harness for dependable agents.**

<img src="assets/header image.png" alt="Prax">

</div>

[Agents](docs/agents/README.md) · [Checkpointing](docs/agents/checkpointing.md#save-now)
[Local section](#quick-start) · [Configuration](config.py) · [Source tree](prax/)

## Quick Start

Start here.

### [Research](docs/research/README.md)

Academic foundations for agentic workflow design — embryonic material.

## Memory System

This later top-level section must remain.

### [Guides](docs/guides/README.md)

Practical documentation.
""",
        )
        self._write("assets/header image.png", b"not-a-real-png")
        self._write("config.py", "SETTING = True\n")
        self._write("prax/agent.py", "# fixture\n")
        self._write("docs/library.md", "# Library\n\nBrowse the durable knowledge library.\n")

        section_titles = {
            "agents": "Agents",
            "architecture": "Architecture",
            "guides": "Guides",
            "infrastructure": "Infrastructure",
            "research": "Research",
            "security": "Security",
        }
        for section, title in section_titles.items():
            extra = "\n[Channels](channels.md)\n" if section == "guides" else ""
            self._write(
                f"docs/{section}/README.md",
                f"# {title}\n\nLearn about {title.lower()}.\n{extra}",
            )

        self._write(
            "docs/agents/checkpointing.md",
            """# Checkpointing

[← Agents](README.md)

Save work safely. See [Security](../security/README.md#defaults),
[`configuration`](../../config.py), and the [source directory](../../prax/).
The multiline [**security
guide**](../security/README.md#defaults) is imported too.

## Save Now

Checkpoint the state.
""",
        )
        self._write(
            "docs/guides/extending.md",
            "# Extending Prax\n\nRead [the self-modification plan](SELF_MODIFY_PLAN.md) "
            "and [`PluginCapabilities`](#plugin-security).\n",
        )
        self._write(
            "docs/guides/setup.md",
            "# Setup\n\nSee [Quick Start](#quick-start) in the project overview.\n",
        )
        self._write(
            "docs/infrastructure/memory.md",
            "# Memory\n\n[Decay](#memory-decay-ebbinghaus-forgetting-curve)\n\n"
            "## Memory Decay (Dual: Time + Interaction)\n",
        )
        self._write(
            "docs/research/edge-bench-learning-curves.md",
            "# Learning Curves\n\nSee [loops explained](loops-explained-assessment).\n",
        )
        self._write(
            "docs/research/harness-engineering.md",
            "# Harness Engineering\n\nSee [section five](orchestration.md#5).\n",
        )
        self._write(
            "docs/research/orchestration.md",
            "# Orchestration\n\n### 5. Context Window Management and Drift Prevention\n",
        )
        self._write(
            "docs/research/plugin-sandboxing.md",
            "# Plugin Sandboxing\n\nSee [Plugin security](#plugin-security).\n",
        )
        self._write(
            "docs/security/plugin-trust.md",
            "# Plugin Trust\n\n## Plugin security\n",
        )
        # These are intentionally outside the publication allowlist.
        self._write("docs/SELF_MODIFY_PLAN.md", "# Internal plan\n")
        self._write("docs/IDEAS_BACKLOG.md", "# Internal ideas\n")
        self._write("docs/plans/private.md", "# Private plan\n")

        run("git", "init", "-b", "main", cwd=self.root)
        run("git", "config", "user.name", "Importer Test", cwd=self.root)
        run("git", "config", "user.email", "importer@example.invalid", cwd=self.root)
        run("git", "add", ".", cwd=self.root)
        run("git", "commit", "-m", "fixture", cwd=self.root)

    def _write(self, relative: str, content: str | bytes) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")

    @property
    def commit(self) -> str:
        return run("git", "rev-parse", "HEAD", cwd=self.root)

    def commit_change(self, relative: str, content: str, message: str = "change") -> None:
        self._write(relative, content)
        run("git", "add", relative, cwd=self.root)
        run("git", "commit", "-m", message, cwd=self.root)


class ImportPraxDocsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        source = self.base / "source"
        source.mkdir()
        self.fixture = FixtureRepository(source)
        self.content = self.base / "content"
        self.static = self.base / "static"

    def import_docs(self) -> dict[str, object]:
        return importer.import_documentation(
            self.fixture.root,
            self.content,
            self.static,
            repository="https://github.com/praxagent/prax.git",
            ref="main",
        )

    def test_front_matter_h1_readme_routes_assets_and_manifest(self) -> None:
        manifest = self.import_docs()
        root = (self.content / "_index.md").read_text(encoding="utf-8")

        # tempfile.mkdtemp is 0o700 by default; published trees must stay
        # world-traversable or GitHub Pages 404s the whole collection.
        self.assertTrue(self.content.stat().st_mode & 0o005)
        self.assertTrue(self.static.stat().st_mode & 0o005)

        self.assertTrue(root.startswith("+++\ntitle = \"Prax\"\n"))
        self.assertIn('summary = "A teaching harness for dependable agents."', root)
        self.assertIn('layout = "prax-docs-section"', root)
        self.assertIn('aliases = ["/references/prax/"]', root)
        self.assertIn("weight = 30", root)
        self.assertIn('card_label = "Documentation"', root)
        self.assertIn('card_title = "Prax Harness"', root)
        self.assertIn(f'source_commit = "{self.fixture.commit}"', root)
        self.assertIn('source_path = "README.md"', root)
        self.assertIn(
            f'edit_url = "https://github.com/praxagent/prax/edit/{self.fixture.commit}/README.md"',
            root,
        )
        self.assertNotIn("# Prax", root)
        self.assertNotIn("<div", root)
        self.assertNotIn("<img", root)
        self.assertIn("[Agents](/blog/knowledge-base/prax/agents/)", root)
        self.assertIn(
            "[Checkpointing](/blog/knowledge-base/prax/agents/checkpointing/#save-now)", root
        )
        self.assertIn("[Local section](/blog/knowledge-base/prax/#quick-start)", root)
        self.assertNotIn("Academic foundations for agentic workflow design", root)
        self.assertNotIn("[Research]", root)
        self.assertIn("## Memory System", root)
        self.assertIn(
            "![Prax](/blog/prax-docs/assets/header%20image.png)", root
        )
        self.assertEqual(
            (self.static / "assets/header image.png").read_bytes(), b"not-a-real-png"
        )

        on_disk_manifest = json.loads(
            (self.static / importer.MANIFEST_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest, on_disk_manifest)
        self.assertEqual(on_disk_manifest["source_commit"], self.fixture.commit)
        self.assertEqual(on_disk_manifest["counts"]["asset_files"], 1)
        self.assertEqual(
            on_disk_manifest["counts"]["content_files"],
            len(on_disk_manifest["content_files"]),
        )

    def test_readme_sections_relative_links_and_outside_repository_links(self) -> None:
        self.import_docs()
        self.assertTrue((self.content / "agents/_index.md").is_file())
        self.assertTrue((self.content / "agents/checkpointing.md").is_file())
        self.assertTrue((self.content / "library.md").is_file())
        self.assertFalse((self.content / "plans/private.md").exists())
        self.assertFalse((self.content / "research").exists())

        leaf = (self.content / "agents/checkpointing.md").read_text(encoding="utf-8")
        self.assertIn('layout = "prax-doc"', leaf)
        self.assertIn(
            'aliases = ["/references/prax/agents/checkpointing/"]', leaf
        )
        self.assertNotIn("# Checkpointing", leaf)
        self.assertIn("[← Agents](/blog/knowledge-base/prax/agents/)", leaf)
        self.assertIn("[Security](/blog/knowledge-base/prax/security/#defaults)", leaf)
        self.assertIn(
            "[**security\nguide**](/blog/knowledge-base/prax/security/#defaults)", leaf
        )
        self.assertIn(
            f"[`configuration`](https://github.com/praxagent/prax/blob/{self.fixture.commit}/config.py)",
            leaf,
        )
        self.assertIn(
            f"[source directory](https://github.com/praxagent/prax/tree/{self.fixture.commit}/prax)",
            leaf,
        )
        root = (self.content / "_index.md").read_text(encoding="utf-8")
        self.assertIn(
            f"[Configuration](https://github.com/praxagent/prax/blob/{self.fixture.commit}/config.py)",
            root,
        )
        self.assertIn(
            f"[Source tree](https://github.com/praxagent/prax/tree/{self.fixture.commit}/prax)",
            root,
        )

    def test_known_broken_link_overrides(self) -> None:
        self.import_docs()
        guides = (self.content / "guides/_index.md").read_text(encoding="utf-8")
        extending = (self.content / "guides/extending.md").read_text(encoding="utf-8")
        setup = (self.content / "guides/setup.md").read_text(encoding="utf-8")
        memory = (self.content / "infrastructure/memory.md").read_text(encoding="utf-8")
        self.assertIn(
            "[Channels](/blog/knowledge-base/prax/security/configuration/#channel-setup)", guides
        )
        self.assertIn(
            f"[the self-modification plan](https://github.com/praxagent/prax/blob/{self.fixture.commit}/docs/SELF_MODIFY_PLAN.md)",
            extending,
        )
        self.assertIn("[Quick Start](/blog/knowledge-base/prax/#quick-start)", setup)
        self.assertIn(
            "[`PluginCapabilities`](/blog/knowledge-base/prax/security/plugin-trust/#plugin-security)",
            extending,
        )
        self.assertIn(
            "[Decay](/blog/knowledge-base/prax/infrastructure/memory/#memory-decay-dual-time--interaction)",
            memory,
        )

    def test_rejects_dangerous_html_without_disturbing_atomic_outputs(self) -> None:
        (self.content / "stale.md").parent.mkdir(parents=True)
        (self.content / "stale.md").write_text("stale", encoding="utf-8")
        (self.static / "stale.bin").parent.mkdir(parents=True)
        (self.static / "stale.bin").write_bytes(b"stale")
        self.import_docs()
        self.assertFalse((self.content / "stale.md").exists())
        self.assertFalse((self.static / "stale.bin").exists())

        before_content = (self.content / "_index.md").read_bytes()
        before_manifest = (self.static / importer.MANIFEST_NAME).read_bytes()
        self.fixture.commit_change(
            "docs/agents/checkpointing.md",
            "# Checkpointing\n\n<script>alert('no')</script>\n",
            "dangerous upstream documentation",
        )
        with self.assertRaisesRegex(importer.ImportFailure, "raw HTML <script>"):
            self.import_docs()
        self.assertEqual((self.content / "_index.md").read_bytes(), before_content)
        self.assertEqual((self.static / importer.MANIFEST_NAME).read_bytes(), before_manifest)

    def test_rejects_javascript_and_event_attributes(self) -> None:
        cases = (
            "# Unsafe\n\n[click](javascript:alert(1))\n",
            '# Unsafe\n\n<img src="x" onerror="alert(1)">\n',
        )
        for index, unsafe in enumerate(cases):
            with self.subTest(index=index):
                self.fixture.commit_change(
                    "docs/security/configuration.md",
                    unsafe,
                    f"unsafe {index}",
                )
                with self.assertRaises(importer.ImportFailure):
                    self.import_docs()

    def test_sparse_acquisition_resolves_exact_commit(self) -> None:
        cache = self.base / "cache/repo"
        checkout = importer.acquire_checkout(str(self.fixture.root), "main", cache)
        self.assertEqual(run("git", "rev-parse", "HEAD", cwd=checkout), self.fixture.commit)
        self.assertEqual(run("git", "rev-parse", "--is-shallow-repository", cwd=checkout), "true")
        self.assertTrue((checkout / "README.md").is_file())
        self.assertTrue((checkout / "docs/agents/README.md").is_file())
        self.assertTrue((checkout / "assets/header image.png").is_file())


if __name__ == "__main__":
    unittest.main()
