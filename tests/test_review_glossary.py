from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "review_glossary.py"
SPEC = importlib.util.spec_from_file_location("review_glossary", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
review_glossary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = review_glossary
SPEC.loader.exec_module(review_glossary)


class GenericKnowledgeBaseReviewTests(unittest.TestCase):
    def discover(self, relative_entry: str):
        entry = review_glossary.resolve_entry_path(
            review_glossary.ROOT / relative_entry
        )
        paths, notes = review_glossary.discover_entry_artifacts(
            entry, review_glossary.DEFAULT_MAX_ARTIFACT_BYTES
        )
        return entry, paths, notes

    def test_review_rubric_is_topic_neutral(self) -> None:
        prompt = review_glossary.KNOWLEDGE_BASE_REVIEW_INSTRUCTIONS.lower()
        for article_specific_phrase in (
            "principal component",
            "late chunk",
            "seeds dataset",
            "2025-08-02",
        ):
            self.assertNotIn(article_specific_phrase, prompt)

    def test_pca_entry_discovers_complete_text_evidence_bundle(self) -> None:
        entry, paths, notes = self.discover(
            "blog-source/content/knowledge-base/deep-dives/principal-component-analysis"
        )
        self.assertEqual(paths[0], entry)
        self.assertEqual(sum(path.suffix == ".svg" for path in paths), 9)
        self.assertEqual(sum(path.suffix == ".py" for path in paths), 4)
        self.assertIn(entry.parent / "wheat-kernel-pca-colab.ipynb", paths)
        self.assertIn(entry.parent / "wheat-kernel-pca.receipt.json", paths)
        self.assertIn(entry.parent / "fig-wheat-kernel-pca.receipt.json", paths)
        note_by_name = {note.path.name: note for note in notes}
        self.assertIn("reproduce.py.lock", note_by_name)
        self.assertIn("fig-wheat-kernel-pca.png", note_by_name)
        self.assertEqual(
            sum(note.path.suffix == ".png" for note in notes),
            6,
        )

    def test_late_chunking_hashes_oversized_and_lock_artifacts(self) -> None:
        entry, paths, notes = self.discover(
            "blog-source/content/knowledge-base/deep-dives/late-chunking"
        )
        self.assertEqual(sum(path.suffix == ".svg" for path in paths), 5)
        self.assertEqual(sum(path.suffix == ".py" for path in paths), 2)
        note_by_name = {note.path.name: note for note in notes}
        self.assertIn("top-10-rankings.jsonl", note_by_name)
        self.assertIn("oversized data artifact", note_by_name["top-10-rankings.jsonl"].reason)
        self.assertIn("reproduce.py.lock", note_by_name)
        self.assertNotIn(entry.parent / "receipts" / "top-10-rankings.jsonl", paths)

    def test_local_glossary_svg_and_raster_metadata_are_discovered(self) -> None:
        entry, paths, notes = self.discover(
            "blog-source/content/knowledge-base/glossary/transformer"
        )
        self.assertIn(entry.parent / "transformer-overview.svg", paths)
        self.assertIn(
            entry.parent / "transformer-overview.png",
            {note.path for note in notes},
        )

    def test_shared_static_svg_is_discovered(self) -> None:
        _, paths, _ = self.discover(
            "blog-source/content/knowledge-base/glossary/bf16"
        )
        self.assertIn(
            review_glossary.STATIC_ROOT
            / "knowledge-base"
            / "glossary"
            / "fp-bit-layouts.svg",
            paths,
        )

    def test_toml_front_matter_title_is_read_for_imported_docs(self) -> None:
        entry = review_glossary.resolve_entry_path(
            review_glossary.KNOWLEDGE_BASE_DIR
            / "prax"
            / "guides"
            / "authentication.md"
        )
        self.assertEqual(review_glossary.entry_title(entry), "Authentication")

    def test_notebook_packet_strips_outputs_attachments_and_metadata(self) -> None:
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {"kernel": "must not be sent"},
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {"private": "must not be sent"},
                    "execution_count": 7,
                    "source": ["print('kept')\n"],
                    "outputs": [{"output_type": "stream", "text": "must not be sent"}],
                    "attachments": {
                        "plot.png": {"image/png": "must not be sent"}
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "example.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")
            packet = review_glossary.sanitized_notebook_source(path)

        self.assertIn("print('kept')", packet)
        self.assertIn("1 output item(s)", packet)
        self.assertIn("1 attachment(s)", packet)
        self.assertNotIn("must not be sent", packet)
        self.assertNotIn("execution_count", packet)

    def test_notebook_size_limit_uses_sanitized_source_not_large_outputs(self) -> None:
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": 1,
                    "source": ["answer = 42\n"],
                    "outputs": [
                        {"output_type": "stream", "text": "x" * 300_000}
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "large-output.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")
            self.assertGreater(
                path.stat().st_size,
                review_glossary.DEFAULT_MAX_ARTIFACT_BYTES,
            )
            self.assertLess(
                review_glossary.review_payload_size(path),
                review_glossary.DEFAULT_MAX_ARTIFACT_BYTES,
            )

    def test_explicit_lockfile_is_hash_only(self) -> None:
        lockfile = (
            review_glossary.KNOWLEDGE_BASE_DIR
            / "deep-dives"
            / "late-chunking"
            / "reproduce.py.lock"
        )
        paths, notes = review_glossary.augment_explicit_artifacts(
            [lockfile], review_glossary.DEFAULT_MAX_ARTIFACT_BYTES
        )
        self.assertEqual(paths, [])
        self.assertEqual([note.path for note in notes], [lockfile])
        self.assertIn("hash only", notes[0].reason)

    def test_secret_guard_allows_documentation_placeholders_only(self) -> None:
        self.assertFalse(
            review_glossary.contains_possible_credential(
                'clientSecret="YOUR_CLIENT_SECRET"'
            )
        )
        self.assertFalse(
            review_glossary.contains_possible_credential(
                "AUTHENTIK_POSTGRESQL__PASSWORD: authentik-password"
            )
        )
        self.assertFalse(
            review_glossary.contains_possible_credential(
                "token = set_experiment_overrides"
            )
        )
        self.assertTrue(
            review_glossary.contains_possible_credential(
                'clientSecret="Z72nLa9xQp4vT8mRc6fHs3kW"'
            )
        )
        self.assertTrue(
            review_glossary.contains_possible_credential(
                "-----BEGIN " + "PRIVATE KEY-----"
            )
        )
        for assignment in (
            'GOOGLE_API_KEY="Z72nLa9xQp4vT8mRc6fHs3kW"',
            'AWS_SECRET_ACCESS_KEY="Z72nLa9xQp4vT8mRc6fHs3kW"',
            'GITHUB_TOKEN="Z72nLa9xQp4vT8mRc6fHs3kW"',
        ):
            self.assertTrue(review_glossary.contains_possible_credential(assignment))

    def test_common_lockfile_names_are_hash_only_candidates(self) -> None:
        for filename in ("go.sum", "package-lock.json", "pnpm-lock.yaml", "uv.lock"):
            self.assertTrue(review_glossary.is_lockfile(Path(filename)))
            self.assertTrue(review_glossary.is_reviewable_text_path(Path(filename)))

    def test_common_code_and_config_suffixes_are_reviewable(self) -> None:
        for filename in ("analysis.R", "figure.mmd", "run.sh", "config.yaml"):
            self.assertTrue(review_glossary.is_reviewable_text_path(Path(filename)))

    def test_arff_dataset_is_reviewable_but_not_a_core_text_artifact(self) -> None:
        path = Path("Rice_Cammeo_Osmancik.arff")
        self.assertTrue(review_glossary.is_reviewable_text_path(path))
        self.assertFalse(review_glossary.is_core_text_path(path))

    def test_inline_raster_data_is_rejected(self) -> None:
        source = "data:image/" + "png;base64,iVBORw0KGgo="
        self.assertTrue(review_glossary.contains_inline_raster(source))

    def test_hidden_review_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            hidden_directory = temporary_root / ".private-review"
            hidden_directory.mkdir()
            path = hidden_directory / "entry.md"
            path.write_text("---\ntitle: Hidden\n---\n", encoding="utf-8")
            with mock.patch.object(review_glossary, "ROOT", temporary_root):
                with self.assertRaises(ValueError):
                    review_glossary.validate_repository_file(path)

    def test_continuation_is_bound_to_bundle_and_instructions(self) -> None:
        context = review_glossary.review_context_sha256("bundle", "instructions")
        prior = {
            review_glossary.REVIEW_METADATA_KEY: {"context_sha256": context}
        }
        review_glossary.validate_prior_binding(prior, context)
        with self.assertRaises(ValueError):
            review_glossary.validate_prior_binding(prior, "0" * 64)
        with self.assertRaises(ValueError):
            review_glossary.validate_prior_binding({}, context)

    def test_review_json_schema_is_checked(self) -> None:
        valid = json.dumps(
            {
                "verdict": "pass",
                "summary": "No substantive findings.",
                "findings": [],
                "cross_artifact_conflicts": [],
                "checks_that_passed": ["Artifacts agree."],
            }
        )
        review_glossary.validate_review_json(valid)
        with self.assertRaises(ValueError):
            review_glossary.validate_review_json('{"verdict":"pass"}')


if __name__ == "__main__":
    unittest.main()
