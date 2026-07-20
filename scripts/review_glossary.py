#!/usr/bin/env python3
"""Submit Knowledge Base content to GPT-5.6 Sol Pro for technical review.

The script uses only the Python standard library. It reads OPENAI_API_KEY (or
the project's OPENAI_KEY alias) from the process environment first, then from
the repository-root .env file. The key is sent only in the Authorization header
and is never written to an artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
GLOSSARY_DIR = ROOT / "blog-source" / "content" / "knowledge-base" / "glossary"
DIAGRAM_DIR = ROOT / "blog-source" / "static" / "knowledge-base" / "glossary"
DOTENV_PATH = ROOT / ".env"
PRIVATE_OUTPUT_DIR = ROOT / ".cache" / "glossary-review"
DEFAULT_OUTPUT = PRIVATE_OUTPUT_DIR / "gpt-5.6-sol-pro.json"
DEFAULT_CONTINUATION_OUTPUT = PRIVATE_OUTPUT_DIR / "gpt-5.6-sol-pro-continued.json"
API_KEY_NAMES = ("OPENAI_API_KEY", "OPENAI_KEY")
SVG_SRC_RE = re.compile(r'''\bsrc\s*=\s*["']([^"']+[.]svg)(?:[?#][^"']*)?["']''')

REVIEW_INSTRUCTIONS = """\
You are the final scientific and technical reviewer for an educational glossary
about language-model inference, interpretability, and small-sample statistics.
Review the supplied Markdown entries and SVG source as one connected document.

Check, with exceptional care:
1. factual and mathematical accuracy, including equations and numeric examples;
2. whether Python snippets implement the prose and handle the stated boundary;
3. consistency between entries, Mermaid diagrams, SVG labels, captions, and links;
4. every supplied SVG's visible labels, arrows, grouping, encoded relationships,
   title, description, alt text, and caption as substantive technical claims;
5. distinctions among description, readout, prediction, causation, and mechanism;
6. statistical assumptions: pairing, sidedness, ties, dependence, exchangeability,
   selection, effect size, and population generalization;
7. model- or experiment-specific claims that are phrased too generally;
8. terminology that would teach a careful newcomer the wrong mental model;
9. unexplained acronyms, symbols, or specialized terms in prose, equations,
   code, tables, captions, alt text, and SVG source; require first-use expansion
   and a just-in-time definition even when a glossary link is present.

Do not reward length and do not rewrite passages that are already correct. Treat
site-specific empirical claims as claims that need appropriately narrow wording;
do not invent evidence or citations. Return valid JSON only, with this shape:
{
  "verdict": "pass" | "revise",
  "summary": "short overall assessment",
  "findings": [
    {
      "severity": "error" | "important" | "minor",
      "file": "repository-relative path",
      "section": "heading or SVG element",
      "claim": "short exact excerpt or identifier",
      "issue": "specific explanation",
      "recommended_replacement": "ready-to-apply wording or code",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "cross_entry_conflicts": ["specific conflict, if any"],
  "checks_that_passed": ["important point explicitly verified"]
}
Sort findings by severity, then file. If there are no actionable findings, use
an empty findings array and verdict "pass".
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

DEEP_DIVE_REVIEW_INSTRUCTIONS = """\
You are the final scientific, technical, and educational reviewer for a Knowledge
Base Deep Dive about late chunking for retrieval embeddings. Review the supplied
Markdown and SVG source as one connected article.

Check, with exceptional care:
1. fidelity to the cited primary Late Chunking paper, especially the order of
   token contextualization and span pooling, and the long-document extension;
2. mathematical accuracy, including pooling notation, normalization, token spans,
   attention masks, special tokens, truncation, and overlapping macro-windows;
3. whether code and pseudocode implement the prose without relying on an API that
   exposes only already-pooled embeddings;
4. distinctions among contextual influence, reference resolution, cosine
   similarity, retrieval quality, and universal performance claims;
5. whether empirical claims identify the evaluated benchmark, metric, comparison,
   and boundary, without turning a paper result into a product guarantee;
6. consistency among prose, SVG labels, alt text, captions, and links;
7. every supplied SVG's visible labels, arrows, grouping, encoded relationships,
   title, description, alt text, and caption as substantive technical claims;
8. accessibility: each figure needs one finding, useful direct labels, a concise
   alt text, and a caption that is a complete nonvisual explanation;
9. teaching quality for a technically curious newcomer, including a plain-language
   definition, explicit non-goals, implementation pitfalls, and evaluation advice;
10. primary-source citations and scoped wording for model-specific facts;
11. whether the runnable reproduction freezes a real corpus, queries, qrels,
    model, tokenizer, chunker, and query path, while changing only the stated
    document-context treatment across matched arms;
12. whether document ranking, query-level nDCG@10, the whole-document control,
    paired uncertainty, and scope limits are computed and described correctly;
13. whether every empirical figure is generated from compact committed results,
    has a machine-verifiable receipt, agrees with its accessible text equivalent,
    and avoids unsupported statistical or cross-corpus inference.
14. unexplained acronyms, symbols, or specialized terms in prose, equations,
    code, tables, captions, alt text, and SVG source; require first-use expansion
    and a just-in-time definition even when a glossary link is present.

Editorial constraint: the author requires the page to retain its original
publication date, 2025-08-02, with no public `lastmod` or correction notice.
The separately timestamped experiment receipt supplies provenance for the later
run and is not a reason to request an updated page date.

Do not reward length and do not invent evidence, benchmark values, citations, or
implementation guarantees. Return valid JSON only, with this shape:
{
  "verdict": "pass" | "revise",
  "summary": "short overall assessment",
  "findings": [
    {
      "severity": "error" | "important" | "minor",
      "file": "repository-relative path",
      "section": "heading or SVG element",
      "claim": "short exact excerpt or identifier",
      "issue": "specific explanation",
      "recommended_replacement": "ready-to-apply wording or code",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "cross_entry_conflicts": ["specific conflict, if any"],
  "checks_that_passed": ["important point explicitly verified"]
}
Sort findings by severity, then file. If there are no actionable findings, use
an empty findings array and verdict "pass".
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


def selected_source_paths(candidates: list[Path]) -> list[Path]:
    """Resolve explicit review inputs while keeping them inside this repository."""
    selected: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        resolved = resolved.resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(f"review input must stay inside {ROOT}: {candidate}") from exc
        if not resolved.is_file():
            raise ValueError(f"review input is not a file: {candidate}")
        if resolved.suffix.lower() not in {
            ".csv",
            ".json",
            ".jsonl",
            ".lock",
            ".md",
            ".py",
            ".svg",
            ".tsv",
        }:
            raise ValueError(f"unsupported review input type: {candidate}")
        if resolved not in selected:
            selected.append(resolved)
    return selected


def referenced_svg_paths(markdown_paths: list[Path]) -> list[Path]:
    """Resolve local SVGs referenced by selected Markdown artifacts."""
    referenced: list[Path] = []
    static_root = ROOT / "blog-source" / "static"
    for markdown_path in markdown_paths:
        if markdown_path.suffix.lower() != ".md":
            continue
        source = markdown_path.read_text(encoding="utf-8")
        for match in SVG_SRC_RE.finditer(source):
            raw_reference = match.group(1)
            if "://" in raw_reference:
                continue
            normalized = raw_reference.removeprefix("/blog/").lstrip("/")
            candidates = (
                markdown_path.parent / normalized,
                static_root / normalized,
            )
            resolved = next(
                (candidate.resolve() for candidate in candidates if candidate.is_file()),
                None,
            )
            if resolved is not None and resolved not in referenced:
                referenced.append(resolved)
    return referenced


def build_bundle(paths: list[Path]) -> str:
    chunks = [
        "Review all artifacts below. Repository-relative filenames are authoritative."
    ]
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        chunks.extend(
            (
                f"\n===== BEGIN FILE: {relative} =====",
                path.read_text(encoding="utf-8"),
                f"===== END FILE: {relative} =====",
            )
        )
    return "\n".join(chunks)


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
    resolved = path.expanduser().resolve()
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


def load_prior_response(path: Path) -> dict[str, object]:
    prior_path = resolve_private_path(path)
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
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
        choices=("glossary", "deep-dive"),
        default="glossary",
        help="Review rubric to apply (default: %(default)s)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        action="append",
        default=[],
        help="Repository-local artifact to review; repeat for a connected bundle",
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
    if args.max_output_tokens < 25_000:
        print(
            "--max-output-tokens must be at least 25000 for a reasoning review.",
            file=sys.stderr,
        )
        return 1

    try:
        prior = load_prior_response(args.resume_from) if args.resume_from else None
        output_candidate = args.output
        if prior is not None and output_candidate == DEFAULT_OUTPUT:
            output_candidate = DEFAULT_CONTINUATION_OUTPUT
        output_path = resolve_private_path(output_candidate)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Invalid private review artifact: {exc}", file=sys.stderr)
        return 1

    try:
        paths = (
            selected_source_paths(args.path)
            if args.path
            else glossary_source_paths()
        )
    except (OSError, ValueError) as exc:
        print(f"Invalid review input: {exc}", file=sys.stderr)
        return 1
    if args.review_profile == "deep-dive" and not args.path:
        print("--review-profile deep-dive requires at least one --path.", file=sys.stderr)
        return 1
    if not paths:
        print("No review artifacts found.", file=sys.stderr)
        return 1
    if args.path:
        referenced_svgs = referenced_svg_paths(paths)
        missing_svgs = [path for path in referenced_svgs if path not in paths]
        if missing_svgs:
            print(
                "Explicit review bundle omitted referenced SVG source(s):",
                file=sys.stderr,
            )
            for path in missing_svgs:
                print(f"  {path.relative_to(ROOT).as_posix()}", file=sys.stderr)
            print("Add each file with another --path argument.", file=sys.stderr)
            return 1

    bundle = build_bundle(paths)
    if args.dry_run:
        counts = {
            suffix: sum(path.suffix == suffix for path in paths)
            for suffix in (
                ".md",
                ".svg",
                ".py",
                ".json",
                ".lock",
                ".csv",
                ".jsonl",
                ".tsv",
            )
        }
        print(
            "Review bundle ready: "
            f"{counts['.md']} Markdown, {counts['.svg']} SVG, "
            f"{counts['.py']} Python, {counts['.json']} JSON, "
            f"{counts['.lock']} lock, {counts['.csv']} CSV, "
            f"{counts['.jsonl']} JSONL, {counts['.tsv']} TSV, "
            f"{len(bundle.encode('utf-8')):,} bytes."
        )
        print("Artifacts:")
        for path in paths:
            print(f"  {path.relative_to(ROOT).as_posix()}")
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
    review_instructions = (
        DEEP_DIVE_REVIEW_INSTRUCTIONS
        if args.review_profile == "deep-dive"
        else REVIEW_INSTRUCTIONS
    )
    if args.review_round > 1:
        review_instructions += "\n\n" + FOLLOWUP_REVIEW_INSTRUCTIONS.format(
            round_number=args.review_round
        )

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

    write_private_json(output_path, response)

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
        print(review_text)
    else:
        print("The response contained no output_text item.", file=sys.stderr)
        print(f"Raw response saved privately to {output_path}", file=sys.stderr)
        return 4
    print(f"\nRaw response saved privately to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
