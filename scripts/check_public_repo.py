#!/usr/bin/env python3
"""Fail on high-confidence publication hazards in repository files.

This is deliberately conservative: it reports filenames and detector names,
never matching credential text. GitHub secret scanning remains the primary
defense for the public repository.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 2 * 1024 * 1024

FORBIDDEN_PARTS = {
    ".cache",
    ".claude",
    ".codex",
    ".cursor",
    ".pids",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
FORBIDDEN_BASENAMES = {
    ".DS_Store",
    ".env",
    ".envrc",
    ".hugo_build.lock",
    ".npmrc",
    ".pypirc",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
FORBIDDEN_SUFFIXES = {".key", ".kdbx", ".p12", ".pem", ".pfx", ".swo", ".swp"}
FORBIDDEN_PATH_PREFIXES = ("blog-source/public/", "pages-artifact/")
RAW_REVIEW_NAME = re.compile(
    r"(?:gpt-.+-(?:response|review)|pro-review-(?:findings|response)).*\.json$",
    re.IGNORECASE,
)

CREDENTIAL_PATTERNS = (
    (
        "private-key block",
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    ("OpenAI API key", re.compile(rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b")),
    ("GitHub token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("GitHub fine-grained token", re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("AWS access key", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(rb"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("Slack token", re.compile(rb"\bxox[baprs]-[0-9A-Za-z-]{20,}\b")),
)


def candidate_paths() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(
        ROOT / raw.decode("utf-8", errors="surrogateescape")
        for raw in result.stdout.split(b"\0")
        if raw
    )


def check_paths(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        parts = path.relative_to(ROOT).parts
        basename = path.name

        if relative.startswith(FORBIDDEN_PATH_PREFIXES):
            errors.append(f"{relative}: generated/local-only path must not be tracked")
        if any(part in FORBIDDEN_PARTS for part in parts):
            errors.append(f"{relative}: local-only directory must not be tracked")
        if basename in FORBIDDEN_BASENAMES or (
            basename.startswith(".env.") and basename != ".env.example"
        ):
            errors.append(f"{relative}: sensitive/local filename must not be tracked")
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES or basename.endswith("~"):
            errors.append(f"{relative}: credential or editor-backup suffix is forbidden")
        if RAW_REVIEW_NAME.search(basename):
            errors.append(f"{relative}: raw model-review output must stay in ignored cache")

        try:
            size = path.stat().st_size
            if size > MAX_TEXT_BYTES:
                continue
            data = path.read_bytes()
        except OSError as exc:
            errors.append(f"{relative}: could not inspect file: {exc}")
            continue
        if b"\0" in data:
            continue
        for detector, pattern in CREDENTIAL_PATTERNS:
            if pattern.search(data):
                errors.append(f"{relative}: possible {detector}; value withheld")

    return errors


def main() -> None:
    errors = check_paths(candidate_paths())
    if errors:
        print("Public-repository safety check failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("Public-repository safety check passed.")


if __name__ == "__main__":
    main()
