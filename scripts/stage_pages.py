#!/usr/bin/env python3
"""Assemble the intentionally public GitHub Pages artifact.

The repository is public, but the deployed website should still contain only
the files that are part of the website.  This script copies an explicit
allowlist into a disposable staging directory instead of publishing the
working tree.
"""

from __future__ import annotations

import argparse
import shutil
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "pages-artifact"

PUBLIC_FILES = (
    ".nojekyll",
    "CNAME",
    "apps.html",
    "favicon.svg",
    "index.html",
    "script.js",
    "stochastic-mountain.js",
    "styles-v2.css",
    "styles.css",
)

PUBLIC_DIRECTORIES = (
    "assets",
    "blog",
    "data",
    "internships",
    "methods",
    "research",
    "tippytip",
    "work",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="staging directory (must be named pages-artifact)",
    )
    return parser.parse_args()


def _safe_output(raw_output: Path) -> Path:
    output = raw_output if raw_output.is_absolute() else ROOT / raw_output
    output = output.resolve(strict=False)
    if output.name != "pages-artifact":
        raise SystemExit(
            f"refusing unsafe output {output}: directory must be named "
            "'pages-artifact'"
        )
    try:
        output.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit(
            f"refusing unsafe output {output}: it must be inside {ROOT}"
        ) from exc
    if output == ROOT:
        raise SystemExit("refusing to replace the repository root")
    return output


def _assert_regular_tree(source: Path) -> None:
    candidates = (source, *source.rglob("*")) if source.is_dir() else (source,)
    for candidate in candidates:
        if candidate.is_symlink():
            raise SystemExit(
                f"refusing to publish symlink: {candidate.relative_to(ROOT)}"
            )


def _make_publicly_readable(output: Path) -> None:
    for path in (output, *output.rglob("*")):
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            path.chmod(mode | 0o555)
        else:
            path.chmod(mode | 0o444)


def stage(output: Path) -> None:
    sources = tuple(ROOT / name for name in (*PUBLIC_FILES, *PUBLIC_DIRECTORIES))
    missing = [path.relative_to(ROOT) for path in sources if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"cannot stage Pages; required inputs are missing:\n{formatted}")

    for source in sources:
        _assert_regular_tree(source)

    if output.exists():
        if output.is_symlink():
            raise SystemExit(f"refusing to replace symlinked output: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for name in PUBLIC_FILES:
        shutil.copy2(ROOT / name, output / name)
    for name in PUBLIC_DIRECTORIES:
        shutil.copytree(ROOT / name, output / name)

    _make_publicly_readable(output)
    file_count = sum(path.is_file() for path in output.rglob("*"))
    total_bytes = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    print(
        f"Staged {file_count} public file(s) ({total_bytes:,} bytes) in "
        f"{output.relative_to(ROOT)}."
    )


def main() -> None:
    stage(_safe_output(parse_args().output))


if __name__ == "__main__":
    main()
