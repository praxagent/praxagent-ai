#!/usr/bin/env python3
"""Compile repository Python source in memory without creating bytecode files."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def candidate_paths() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
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


def main() -> None:
    errors: list[str] = []
    checked = 0
    for path in candidate_paths():
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(relative), "exec")
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"{relative}: {exc}")
        else:
            checked += 1

    if errors:
        print("Python syntax validation failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print(f"Python syntax validation passed ({checked} file(s), no bytecode written).")


if __name__ == "__main__":
    main()
