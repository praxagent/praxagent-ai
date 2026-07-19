#!/usr/bin/env python3
"""Require deploy-relevant inputs to match the Git index.

Local builds can see untracked files and unstaged edits that are absent from a
fresh GitHub Actions checkout.  Refuse that split-brain state so a successful
release check describes the commit that will actually be pushed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from stage_pages import PUBLIC_DIRECTORIES, PUBLIC_FILES


ROOT = Path(__file__).resolve().parents[1]

# These paths either drive the site build or are copied directly into the
# Pages artifact. Generated blog/ output is intentionally excluded: Hugo
# rebuilds it from blog-source/ in CI.
DIRECT_PUBLIC_INPUTS = tuple(
    path for path in (*PUBLIC_FILES, *PUBLIC_DIRECTORIES) if path != "blog"
)
RELEASE_INPUTS = (
    ".github",
    ".gitignore",
    "Makefile",
    "blog-source",
    "scripts",
    *DIRECT_PUBLIC_INPUTS,
)


def _git_paths(*arguments: str) -> list[str]:
    result = subprocess.run(
        ["git", *arguments, "-z", "--", *RELEASE_INPUTS],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(
        raw.decode("utf-8", errors="surrogateescape")
        for raw in result.stdout.split(b"\0")
        if raw
    )


def main() -> None:
    untracked = _git_paths("ls-files", "--others", "--exclude-standard")
    unstaged = _git_paths("diff", "--name-only")

    if untracked or unstaged:
        print("Release-input check failed: the working tree differs from the Git index.")
        if untracked:
            print("  Untracked release inputs (stage or ignore them):")
            for path in untracked:
                print(f"    - {path}")
        if unstaged:
            print("  Unstaged release-input changes (stage or restore them):")
            for path in unstaged:
                print(f"    - {path}")
        print("Stage the intended release inputs, then rerun make ci.")
        raise SystemExit(1)

    print("Release-input check passed: deploy-relevant files match the Git index.")


if __name__ == "__main__":
    main()
