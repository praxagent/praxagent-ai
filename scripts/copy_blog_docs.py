#!/usr/bin/env python3
"""Copy raw Markdown evidence files beside their dated Hugo post output."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


DATE_RE = re.compile(r"^date:\s*[\"']?(\d{4})-(\d{2})-\d{2}", re.MULTILINE)
SLUG_RE = re.compile(r"^slug:\s*[\"']?([^\"'\s]+)", re.MULTILINE)
RESOURCE_NAMES = {"README.md", "WEB.md"}


def front_matter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path}: expected YAML front matter")

    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError(f"{path}: unterminated YAML front matter")
    return parts[1]


def copy_resources(source_root: Path, output_root: Path) -> int:
    copied = 0

    for index_path in sorted(source_root.rglob("index.md")):
        header = front_matter(index_path)
        date_match = DATE_RE.search(header)
        if date_match is None:
            raise ValueError(f"{index_path}: date must start with YYYY-MM-DD")

        slug_match = SLUG_RE.search(header)
        slug = slug_match.group(1) if slug_match else index_path.parent.name
        year, month = date_match.groups()
        destination_bundle = output_root / year / month / slug

        for resource in sorted(index_path.parent.rglob("*")):
            if not resource.is_file() or resource.name not in RESOURCE_NAMES:
                continue

            relative_path = resource.relative_to(index_path.parent)
            destination = destination_bundle / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resource, destination)
            copied += 1

    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Hugo posts content directory")
    parser.add_argument("output", type=Path, help="Generated posts directory")
    args = parser.parse_args()

    copied = copy_resources(args.source, args.output)
    print(f"Copied {copied} raw Markdown evidence file(s) to dated post routes.")


if __name__ == "__main__":
    main()
