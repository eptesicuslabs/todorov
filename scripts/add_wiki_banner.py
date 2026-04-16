#!/usr/bin/env python3
"""Batch-add banners to wiki articles per OPERATING_DIRECTIVE.md.

Usage:
    python scripts/add_wiki_banner.py <glob> <banner_line>
    python scripts/add_wiki_banner.py --per-file <map.json>

banner is inserted on the line immediately after the title heading,
separated from the title by one blank line and from the following
content by one blank line.

skips any file that already has a line starting with 'status:' in the
first 10 non-heading lines. skips files whose basename starts with _.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def has_banner(content: str) -> bool:
    for line in content.split("\n")[:15]:
        if line.startswith("status:"):
            return True
    return False


def add_banner(content: str, banner: str) -> str:
    lines = content.split("\n")
    if not lines:
        return content
    title_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title_idx = i
            break
    if title_idx < 0:
        new_lines = [banner, ""] + lines
        return "\n".join(new_lines)
    new_lines = lines[: title_idx + 1]
    new_lines.append("")
    new_lines.append(banner)
    rest = lines[title_idx + 1 :]
    if rest and rest[0] == "":
        new_lines.append("")
        new_lines.extend(rest[1:])
    else:
        new_lines.append("")
        new_lines.extend(rest)
    return "\n".join(new_lines)


def process_file(path: Path, banner: str) -> str:
    if path.name.startswith("_"):
        return "skip-underscore"
    content = path.read_text(encoding="utf-8")
    if has_banner(content):
        return "skip-has-banner"
    new_content = add_banner(content, banner)
    path.write_text(new_content, encoding="utf-8")
    return "added"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("glob", help="glob pattern relative to cwd (e.g. neuroloc/wiki/synthesis/*.md)")
    parser.add_argument("banner", help="banner text (one line, starting with 'status:')")
    parser.add_argument("--exclude", action="append", default=[], help="filename basenames to exclude")
    args = parser.parse_args()
    if not args.banner.startswith("status:"):
        print(f"ERROR: banner must start with 'status:' — got {args.banner!r}", file=sys.stderr)
        return 2
    root = Path.cwd()
    paths = sorted(root.glob(args.glob))
    if not paths:
        print(f"ERROR: glob matched no files: {args.glob}", file=sys.stderr)
        return 2
    excluded = set(args.exclude)
    results: dict[str, list[str]] = {"added": [], "skip-underscore": [], "skip-has-banner": [], "skip-excluded": []}
    for p in paths:
        if p.name in excluded:
            results["skip-excluded"].append(p.name)
            continue
        status = process_file(p, args.banner)
        results[status].append(p.name)
    for key, names in results.items():
        print(f"{key}: {len(names)}")
        for name in names:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
