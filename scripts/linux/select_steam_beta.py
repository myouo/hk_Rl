#!/usr/bin/env python3
"""Select Hollow Knight's supported Steam beta in an appmanifest.

Steam should be fully stopped before applying this change. The script makes a
timestamped backup and atomically updates both UserConfig and MountedConfig.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BRANCH = "1.5.78.11833"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    manifest = args.manifest.expanduser().resolve()
    branch = _validate_branch(args.branch)
    original = manifest.read_text(encoding="utf-8")
    updated = _set_beta_key(original, branch)
    changed = updated != original
    backup: Path | None = None

    if changed and not args.dry_run:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = manifest.with_name(f"{manifest.name}.bak-{timestamp}")
        shutil.copy2(manifest, backup)
        _atomic_write(manifest, updated)

    print(
        json.dumps(
            {
                "backup": None if backup is None else str(backup),
                "branch": branch,
                "changed": changed,
                "dry_run": args.dry_run,
                "manifest": str(manifest),
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_branch(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError(
            "branch must contain only letters, digits, dot, underscore, or dash"
        )
    return value


def _set_beta_key(text: str, branch: str) -> str:
    updated = text
    for section in ("UserConfig", "MountedConfig"):
        updated = _set_section_beta(updated, section, branch)
    return updated


def _set_section_beta(text: str, section: str, branch: str) -> str:
    pattern = re.compile(
        rf'(?P<header>^[ \t]*"{re.escape(section)}"[ \t]*\r?\n'
        rf"^[ \t]*\{{[ \t]*\r?\n)"
        rf"(?P<body>.*?)"
        rf"(?P<footer>^[ \t]*\}})",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"manifest is missing {section}")

    body = match.group("body")
    beta_pattern = re.compile(
        r'^(?P<indent>[ \t]*)"betakey"[ \t]+"[^"]*"[ \t]*\r?$',
        re.IGNORECASE | re.MULTILINE,
    )
    if beta_pattern.search(body):
        body = beta_pattern.sub(
            lambda item: f'{item.group("indent")}"betakey"\t\t"{branch}"',
            body,
            count=1,
        )
    else:
        indent_match = re.search(r"^(?P<indent>[ \t]+)\S", body, re.MULTILINE)
        indent = indent_match.group("indent") if indent_match else "\t\t"
        body = f'{body}{indent}"betakey"\t\t"{branch}"\n'

    return (
        text[: match.start()]
        + match.group("header")
        + body
        + match.group("footer")
        + text[match.end() :]
    )


def _atomic_write(path: Path, text: str) -> None:
    mode = path.stat().st_mode
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
