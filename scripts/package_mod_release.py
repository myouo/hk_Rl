#!/usr/bin/env python3
"""Package or verify the versioned HKRLEnvMod release archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hkrl.utils.mod_release import (
    ModReleaseError,
    build_mod_release,
    load_mod_release_metadata,
    verify_mod_release,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)

    package = subparsers.add_parser("package", help="build and self-verify the release ZIP")
    package.add_argument(
        "--dll",
        default="mod/HKRLEnvMod/bin/Release/HKRLEnvMod.dll",
    )
    package.add_argument(
        "--flatbuffers-dll",
        default="mod/HKRLEnvMod/bin/Release/Google.FlatBuffers.dll",
    )
    package.add_argument(
        "--evidence-json",
        help="default: versioned Hall of Gods JSON under runs/live",
    )
    package.add_argument(
        "--evidence-report",
        help="default: versioned Hall of Gods Markdown under runs/live",
    )
    package.add_argument(
        "--walk-evidence",
        default="runs/live/walk-smoothness-post-fix.json",
    )
    package.add_argument("--output-dir", default="dist/mod")
    package.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development-only: permit a dirty source worktree",
    )
    package.add_argument(
        "--allow-missing-tag",
        action="store_true",
        help="development-only: package before the version tag exists",
    )

    verify = subparsers.add_parser("verify", help="verify archive metadata and every hash")
    verify.add_argument(
        "--archive",
        help="default: versioned release ZIP under dist/mod",
    )
    verify.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a development archive whose manifest says git_dirty=true",
    )
    verify.add_argument(
        "--allow-missing-tag",
        action="store_true",
        help="development-only: verify an archive before the version tag exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    root = Path(args.repo_root).expanduser().resolve()
    try:
        if args.command == "package":
            metadata = load_mod_release_metadata(root)
            evidence_json = (
                args.evidence_json
                or f"runs/live/godhome-all-boss-sweep-v{metadata.mod_version}.json"
            )
            evidence_report = (
                args.evidence_report
                or f"runs/live/godhome-all-boss-sweep-v{metadata.mod_version}.md"
            )
            result = build_mod_release(
                repo_root=root,
                dll_path=_under_root(root, args.dll),
                flatbuffers_dll_path=_under_root(root, args.flatbuffers_dll),
                evidence_json_path=_under_root(root, evidence_json),
                evidence_report_path=_under_root(root, evidence_report),
                walk_evidence_path=_under_root(root, args.walk_evidence),
                output_dir=_under_root(root, args.output_dir),
                allow_dirty=args.allow_dirty,
                require_tag=not args.allow_missing_tag,
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "mod_version": metadata.mod_version,
                        "protocol_schema_version": metadata.protocol_schema_version,
                        "archive": str(result.archive.resolve()),
                        "archive_sha256": result.archive_sha256,
                        "manifest": str(result.manifest.resolve()),
                        "checksum": str(result.checksum.resolve()),
                    },
                    sort_keys=True,
                )
            )
            return 0

        metadata = load_mod_release_metadata(root)
        archive = args.archive or f"dist/mod/{metadata.archive_stem}.zip"
        manifest = verify_mod_release(
            _under_root(root, archive),
            repo_root=root,
            allow_dirty=args.allow_dirty,
            require_tag=not args.allow_missing_tag,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "archive": str(_under_root(root, archive).resolve()),
                    "mod_version": manifest["mod_version"],
                    "protocol_schema_version": manifest["protocol_schema_version"],
                    "git_sha": manifest["git_sha"],
                    "git_dirty": manifest["git_dirty"],
                },
                sort_keys=True,
            )
        )
        return 0
    except ModReleaseError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2


def _under_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    raise SystemExit(main())
