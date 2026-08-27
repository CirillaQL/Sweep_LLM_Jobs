#!/usr/bin/env python3
"""Resolve the newest accepted prerequisite unless an explicit path is supplied."""

import argparse
import json
from pathlib import Path


def accepted(path: Path, kind: str) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not value.get("valid"):
        return False
    if kind == "history":
        return bool(
            value.get("ready_for_phase4b")
            and value.get("configuration_aggregates")
            and not value.get("models_trained_or_used")
        )
    return str(value.get("phase", "")).startswith("3D-A_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("actuator", "history"), required=True)
    parser.add_argument("--override")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.override:
        candidate = Path(args.override)
        if not candidate.is_file() or not accepted(candidate, args.kind):
            raise SystemExit(f"explicit {args.kind} evidence is not accepted: {candidate}")
        print(candidate)
        return 0
    name = "actuator_audit.json" if args.kind == "actuator" else "phase4a_summary.json"
    candidates = [path for path in args.root.rglob(name) if accepted(path, args.kind)]
    if not candidates:
        raise SystemExit(f"no accepted {args.kind} evidence found under {args.root}")
    print(max(candidates, key=lambda path: path.stat().st_mtime))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
