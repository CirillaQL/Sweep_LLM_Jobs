#!/usr/bin/env python3
"""Read-only, bounded artifact collection and diagnosis for XPYD Job 255388."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable


MAX_CAPTURE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_CAPTURE_BYTES = 20 * 1024 * 1024
ERROR_RE = re.compile(
    r"(?i)(traceback|exception|\berror\b|failed|failure|timeout|timed out|"
    r"out of memory|oom|nccl|assert|exit code|systemexit)"
)
TEXT_SUFFIXES = {".log", ".out", ".err", ".txt", ".json", ".jsonl", ".csv", ".md"}


def tail_bytes(path: Path, limit: int = MAX_CAPTURE_BYTES) -> bytes:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        data = stream.read()
    if size > limit:
        newline = data.find(b"\n")
        if newline >= 0:
            data = data[newline + 1 :]
    return data


def safe_text(path: Path, limit: int = MAX_CAPTURE_BYTES) -> str:
    return tail_bytes(path, limit).decode("utf-8", errors="replace")


def inventory(roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            rows.append({"path": str(root), "exists": False})
            continue
        if root.is_file():
            paths = [root]
        else:
            paths = sorted(path for path in root.rglob("*") if path.is_file())
        for path in paths:
            try:
                stat = path.stat()
                rows.append({
                    "path": str(path),
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                })
            except OSError as exc:
                rows.append({"path": str(path), "exists": True, "error": str(exc)})
    return rows


def capture_name(path: Path, target_job_dir: Path, raw_root: Path) -> str:
    if path == target_job_dir or target_job_dir in path.parents:
        relative = path.relative_to(target_job_dir)
        prefix = "broker"
    else:
        relative = path.relative_to(raw_root)
        prefix = "raw"
    flattened = "__".join(relative.parts)
    return f"{prefix}__{flattened}"


def selected_logs(target_job_dir: Path, raw_root: Path, target_job_id: str) -> list[Path]:
    explicit = [
        target_job_dir / f"slurm-{target_job_id}.out",
        target_job_dir / f"slurm-{target_job_id}.err",
        target_job_dir / "status.json",
        target_job_dir / "status.log",
        raw_root / "frequency_table.json",
        raw_root / "feedback_events.jsonl",
    ]
    discovered: list[Path] = []
    if raw_root.is_dir():
        for path in raw_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            lowered = path.name.lower()
            if any(token in lowered for token in (
                "proxy", "server", "vllm", "audit", "summary", "request",
                "route", "clock", "frequency", "feedback", "error", "log",
            )):
                discovered.append(path)
    unique: dict[Path, Path] = {}
    for path in explicit + sorted(discovered, key=lambda item: str(item)):
        if path.is_file():
            unique.setdefault(path.resolve(), path)
    return list(unique.values())


def parse_table(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries", {})
        populated = {
            key: entry.get("value")
            for key, entry in entries.items()
            if isinstance(entry, dict) and entry.get("value") is not None
        }
        return {
            "available": True,
            "path": str(path),
            "entry_count": len(entries),
            "populated_count": len(populated),
            "missing": sorted(set(entries) - set(populated)),
            "populated": populated,
        }
    except Exception as exc:
        return {"available": True, "path": str(path), "parse_error": str(exc)}


def parse_events(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    counts: Counter[str] = Counter()
    parsed = 0
    malformed = 0
    last_events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            parsed += 1
            kind = str(event.get("event") or event.get("type") or "unknown")
            counts[kind] += 1
            last_events.append(event)
            last_events = last_events[-20:]
    return {
        "available": True,
        "path": str(path),
        "parsed_events": parsed,
        "malformed_lines": malformed,
        "event_counts": dict(sorted(counts.items())),
        "last_events": last_events,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-job-id", required=True)
    parser.add_argument("--target-job-dir", type=Path, required=True)
    parser.add_argument("--target-raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    captures = args.output_dir / "log_tails"
    captures.mkdir(exist_ok=True)

    manifest = inventory((args.target_job_dir, args.target_raw_root))
    logs = selected_logs(args.target_job_dir, args.target_raw_root, args.target_job_id)
    error_hits: list[dict[str, Any]] = []
    captured: list[dict[str, Any]] = []
    remaining_capture_bytes = MAX_TOTAL_CAPTURE_BYTES
    for source in logs:
        if remaining_capture_bytes <= 0:
            break
        capture_limit = min(MAX_CAPTURE_BYTES, remaining_capture_bytes)
        text = safe_text(source, capture_limit)
        captured_size = len(text.encode("utf-8"))
        remaining_capture_bytes -= captured_size
        destination = captures / capture_name(
            source, args.target_job_dir, args.target_raw_root
        )
        destination.write_text(text, encoding="utf-8")
        lines = text.splitlines()
        matches = [line[-1000:] for line in lines if ERROR_RE.search(line)]
        if matches:
            error_hits.append({"source": str(source), "matches": matches[-50:]})
        captured.append({
            "source": str(source),
            "capture": str(destination.relative_to(args.output_dir)),
            "source_size_bytes": source.stat().st_size,
            "captured_bytes": captured_size,
            "tail_only": source.stat().st_size > capture_limit,
        })

    table = parse_table(args.target_raw_root / "frequency_table.json")
    events = parse_events(args.target_raw_root / "feedback_events.jsonl")
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "read_only": True,
        "target_job_id": args.target_job_id,
        "target_job_dir": str(args.target_job_dir),
        "target_raw_root": str(args.target_raw_root),
        "target_job_dir_exists": args.target_job_dir.is_dir(),
        "target_raw_root_exists": args.target_raw_root.is_dir(),
        "frequency_table": table,
        "feedback_events": events,
        "error_excerpts": error_hits,
        "captured_logs": captured,
        "artifact_count": sum(1 for row in manifest if row.get("exists")),
    }
    (args.output_dir / "diagnosis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    markdown = [
        f"# Read-only diagnosis for Slurm Job {args.target_job_id}",
        "",
        f"- Raw root available: `{args.target_raw_root.is_dir()}`",
        f"- Discovered artifacts: `{result['artifact_count']}`",
        f"- Captured bounded logs: `{len(captured)}`",
        f"- Logs with error-pattern matches: `{len(error_hits)}`",
        f"- Frequency table available: `{table.get('available')}`",
        f"- Frequency table populated keys: `{table.get('populated_count', 0)}`",
        f"- Feedback events parsed: `{events.get('parsed_events', 0)}`",
        "",
        "See `diagnosis.json`, `artifact_manifest.json`, and `log_tails/` for evidence.",
    ]
    (args.output_dir / "diagnosis.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
