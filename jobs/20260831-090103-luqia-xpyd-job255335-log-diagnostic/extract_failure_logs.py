#!/usr/bin/env python3
"""Extract bounded, redacted failure evidence from an XPYD cache directory."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
from pathlib import Path


MAX_INVENTORY_FILES = 20000
MAX_SELECTED_LOGS = 40
TAIL_LINES_PER_FILE = 500
MAX_LINE_CHARS = 4000
MAX_SCAN_BYTES_PER_FILE = 24 * 1024 * 1024
MAX_SCAN_BYTES_TOTAL = 192 * 1024 * 1024
MAX_ERROR_MATCHES = 2500
MAX_SMALL_ARTIFACT_BYTES = 256 * 1024
MAX_TAIL_OUTPUT_CHARS = 3 * 1024 * 1024
MAX_ERROR_OUTPUT_CHARS = 2 * 1024 * 1024

TEXT_SUFFIXES = {".log", ".out", ".err", ".txt"}
ERROR_RE = re.compile(
    r"traceback|exception|error|failed|failure|timeout|timed out|killed|oom|"
    r"out of memory|cuda error|nccl|connection refused|address already in use|"
    r"binarydvfserror|assertion|invalid|does not satisfy|did not converge|"
    r"not found|missing",
    re.IGNORECASE,
)
SECRET_PATTERNS = [
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+"),
    re.compile(r"(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"\bhf_[A-Za-z0-9]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
]
KEY_JSON_NAMES = {
    "job_failure.json",
    "preflight.json",
    "binary_dvfs_audit.json",
    "frequency_grids.json",
    "selected_frequencies.json",
    "capabilities.json",
}


def redact(value: str) -> str:
    value = value.replace("\x00", "")
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            value = pattern.sub(r"\1<REDACTED>", value)
        else:
            value = pattern.sub("<REDACTED>", value)
    return value[:MAX_LINE_CHARS]


def safe_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def inventory(root: Path) -> list[dict]:
    rows: list[dict] = []
    for directory, dirnames, filenames in os.walk(str(root), followlinks=False):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            path = Path(directory) / filename
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                stat = path.stat()
                rows.append({
                    "path": safe_relative(path, root),
                    "size_bytes": stat.st_size,
                    "mtime_unix_s": stat.st_mtime,
                })
            except (OSError, ValueError):
                continue
            if len(rows) >= MAX_INVENTORY_FILES:
                return rows
    return rows


def priority(row: dict) -> tuple:
    rel = row["path"].lower()
    name = Path(rel).name
    if name == "job.log":
        rank = 0
    elif name in {"slurm-255335.err", "slurm-255335.out"}:
        rank = 1
    elif name.startswith("vllm_") and name.endswith(".log"):
        rank = 2
    elif "monitor.log" in name or "proxy" in name:
        rank = 3
    else:
        rank = 4
    return (rank, -float(row["mtime_unix_s"]), rel)


def select_logs(rows: list[dict]) -> list[dict]:
    candidates = []
    for row in rows:
        rel = row["path"]
        path = Path(rel)
        lower = rel.lower()
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if "live_clock/" in lower and not lower.endswith("monitor.log"):
            continue
        candidates.append(row)
    return sorted(candidates, key=priority)[:MAX_SELECTED_LOGS]


def tail_lines(path: Path) -> list[str]:
    tail = collections.deque(maxlen=TAIL_LINES_PER_FILE)
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            tail.append(redact(line.rstrip("\n")))
    return list(tail)


def scan_errors(path: Path, byte_budget: int) -> tuple[list[tuple[int, str]], int]:
    matches: list[tuple[int, str]] = []
    consumed = 0
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if consumed >= byte_budget:
                break
            consumed += len(raw)
            text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if ERROR_RE.search(text):
                matches.append((line_number, redact(text)))
                if len(matches) >= MAX_ERROR_MATCHES:
                    break
    return matches, consumed


def copy_small_artifacts(root: Path, output: Path, rows: list[dict]) -> list[str]:
    copied: list[str] = []
    artifact_dir = output / "key_artifacts"
    for row in rows:
        rel = row["path"]
        if Path(rel).name not in KEY_JSON_NAMES:
            continue
        if int(row["size_bytes"]) > MAX_SMALL_ARTIFACT_BYTES:
            continue
        source = root / rel
        digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:10]
        target = artifact_dir / (digest + "-" + Path(rel).name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(rel)
    return copied


def run(target: Path, output: Path) -> None:
    target = target.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = inventory(target)
    selected = select_logs(rows)

    with (output / "file_inventory.tsv").open("w", encoding="utf-8") as stream:
        stream.write("path\tsize_bytes\tmtime_unix_s\n")
        for row in rows:
            stream.write("{path}\t{size_bytes}\t{mtime_unix_s:.6f}\n".format(**row))

    error_count = 0
    bytes_scanned = 0
    tail_chars_written = 0
    error_chars_written = 0
    tails_truncated = False
    errors_truncated_by_size = False
    matched_files: dict[str, int] = {}
    with (output / "log_tails.txt").open("w", encoding="utf-8") as tails, \
            (output / "error_matches.txt").open("w", encoding="utf-8") as errors:
        for row in selected:
            rel = row["path"]
            path = target / rel
            heading = "\n===== {} (last {} lines) =====\n".format(
                rel, TAIL_LINES_PER_FILE
            )
            if tail_chars_written + len(heading) <= MAX_TAIL_OUTPUT_CHARS:
                tails.write(heading)
                tail_chars_written += len(heading)
            else:
                tails_truncated = True
            try:
                for line in tail_lines(path):
                    rendered = line + "\n"
                    if tail_chars_written + len(rendered) > MAX_TAIL_OUTPUT_CHARS:
                        tails_truncated = True
                        break
                    tails.write(rendered)
                    tail_chars_written += len(rendered)
            except OSError as exc:
                rendered = "<READ_ERROR: {}>\n".format(redact(str(exc)))
                if tail_chars_written + len(rendered) <= MAX_TAIL_OUTPUT_CHARS:
                    tails.write(rendered)
                    tail_chars_written += len(rendered)
                else:
                    tails_truncated = True

            remaining = MAX_SCAN_BYTES_TOTAL - bytes_scanned
            if remaining <= 0 or error_count >= MAX_ERROR_MATCHES:
                continue
            budget = min(MAX_SCAN_BYTES_PER_FILE, remaining)
            try:
                matches, consumed = scan_errors(path, budget)
            except OSError as exc:
                errors.write("{}:<READ_ERROR>:{}\n".format(rel, redact(str(exc))))
                continue
            bytes_scanned += consumed
            if matches:
                matched_files[rel] = len(matches)
            for line_number, line in matches:
                if error_count >= MAX_ERROR_MATCHES:
                    break
                rendered = "{}:{}:{}\n".format(rel, line_number, line)
                if error_chars_written + len(rendered) > MAX_ERROR_OUTPUT_CHARS:
                    errors_truncated_by_size = True
                    break
                errors.write(rendered)
                error_chars_written += len(rendered)
                error_count += 1

    copied = copy_small_artifacts(target, output, rows)
    summary = {
        "valid": True,
        "read_only": True,
        "target_cache": str(target),
        "inventory_file_count": len(rows),
        "inventory_truncated": len(rows) >= MAX_INVENTORY_FILES,
        "selected_log_count": len(selected),
        "selected_logs": [row["path"] for row in selected],
        "error_match_count": error_count,
        "error_matches_truncated": (
            error_count >= MAX_ERROR_MATCHES or errors_truncated_by_size
        ),
        "log_tails_truncated": tails_truncated,
        "error_match_files": matched_files,
        "bytes_scanned": bytes_scanned,
        "copied_key_artifacts": copied,
        "limits": {
            "tail_lines_per_file": TAIL_LINES_PER_FILE,
            "max_selected_logs": MAX_SELECTED_LOGS,
            "max_error_matches": MAX_ERROR_MATCHES,
            "max_scan_bytes_total": MAX_SCAN_BYTES_TOTAL,
            "max_tail_output_chars": MAX_TAIL_OUTPUT_CHARS,
            "max_error_output_chars": MAX_ERROR_OUTPUT_CHARS,
        },
    }
    with (output / "diagnostic_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.target.is_dir():
        parser.error("target must be an existing directory")
    run(args.target, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
