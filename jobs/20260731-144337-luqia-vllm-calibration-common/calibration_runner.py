#!/usr/bin/env python3
"""Run one resumable full-grid vLLM DVFS calibration shard."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


UTC = dt.timezone.utc
STOP_REQUESTED = threading.Event()


class UploadPending(RuntimeError):
    """The benchmark completed, but its durable ClickHouse upload is pending."""


def utc_now() -> str:
    return dt.datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f000")


def json_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        for row in rows
    )


class ClickHouse:
    def __init__(self) -> None:
        self.url = os.environ["CLICKHOUSE_URL"].rstrip("/") + "/"
        self.user = os.environ["CLICKHOUSE_USER"]
        self.password = os.environ["CLICKHOUSE_PASSWORD"]
        self.database = os.environ.get("CLICKHOUSE_DATABASE", "vllm_observability")

    def request(
        self,
        query: str,
        body: bytes = b"",
        *,
        settings: dict[str, str] | None = None,
        query_id: str = "",
        compressed: bool = False,
        timeout: int = 180,
    ) -> bytes:
        params = {"database": self.database, "query": query, "wait_end_of_query": "1"}
        if settings:
            params.update(settings)
        request = urllib.request.Request(
            self.url + "?" + urllib.parse.urlencode(params), data=body, method="POST"
        )
        request.add_header("X-ClickHouse-User", self.user)
        request.add_header("X-ClickHouse-Key", self.password)
        request.add_header("User-Agent", "vllm-calibration-uploader/1.0")
        if query_id:
            request.add_header("X-ClickHouse-Query-Id", query_id)
        if compressed:
            request.add_header("Content-Encoding", "gzip")
            request.add_header("Content-Type", "application/x-ndjson")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read(8192).decode("utf-8", "replace")
            raise RuntimeError(f"ClickHouse HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ClickHouse connection failed: {exc}") from exc

    def initialize(self, schema_path: Path) -> str:
        version = self.request("SELECT version() FORMAT TabSeparated", timeout=30).decode().strip()
        schema = schema_path.read_text(encoding="utf-8")
        for statement in (part.strip() for part in schema.split(";")):
            if statement:
                self.request(statement, timeout=60)
        return version

    def insert(self, table: str, rows: list[dict[str, Any]], label: str) -> None:
        if not rows:
            return
        payload = json_bytes(rows)
        digest = hashlib.sha256(table.encode() + b"\0" + label.encode() + b"\0" + payload).hexdigest()
        body = gzip.compress(payload, compresslevel=6, mtime=0)
        last_error: Exception | None = None
        for attempt, delay in enumerate((0, 2, 5, 10, 20, 40), start=1):
            if delay:
                time.sleep(delay)
            try:
                self.request(
                    f"INSERT INTO {self.database}.{table} FORMAT JSONEachRow",
                    body,
                    settings={"async_insert": "0", "insert_deduplication_token": digest},
                    query_id=f"cal-{table}-{digest[:24]}",
                    compressed=True,
                )
                return
            except RuntimeError as exc:
                last_error = exc
                retryable = any(
                    marker in str(exc)
                    for marker in ("connection failed", "HTTP 408", "HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")
                )
                if not retryable or attempt == 6:
                    break
                print(f"clickhouse_retry table={table} attempt={attempt}", flush=True)
        raise RuntimeError(f"ClickHouse insert failed for {table}: {last_error}")

    def completed(self, campaign: str, gpu_type: str, shard: int) -> set[tuple[str, int, int]]:
        campaign_q = campaign.replace("'", "''")
        gpu_q = gpu_type.replace("'", "''")
        query = f"""
            SELECT config_id, repeat_no, segment_no
            FROM {self.database}.calibration_runs
            WHERE campaign_id='{campaign_q}' AND gpu_type='{gpu_q}'
              AND shard_id={int(shard)} AND status='success'
            GROUP BY config_id, repeat_no, segment_no
            FORMAT JSONEachRow
        """
        raw = self.request(query).decode("utf-8", "replace")
        return {
            (row["config_id"], int(row["repeat_no"]), int(row["segment_no"]))
            for row in (json.loads(line) for line in raw.splitlines() if line.strip())
        }


def run_command(command: list[str], *, timeout: float = 120, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(
            f"command failed rc={result.returncode}: {shlex.join(command)}\n"
            f"stdout={result.stdout[-4000:]}\nstderr={result.stderr[-4000:]}"
        )
    return result


def gpu_query() -> list[dict[str, str]]:
    fields = "index,name,uuid,pstate,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu"
    result = run_command(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], timeout=30
    )
    keys = fields.split(",")
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in next(csv.reader([line]))]
        if len(values) == len(keys):
            rows.append(dict(zip(keys, values)))
    return rows


def set_clock(indices: range | list[int], target_mhz: int) -> None:
    for index in indices:
        run_command(
            ["sudo", "-n", "nvidia-smi", "-i", str(index), "-lgc", f"{target_mhz},{target_mhz}"],
            timeout=30,
        )


def reset_clocks(indices: range | list[int]) -> list[str]:
    errors = []
    for index in indices:
        result = run_command(
            ["sudo", "-n", "nvidia-smi", "-i", str(index), "-rgc"], timeout=30, check=False
        )
        if result.returncode:
            errors.append(f"gpu={index} rc={result.returncode} {result.stderr.strip()}")
    return errors


def value(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


class GPUMonitor:
    def __init__(self, gpu_count: int, interval_ms: int = 500) -> None:
        self.gpu_count = gpu_count
        self.interval_ms = interval_ms
        self.samples: list[dict[str, Any]] = []
        self.error = ""
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self.stop_event.set()
        self.thread.join(timeout=10)
        return self.samples

    def _run(self) -> None:
        sequence = 0
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                now_ns = time.time_ns()
                now = dt.datetime.fromtimestamp(now_ns / 1e9, UTC).strftime("%Y-%m-%d %H:%M:%S.%f000")
                rows = gpu_query()
                for row in rows[: self.gpu_count]:
                    self.samples.append(
                        {
                            "sample_seq": sequence,
                            "gpu_index": int(row["index"]),
                            "sampled_at": now,
                            "unix_ns": now_ns,
                            "sm_clock_mhz": value(row, "clocks.sm"),
                            "mem_clock_mhz": value(row, "clocks.mem"),
                            "gpu_util_pct": value(row, "utilization.gpu"),
                            "mem_util_pct": value(row, "utilization.memory"),
                            "memory_used_mib": value(row, "memory.used"),
                            "memory_total_mib": value(row, "memory.total"),
                            "power_w": value(row, "power.draw"),
                            "temperature_c": value(row, "temperature.gpu"),
                            "pstate": row["pstate"],
                        }
                    )
                sequence += 1
            except Exception as exc:  # monitoring must not kill the benchmark
                self.error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
            self.stop_event.wait(max(0.05, self.interval_ms / 1000 - elapsed))


def mean(values: list[float]) -> float:
    finite = [item for item in values if math.isfinite(item)]
    return statistics.fmean(finite) if finite else 0.0


def summarize_samples(samples: list[dict[str, Any]], target: int, tolerance: int, tp: int) -> dict[str, Any]:
    active = [sample for sample in samples if sample["gpu_util_pct"] >= 10 and math.isfinite(sample["sm_clock_mhz"])]
    clocks = [sample["sm_clock_mhz"] for sample in active]
    per_gpu = {}
    verified = True
    within_total = 0
    for gpu in range(tp):
        rows = [sample for sample in active if sample["gpu_index"] == gpu]
        within = sum(abs(sample["sm_clock_mhz"] - target) <= tolerance for sample in rows)
        ratio = within / len(rows) if rows else 0.0
        per_gpu[str(gpu)] = {"active_samples": len(rows), "within_tolerance_ratio": ratio}
        within_total += within
        if len(rows) < 3 or ratio < 0.90:
            verified = False
    grouped: dict[int, float] = {}
    for sample in samples:
        if math.isfinite(sample["power_w"]):
            grouped[sample["sample_seq"]] = grouped.get(sample["sample_seq"], 0.0) + sample["power_w"]
    total_power = list(grouped.values())
    timestamps = sorted({sample["unix_ns"] for sample in samples})
    monitor_duration = (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else 0.0
    return {
        "avg_total_power_w": mean(total_power),
        "min_total_power_w": min(total_power, default=0.0),
        "max_total_power_w": max(total_power, default=0.0),
        "energy_j": mean(total_power) * monitor_duration,
        "avg_gpu_util_pct": mean([sample["gpu_util_pct"] for sample in samples]),
        "avg_mem_util_pct": mean([sample["mem_util_pct"] for sample in samples]),
        "actual_sm_clock_min_mhz": min(clocks, default=0.0),
        "actual_sm_clock_mean_mhz": mean(clocks),
        "actual_sm_clock_max_mhz": max(clocks, default=0.0),
        "active_clock_sample_count": len(active),
        "active_clock_within_tolerance_ratio": within_total / len(active) if active else 0.0,
        "frequency_verified": verified,
        "per_gpu_frequency_verification": per_gpu,
    }


def metric(result: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in result and result[key] is not None:
            try:
                return float(result[key])
            except (TypeError, ValueError):
                pass
    return default


class VLLMServer:
    def __init__(self, args: argparse.Namespace, logs_dir: Path, max_freq: int, all_gpus: int) -> None:
        self.args = args
        self.logs_dir = logs_dir
        self.max_freq = max_freq
        self.all_gpus = all_gpus
        self.process: subprocess.Popen | None = None
        self.log_handle = None
        self.tp = 0

    def ensure(self, tp: int) -> None:
        if self.process and self.process.poll() is None and self.tp == tp:
            return
        self.stop()
        reset_errors = reset_clocks(range(self.all_gpus))
        if reset_errors:
            raise RuntimeError(f"cannot reset GPU clocks before server start: {reset_errors}")
        set_clock(range(tp), self.max_freq)
        self.tp = tp
        log_path = self.logs_dir / f"vllm-server-tp{tp}-{int(time.time())}.log"
        self.log_handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(tp))
        env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
        command = [
            self.args.vllm_bin,
            "serve",
            self.args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.args.port),
            "--tensor-parallel-size",
            str(tp),
            "--gpu-memory-utilization",
            str(self.args.gpu_memory_utilization),
            "--max-model-len",
            str(self.args.max_model_len),
            "--dtype",
            "float16",
        ]
        print(f"server_start tp={tp} command={shlex.join(command)}", flush=True)
        self.process = subprocess.Popen(
            command,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.args.server_ready_timeout_s
        url = f"http://127.0.0.1:{self.args.port}/v1/models"
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM server exited rc={self.process.returncode}; see {log_path}")
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        print(f"server_ready tp={tp}", flush=True)
                        return
            except Exception as exc:
                last_error = exc
            time.sleep(3)
        raise RuntimeError(f"vLLM readiness timeout: {last_error}; see {log_path}")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        self.process = None
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None
        self.tp = 0


def build_run_row(
    manifest: dict,
    config: dict,
    segment: dict,
    repeat_no: int,
    attempt: int,
    run_id: str,
    args: argparse.Namespace,
    gpu_names: list[str],
    started_at: str,
    finished_at: str,
    bench_rc: int,
    duration_s: float,
    result: dict[str, Any],
    summary: dict[str, Any],
    error: str,
    run_seed: int,
) -> dict[str, Any]:
    success = bench_rc == 0 and not error and summary["frequency_verified"]
    if bench_rc == 0 and not summary["frequency_verified"]:
        error = "target frequency verification failed: " + json.dumps(
            summary["per_gpu_frequency_verification"], sort_keys=True
        )
    return {
        "campaign_id": manifest["campaign_id"],
        "gpu_type": manifest["gpu_type"],
        "config_id": config["config_id"],
        "repeat_no": repeat_no,
        "segment_no": segment["segment_no"],
        "segment_count": segment["segment_count"],
        "infra_attempt": attempt,
        "shard_id": args.shard_id,
        "run_id": run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
        "slurm_array_task_id": int(os.environ.get("SLURM_ARRAY_TASK_ID", args.shard_id)),
        "hostname": socket.gethostname().split(".")[0],
        "model": args.model,
        "vllm_version": args.vllm_version,
        "gpu_names": gpu_names,
        "tp_degree": config["tp_degree"],
        "target_gpu_freq_mhz": config["gpu_freq_mhz"],
        "historical_mem_freq_mhz": config["mem_freq_mhz"],
        "input_len": config["input_len"],
        "output_len": config["output_len"],
        "request_rate": config["request_rate"],
        "num_prompts": segment["num_prompts"],
        "seed": run_seed,
        "source_steps": config["source_steps"],
        "started_at": started_at,
        "finished_at": finished_at,
        "status": "success" if success else "failed",
        "benchmark_rc": bench_rc,
        "benchmark_duration_s": duration_s,
        "completed_requests": int(metric(result, "completed", "completed_requests")),
        "failed_requests": int(metric(result, "failed", "failed_requests")),
        "total_input_tokens": int(metric(result, "total_input_tokens")),
        "total_output_tokens": int(metric(result, "total_output_tokens")),
        "request_throughput_rps": metric(result, "request_throughput", "request_throughput_rps"),
        "output_token_throughput_tps": metric(result, "output_throughput", "output_token_throughput_tps"),
        "total_token_throughput_tps": metric(result, "total_token_throughput", "total_token_throughput_tps"),
        "mean_ttft_ms": metric(result, "mean_ttft_ms"),
        "median_ttft_ms": metric(result, "median_ttft_ms"),
        "p99_ttft_ms": metric(result, "p99_ttft_ms"),
        "mean_tpot_ms": metric(result, "mean_tpot_ms"),
        "median_tpot_ms": metric(result, "median_tpot_ms"),
        "p99_tpot_ms": metric(result, "p99_tpot_ms"),
        "mean_itl_ms": metric(result, "mean_itl_ms"),
        "median_itl_ms": metric(result, "median_itl_ms"),
        "p99_itl_ms": metric(result, "p99_itl_ms"),
        **{key: summary[key] for key in (
            "avg_total_power_w", "min_total_power_w", "max_total_power_w", "energy_j",
            "avg_gpu_util_pct", "avg_mem_util_pct", "actual_sm_clock_min_mhz",
            "actual_sm_clock_mean_mhz", "actual_sm_clock_max_mhz", "active_clock_sample_count",
            "active_clock_within_tolerance_ratio", "frequency_verified",
        )},
        "frequency_tolerance_mhz": args.frequency_tolerance_mhz,
        "result_json": json.dumps(result, separators=(",", ":"), ensure_ascii=False),
        "error": error,
        "updated_at": utc_now(),
    }


def decorate_samples(
    samples: list[dict[str, Any]], manifest: dict, config: dict, segment: dict,
    repeat_no: int, run_id: str, shard_id: int
) -> list[dict[str, Any]]:
    ingested_at = utc_now()
    common = {
        "campaign_id": manifest["campaign_id"],
        "gpu_type": manifest["gpu_type"],
        "run_id": run_id,
        "config_id": config["config_id"],
        "repeat_no": repeat_no,
        "segment_no": segment["segment_no"],
        "shard_id": shard_id,
        "target_gpu_freq_mhz": config["gpu_freq_mhz"],
        "ingested_at": ingested_at,
    }
    return [{**common, **sample} for sample in samples]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def upload_bundle(client: ClickHouse, bundle_path: Path) -> dict[str, Any]:
    with gzip.open(bundle_path, "rt", encoding="utf-8") as handle:
        bundle = json.load(handle)
    row = bundle["run"]
    samples = bundle["samples"]
    run_id = row["run_id"]
    for index in range(0, len(samples), 5000):
        client.insert(
            "calibration_gpu_samples", samples[index:index + 5000],
            f"{run_id}-samples-{index // 5000}",
        )
    client.insert("calibration_runs", [row], f"{run_id}-run")
    bundle_path.unlink()
    if row["status"] == "success":
        for path in bundle_path.parent.iterdir():
            path.unlink()
        bundle_path.parent.rmdir()
    return row


def replay_spool(client: ClickHouse, output_dir: Path) -> int:
    replayed = 0
    for bundle_path in sorted((output_dir / "spool").glob("*/upload_bundle.json.gz")):
        row = upload_bundle(client, bundle_path)
        replayed += 1
        print(f"spool_replayed run_id={row['run_id']} status={row['status']}", flush=True)
    return replayed


def run_segment(
    client: ClickHouse | None,
    server: VLLMServer,
    manifest: dict,
    config: dict,
    segment: dict,
    repeat_no: int,
    attempt: int,
    args: argparse.Namespace,
    output_dir: Path,
    gpu_names: list[str],
) -> tuple[dict[str, Any], bool]:
    run_id = (
        f"{manifest['campaign_id']}-{manifest['gpu_type']}-{config['config_id']}-"
        f"rep{repeat_no}-seg{segment['segment_no']}-attempt{attempt}"
    )
    set_clock(range(config["tp_degree"]), config["gpu_freq_mhz"])
    for inactive in range(config["tp_degree"], manifest["gpus_per_node"]):
        reset_clocks([inactive])
    lock_state = gpu_query()[: config["tp_degree"]]
    print(
        f"segment_start run_id={run_id} tp={config['tp_degree']} target_mhz={config['gpu_freq_mhz']} "
        f"prompts={segment['num_prompts']} lock_state={json.dumps(lock_state, separators=(',', ':'))}",
        flush=True,
    )

    segment_dir = output_dir / "spool" / run_id
    segment_dir.mkdir(parents=True, exist_ok=True)
    result_path = segment_dir / "vllm_bench_result.json"
    bench_log = segment_dir / "vllm_bench.log"
    request_prefix = f"{run_id}-"
    run_seed = args.seed + (repeat_no - 1) * 1000 + segment["segment_no"] - 1
    command = [
        args.vllm_bin, "bench", "serve", "--backend", "vllm", "--model", args.model,
        "--base-url", f"http://127.0.0.1:{args.port}", "--dataset-name", "random",
        "--random-input-len", str(config["input_len"]), "--random-output-len", str(config["output_len"]),
        "--random-range-ratio", "0.0", "--num-prompts", str(segment["num_prompts"]),
        "--request-rate", str(config["request_rate"]), "--seed", str(run_seed), "--ignore-eos",
        "--request-id-prefix", request_prefix, "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,99", "--save-result", "--result-dir", str(segment_dir),
        "--result-filename", result_path.name,
    ]
    monitor = GPUMonitor(config["tp_degree"], args.monitor_interval_ms)
    started_at = utc_now()
    started = time.monotonic()
    bench_rc = 1
    error = ""
    monitor.start()
    try:
        timeout_s = max(
            args.minimum_benchmark_timeout_s,
            int(segment["historical_estimated_duration_s"] * 2 + 600),
        )
        with bench_log.open("w", encoding="utf-8") as handle:
            result_process = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, text=True, timeout=timeout_s
            )
        bench_rc = result_process.returncode
        if bench_rc:
            error = f"vllm bench serve exited with rc={bench_rc}; see {bench_log}"
    except subprocess.TimeoutExpired:
        bench_rc = 124
        error = "benchmark timeout"
    except Exception as exc:
        bench_rc = 1
        error = f"{type(exc).__name__}: {exc}"
    finally:
        samples = monitor.stop()
    duration_s = time.monotonic() - started
    finished_at = utc_now()
    result: dict[str, Any] = {}
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            error = error or f"result parse failed: {exc}"
    elif not error:
        error = "benchmark result JSON missing"
    summary = summarize_samples(
        samples, config["gpu_freq_mhz"], args.frequency_tolerance_mhz, config["tp_degree"]
    )
    if monitor.error:
        error = error or f"GPU monitor failed: {monitor.error}"
    row = build_run_row(
        manifest, config, segment, repeat_no, attempt, run_id, args, gpu_names,
        started_at, finished_at, bench_rc, duration_s, result, summary, error, run_seed,
    )
    sample_rows = decorate_samples(
        samples, manifest, config, segment, repeat_no, run_id, args.shard_id
    )
    append_jsonl(output_dir / "run_results.jsonl", row)
    append_jsonl(
        output_dir / "frequency_verification.jsonl",
        {"run_id": run_id, **summary, "monitor_error": monitor.error},
    )
    if client:
        bundle_path = segment_dir / "upload_bundle.json.gz"
        with gzip.open(bundle_path, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump({"run": row, "samples": sample_rows}, handle, separators=(",", ":"))
        try:
            upload_bundle(client, bundle_path)
        except Exception as exc:
            raise UploadPending(f"durable spool retained at {bundle_path}: {exc}") from exc
    print(
        f"segment_finish run_id={run_id} status={row['status']} duration_s={duration_s:.3f} "
        f"frequency_verified={row['frequency_verified']} samples={len(samples)}",
        flush=True,
    )
    return row, row["status"] == "success"


def shard_row(manifest: dict, args: argparse.Namespace, state: str, planned: int, completed: int,
              failed: int, skipped: int, message: str = "") -> dict[str, Any]:
    return {
        "campaign_id": manifest["campaign_id"], "gpu_type": manifest["gpu_type"],
        "shard_id": args.shard_id, "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
        "hostname": socket.gethostname().split(".")[0], "state": state,
        "planned_runs": planned, "completed_runs": completed, "failed_runs": failed,
        "skipped_runs": skipped, "message": message, "updated_at": utc_now(),
    }


def check_bench_flags(vllm_bin: str) -> str:
    version = run_command([vllm_bin, "--version"], timeout=60, check=False).stdout.strip()
    help_result = run_command([vllm_bin, "bench", "serve", "--help"], timeout=120)
    required = (
        "--random-range-ratio", "--request-id-prefix", "--percentile-metrics",
        "--metric-percentiles", "--save-result", "--result-filename",
    )
    missing = [flag for flag in required if flag not in help_result.stdout]
    if missing:
        raise RuntimeError(f"installed vLLM bench serve lacks required flags: {missing}")
    return version or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-bin", default="/data/users/chjing/miniforge3/envs/cuda-env/bin/vllm")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--monitor-interval-ms", type=int, default=500)
    parser.add_argument("--frequency-tolerance-mhz", type=int, default=30)
    parser.add_argument("--server-ready-timeout-s", type=int, default=900)
    parser.add_argument("--minimum-benchmark-timeout-s", type=int, default=1800)
    parser.add_argument("--deadline-seconds", type=int, default=84600)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--skip-clickhouse", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not 0 <= args.shard_id < manifest["shard_count"]:
        raise SystemExit(f"invalid shard {args.shard_id}; expected 0..{manifest['shard_count'] - 1}")
    configs = manifest["shards"][args.shard_id]
    units = [
        (config, repeat_no, segment)
        for config in configs
        for repeat_no in range(1, config["repeats"] + 1)
        for segment in config["segments"]
    ]
    if args.max_segments:
        units = units[: args.max_segments]
    print(
        f"plan campaign={manifest['campaign_id']} gpu_type={manifest['gpu_type']} shard={args.shard_id} "
        f"configs={len(configs)} segments={len(units)}",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps([{"config_id": c["config_id"], "repeat": r, **s} for c, r, s in units], indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "spool").mkdir(exist_ok=True)
    all_gpu_rows = gpu_query()
    if len(all_gpu_rows) < manifest["gpus_per_node"]:
        raise RuntimeError(
            f"expected {manifest['gpus_per_node']} GPUs, nvidia-smi returned {len(all_gpu_rows)}"
        )
    expected_name = "L40S" if manifest["gpu_type"] == "l40s" else "L4"
    gpu_names = [row["name"] for row in all_gpu_rows[: manifest["gpus_per_node"]]]
    if any(expected_name.lower() not in name.lower() for name in gpu_names):
        raise RuntimeError(f"wrong GPU pool: expected {expected_name}, found {gpu_names}")

    args.vllm_version = check_bench_flags(args.vllm_bin)
    client = None if args.skip_clickhouse else ClickHouse()
    if client:
        version = client.initialize(args.schema)
        print(f"clickhouse_preflight_ok version={version}", flush=True)
        replayed = replay_spool(client, args.output_dir)
        print(f"spool_replay_count={replayed}", flush=True)
        completed_keys = client.completed(manifest["campaign_id"], manifest["gpu_type"], args.shard_id)
    else:
        completed_keys = set()
    planned = len(units)
    completed = failed = skipped = 0
    start_monotonic = time.monotonic()
    max_freq = max(item["gpu_freq_mhz"] for item in configs)
    server = VLLMServer(args, args.output_dir, max_freq, manifest["gpus_per_node"])

    def stop_handler(signum: int, _frame: Any) -> None:
        print(f"signal_received={signum}", flush=True)
        STOP_REQUESTED.set()

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    if client:
        client.insert("calibration_shards", [shard_row(manifest, args, "running", planned, 0, 0, 0)], "start")
    try:
        for config, repeat_no, segment in units:
            key = (config["config_id"], repeat_no, segment["segment_no"])
            if key in completed_keys:
                skipped += 1
                continue
            remaining = args.deadline_seconds - (time.monotonic() - start_monotonic)
            required = max(900, segment["historical_estimated_duration_s"] * 1.8 + 600)
            if STOP_REQUESTED.is_set() or remaining < required:
                message = f"checkpointed before next segment; remaining_s={remaining:.1f} required_s={required:.1f}"
                print(message, flush=True)
                if client:
                    client.insert(
                        "calibration_shards",
                        [shard_row(manifest, args, "checkpointed", planned, completed, failed, skipped, message)],
                        f"checkpoint-{completed}-{failed}-{skipped}",
                    )
                return 75
            try:
                server.ensure(config["tp_degree"])
            except Exception as exc:
                failed += 1
                print(f"server_error config={config['config_id']} error={type(exc).__name__}:{exc}", flush=True)
                server.stop()
                continue
            success = False
            for attempt in (1, 2):
                try:
                    _row, success = run_segment(
                        client, server, manifest, config, segment, repeat_no, attempt,
                        args, args.output_dir, gpu_names,
                    )
                except UploadPending as exc:
                    message = str(exc)
                    print(f"upload_checkpoint={message}", flush=True)
                    if client:
                        with contextlib.suppress(Exception):
                            client.insert(
                                "calibration_shards",
                                [shard_row(manifest, args, "upload_pending", planned, completed, failed, skipped, message)],
                                f"upload-pending-{completed}-{failed}-{skipped}",
                            )
                    return 75
                except Exception as exc:
                    print(
                        f"segment_exception config={config['config_id']} repeat={repeat_no} "
                        f"segment={segment['segment_no']} attempt={attempt} error={type(exc).__name__}:{exc}",
                        flush=True,
                    )
                    success = False
                if success:
                    completed += 1
                    break
                server.stop()
                if attempt == 1 and not STOP_REQUESTED.is_set():
                    server.ensure(config["tp_degree"])
            if not success:
                failed += 1
        state = "complete" if failed == 0 else "complete_with_failures"
        if client:
            client.insert(
                "calibration_shards",
                [shard_row(manifest, args, state, planned, completed, failed, skipped)],
                f"finish-{completed}-{failed}-{skipped}",
            )
        return 0 if failed == 0 else 2
    finally:
        server.stop()
        errors = reset_clocks(range(manifest["gpus_per_node"]))
        print(f"clock_reset_errors={json.dumps(errors)}", flush=True)
        if errors:
            raise RuntimeError(f"failed to restore automatic GPU clocks: {errors}")


if __name__ == "__main__":
    raise SystemExit(main())
