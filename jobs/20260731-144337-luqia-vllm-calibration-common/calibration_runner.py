#!/usr/bin/env python3
"""Run one resumable full-grid vLLM DVFS calibration shard."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


UTC = dt.timezone.utc
STOP_REQUESTED = threading.Event()


def utc_now() -> str:
    return dt.datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f000")


def run_command(command: list[str], *, timeout: float = 120, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(
            f"command failed rc={result.returncode}: {shlex.join(command)}\n"
            f"stdout={result.stdout[-4000:]}\nstderr={result.stderr[-4000:]}"
        )
    return result


def gpu_query() -> list[dict[str, str]]:
    fields = "index,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu"
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
        self.network_interface = default_network_interface()
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
                rx_bytes, tx_bytes = network_counters(self.network_interface)
                rows = gpu_query()
                for row in rows[: self.gpu_count]:
                    self.samples.append(
                        {
                            "sample_seq": sequence,
                            "gpu_index": int(row["index"]),
                            "gpu_uuid": row["uuid"],
                            "sampled_at": now,
                            "unix_ns": now_ns,
                            "sm_clock_mhz": value(row, "clocks.sm"),
                            "mem_clock_mhz": value(row, "clocks.mem"),
                            "gpu_util_pct": value(row, "utilization.gpu"),
                            "mem_util_pct": value(row, "utilization.memory"),
                            "memory_used_mib": value(row, "memory.used"),
                            "memory_total_mib": value(row, "memory.total"),
                            "power_w": value(row, "power.draw"),
                            "power_limit_w": value(row, "power.limit"),
                            "temperature_c": value(row, "temperature.gpu"),
                            "pstate": row["pstate"],
                            "network_interface": self.network_interface,
                            "rx_bytes": rx_bytes,
                            "tx_bytes": tx_bytes,
                        }
                    )
                sequence += 1
            except Exception as exc:  # monitoring must not kill the benchmark
                self.error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
            self.stop_event.wait(max(0.05, self.interval_ms / 1000 - elapsed))


def default_network_interface() -> str:
    try:
        for line in Path("/proc/net/route").read_text().splitlines()[1:]:
            columns = line.split()
            if len(columns) >= 4 and columns[1] == "00000000":
                return columns[0]
    except OSError:
        pass
    return ""


def network_counters(interface: str) -> tuple[int, int]:
    if not interface:
        return 0, 0


def read_int(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return 0
    base = Path("/sys/class/net") / interface / "statistics"
    try:
        return (
            int((base / "rx_bytes").read_text().strip()),
            int((base / "tx_bytes").read_text().strip()),
        )
    except (OSError, ValueError):
        return 0, 0


def stop_process(process: subprocess.Popen | None, timeout: int = 15) -> None:
    if process is None or process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def wait_http(url: str, timeout_s: int = 60) -> None:
    deadline = time.monotonic() + timeout_s
    error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            error = exc
        time.sleep(1)
    raise RuntimeError(f"readiness timeout url={url}: {error}")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def csv_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def write_telemetry_csv(
    path: Path, samples: list[dict[str, Any]], hostname: str
) -> None:
    fields = [
        "unix_ts", "role", "hostname", "gpu_index", "gpu_uuid",
        "network_interface", "rx_bytes", "tx_bytes", "rx_bytes_per_s",
        "tx_bytes_per_s", "gpu_util_pct", "gpu_power_w", "gpu_sm_mhz",
        "gpu_memory_used_mib", "gpu_mem_util_pct", "gpu_power_limit_w",
        "gpu_mem_clock_mhz", "gpu_temperature_c", "gpu_memory_total_mib",
        "gpu_pstate", "dvfs_mode",
    ]
    previous: dict[int, dict[str, Any]] = {}
    rows = []
    for sample in samples:
        index = int(sample["gpu_index"])
        prior = previous.get(index)
        elapsed = (
            (sample["unix_ns"] - prior["unix_ns"]) / 1e9 if prior else 0.0
        )
        rows.append(
            {
                "unix_ts": sample["unix_ns"] / 1e9,
                "role": "combined",
                "hostname": hostname,
                "gpu_index": index,
                "gpu_uuid": sample["gpu_uuid"],
                "network_interface": sample["network_interface"],
                "rx_bytes": sample["rx_bytes"] if index == 0 else 0,
                "tx_bytes": sample["tx_bytes"] if index == 0 else 0,
                "rx_bytes_per_s": (
                    (sample["rx_bytes"] - prior["rx_bytes"]) / elapsed
                    if index == 0 and prior and elapsed > 0 else 0
                ),
                "tx_bytes_per_s": (
                    (sample["tx_bytes"] - prior["tx_bytes"]) / elapsed
                    if index == 0 and prior and elapsed > 0 else 0
                ),
                "gpu_util_pct": sample["gpu_util_pct"],
                "gpu_power_w": sample["power_w"],
                "gpu_sm_mhz": sample["sm_clock_mhz"],
                "gpu_memory_used_mib": sample["memory_used_mib"],
                "gpu_mem_util_pct": sample["mem_util_pct"],
                "gpu_power_limit_w": sample["power_limit_w"],
                "gpu_mem_clock_mhz": sample["mem_clock_mhz"],
                "gpu_temperature_c": sample["temperature_c"],
                "gpu_memory_total_mib": sample["memory_total_mib"],
                "gpu_pstate": sample["pstate"],
                "dvfs_mode": "fixed_core_clock",
            }
        )
        previous[index] = sample
    write_csv(path, fields, rows)


PROMETHEUS_SAMPLE_RE = re.compile(
    r'^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)'
)
PROMETHEUS_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')


def capture_histograms(
    url: str, output: Path, window_id: str, capture_kind: str
) -> None:
    with urllib.request.urlopen(url, timeout=10) as response:
        text_body = response.read().decode("utf-8", "replace")
    buckets: list[tuple[str, dict[str, str], str, int]] = []
    counts: dict[tuple[str, str], int] = {}
    sums: dict[tuple[str, str], float] = {}
    for line in text_body.splitlines():
        match = PROMETHEUS_SAMPLE_RE.match(line)
        if not match or line.startswith("#"):
            continue
        metric, raw_labels, raw_value = match.groups()
        labels = {
            key: value.replace(r'\"', '"').replace(r"\\", "\\")
            for key, value in PROMETHEUS_LABEL_RE.findall(raw_labels or "")
        }
        try:
            numeric = float(raw_value)
        except ValueError:
            continue
        if metric.endswith("_bucket") and "le" in labels:
            le = labels.pop("le")
            buckets.append((metric[:-7], labels, le, int(numeric)))
        elif metric.endswith("_count"):
            key = (metric[:-6], json.dumps(labels, sort_keys=True))
            counts[key] = int(numeric)
        elif metric.endswith("_sum"):
            key = (metric[:-4], json.dumps(labels, sort_keys=True))
            sums[key] = numeric
    captured_ns = time.time_ns()
    rows = []
    for metric, labels, raw_le, cumulative in buckets:
        labels_json = json.dumps(labels, sort_keys=True)
        rows.append(
            {
                "window_id": window_id,
                "captured_unix_ns": captured_ns,
                "role": "combined",
                "capture_kind": capture_kind,
                "metric": metric,
                "labels_json": labels_json,
                "bucket_le": 1e308 if raw_le == "+Inf" else float(raw_le),
                "cumulative_count": cumulative,
                "histogram_count": counts.get((metric, labels_json), 0),
                "histogram_sum": sums.get((metric, labels_json), 0.0),
            }
        )
    fields = [
        "window_id", "captured_unix_ns", "role", "capture_kind",
        "metric", "labels_json", "bucket_le", "cumulative_count",
        "histogram_count", "histogram_sum",
    ]
    existing = []
    if output.exists() and output.stat().st_size:
        with output.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    write_csv(output, fields, existing + rows)


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


class VLLMServer:
    def __init__(self, args: argparse.Namespace, logs_dir: Path, max_freq: int, all_gpus: int) -> None:
        self.args = args
        self.logs_dir = logs_dir
        self.max_freq = max_freq
        self.all_gpus = all_gpus
        self.process: subprocess.Popen | None = None
        self.log_handle = None
        self.log_path: Path | None = None
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
        self.log_path = log_path
        self.log_handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(tp))
        env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
        env["PYTHONPATH"] = str(self.args.otel_bundle) + (
            f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
        )
        # vLLM 0.15 defaults to the OTLP gRPC exporter.  The cluster's
        # deliberately self-contained bundle provides the HTTP/protobuf
        # exporter, matching the local collector's /v1/traces endpoint.
        # Select it explicitly so vLLM does not try to import the absent
        # opentelemetry-exporter-otlp-proto-grpc package.
        env["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "http/protobuf"
        env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = (
            self.args.otlp_traces_endpoint
        )
        env["OTEL_EXPORTER_OTLP_TRACES_INSECURE"] = "true"
        env["OTEL_SERVICE_NAME"] = "vllm-single-pool"
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
            "--max-num-seqs",
            str(self.args.max_num_seqs),
            "--max-num-batched-tokens",
            str(self.args.max_num_batched_tokens),
            "--enable-request-id-headers",
            "--enable-prefix-caching",
            "--kv-cache-metrics",
            "--kv-cache-metrics-sample",
            "1.0",
            "--enable-logging-iteration-details",
            "--collect-detailed-traces",
            "all",
            "--otlp-traces-endpoint",
            self.args.otlp_traces_endpoint,
            "--enable-mfu-metrics",
            "--enable-log-requests",
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_segment(
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

    segment_dir = output_dir / "full_observability" / run_id
    segment_dir.mkdir(parents=True, exist_ok=True)
    workload_path = segment_dir / "workload.csv"
    run_seed = args.seed + (repeat_no - 1) * 1000 + segment["segment_no"] - 1
    duration_s = segment["num_prompts"] / config["request_rate"]
    workload_fields = [
        "window_id", "duration_s", "request_rate", "input_tokens",
        "max_tokens", "max_concurrency", "timeout_s", "retries",
        "cancel_fraction", "cancel_after_s", "shared_prefix_tokens",
        "inter_window_pause_s", "tp_degree", "target_gpu_freq_mhz",
        "config_id", "manual_frequency_control", "scheduler_prediction",
        "random_seed",
    ]
    write_csv(
        workload_path,
        workload_fields,
        [{
            "window_id": run_id,
            "duration_s": f"{duration_s:.9f}",
            "request_rate": config["request_rate"],
            "input_tokens": config["input_len"],
            "max_tokens": config["output_len"],
            "max_concurrency": min(2048, max(32, segment["num_prompts"])),
            "timeout_s": max(
                args.minimum_benchmark_timeout_s,
                int(segment["historical_estimated_duration_s"] * 2 + 600),
            ),
            "retries": 1,
            "cancel_fraction": 0,
            "cancel_after_s": 0,
            "shared_prefix_tokens": 1,
            "inter_window_pause_s": 0,
            "tp_degree": config["tp_degree"],
            "target_gpu_freq_mhz": config["gpu_freq_mhz"],
            "config_id": config["config_id"],
            "manual_frequency_control": "true",
            "scheduler_prediction": "false",
            "random_seed": run_seed,
        }],
    )

    tools = args.observability_tools_dir
    python = args.python_bin
    otel_env = os.environ.copy()
    otel_env["PYTHONPATH"] = str(args.otel_bundle) + (
        f":{otel_env['PYTHONPATH']}" if otel_env.get("PYTHONPATH") else ""
    )
    otel_log = (segment_dir / "otel_collector.log").open("w", encoding="utf-8")
    metrics_log = (segment_dir / "metrics_collector.log").open("w", encoding="utf-8")
    otel_process: subprocess.Popen | None = None
    metrics_process: subprocess.Popen | None = None
    monitor = GPUMonitor(config["tp_degree"], args.monitor_interval_ms)
    monitor_started = False
    started_unix_ns = time.time_ns()
    started_at = utc_now()
    started = time.monotonic()
    bench_rc = 1
    upload_rc = 1
    error = ""
    server_log_offset = (
        server.log_path.stat().st_size
        if server.log_path and server.log_path.exists() else 0
    )
    try:
        otel_process = subprocess.Popen(
            [
                python, "-u", str(tools / "otel_collector.py"),
                "--host", "127.0.0.1", "--port", str(args.otlp_port),
                "--output-dir", str(segment_dir),
            ],
            stdout=otel_log,
            stderr=subprocess.STDOUT,
            env=otel_env,
            start_new_session=True,
        )
        wait_http(f"http://127.0.0.1:{args.otlp_port}/health", 60)
        metrics_process = subprocess.Popen(
            [
                python, "-u", str(tools / "metrics_collector.py"),
                "--endpoint", f"combined=http://127.0.0.1:{args.port}/metrics",
                "--output-dir", str(segment_dir), "--interval", "0.5",
            ],
            stdout=metrics_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        monitor.start()
        monitor_started = True
        capture_histograms(
            f"http://127.0.0.1:{args.port}/metrics",
            segment_dir / "histogram_buckets.csv",
            run_id,
            "start",
        )
        timeout_s = max(
            args.minimum_benchmark_timeout_s,
            int(segment["historical_estimated_duration_s"] * 2 + 600),
        )
        with (segment_dir / "stream_bench.log").open("w", encoding="utf-8") as handle:
            result_process = subprocess.run(
                [
                    python, "-u", str(tools / "stream_bench.py"),
                    "--endpoint", f"http://127.0.0.1:{args.port}/v1/completions",
                    "--model", args.model, "--workloads", str(workload_path),
                    "--output-dir", str(segment_dir), "--seed", str(run_seed),
                ],
                stdout=handle, stderr=subprocess.STDOUT, text=True,
                timeout=timeout_s,
            )
        bench_rc = result_process.returncode
        capture_histograms(
            f"http://127.0.0.1:{args.port}/metrics",
            segment_dir / "histogram_buckets.csv",
            run_id,
            "end",
        )
        if bench_rc:
            error = f"stream benchmark exited with rc={bench_rc}"
    except subprocess.TimeoutExpired:
        bench_rc = 124
        error = "benchmark timeout"
    except Exception as exc:
        bench_rc = 1
        error = f"{type(exc).__name__}: {exc}"
    finally:
        samples = monitor.stop() if monitor_started else []
        stop_process(metrics_process)
        with contextlib.suppress(Exception), (
            segment_dir / "drain_observer.log"
        ).open("w", encoding="utf-8") as drain_log:
            subprocess.run(
                [
                    python, "-u", str(tools / "drain_observer.py"),
                    "--endpoint", f"combined=http://127.0.0.1:{args.port}/metrics",
                    "--output-dir", str(segment_dir), "--interval", "0.5",
                    "--max-seconds", "180", "--zero-samples", "3",
                ],
                stdout=drain_log, stderr=subprocess.STDOUT, timeout=190,
            )
        # BatchSpanProcessor commonly exports on a five-second cadence.
        time.sleep(7)
        stop_process(otel_process)
        otel_log.close()
        metrics_log.close()
    elapsed_s = time.monotonic() - started
    finished_at = utc_now()
    summary = summarize_samples(
        samples, config["gpu_freq_mhz"], args.frequency_tolerance_mhz, config["tp_degree"]
    )
    if monitor.error:
        error = error or f"GPU monitor failed: {monitor.error}"

    hostname = socket.gethostname().split(".")[0]
    write_telemetry_csv(
        segment_dir / f"combined_{hostname}_telemetry.csv", samples, hostname
    )
    if server.log_handle:
        server.log_handle.flush()
    if server.log_path and server.log_path.exists():
        with server.log_path.open("rb") as source:
            source.seek(server_log_offset)
            (segment_dir / "combined_server.log").write_bytes(source.read())

    metadata_fields = [
        "experiment_id", "dataset_name", "description", "variant_id",
        "repeat_no", "slurm_job_id", "model", "topology", "hostname",
        "prefill_node", "decode_node", "attention_backend", "kv_connector",
        "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
        "tensor_parallel_size", "dvfs_mode", "manual_frequency_control",
        "scheduler_prediction", "policy_variant", "kv_cache_metrics",
        "kv_cache_metrics_sample", "enable_logging_iteration_details",
        "collect_detailed_traces", "enable_prefix_caching",
        "request_id_headers", "config_id", "segment_no",
        "target_gpu_freq_mhz", "gpu_type", "queue_state_source", "ingestion",
    ]
    write_csv(
        segment_dir / "job_metadata.csv",
        metadata_fields,
        [{
            "experiment_id": manifest["campaign_id"],
            "dataset_name": "calibration_data_full_observability",
            "description": "Full single-pool DVFS calibration observability",
            "variant_id": config["config_id"],
            "repeat_no": repeat_no,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            "model": args.model,
            "topology": "single_pool",
            "hostname": hostname,
            "prefill_node": hostname,
            "decode_node": hostname,
            "attention_backend": "FLASH_ATTN",
            "kv_connector": "",
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": config["tp_degree"],
            "dvfs_mode": "fixed_core_clock",
            "manual_frequency_control": "true",
            "scheduler_prediction": "false",
            "policy_variant": "vllm_default_fixed_frequency",
            "kv_cache_metrics": "true",
            "kv_cache_metrics_sample": 1.0,
            "enable_logging_iteration_details": "true",
            "collect_detailed_traces": "all",
            "enable_prefix_caching": "true",
            "request_id_headers": "true",
            "config_id": config["config_id"],
            "segment_no": segment["segment_no"],
            "target_gpu_freq_mhz": config["gpu_freq_mhz"],
            "gpu_type": manifest["gpu_type"],
            "queue_state_source": "combined_vllm_metrics",
            "ingestion": "segment_batch",
        }],
    )
    interface = default_network_interface()
    environment_fields = [
        "unix_ns", "hostname", "role", "node_group", "node_ip", "peer_ip",
        "interface", "link_speed_mbps", "link_mtu", "kernel", "cpu_count",
        "gpu", "expected_gpu", "model", "max_num_seqs",
        "max_num_batched_tokens", "gpu_memory_utilization", "kv_connector",
        "kv_cache_metrics", "kv_cache_metrics_sample",
        "enable_logging_iteration_details", "collect_detailed_traces",
        "enable_prefix_caching", "enable_request_id_headers", "otlp_endpoint",
        "dvfs_mode", "manual_frequency_control", "scheduler_prediction",
        "attention_backend", "node_work_dir", "runtime_cwd",
        "flashinfer_workspace_base", "flashinfer_workspace", "xdg_cache_home",
        "xdg_config_home", "slurm_job_id", "slurm_node_list", "driver_version",
        "vllm_version", "gpu_memory_total_mib",
    ]
    current_gpu_rows = gpu_query()[: config["tp_degree"]]
    network_base = Path("/sys/class/net") / interface
    write_csv(
        segment_dir / f"environment_{hostname}.csv",
        environment_fields,
        [{
            "unix_ns": started_unix_ns,
            "hostname": hostname,
            "role": "combined",
            "node_group": manifest["gpu_type"],
            "node_ip": "", "peer_ip": "", "interface": interface,
            "link_speed_mbps": read_int(network_base / "speed"),
            "link_mtu": read_int(network_base / "mtu"),
            "kernel": os.uname().release, "cpu_count": os.cpu_count() or 0,
            "gpu": json.dumps(current_gpu_rows),
            "expected_gpu": gpu_names[0] if gpu_names else manifest["gpu_type"],
            "model": args.model, "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_connector": "", "kv_cache_metrics": "true",
            "kv_cache_metrics_sample": 1.0,
            "enable_logging_iteration_details": "true",
            "collect_detailed_traces": "all", "enable_prefix_caching": "true",
            "enable_request_id_headers": "true",
            "otlp_endpoint": args.otlp_traces_endpoint,
            "dvfs_mode": "fixed_core_clock", "manual_frequency_control": "true",
            "scheduler_prediction": "false", "attention_backend": "FLASH_ATTN",
            "node_work_dir": str(Path.cwd()), "runtime_cwd": str(Path.cwd()),
            "flashinfer_workspace_base": os.environ.get(
                "FLASHINFER_WORKSPACE_BASE", ""
            ),
            "flashinfer_workspace": os.environ.get(
                "FLASHINFER_WORKSPACE_BASE", ""
            ),
            "xdg_cache_home": os.environ.get("XDG_CACHE_HOME", ""),
            "xdg_config_home": os.environ.get("XDG_CONFIG_HOME", ""),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST", ""),
            "driver_version": (
                current_gpu_rows[0].get("driver_version", "")
                if current_gpu_rows else ""
            ),
            "vllm_version": args.vllm_version,
            "gpu_memory_total_mib": (
                value(current_gpu_rows[0], "memory.total")
                if current_gpu_rows else 0
            ),
        }],
    )

    post_commands = [
        [
            python, str(tools / "join_queue_state.py"),
            "--requests", str(segment_dir / "requests.csv"),
            "--metrics", str(segment_dir / "vllm_metrics_snapshots.csv"),
            "--output", str(segment_dir / "queue_state_at_arrival.csv"),
        ],
        [
            python, str(tools / "vllm_log_to_csv.py"),
            "--log", f"combined={segment_dir / 'combined_server.log'}",
            "--output", str(segment_dir / "vllm_observability_log_events.csv"),
        ],
    ]
    for command in post_commands:
        result_process = subprocess.run(command, text=True, capture_output=True)
        if result_process.returncode:
            error = error or (
                f"postprocess failed: {shlex.join(command)}: "
                f"{result_process.stderr[-1000:]}"
            )

    integrity_counts = {
        name: csv_row_count(segment_dir / name)
        for name in (
            "requests.csv", "token_timestamps.csv", "client_events.csv",
            "window_summary.csv", "vllm_metrics_snapshots.csv",
            f"combined_{hostname}_telemetry.csv", "histogram_buckets.csv",
            "otel_spans.csv", "drain_samples.csv", "queue_state_at_arrival.csv",
            "vllm_observability_log_events.csv",
        )
    }
    required_nonempty = set(integrity_counts) - {"token_timestamps.csv"}
    incomplete = [
        name for name in sorted(required_nonempty)
        if integrity_counts[name] <= 0
    ]
    if integrity_counts["requests.csv"] != segment["num_prompts"]:
        incomplete.append(
            f"requests.csv expected={segment['num_prompts']} "
            f"actual={integrity_counts['requests.csv']}"
        )
    print(
        f"observability_integrity run_id={run_id} "
        f"counts={json.dumps(integrity_counts, sort_keys=True)} "
        f"incomplete={json.dumps(incomplete)}",
        flush=True,
    )
    if incomplete:
        error = error or f"incomplete observability artifacts: {incomplete}"

    if not incomplete:
        segment_job_id = (
            f"{os.environ.get('SLURM_JOB_ID', 'manual')}-{run_id}"
        )
        uploader_log_path = segment_dir / "clickhouse_final_upload.log"
        with uploader_log_path.open("w", encoding="utf-8") as uploader_log:
            uploader_result = subprocess.run(
                [
                    python, "-u", str(tools / "clickhouse_batch_uploader.py"),
                    "--mode", "final", "--output-dir", str(segment_dir),
                    "--workloads", str(workload_path), "--job-id", segment_job_id,
                    "--job-name", run_id, "--started-unix-ns", str(started_unix_ns),
                    "--state-file", str(segment_dir / "clickhouse_upload_state.json"),
                ],
                stdout=uploader_log, stderr=subprocess.STDOUT, text=True,
                env=otel_env,
            )
        upload_rc = uploader_result.returncode
        if upload_rc:
            error = error or f"ClickHouse full-observability upload rc={upload_rc}"

    success = bench_rc == 0 and upload_rc == 0 and summary["frequency_verified"]
    if not summary["frequency_verified"]:
        error = error or "target frequency verification failed"
    row = {
        "campaign_id": manifest["campaign_id"], "gpu_type": manifest["gpu_type"],
        "run_id": run_id, "config_id": config["config_id"],
        "repeat_no": repeat_no, "segment_no": segment["segment_no"],
        "started_at": started_at, "finished_at": finished_at,
        "status": "success" if success else "failed", "benchmark_rc": bench_rc,
        "clickhouse_upload_rc": upload_rc, "duration_s": elapsed_s,
        "error": error, **summary,
    }
    append_jsonl(output_dir / "run_results.jsonl", row)
    print(
        f"segment_finish run_id={run_id} status={row['status']} duration_s={elapsed_s:.3f} "
        f"frequency_verified={summary['frequency_verified']} samples={len(samples)} "
        f"clickhouse_upload_rc={upload_rc}",
        flush=True,
    )
    return row, success


def detect_vllm_version(vllm_bin: str) -> str:
    version_result = run_command([vllm_bin, "--version"], timeout=60, check=False)
    version = (version_result.stdout + "\n" + version_result.stderr).strip()
    return version or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-bin", default="/data/users/chjing/miniforge3/envs/cuda-env/bin/vllm")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--observability-tools-dir", type=Path, required=True)
    parser.add_argument("--otel-bundle", type=Path, required=True)
    parser.add_argument("--otlp-port", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--monitor-interval-ms", type=int, default=500)
    parser.add_argument("--frequency-tolerance-mhz", type=int, default=30)
    parser.add_argument("--server-ready-timeout-s", type=int, default=900)
    parser.add_argument("--minimum-benchmark-timeout-s", type=int, default=1800)
    parser.add_argument("--deadline-seconds", type=int, default=84600)
    parser.add_argument("--max-segments", type=int, default=0)
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
    (args.output_dir / "full_observability").mkdir(exist_ok=True)
    required_tools = (
        "stream_bench.py", "metrics_collector.py", "drain_observer.py",
        "otel_collector.py", "join_queue_state.py", "vllm_log_to_csv.py",
        "clickhouse_batch_uploader.py",
    )
    missing = [
        str(args.observability_tools_dir / name)
        for name in required_tools
        if not (args.observability_tools_dir / name).is_file()
    ]
    if missing or not args.otel_bundle.is_file():
        raise RuntimeError(
            f"full observability tools missing: {missing}; "
            f"otel_bundle={args.otel_bundle}"
        )
    if args.otlp_port <= 0:
        slurm_id = int(os.environ.get("SLURM_JOB_ID", "0") or 0)
        args.otlp_port = 36000 + slurm_id % 1000
    args.otlp_traces_endpoint = (
        f"http://127.0.0.1:{args.otlp_port}/v1/traces"
    )
    all_gpu_rows = gpu_query()
    if len(all_gpu_rows) < manifest["gpus_per_node"]:
        raise RuntimeError(
            f"expected {manifest['gpus_per_node']} GPUs, nvidia-smi returned {len(all_gpu_rows)}"
        )
    expected_name = "L40S" if manifest["gpu_type"] == "l40s" else "L4"
    gpu_names = [row["name"] for row in all_gpu_rows[: manifest["gpus_per_node"]]]
    if any(expected_name.lower() not in name.lower() for name in gpu_names):
        raise RuntimeError(f"wrong GPU pool: expected {expected_name}, found {gpu_names}")

    args.vllm_version = detect_vllm_version(args.vllm_bin)
    completed_keys: set[tuple[str, int, int]] = set()
    print(
        "collection_mode=full_observability canonical_tables=true "
        "calibration_tables=false segment_batch_upload=true",
        flush=True,
    )
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
                return 75
            try:
                server.ensure(config["tp_degree"])
            except Exception as exc:
                failed += 1
                message = f"server startup failed: {type(exc).__name__}: {exc}"
                print(f"server_error config={config['config_id']} error={type(exc).__name__}:{exc}", flush=True)
                server.stop()
                return 2
            success = False
            for attempt in (1, 2):
                try:
                    _row, success = run_segment(
                        server, manifest, config, segment, repeat_no, attempt,
                        args, args.output_dir, gpu_names,
                    )
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
        print(
            f"shard_finish planned={planned} completed={completed} "
            f"failed={failed} skipped={skipped}",
            flush=True,
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
