#!/usr/bin/env python3
"""Single-GPU vLLM characterization driver for CanTuning (K2 prefill / K3 decode).

The driver talks to one plain (non-disaggregated) vLLM OpenAI server on the
same node, locks the allocated GPU's clocks through `sudo -n nvidia-smi`, and
records request timings plus NVML energy.  Every measurement row is appended to
CSV immediately so a Slurm time-limit kill still leaves usable partial data.

Phases are ordered by priority and skipped once the wall-clock deadline would be
exceeded; the summary always records which phases ran.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# Pure helpers (unit tested locally)
# --------------------------------------------------------------------------

def percentile(values: Iterable[float], q: float) -> float | None:
    data = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def make_prompt(rng: random.Random, length: int) -> list[int]:
    """Token-id prompt of exact length; ids avoid special/low ids."""
    return [rng.randrange(1000, 30000) for _ in range(int(length))]


def burst_plan(
    freqs: Sequence[int], lengths: Sequence[int], sizes: Sequence[int], reps: int, seed: int,
) -> list[tuple[int, int, int, int]]:
    """(freq, input_len, burst_size, rep); frequency-major so clocks change rarely,
    shuffled within each frequency to decorrelate order effects."""
    plan: list[tuple[int, int, int, int]] = []
    for f_index, freq in enumerate(freqs):
        block = [(int(freq), int(l), int(n), r) for l in lengths for n in sizes for r in range(reps)]
        random.Random(seed + f_index).shuffle(block)
        plan.extend(block)
    return plan


def poisson_arrivals(rate_rps: float, duration_s: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    t, out = 0.0, []
    while True:
        t += rng.expovariate(rate_rps)
        if t >= duration_s:
            return out
        out.append(t)


def parse_prom_counter(text: str, name: str) -> float | None:
    total, seen = 0.0, False
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            rest = line[len(name):]
            if rest[:1] not in ("{", " "):
                continue
            try:
                total += float(line.rsplit(" ", 1)[1])
                seen = True
            except (ValueError, IndexError):
                continue
    return total if seen else None


def tpot_from_token_times(times: Sequence[float]) -> float | None:
    if len(times) < 2:
        return None
    return (times[-1] - times[0]) * 1000.0 / (len(times) - 1)


# --------------------------------------------------------------------------
# GPU access (NVML read, sudo nvidia-smi for clock locks)
# --------------------------------------------------------------------------

class Gpu:
    def __init__(self, index: int, mem_mhz: int, dry_run: bool = False) -> None:
        self.index = int(index)
        self.mem_mhz = int(mem_mhz)
        self.dry_run = dry_run
        self.locked_mhz: int | None = None
        self.commands: list[dict[str, Any]] = []
        if dry_run:
            self._t0 = time.monotonic()
            return
        import pynvml  # provided by nvidia-ml-py
        pynvml.nvmlInit()
        self.nv = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        self.name = pynvml.nvmlDeviceGetName(self.handle)
        self.uuid = pynvml.nvmlDeviceGetUUID(self.handle)

    def energy_mj(self) -> float:
        if self.dry_run:
            return (time.monotonic() - self._t0) * 50_000.0  # 50 W
        return float(self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle))

    def power_w(self) -> float:
        if self.dry_run:
            return 50.0
        return self.nv.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def sm_clock(self) -> int:
        if self.dry_run:
            return int(self.locked_mhz or 0)
        return int(self.nv.nvmlDeviceGetClockInfo(self.handle, self.nv.NVML_CLOCK_SM))

    def _smi(self, *args: str) -> None:
        cmd = ["sudo", "-n", "nvidia-smi", "-i", str(self.index), *args]
        if self.dry_run:
            self.commands.append({"cmd": cmd, "rc": 0})
            return
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        text = res.stdout + res.stderr
        self.commands.append({"cmd": cmd, "rc": res.returncode, "out": text[-300:]})
        # nvidia-smi has been seen to return a non-zero status after a successful
        # clock change (K3 job 267572, exit 9); judge by its own confirmation text.
        if res.returncode != 0 and "All done" not in text:
            raise RuntimeError("clock command failed: %s -> rc=%d %s" % (" ".join(cmd), res.returncode, text))

    def lock(self, mhz: int, wait_s: float = 3.0, tol: int = 20) -> float | None:
        """Lock graphics (and memory) clocks; return seconds until SM clock reads back
        within `tol` MHz of target (None if never within wait_s)."""
        t0 = time.monotonic()
        self._smi("-lgc", "%d,%d" % (mhz, mhz))
        if self.locked_mhz is None:
            self._smi("-lmc", "%d,%d" % (self.mem_mhz, self.mem_mhz))
        self.locked_mhz = int(mhz)
        while time.monotonic() - t0 < wait_s:
            if abs(self.sm_clock() - mhz) <= tol:
                return time.monotonic() - t0
            time.sleep(0.005)
        return None

    def reset(self) -> None:
        for args in (("-rgc",), ("-rmc",)):
            try:
                self._smi(*args)
            except Exception as exc:  # reset must never raise during cleanup
                self.commands.append({"cmd": list(args), "error": str(exc)})
        self.locked_mhz = None


class PowerSampler(threading.Thread):
    def __init__(self, gpu: Gpu, path: Path, period_s: float = 0.1) -> None:
        super().__init__(daemon=True)
        self.gpu, self.path, self.period_s = gpu, path, period_s
        self.stop_event = threading.Event()

    def run(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            w = csv.writer(stream)
            w.writerow(["mono_s", "power_w", "sm_mhz", "energy_mj"])
            while not self.stop_event.is_set():
                try:
                    w.writerow(["%.3f" % time.monotonic(), "%.2f" % self.gpu.power_w(),
                                self.gpu.sm_clock(), "%.0f" % self.gpu.energy_mj()])
                except Exception:
                    pass
                self.stop_event.wait(self.period_s)


# --------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------

class RowSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fields: list[str] | None = None
        self.rows: list[dict[str, Any]] = []

    def add(self, row: Mapping[str, Any]) -> None:
        row = dict(row)
        self.rows.append(row)
        if self.fields is None:
            self.fields = list(row)
            with self.path.open("w", newline="", encoding="utf-8") as s:
                csv.DictWriter(s, fieldnames=self.fields, extrasaction="ignore").writeheader()
        with self.path.open("a", newline="", encoding="utf-8") as s:
            csv.DictWriter(s, fieldnames=self.fields, extrasaction="ignore").writerow(row)


async def stream_completion(session: Any, url: str, model: str, prompt: list[int],
                            max_tokens: int, timeout_s: float) -> dict[str, Any]:
    body = {"model": model, "prompt": prompt, "max_tokens": int(max_tokens),
            "temperature": 0.0, "ignore_eos": True, "stream": True}
    send = time.monotonic()
    token_times: list[float] = []
    status, error = "ok", None
    try:
        async with session.post(url + "/v1/completions", json=body, timeout=timeout_s) as resp:
            if resp.status != 200:
                raise RuntimeError("HTTP %d: %s" % (resp.status, (await resp.text())[:200]))
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    choices = json.loads(payload).get("choices") or []
                except json.JSONDecodeError:
                    continue
                # a generated token can decode to "" (seen in K2 poisson rows); count every choice chunk
                if choices and choices[0].get("text") is not None:
                    token_times.append(time.monotonic())
    except Exception as exc:
        status, error = "error", "%s: %s" % (type(exc).__name__, str(exc)[:200])
    done = time.monotonic()
    return {
        "send_mono_s": send, "status": status, "error": error,
        "ttft_ms": (token_times[0] - send) * 1000.0 if token_times else None,
        "tpot_ms": tpot_from_token_times(token_times),
        "tokens": len(token_times), "latency_ms": (done - send) * 1000.0,
    }


async def fetch_metrics(session: Any, url: str) -> str:
    try:
        async with session.get(url + "/metrics", timeout=10) as resp:
            return await resp.text()
    except Exception:
        return ""


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------

@dataclass
class Ctx:
    cfg: Mapping[str, Any]
    gpu: Gpu
    session: Any
    url: str
    model: str
    out: Path
    deadline: float
    rng: random.Random
    phases: list[dict[str, Any]] = field(default_factory=list)

    def time_left(self) -> float:
        return self.deadline - time.monotonic()


async def phase_idle_and_switch(ctx: Ctx, sink: RowSink) -> None:
    c = ctx.cfg["idle_and_switch"]
    for mode in c["idle_modes"]:
        if mode == "unlocked":
            await asyncio.to_thread(ctx.gpu.reset)
        else:
            await asyncio.to_thread(ctx.gpu.lock, int(mode))
        await asyncio.sleep(float(c["settle_s"]))
        e0, t0 = ctx.gpu.energy_mj(), time.monotonic()
        clocks = []
        while time.monotonic() - t0 < float(c["measure_s"]):
            clocks.append(ctx.gpu.sm_clock())
            await asyncio.sleep(0.2)
        e1, t1 = ctx.gpu.energy_mj(), time.monotonic()
        sink.add({"kind": "idle_power", "mode": str(mode), "from_mhz": None, "to_mhz": None,
                  "value": (e1 - e0) / 1000.0 / (t1 - t0), "unit": "W",
                  "sm_mhz_median": statistics.median(clocks) if clocks else None})
    lo, hi = int(c["switch_pair"][0]), int(c["switch_pair"][1])
    await asyncio.to_thread(ctx.gpu.lock, lo)
    await asyncio.sleep(1.0)
    for rep in range(int(c["switch_reps"])):
        for a, b in ((lo, hi), (hi, lo)):
            dt = await asyncio.to_thread(ctx.gpu.lock, b)
            sink.add({"kind": "switch_latency", "mode": "rep%d" % rep, "from_mhz": a, "to_mhz": b,
                      "value": None if dt is None else dt * 1000.0, "unit": "ms",
                      "sm_mhz_median": ctx.gpu.sm_clock()})
            await asyncio.sleep(0.5)


async def run_burst(ctx: Ctx, input_len: int, size: int, max_tokens: int) -> list[dict[str, Any]]:
    prompts = [make_prompt(ctx.rng, input_len) for _ in range(size)]
    e0 = ctx.gpu.energy_mj()
    t0 = time.monotonic()
    results = await asyncio.gather(*[
        stream_completion(ctx.session, ctx.url, ctx.model, p, max_tokens, 120.0) for p in prompts
    ])
    t1 = time.monotonic()
    e1 = ctx.gpu.energy_mj()
    for rank, row in enumerate(results):
        row.update({"rank": rank, "burst_energy_j": (e1 - e0) / 1000.0,
                    "burst_duration_ms": (t1 - t0) * 1000.0})
    return results


async def phase_bursts(ctx: Ctx, sink: RowSink, section: str) -> None:
    c = ctx.cfg[section]
    plan = burst_plan(c["freqs_mhz"], c["input_lens"], c["burst_sizes"], int(c["reps"]),
                      int(ctx.cfg["seed"]))
    current = None
    for freq, input_len, size, rep in plan:
        if ctx.time_left() < float(ctx.cfg["reserve_s"]):
            ctx.phases.append({"phase": section, "truncated_at": [freq, input_len, size, rep]})
            return
        if freq != current:
            await asyncio.to_thread(ctx.gpu.lock, freq)
            await asyncio.sleep(float(c["settle_s"]))
            current = freq
        rows = await run_burst(ctx, input_len, size, int(c["max_tokens"]))
        for row in rows:
            sink.add({"phase": section, "freq_mhz": freq, "input_len": input_len,
                      "burst_size": size, "rep": rep, **row})
        await asyncio.sleep(float(c["gap_s"]))


async def phase_poisson(ctx: Ctx, sink: RowSink) -> None:
    c = ctx.cfg["poisson"]
    for f_index, freq in enumerate(c["freqs_mhz"]):
        await asyncio.to_thread(ctx.gpu.lock, int(freq))
        await asyncio.sleep(float(c["settle_s"]))
        for r_index, rate in enumerate(c["rates_rps"]):
            if ctx.time_left() < float(c["duration_s"]) + float(ctx.cfg["reserve_s"]):
                ctx.phases.append({"phase": "poisson", "truncated_at": [freq, rate]})
                return
            arrivals = poisson_arrivals(float(rate), float(c["duration_s"]),
                                        int(ctx.cfg["seed"]) + 97 * f_index + r_index)
            inflight: dict[int, int] = {}
            tasks = []
            start = time.monotonic()
            e0 = ctx.gpu.energy_mj()

            async def one(i: int, at: float) -> None:
                delay = start + at - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                length = ctx.rng.choice(c["input_lens"])
                state = {"inflight_requests": len(inflight),
                         "inflight_tokens": sum(inflight.values())}
                inflight[i] = length
                try:
                    row = await stream_completion(ctx.session, ctx.url, ctx.model,
                                                  make_prompt(ctx.rng, length),
                                                  int(c["max_tokens"]), 120.0)
                finally:
                    inflight.pop(i, None)
                sink.add({"phase": "poisson", "freq_mhz": int(freq), "rate_rps": float(rate),
                          "input_len": length, "arrival_s": at, **state, **row})

            for i, at in enumerate(arrivals):
                tasks.append(asyncio.create_task(one(i, at)))
            await asyncio.gather(*tasks)
            e1 = ctx.gpu.energy_mj()
            ctx.phases.append({"phase": "poisson", "freq_mhz": int(freq), "rate_rps": float(rate),
                               "requests": len(arrivals),
                               "window_energy_j": (e1 - e0) / 1000.0,
                               "window_s": time.monotonic() - start})


async def phase_decode(ctx: Ctx, sink: RowSink) -> None:
    c = ctx.cfg["decode"]
    for freq in c["freqs_mhz"]:
        await asyncio.to_thread(ctx.gpu.lock, int(freq))
        await asyncio.sleep(float(c["settle_s"]))
        for input_len, sizes in c["grid"]:
            for size in sizes:
                if ctx.time_left() < float(c["estimated_batch_s"]) + float(ctx.cfg["reserve_s"]):
                    ctx.phases.append({"phase": "decode", "truncated_at": [freq, input_len, size]})
                    return
                pre = parse_prom_counter(await fetch_metrics(ctx.session, ctx.url),
                                         "vllm:num_preemptions_total")
                rows = await run_burst(ctx, int(input_len), int(size), int(c["max_tokens"]))
                post = parse_prom_counter(await fetch_metrics(ctx.session, ctx.url),
                                          "vllm:num_preemptions_total")
                preempted = None if pre is None or post is None else post - pre
                for row in rows:
                    sink.add({"phase": "decode", "freq_mhz": int(freq), "input_len": int(input_len),
                              "burst_size": int(size), "preemptions": preempted, **row})
                await asyncio.sleep(float(c["gap_s"]))


def summarize(rows: list[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        entry = dict(zip(keys, key))
        entry["n"] = len(items)
        for metric in ("ttft_ms", "tpot_ms", "latency_ms"):
            vals = [r[metric] for r in items if r.get(metric) is not None]
            entry[metric + "_p50"] = percentile(vals, 0.5)
            entry[metric + "_p95"] = percentile(vals, 0.95)
        out.append(entry)
    return out


async def main_async(cfg: Mapping[str, Any], out: Path, deadline: float, dry_run: bool) -> dict[str, Any]:
    import aiohttp

    out.mkdir(parents=True, exist_ok=True)
    gpu = Gpu(int(cfg["gpu_index"]), int(cfg["memory_mhz"]), dry_run=dry_run)
    sampler = PowerSampler(gpu, out / "power_trace.csv")
    sampler.start()
    sinks = {kind: RowSink(out / ("%s.csv" % kind)) for kind in ("bursts", "poisson", "decode")}
    gpu_sink = RowSink(out / "gpu_events.csv")
    connector = aiohttp.TCPConnector(limit=0)
    started = datetime.now(timezone.utc).isoformat()
    error = None
    ctx = None
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            ctx = Ctx(cfg, gpu, session, str(cfg["server_url"]), str(cfg["model"]), out,
                      deadline, random.Random(int(cfg["seed"])))
            # Smoke: one request must complete before measurements start.
            await asyncio.to_thread(gpu.lock, int(cfg["smoke_freq_mhz"]))
            smoke = await stream_completion(session, ctx.url, ctx.model,
                                            make_prompt(ctx.rng, 128), 4, 300.0)
            if smoke["status"] != "ok" or not smoke["tokens"]:
                raise RuntimeError("smoke request failed: %s" % smoke)
            ctx.phases.append({"phase": "smoke", **smoke})
            for name in cfg["phase_order"]:
                if ctx.time_left() < float(cfg["reserve_s"]):
                    ctx.phases.append({"phase": name, "skipped": "deadline"})
                    continue
                t0 = time.monotonic()
                if name == "idle_and_switch":
                    await phase_idle_and_switch(ctx, gpu_sink)
                elif name.startswith("bursts"):
                    await phase_bursts(ctx, sinks["bursts"], name)
                elif name == "poisson":
                    await phase_poisson(ctx, sinks["poisson"])
                elif name == "decode":
                    await phase_decode(ctx, sinks["decode"])
                else:
                    raise ValueError("unknown phase %s" % name)
                ctx.phases.append({"phase": name, "completed_s": time.monotonic() - t0})
    except Exception as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
    finally:
        await asyncio.to_thread(gpu.reset)
        sampler.stop_event.set()
        sampler.join(timeout=2)
    rows = [row for sink in sinks.values() for row in sink.rows]
    summary = {
        "schema_version": SCHEMA_VERSION, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "valid": error is None and bool(rows or gpu_sink.rows), "error": error,
        "gpu": {"index": gpu.index, "name": getattr(gpu, "name", "dry-run"),
                "uuid": getattr(gpu, "uuid", "dry-run")},
        "phases": ctx.phases if ctx else [],
        "request_rows": len(rows), "failed_requests": sum(r.get("status") != "ok" for r in rows),
        "gpu_events": gpu_sink.rows,
        "burst_summary": summarize([r for r in rows if str(r.get("phase", "")).startswith("bursts")],
                                   ("phase", "freq_mhz", "input_len", "burst_size")),
        "poisson_summary": summarize([r for r in rows if r.get("phase") == "poisson"],
                                     ("freq_mhz", "rate_rps", "input_len")),
        "decode_summary": summarize([r for r in rows if r.get("phase") == "decode"],
                                    ("freq_mhz", "input_len", "burst_size")),
        "clock_commands": gpu.commands[-50:],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    # --output/--model default to environment variables so the command line of this
    # process never contains a path with "vllm" in it (see serve.py).
    ap.add_argument("--output", type=Path, default=os.environ.get("GPU_CHAR_OUTPUT"))
    ap.add_argument("--time-budget-s", required=True, type=float)
    ap.add_argument("--model", default=os.environ.get("GPU_CHAR_MODEL"),
                    help="served model name (the --model path)")
    ap.add_argument("--gpu-index", required=True, type=int, help="physical NVML/nvidia-smi index")
    ap.add_argument("--dry-run", action="store_true", help="fake GPU (local testing only)")
    args = ap.parse_args()
    if args.output is None or args.model is None:
        ap.error("--output/--model (or GPU_CHAR_OUTPUT/GPU_CHAR_MODEL) are required")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg["model"], cfg["gpu_index"] = args.model, args.gpu_index
    deadline = time.monotonic() + args.time_budget_s
    summary = asyncio.run(main_async(cfg, args.output, deadline, args.dry_run))
    print(json.dumps({"valid": summary["valid"], "error": summary["error"],
                      "request_rows": summary["request_rows"]}, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
