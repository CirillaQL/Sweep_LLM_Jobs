"""K1/K1b: P0->D0 burst prefill overhead, decode-busy delay and Poisson load.

K1b adds `poisson_*` phases (open-loop Poisson arrivals through the real P/D
path, recording the in-flight state seen at each arrival) and runs with the
connector send type taken from XPYD_P_SEND_TYPE (PUT_ASYNC in K1b).

Reuses the r3 substrate (launcher, vLLM patches, P2pNcclConnector PUT, proxy
core, clock actuator).  Questions:

* bursts:      does P/D prefill time grow with the number of co-arriving
               requests (not only their tokens)?  Compared offline with the
               plain-vLLM K2 job on the same hardware.
* decode_busy: how much does an already-decoding D delay a new request's first
               token, and does D frequency change it?
* burst_freq:  does P frequency change the burst overhead?

Every request row is appended to CSV immediately; phases stop cleanly before
the deadline given by XPYD_DRIVER_DEADLINE_EPOCH.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from xpyd.joint_grid_validation import _diagnostic_sink, _exception_chain, _expand, _write_json


def percentile(values: Sequence[float], q: float) -> float | None:
    data = sorted(float(v) for v in values if v is not None)
    if not data:
        return None
    pos = (len(data) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def burst_plan(lengths: Sequence[int], sizes: Sequence[int], reps: int, seed: int) -> list[tuple[int, int, int]]:
    plan = [(int(l), int(n), r) for l in lengths for n in sizes for r in range(reps)]
    random.Random(seed).shuffle(plan)
    return plan


class Sink:
    """Append-only CSV with a fixed superset header."""

    FIELDS = [
        "phase", "request_id", "p_mhz", "d_mhz", "input_len", "output_len", "burst_size", "rank",
        "rep", "background_target", "background_active", "status", "error", "ttft_ms", "tpot_ms",
        "client_first_chunk_ms", "client_done_ms", "chunks",
        "t_request_received_ms", "t_route_selected_ms", "t_prefill_started_ms",
        "t_prefill_completed_ms", "t_kv_handoff_completed_ms", "t_decode_request_started_ms",
        "t_decode_response_headers_received_ms", "t_decode_first_real_chunk_received_ms",
        "t_decode_first_real_chunk_forwarded_ms", "t_decode_last_chunk_received_ms",
        "t_response_completed_ms", "rate_rps", "arrival_s", "awaiting_first_requests",
        "awaiting_first_tokens", "decoding_requests",
    ]

    def __init__(self, path: Path) -> None:
        self.path, self.rows = path, []
        with path.open("w", newline="", encoding="utf-8") as s:
            csv.DictWriter(s, fieldnames=self.FIELDS).writeheader()

    def add(self, row: Mapping[str, Any]) -> None:
        self.rows.append(dict(row))
        with self.path.open("a", newline="", encoding="utf-8") as s:
            csv.DictWriter(s, fieldnames=self.FIELDS, extrasaction="ignore").writerow(row)


class Driver:
    def __init__(self, cfg: Mapping[str, Any], core: Any, runtime: Any, prompts: Mapping[int, str],
                 out: Path, deadline_epoch: float, metrics_fn: Any, strip_fn: Any) -> None:
        self.cfg, self.core, self.runtime, self.prompts = cfg, core, runtime, prompts
        self.out, self.deadline = out, deadline_epoch
        self.metrics_fn, self.strip_fn = metrics_fn, strip_fn
        self.sink = Sink(out / "pd_requests.csv")
        self.phase_log: list[dict[str, Any]] = []
        self.seq = 0

    def time_left(self) -> float:
        return self.deadline - time.time()

    def body(self, input_len: int, output_len: int, workload: str) -> dict[str, Any]:
        return {
            "model": str(self.cfg["model"]), "prompt": self.prompts[int(input_len)],
            "max_tokens": int(output_len), "temperature": 0.0, "top_p": 1.0,
            "ignore_eos": True, "stream": True, "xpyd_input_len": int(input_len),
            "xpyd_output_len": int(output_len), "xpyd_workload_id": workload,
        }

    async def request(self, input_len: int, output_len: int, tag: Mapping[str, Any],
                      record: bool = True, on_first: Any = None) -> dict[str, Any]:
        self.seq += 1
        request_id = "k1-%s-%06d" % (tag.get("phase", "x"), self.seq)
        row: dict[str, Any] = {"request_id": request_id, "input_len": input_len,
                               "output_len": output_len, "status": "ok", "error": None, **tag}
        send = time.monotonic()
        try:
            prepared = await self.core.prepare(
                self.strip_fn(self.body(input_len, output_len, str(tag.get("phase")))), request_id)
            if prepared.stream is None:
                raise RuntimeError("proxy did not return a stream")
            first, chunks = None, 0
            async for _ in prepared.stream:
                chunks += 1
                if first is None:
                    first = time.monotonic()
                    if on_first is not None:
                        on_first()
            done = time.monotonic()
            stamps = dict(prepared.diagnostics.timestamps_monotonic_s or {})
            ttft, tpot = self.metrics_fn(stamps, int(output_len))
            row.update(ttft_ms=ttft, tpot_ms=tpot, chunks=chunks,
                       client_first_chunk_ms=None if first is None else (first - send) * 1e3,
                       client_done_ms=(done - send) * 1e3)
            for key, value in stamps.items():
                if isinstance(value, (int, float)):
                    row["t_%s_ms" % key] = (float(value) - send) * 1e3
        except Exception as exc:
            row.update(status="error", error="%s: %s" % (type(exc).__name__, str(exc)[:300]))
        if record:
            self.sink.add(row)
        return row

    async def set_clocks(self, p_mhz: int, d_mhz: int) -> None:
        changed = await self.runtime.actuate("experiment", int(p_mhz), int(d_mhz))
        if changed:
            await asyncio.sleep(float(self.cfg["burst_overhead"]["frequency_settle_s"]))

    async def energy(self) -> dict[str, float]:
        p, d = await asyncio.gather(asyncio.to_thread(self.runtime._energy_mj, "P0"),
                                    asyncio.to_thread(self.runtime._energy_mj, "D0"))
        return {"P0_mj": p, "D0_mj": d, "epoch": time.time()}

    async def bursts(self, name: str, p_mhz: int, d_mhz: int, lengths: Sequence[int],
                     sizes: Sequence[int], reps: int) -> bool:
        s = self.cfg["burst_overhead"]
        await self.set_clocks(p_mhz, d_mhz)
        e0 = await self.energy()
        for input_len, size, rep in burst_plan(lengths, sizes, reps, int(s["seed"]) + p_mhz):
            if self.time_left() < float(s["reserve_s"]):
                self.phase_log.append({"phase": name, "truncated": True, "p_mhz": p_mhz})
                return False
            tag = {"phase": name, "p_mhz": p_mhz, "d_mhz": d_mhz, "burst_size": size, "rep": rep}
            await asyncio.gather(*[
                self.request(input_len, int(s["probe_output_len"]), tag | {"rank": rank})
                for rank in range(size)
            ])
            await asyncio.sleep(float(s["idle_gap_s"]))
        e1 = await self.energy()
        self.phase_log.append({"phase": name, "p_mhz": p_mhz, "d_mhz": d_mhz,
                               "energy_start": e0, "energy_end": e1})
        return True

    async def decode_busy(self, d_mhz: int) -> bool:
        s = self.cfg["burst_overhead"]
        p_mhz = int(s["reference_p_mhz"])
        await self.set_clocks(p_mhz, d_mhz)
        for target in s["background_levels"]:
            needed = float(s["background_warmup_s"]) + len(s["busy_probe_lengths"]) * \
                int(s["busy_probe_reps"]) * float(s["busy_probe_gap_s"] + 2.0) + 30.0
            if self.time_left() < needed + float(s["reserve_s"]):
                self.phase_log.append({"phase": "decode_busy", "truncated": True, "d_mhz": d_mhz,
                                       "background_target": target})
                return False
            background = [
                asyncio.create_task(self.request(
                    int(s["background_input_len"]), int(s["background_output_len"]),
                    {"phase": "background", "p_mhz": p_mhz, "d_mhz": d_mhz,
                     "background_target": int(target)}))
                for _ in range(int(target))
            ]
            await asyncio.sleep(float(s["background_warmup_s"]))
            probes = [(l, r) for l in s["busy_probe_lengths"] for r in range(int(s["busy_probe_reps"]))]
            random.Random(int(s["seed"]) + int(target) + d_mhz).shuffle(probes)
            for input_len, rep in probes:
                active = sum(not t.done() for t in background)
                await self.request(int(input_len), int(s["probe_output_len"]), {
                    "phase": "decode_busy", "p_mhz": p_mhz, "d_mhz": d_mhz, "rep": rep,
                    "background_target": int(target), "background_active": active})
                await asyncio.sleep(float(s["busy_probe_gap_s"]))
            await asyncio.gather(*background)
            await asyncio.sleep(float(s["idle_gap_s"]))
        self.phase_log.append({"phase": "decode_busy", "d_mhz": d_mhz, "completed": True})
        return True

    async def poisson(self, name: str, p_mhz: int, d_mhz: int) -> bool:
        """Open-loop Poisson arrivals; state at arrival = requests still waiting for
        their first token (prefill + KV) and requests currently decoding."""
        s = self.cfg["burst_overhead"]
        c = s["poisson"]
        await self.set_clocks(p_mhz, d_mhz)
        for r_index, rate in enumerate(c["rates_rps"]):
            if self.time_left() < float(c["duration_s"]) + float(c["drain_allowance_s"]) + float(s["reserve_s"]):
                self.phase_log.append({"phase": name, "truncated": True, "rate_rps": rate})
                return False
            rng = random.Random(int(s["seed"]) + 31 * r_index + p_mhz)
            arrivals, t = [], 0.0
            while True:
                t += rng.expovariate(float(rate))
                if t >= float(c["duration_s"]):
                    break
                arrivals.append(t)
            awaiting: dict[int, int] = {}
            decoding: set[int] = set()
            start = time.monotonic()
            e0 = await self.energy()

            async def one(i: int, at: float) -> None:
                delay = start + at - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                length = int(rng.choice(c["input_lens"]))
                tag = {"phase": name, "p_mhz": p_mhz, "d_mhz": d_mhz, "rate_rps": float(rate),
                       "arrival_s": at, "awaiting_first_requests": len(awaiting),
                       "awaiting_first_tokens": sum(awaiting.values()),
                       "decoding_requests": len(decoding)}
                awaiting[i] = length

                def first_token() -> None:
                    awaiting.pop(i, None)
                    decoding.add(i)

                try:
                    await self.request(length, int(c["output_len"]), tag, on_first=first_token)
                finally:
                    awaiting.pop(i, None)
                    decoding.discard(i)

            await asyncio.gather(*[one(i, at) for i, at in enumerate(arrivals)])
            e1 = await self.energy()
            self.phase_log.append({"phase": name, "rate_rps": float(rate), "requests": len(arrivals),
                                   "window_s": time.monotonic() - start,
                                   "energy_start": e0, "energy_end": e1})
            await asyncio.sleep(float(s["idle_gap_s"]))
        return True

    async def run(self) -> None:
        s = self.cfg["burst_overhead"]
        ref_p, ref_d = int(s["reference_p_mhz"]), int(s["reference_d_mhz"])
        for step in s["phase_order"]:
            if self.time_left() < float(s["reserve_s"]):
                self.phase_log.append({"phase": step, "skipped": "deadline"})
                continue
            if step == "bursts_reference":
                await self.bursts(step, ref_p, ref_d, s["burst_lengths"], s["burst_sizes"], int(s["burst_reps"]))
            elif step.startswith("decode_busy_"):
                await self.decode_busy(int(step.rsplit("_", 1)[1]))
            elif step.startswith("poisson_"):
                await self.poisson(step, int(step.rsplit("_", 1)[1]), ref_d)
            elif step.startswith("bursts_p"):
                await self.bursts(step, int(step[len("bursts_p"):]), ref_d, s["freq_burst_lengths"],
                                  s["freq_burst_sizes"], int(s["freq_burst_reps"]))
            else:
                raise ValueError("unknown step %s" % step)


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("phase") == "background":
            continue
        key = (row.get("phase"), row.get("p_mhz"), row.get("d_mhz"), row.get("input_len"),
               row.get("burst_size") if row.get("rate_rps") is None else row.get("rate_rps"),
               row.get("background_target"))
        groups.setdefault(key, []).append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        entry = dict(zip(("phase", "p_mhz", "d_mhz", "input_len", "burst_size_or_rate", "background_target"), key))
        entry["n"] = len(items)
        for metric in ("ttft_ms", "t_prefill_completed_ms", "t_decode_first_real_chunk_received_ms"):
            vals = [r.get(metric) for r in items if r.get(metric) is not None]
            entry[metric + "_p50"] = percentile(vals, 0.5)
            entry[metric + "_p95"] = percentile(vals, 0.95)
        prefill = [r["t_prefill_completed_ms"] - r["t_prefill_started_ms"] for r in items
                   if r.get("t_prefill_completed_ms") is not None and r.get("t_prefill_started_ms") is not None]
        dfirst = [r["t_decode_first_real_chunk_received_ms"] - r["t_kv_handoff_completed_ms"] for r in items
                  if r.get("t_decode_first_real_chunk_received_ms") is not None
                  and r.get("t_kv_handoff_completed_ms") is not None]
        entry["prefill_ms_p50"], entry["prefill_ms_p95"] = percentile(prefill, 0.5), percentile(prefill, 0.95)
        entry["decode_first_ms_p50"], entry["decode_first_ms_p95"] = percentile(dfirst, 0.5), percentile(dfirst, 0.95)
        out.append(entry)
    return out


async def run(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    import aiohttp
    from replay_synthetic_trace import build_prompt_cache
    from transformers import AutoTokenizer
    from xpyd.disagg_proxy import _build_multi_core
    from xpyd.online_feedback_controller import (
        PhysicalFeedbackRuntime, pd_inference_metrics, strip_feedback_metadata,
    )

    s = config["burst_overhead"]
    output.mkdir(parents=True, exist_ok=True)
    deadline = float(os.environ.get("XPYD_DRIVER_DEADLINE_EPOCH", time.time() + 1200))
    diagnostics_path = output / "proxy_diagnostics.jsonl"
    core = _build_multi_core(
        config, lambda: aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=float(s.get("request_timeout_s", 300.0)))),
        diagnostic_sink=_diagnostic_sink(diagnostics_path),
    )
    experiment_core = core.cores[("P0", "D0")]
    runtime = PhysicalFeedbackRuntime(config, experiment_core)
    tokenizer = AutoTokenizer.from_pretrained(str(config["tokenizer_model"]))
    lengths = set(s["burst_lengths"]) | set(s["freq_burst_lengths"]) | set(s["busy_probe_lengths"]) | {
        int(s["background_input_len"])} | set(s.get("poisson", {}).get("input_lens", []))
    prompts = build_prompt_cache(tokenizer, {int(l) for l in lengths})
    driver = Driver(config, experiment_core, runtime, prompts, output, deadline,
                    pd_inference_metrics, strip_feedback_metadata)

    await driver.set_clocks(int(s["reference_p_mhz"]), int(s["reference_d_mhz"]))
    smoke = await driver.request(128, 16, {"phase": "smoke", "p_mhz": int(s["reference_p_mhz"]),
                                           "d_mhz": int(s["reference_d_mhz"])})
    if smoke["status"] != "ok":
        failure = {"stage": "p0d0_end_to_end_smoke", "row": smoke,
                   "created_utc": datetime.now(timezone.utc).isoformat()}
        _write_json(output / "smoke_failure.json", failure)
        _write_json(output / "audit.json", {"valid": False, "failure_stage": "smoke"})
        raise RuntimeError("P0/D0 smoke failed; experiment not started: %s" % smoke["error"])
    _write_json(output / "smoke_success.json", {"row": smoke})

    error = None
    try:
        await driver.run()
    except Exception as exc:
        error = {"exception_chain": _exception_chain(exc)}
    rows = driver.sink.rows
    measured = [r for r in rows if r.get("phase") not in ("smoke", "background")]
    summary = {
        "schema_version": 1, "send_type": os.environ.get("XPYD_P_SEND_TYPE", "PUT"), "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": error is None and bool(measured), "error": error,
        "phase_log": driver.phase_log, "request_rows": len(rows),
        "failed_requests": sum(r.get("status") != "ok" for r in rows),
        "groups": summarize(rows),
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "audit.json", {"valid": summary["valid"], "request_rows": len(rows),
                                        "failed_measurements": summary["failed_requests"]})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = _expand(json.loads(args.config.read_text(encoding="utf-8")))
    summary = asyncio.run(run(config, args.output))
    print(json.dumps({"valid": summary["valid"], "request_rows": summary["request_rows"]}, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
