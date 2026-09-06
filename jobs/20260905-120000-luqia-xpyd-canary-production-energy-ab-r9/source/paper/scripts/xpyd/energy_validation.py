"""Cold-start Canary + continuous observed Production feedback experiment.

Only this process owns the four GPUs' clock controller. NVML is sampled by
two persistent node-local processes; energy reads never spawn per-request srun.
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time


SAMPLER = r'''
import json, sys, time, pynvml
pynvml.nvmlInit()
handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in (0, 1)]
while True:
    for i, h in enumerate(handles):
        a = time.time()
        energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
        clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        b = time.time()
        print(json.dumps(dict(gpu=i, t=(a+b)/2, energy_j=energy,
                             clock_mhz=clock, power_w=power, read_duration_s=b-a)), flush=True)
    time.sleep(0.1)
'''


def append(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def percentile(values, q=0.95):
    values = sorted(values)
    pos = (len(values)-1)*q
    lo = int(pos)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(pos-lo)


async def consume(prepared, expected_tokens):
    if prepared.stream is None or prepared.status_code != 200:
        raise RuntimeError("missing successful SSE response")
    chunks = []
    async for chunk in prepared.stream:
        chunks.append(chunk)
    tokens, done = None, False
    for line in b"".join(chunks).decode("utf-8").splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
        else:
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError("upstream SSE error")
            if event.get("usage"):
                tokens = event["usage"].get("completion_tokens")
    if not done or tokens != expected_tokens:
        raise RuntimeError(f"incomplete generation: done={done}, tokens={tokens}, expected={expected_tokens}")
    return tokens


def integrate(rows, start, end, target=None):
    """Interpolate cumulative NVML counters, with bracketing and clock gates."""
    times = [r["t"] for r in rows]
    left = bisect.bisect_right(times, start)-1
    right = bisect.bisect_left(times, end)
    if end <= start or left < 0 or right >= len(rows):
        raise ValueError("energy interval lacks bracketing samples")
    window = rows[left:right+1]
    if len(window) < 2:
        raise ValueError("too few energy samples")
    for a, b in zip(window, window[1:]):
        if not 0 < b["t"]-a["t"] <= 0.5 or b["energy_j"] < a["energy_j"]:
            raise ValueError("energy counter reset or sample gap > 0.5 s")
    def at(t):
        i = bisect.bisect_right(times, t)-1
        if times[i] == t:
            return rows[i]["energy_j"]
        a, b = rows[i], rows[i+1]
        return a["energy_j"]+(b["energy_j"]-a["energy_j"])*(t-a["t"])/(b["t"]-a["t"])
    energy = at(end)-at(start)
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("nonpositive or invalid energy")
    interior = [r for r in window if start <= r["t"] <= end]
    match = (sum(r["clock_mhz"] == target for r in interior)/len(interior)
             if target is not None and interior else None)
    if target is not None and (match is None or match < 0.95):
        raise ValueError("actual GPU clock does not match target for >=95% of request")
    return {"energy_j": energy, "clock_match_fraction": match,
            "mean_power_w": energy/(end-start), "samples": len(window)}


class Meter:
    def __init__(self, root):
        self.root = root
        self.samples = {e: [] for e in ("P0", "P1", "D0", "D1")}
        self.errors = []
        self.processes = []
        self.threads = []

    def start(self):
        for node, role in (("uranus", "P"), ("ganymede", "D")):
            err = (self.root / (node+"_sampler.stderr")).open("w")
            process = subprocess.Popen([
                "srun", "--overlap", "--nodes=1", "--ntasks=1", "--mem=128M",
                "--nodelist="+node, "--gpus-per-node=2", "--gpu-bind=none",
                os.environ.get("PYTHON_BIN", "python"), "-u", "-c", SAMPLER,
            ], stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
            self.processes.append((process, err))
            def read(p=process, prefix=role):
                try:
                    with (self.root / (prefix+"_energy_raw.jsonl")).open("w") as out:
                        for line in p.stdout:
                            row = json.loads(line)
                            row["received_t"] = time.time()
                            # Bound cross-node time skew plus transport latency.
                            if abs(row["received_t"]-row["t"]) > 0.25:
                                raise ValueError("sampler clock skew/transport delay >250ms")
                            out.write(json.dumps(row)+"\n")
                            out.flush()
                            self.samples[prefix+str(row["gpu"])].append(row)
                except Exception as exc:
                    self.errors.append(str(exc))
            thread = threading.Thread(target=read, daemon=True)
            thread.start()
            self.threads.append(thread)

    async def ready(self, end=None):
        deadline = time.monotonic()+30
        while True:
            if self.errors:
                raise RuntimeError(self.errors)
            if all(v and (end is None or v[-1]["t"] >= end) for v in self.samples.values()):
                return
            if any(p.poll() is not None for p, _ in self.processes):
                raise RuntimeError("persistent energy sampler exited")
            if time.monotonic() > deadline:
                raise TimeoutError("energy samples unavailable")
            await asyncio.sleep(0.05)

    async def energy(self, pair, start, end, targets=None):
        await self.ready(end)
        results = {ep: integrate(list(self.samples[ep]), start, end,
                   targets[i] if targets else None) for i, ep in enumerate(pair)}
        return {"start": start, "end": end, "duration_s": end-start,
                "energy_j": sum(v["energy_j"] for v in results.values()),
                "mean_power_w": sum(v["energy_j"] for v in results.values())/(end-start),
                "endpoints": results}

    def stop(self):
        for p, err in self.processes:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            err.close()
        for thread in self.threads:
            thread.join(timeout=5)



def observed_summary(rows):
    """Descriptive before/after summaries, without assuming randomized groups."""
    results = []
    for w in sorted({r["workload_id"] for r in rows}):
        result = {"workload_id": w}
        for source in ("safe_high", "table"):
            rr = [r for r in rows if r["workload_id"] == w and r["frequency_source"] == source]
            if not rr:
                result[source] = {"n": 0}
                continue
            duration = sum(r["duration_s"] for r in rr)
            energy = sum(r["energy_j"] for r in rr)
            inf_duration = sum(r["inference"]["duration_s"] for r in rr)
            inf_energy = sum(r["inference"]["energy_j"] for r in rr)
            result[source] = {
                "n": len(rr), "mean_duration_s": duration/len(rr),
                "mean_power_w": energy/duration, "joules_per_request": energy/len(rr),
                "inference_mean_power_w": inf_energy/inf_duration,
                "inference_joules_per_request": inf_energy/len(rr),
                "ttft_p95_ms": percentile([r["ttft_ms"] for r in rr]),
                "control_inclusive_ttft_p95_ms": percentile([r["control_inclusive_ttft_ms"] for r in rr]),
                "tpot_p95_ms": percentile([r["tpot_ms"] for r in rr]),
                "first_request_start": min(r["start"] for r in rr),
                "last_request_end": max(r["end"] for r in rr),
                "frequency_pairs": sorted({(r["p_mhz"], r["d_mhz"]) for r in rr}),
            }
        a, b = result["safe_high"], result["table"]
        result["observed_energy_reduction_percent"] = (
            100*(a["joules_per_request"]-b["joules_per_request"])/a["joules_per_request"]
            if a["n"] and b["n"] else None)
        result["observed_power_reduction_percent"] = (
            100*(a["mean_power_w"]-b["mean_power_w"])/a["mean_power_w"]
            if a["n"] and b["n"] else None)
        result["evidence"] = "observational; unequal sample counts, time drift and Canary activity may confound comparisons"
        results.append(result)
    return results


def probe_metadata(probe_id):
    match = re.search(r"-(binary-slo|energy-refine)-([PD])-(\d+)(?:-sample-(\d+))?", probe_id)
    repeat = re.search(r"-sample-(\d+)$", probe_id)
    return {
        "search_stage": match[1] if match else ("confirmation" if "-confirm" in probe_id else "warmup"),
        "axis": match[2] if match else None,
        "grid_index": int(match[3]) if match else None,
        "repeat_index": int(repeat[1]) if repeat else None,
        "candidate_id": re.sub(r"-sample-\d+$", "", probe_id),
    }


def export_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


async def run(config, root):
    import aiohttp
    from transformers import AutoTokenizer
    from replay_synthetic_trace import build_prompt_cache
    from xpyd.disagg_proxy import _build_multi_core
    from xpyd.online_feedback_controller import (
        OnlineFeedbackController, PhysicalFeedbackRuntime, ProbeResult,
        strip_feedback_metadata, pd_inference_metrics)

    if Path(config["frequency_table_path"]).exists():
        raise RuntimeError("cold start requires a nonexistent frequency table")
    settings = config["online_feedback"]
    settings["event_log"] = str(root/"feedback_events.jsonl")
    settings["service_request_log"] = str(root/"service_dispatch.jsonl")
    protocol = config.get("energy_validation_protocol", {})
    required_hits = int(protocol.get("minimum_table_requests_per_class", 30))
    if required_hits < 1:
        raise ValueError("minimum_table_requests_per_class must be positive")
    diagnostics, dispatches = {}, {}
    requests, service_rows, attempts, candidate_events = [], [], [], []
    attempt_numbers = {}
    def sink(row):
        append(root/"diagnostics.jsonl", row)
        diagnostics[row["request_id"]] = row
    timeout = aiohttp.ClientTimeout(total=600, connect=10)
    core = _build_multi_core(config, lambda: aiohttp.ClientSession(timeout=timeout), diagnostic_sink=sink)
    meter = Meter(root)
    meter.start()
    controller = runtime = None

    async def request(pair, body, request_id, targets, **meta):
        start = time.time()
        prepared = await core.cores[tuple(pair)].prepare(strip_feedback_metadata(body), request_id)
        tokens = await consume(prepared, int(body["max_tokens"]))
        end = time.time()
        if request_id not in diagnostics:
            raise RuntimeError("completed stream missing diagnostic: "+request_id)
        ttft, tpot = pd_inference_metrics(prepared.diagnostics.timestamps_monotonic_s, int(body["max_tokens"]))
        row = dict(await meter.energy(pair, start, end, targets), request_id=request_id,
                   workload_id=body["xpyd_workload_id"], ttft_ms=ttft, tpot_ms=tpot,
                   p_mhz=targets[0], d_mhz=targets[1], observed_output_tokens=tokens, **meta)
        row["mean_power_w"] = row["energy_j"]/row["duration_s"]
        requests.append(row)
        append(root/"requests_energy.jsonl", row)
        return row

    class ContinuousRuntime(PhysicalFeedbackRuntime):
        async def probe(self, body, workload_id, p_mhz, d_mhz, probe_id):
            control_start = time.time()
            event = dict(request_id=probe_id, workload_id=workload_id, p_mhz=p_mhz, d_mhz=d_mhz,
                         start=control_start, attempt=attempt_numbers[workload_id], **probe_metadata(probe_id))
            append(root/"canary_probe_events.jsonl", dict(event, event="started"))
            try:
                changed = await self.actuate("experiment", p_mhz, d_mhz)
                if changed:
                    await asyncio.sleep(.5)
                row = await request(("P0", "D0"), body, probe_id, (p_mhz, d_mhz),
                                    phase="canary", attempt=attempt_numbers[workload_id], **probe_metadata(probe_id))
                inclusive = await meter.energy(("P0", "D0"), control_start, row["end"])
                row["including_control"] = inclusive
                append(root/"canary_probes.jsonl", row)
                append(root/"canary_probe_events.jsonl", dict(event, event="completed", end=row["end"]))
                return ProbeResult(p_mhz, d_mhz, row["mean_power_w"], row["energy_j"],
                                   row["ttft_ms"], row["tpot_ms"])
            except Exception as exc:
                append(root/"canary_probe_events.jsonl", dict(event, event="failed", end=time.time(), error=str(exc)))
                raise

    class TimedController(OnlineFeedbackController):
        def _record_service_request(self, value):
            dispatches[value["request_id"]] = dict(value)
            super()._record_service_request(value)

        def _log(self, value):
            super()._log(value)
            if value.get("event") == "probe_candidate_aggregated":
                candidate_id = value["probe_id"]
                rr = [r for r in requests if r.get("candidate_id") == candidate_id]
                row = dict(value, attempt=attempt_numbers[value["workload_id"]], **probe_metadata(candidate_id))
                if rr:
                    row.update(start=min(r["start"] for r in rr), end=max(r["end"] for r in rr),
                               request_duration_sum_s=sum(r["duration_s"] for r in rr),
                               request_energy_sum_j=sum(r["energy_j"] for r in rr))
                    row["duration_s"] = row["end"]-row["start"]
                    row["control_inclusive_energy_sum_j"] = sum(r["including_control"]["energy_j"] for r in rr)
                    row["control_inclusive_duration_sum_s"] = sum(r["including_control"]["duration_s"] for r in rr)
                    row["mean_power_w"] = row["request_energy_sum_j"]/row["request_duration_sum_s"]
                candidate_events.append(row)
                append(root/"canary_candidates.jsonl", row)

        async def _explore(self, workload_id, body, request_id):
            number = attempt_numbers.get(workload_id, 0)+1
            attempt_numbers[workload_id] = number
            start = time.time()
            success = False
            try:
                # Unique IDs across retries preserve every measured attempt.
                await super()._explore(workload_id, body, request_id+f"-attempt-{number}")
                success = True
            finally:
                end = time.time()
                row = dict(await meter.energy(("P0", "D0"), start, end),
                           workload_id=workload_id, attempt=number, success=success)
                row["mean_power_w"] = row["energy_j"]/row["duration_s"]
                attempts.append(row)
                append(root/"canary_attempts.jsonl", row)

    async def production(body, request_id):
        start = time.time()
        active_at_start = controller.status()["active_workload"]
        prepared = await controller.handle(body, request_id)
        tokens = await consume(prepared, int(body["max_tokens"]))
        end = time.time()
        if request_id not in diagnostics or request_id not in dispatches:
            raise RuntimeError("Production stream missing diagnostic or dispatch")
        d = dispatches[request_id]  # Actual dispatch snapshot, immune to concurrent table writes.
        targets = (d["prefill_frequency_mhz"], d["decode_frequency_mhz"])
        wall = prepared.diagnostics.timestamps_wall_s
        inference_start = wall["request_received"]
        first = wall["decode_first_real_chunk_received"]
        ttft, tpot = pd_inference_metrics(prepared.diagnostics.timestamps_monotonic_s, int(body["max_tokens"]))
        row = dict(await meter.energy(("P1", "D1"), start, end), request_id=request_id,
                   workload_id=body["xpyd_workload_id"], service_sequence=d["service_sequence"],
                   phase="production", frequency_source=d["frequency_source"],
                   table_hit=d["table_hit"], table_revision=d["table_revision"],
                   p_mhz=targets[0], d_mhz=targets[1], frequency_changed=d["frequency_changed"],
                   settle_wait_s=d["settle_wait_s"], observed_output_tokens=tokens,
                   ttft_ms=ttft, tpot_ms=tpot, control_inclusive_ttft_ms=(first-start)*1000,
                   canary_active_at_start=active_at_start,
                   canary_active_at_end=controller.status()["active_workload"])
        row["mean_power_w"] = row["energy_j"]/row["duration_s"]
        row["inference"] = await meter.energy(("P1", "D1"), inference_start, end, targets)
        row["inference"]["mean_power_w"] = row["inference"]["energy_j"]/row["inference"]["duration_s"]
        service_rows.append(row)
        append(root/"production_requests.jsonl", row)
        return row

    try:
        await meter.ready()
        runtime = ContinuousRuntime(config, core.cores[("P0", "D0")])
        high = (runtime.prefill_grid[-1], runtime.decode_grid[-1])
        write(root/"hardware_grids.json", {"P": runtime.prefill_grid, "D": runtime.decode_grid})
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_model"], local_files_only=True)
        prompts = build_prompt_cache(tokenizer, {int(w["input_len"]) for w in config["workloads"]})
        bodies = {w["id"]: {"model": config["model"], "prompt": prompts[w["input_len"]],
                  "max_tokens": w["output_len"], "ignore_eos": True, "temperature": 0,
                  "stream": True, "xpyd_input_len": w["input_len"], "xpyd_output_len": w["output_len"],
                  "xpyd_workload_id": w["id"]} for w in config["workloads"]}
        controller = TimedController(core.frequency_table, core.cores[("P1", "D1")],
            runtime.actuate, runtime.probe, runtime.prefill_grid, runtime.decode_grid,
            service_settle_s=.5, service_warmup_requests=0, experiment_warmup_requests=1,
            energy_refinement_candidate_budget=9, exploration_max_attempts=3,
            event_log=settings["event_log"], service_request_log=settings["service_request_log"],
            sleep=asyncio.sleep)
        # Only setup warmup precedes the natural Production trace. No baseline stage.
        await runtime.actuate("service", *high)
        await asyncio.sleep(.5)
        await request(("P1", "D1"), bodies["small_light"], "setup-service", high, phase="setup")
        await controller.start()
        exploration_start = production_start = time.time()
        table_counts = {w: 0 for w in bodies}
        sequence = 0
        deadline = time.monotonic()+36000
        while not all(n >= required_hits for n in table_counts.values()):
            if time.monotonic() > deadline:
                raise TimeoutError("not all classes reached required Table-hit count within 10h")
            for w, body in bodies.items():
                sequence += 1
                row = await production(body, "production-%05d" % sequence)
                if row["table_hit"]:
                    table_counts[w] += 1
                await asyncio.sleep(float(settings.get("service_request_interval_s", 5)))
        production_end = time.time()
        await controller.stop()
        exploration_end = max(a["end"] for a in attempts)
        exploration = await meter.energy(("P0", "D0"), exploration_start, exploration_end)
        trace_energy = await meter.energy(("P1", "D1"), production_start, production_end)
        per_class = []
        for w in bodies:
            aa = [a for a in attempts if a["workload_id"] == w]
            duration = sum(a["duration_s"] for a in aa)
            gross = sum(a["energy_j"] for a in aa)
            per_class.append({"workload_id": w, "attempts": len(aa), "duration_s": duration,
                              "gross_j": gross, "mean_power_w": gross/duration})
        overhead = {
            "scope": "P0+D0 GPU boards; excludes model loading, CPU, NIC",
            "exploration": exploration,
            "exploration_mean_power_w": exploration["energy_j"]/exploration["duration_s"],
            "per_class": per_class,
            "inter_attempt_and_queue_energy_j": exploration["energy_j"]-sum(r["gross_j"] for r in per_class),
            "interpretation": "gross extra GPU budget of exploration including warmup, tuning, gaps and retries; no idle-subtracted estimate without a dedicated counterfactual baseline",
        }
        comparison = observed_summary(service_rows)
        for row in comparison:
            if row["table"]["n"] < required_hits or row["safe_high"]["n"] < 1:
                raise RuntimeError("missing natural high/Table observations")
        summary = {
            "valid": True, "design": "continuous_feedback_observation",
            "stop_rule": {"minimum_table_requests_per_class": required_hits},
            "comparison": comparison, "canary_overhead": overhead,
            "production_whole_trace": trace_energy,
            "production_requests": len(service_rows),
            "slo_policy": "report_only_per_user",
            "evidence_boundary": "natural before/after observation, not randomized causal estimate; report sample imbalance and Canary overlap",
        }
        write(root/"canary_overhead.json", overhead)
        write(root/"summary.json", summary)
        export_csv(root/"production_requests.csv", service_rows)
        export_csv(root/"canary_probes.csv", [r for r in requests if r.get("phase") == "canary"])
        export_csv(root/"canary_candidates.csv", candidate_events)
        lines = ["# Continuous Production feedback observation", "",
                 "| Workload | High n | Table n | High W | Table W | High J/request | Table J/request |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for r in comparison:
            a, b = r["safe_high"], r["table"]
            lines.append(f"| {r['workload_id']} | {a['n']} | {b['n']} | {a['mean_power_w']:.3f} | "
                         f"{b['mean_power_w']:.3f} | {a['joules_per_request']:.3f} | {b['joules_per_request']:.3f} |")
        lines.extend(["", f"Canary exploration: {exploration['duration_s']:.3f} seconds, "
                      f"{exploration['energy_j']:.3f} J, {overhead['exploration_mean_power_w']:.3f} W.",
                      "Per-request numbers include clock control; inference-only metrics are separately recorded.",
                      "No fixed-duration baseline or forced high/Table alternation. Unequal sample counts and time drift remain.",
                      "Gross Canary GPU cost includes retries and gaps; no idle-subtracted incremental claim."])
        (root/"summary.md").write_text("\n".join(lines)+"\n")
        export_power_trace(root, getattr(meter, "samples", {}))
        write(root/"audit.json", {"valid": True, "production_requests": len(service_rows),
                                 "table_requests_per_class": table_counts,
                                 "all_energy_and_clock_windows_valid": True,
                                 "slo_is_reported_not_a_failure_gate": True})
    finally:
        if controller is not None and controller._worker is not None:
            controller._worker.cancel()
            await asyncio.gather(controller._worker, return_exceptions=True)
        cleanup_errors = []
        if runtime is not None:
            for group in ("experiment", "service"):
                try:
                    await runtime.actuate(group, runtime.prefill_grid[-1], runtime.decode_grid[-1])
                except Exception as exc:
                    append(root/"cleanup_errors.jsonl", {"group": group, "error": str(exc)})
                    cleanup_errors.append(str(exc))
        meter.stop()
        if cleanup_errors:
            raise RuntimeError("failed to restore GPU clocks: "+repr(cleanup_errors))


def export_power_trace(root, samples):
    """Compact 1-second sensor averages; raw 100-ms samples remain in cache."""
    fields = ["unix_second", "endpoint", "mean_sensor_power_w", "mean_clock_mhz", "samples"]
    with (root/"power_trace_1s.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for endpoint, rows in samples.items():
            bins = {}
            for row in list(rows):
                bins.setdefault(int(row["t"]), []).append(row)
            for second, rr in sorted(bins.items()):
                writer.writerow(dict(unix_second=second, endpoint=endpoint,
                    mean_sensor_power_w=statistics.mean(r["power_w"] for r in rr),
                    mean_clock_mhz=statistics.mean(r["clock_mhz"] for r in rr), samples=len(rr)))


def main():
    from xpyd.phase3c_substrate import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"requests_energy.jsonl").exists():
        raise RuntimeError("refusing to reuse an existing run directory")
    config = load_config(args.config)
    write(args.output/"experiment_config.json", config)
    try:
        asyncio.run(asyncio.wait_for(run(config, args.output), timeout=39600))
    except BaseException as exc:
        write(args.output/"audit.json", {"valid": False, "error": type(exc).__name__+": "+str(exc)})
        raise


if __name__ == "__main__":
    main()
