#!/usr/bin/env python3
"""Force one full re-calibration after the length shift (POST /canatune/canary, "full").

Waits for the replay to start (bench_production's "replaying" console line), then until
  * the Canary's length window holds phase S requests alone (the arrival of the
    window-th phase S request, a margin, and the live summary within 10 % of phase S), and
  * every window of the cold start is older than the locator's cache (a manual "full" does
    not drop the cache, so younger windows would be reused with the old lengths),
and then asks for a full calibration. The start conditions of the scheduler still apply
(energy mode, no rejection for quiet_s, min_interval_s): a refusal is logged and retried.
An aborted or failed experiment is requested again. Every step is one JSON line in --log;
the tier table before and after is saved next to it.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def call(base: str, path: str, body: dict | None = None) -> dict | None:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--console", type=Path, required=True)
    ap.add_argument("--console-offset", type=int, required=True, help="bytes to skip")
    ap.add_argument("--report", type=Path, required=True, help="replay report json")
    ap.add_argument("--out", type=Path, required=True, help="directory for the log and tables")
    ap.add_argument("--cache-ttl-s", type=float, required=True)
    ap.add_argument("--margin-s", type=float, default=90.0)
    ap.add_argument("--not-before-s", type=float, default=0.0,
                    help="replay second before which no calibration is requested")
    ap.add_argument("--give-up-before-end-s", type=float, default=2400.0)
    ap.add_argument("--max-attempts", type=int, default=5)
    args = ap.parse_args()

    report = json.loads(args.report.read_text())
    short = report["S"]
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "forced_recal.jsonl"
    t0: float | None = None

    def log(event: str, **fields) -> None:
        rel = None if t0 is None else round(time.time() - t0, 1)
        with open(log_path, "a") as handle:
            handle.write(json.dumps({"t": time.time(), "replay_s": rel, "event": event,
                                     **fields}) + "\n")
        print(f"force_recal: replay_s={rel} {event} {json.dumps(fields)[:400]}", flush=True)

    # 1. the replay's start
    while t0 is None:
        try:
            with open(args.console, "rb") as handle:
                handle.seek(args.console_offset)
                if b": replaying " in handle.read():
                    t0 = time.time()
        except OSError:
            pass
        if t0 is None:
            time.sleep(1.0)
    log("replay_started", shift_at_s=short["start_s"], not_before_s=args.not_before_s,
        window_short_only_after_s=report["window_short_only_after_s"])
    end = t0 + report["span_s"]
    give_up = end - args.give_up_before_end_s

    # 2. the length window holds phase S alone, the cold-start cache has expired
    first = max(report["window_short_only_after_s"] + args.margin_s, args.not_before_s)
    time.sleep(max(0.0, t0 + first - time.time()))
    before = None
    while time.time() < give_up:
        state = call(args.base, "/canatune/state") or {}
        tiers = call(args.base, "/canatune/tiers") or {}
        lengths, table = state.get("lengths") or {}, tiers.get("table")
        if table is not None and lengths:
            age = time.time() - float(table.get("published_at", time.time()))
            ok = (lengths.get("samples", 0) >= report["window_requests"]
                  and lengths.get("prompt_mean", 1e9) <= 1.1 * short["prompt_mean"]
                  and lengths.get("output_mean", 1e9) <= 1.1 * short["output_mean"]
                  and age >= args.cache_ttl_s + 30)
            if ok:
                before = tiers
                log("ready", lengths=lengths, table_age_s=round(age, 1), h=table.get("h"),
                    mode=(state.get("controller") or {}).get("mode"))
                break
        time.sleep(10.0)
    if before is None:
        log("gave_up", reason="conditions never met")
        return 1
    (args.out / "tiers_before_forced.json").write_text(json.dumps(before, indent=2) + "\n")

    # 3. request the full calibration; again after an abort or a failure
    attempts, seen, refusal = 0, None, None
    while time.time() < give_up and attempts < args.max_attempts:
        state = call(args.base, "/canatune/state") or {}
        history = (state.get("canary") or {}).get("history") or []
        if seen is None:
            seen = len(history)
        reply = call(args.base, "/canatune/canary", {"action": "full"})
        if not reply or not reply.get("ok"):
            reason = None if reply is None else reply.get("reason")
            if reason != refusal:  # one line per distinct refusal
                refusal = reason
                log("refused", reply=reply, mode=(state.get("controller") or {}).get("mode"))
            time.sleep(10.0)
            continue
        attempts, refusal = attempts + 1, None
        log("requested", attempt=attempts, reply=reply)
        outcome = None
        while outcome is None and time.time() < end + 600:
            time.sleep(10.0)
            state = call(args.base, "/canatune/state")
            if state is None:
                continue
            canary = state.get("canary") or {}
            history = canary.get("history") or []
            new = [h for h in history[seen:] if h.get("reason") == "api"]
            if new and not canary.get("running") and canary.get("pending") is None:
                outcome, seen = new[-1], len(history)
        if outcome is None:
            log("no_outcome")
            return 1
        log("outcome", attempt=attempts, **{k: outcome.get(k) for k in ("kind", "outcome", "error")})
        if outcome.get("outcome") == "published":
            after = call(args.base, "/canatune/tiers") or {}
            (args.out / "tiers_after_forced.json").write_text(json.dumps(after, indent=2) + "\n")
            a, b = before.get("table") or {}, after.get("table") or {}
            log("published", h_before=a.get("h"), h_after=b.get("h"),
                park_before=a.get("park"), park_after=b.get("park"),
                alpha_before=a.get("alpha_tokens"), alpha_after=b.get("alpha_tokens"),
                capacity_h_before=a.get("capacity_h"), capacity_h_after=b.get("capacity_h"),
                duration_s=(b.get("evidence") or {}).get("duration_s"),
                windows=(b.get("evidence") or {}).get("windows"),
                reused_windows=(b.get("evidence") or {}).get("reused_windows"))
            return 0
    log("gave_up", reason="no published calibration", attempts=attempts)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
