#!/usr/bin/env python3
"""Two-phase real-time replay for the forced re-calibration job.

Phase A (the mix the cold start assumes): prompts drawn uniformly from the Canary's default
probe prompts, every request generating the Canary's output cap.
Phase S (the shift): short requests, prompt and output lengths from clipped log-normal
distributions with the given means and a hard cap.
Poisson arrivals at a fixed rate per phase. Prompts of exactly the drawn token lengths
(BOS included) are cut from ShareGPT human turns with the helpers of
scripts/make_burstgpt_dataset.py, so the proxy counts the lengths written here.

Rows: arrival_s, prompt, output_tokens, request_tokens, phase (bench_production.py replay).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path


def clipped_lognormal(rng: random.Random, mean: float, lo: int, hi: int, sigma: float = 0.5):
    """-> a sampler of integers in [lo, hi] whose mean is `mean` (mu found by bisection
    on a fixed normal sample, so the result depends only on the seed)."""
    zs = [rng.gauss(0.0, 1.0) for _ in range(20000)]

    def clip(mu: float, z: float) -> int:
        return min(hi, max(lo, round(math.exp(mu + sigma * z))))

    a, b = math.log(lo), math.log(hi)
    for _ in range(60):
        mu = (a + b) / 2
        if statistics.fmean(clip(mu, z) for z in zs) < mean:
            a = mu
        else:
            b = mu
    return lambda: clip(mu, rng.gauss(0.0, 1.0))


def quantile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scripts", required=True, help="CanaTune scripts/ directory")
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--a-s", type=float, default=1800.0, help="phase A length (s)")
    ap.add_argument("--a-rate", type=float, default=1.2, help="phase A req/s")
    ap.add_argument("--a-prompts", default="128,512,1024,2048")
    ap.add_argument("--a-output", type=int, default=256)
    ap.add_argument("--s-s", type=float, default=9000.0, help="phase S length (s)")
    ap.add_argument("--s-rate", type=float, default=2.0, help="phase S req/s")
    ap.add_argument("--s-prompt-mean", type=float, default=64.0)
    ap.add_argument("--s-output-mean", type=float, default=32.0)
    ap.add_argument("--s-max", type=int, default=128, help="phase S cap on prompt and output")
    ap.add_argument("--window", type=int, default=2000, help="LengthStats window (requests)")
    args = ap.parse_args()

    sys.path.insert(0, args.scripts)
    import make_burstgpt_dataset as mb
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    a_prompts = [int(x) for x in args.a_prompts.split(",")]
    stream = mb.token_stream(tokenizer, mb.sharegpt_texts(args.sharegpt),
                             4 * max(a_prompts) + 400_000)
    rng = random.Random(args.seed)
    s_prompt = clipped_lognormal(rng, args.s_prompt_mean, 8, args.s_max)
    s_output = clipped_lognormal(rng, args.s_output_mean, 4, args.s_max)

    def prompt_of(n: int, cap: int | None) -> tuple[str, int]:
        """A prompt of n tokens; if none is exact, the closest found (within the cap)."""
        best = None
        for _ in range(20):
            offset = rng.randrange(0, len(stream) - 2 * max(a_prompts))
            text, got = mb.exact_prompt(tokenizer, stream, offset, n)
            if got == n:
                return text, got
            if (cap is None or got <= cap) and (best is None or abs(got - n) < abs(best[1] - n)):
                best = (text, got)
        if best is None:
            raise SystemExit(f"no prompt of {n} tokens within the cap found")
        return best

    rows, report, t, inexact = [], {}, 0.0, 0
    phases = (("A", args.a_s, args.a_rate), ("S", args.s_s, args.s_rate))
    for tag, span, rate in phases:
        start, sent = t, []
        while True:
            t += rng.expovariate(rate)
            if t >= start + span:
                break
            if tag == "A":
                want, output, cap = rng.choice(a_prompts), args.a_output, None
            else:
                want, output, cap = s_prompt(), s_output(), args.s_max
            text, got = prompt_of(want, cap)
            inexact += got != want
            row = {"arrival_s": round(t, 3), "prompt": text, "output_tokens": output,
                   "request_tokens": got, "phase": tag}
            rows.append(row)
            sent.append(row)
        t = start + span
        p = [r["request_tokens"] for r in sent]
        o = [r["output_tokens"] for r in sent]
        report[tag] = {
            "start_s": start, "span_s": span, "requests": len(sent),
            "rate_rps": round(len(sent) / span, 3),
            "prompt_mean": round(statistics.fmean(p), 1), "prompt_p90": quantile(p, 0.9),
            "prompt_max": max(p), "output_mean": round(statistics.fmean(o), 1),
            "output_p90": quantile(o, 0.9), "output_max": max(o),
        }
    short = [r for r in rows if r["phase"] == "S"]
    if max(r["request_tokens"] for r in short) > args.s_max:
        raise SystemExit("a phase S prompt exceeds the cap")
    if len(short) <= args.window:
        raise SystemExit("phase S is shorter than the length window")
    # The Canary probes with the last `window` finished requests: only after this arrival
    # (plus the time to finish) does the window hold phase S requests alone.
    report["window_requests"] = args.window
    report["window_short_only_after_s"] = short[args.window - 1]["arrival_s"]
    report["span_s"] = t
    report["inexact_prompts"] = inexact
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
