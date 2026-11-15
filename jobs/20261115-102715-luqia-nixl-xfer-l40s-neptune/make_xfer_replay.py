#!/usr/bin/env python3
"""Real-time replay that exercises the KV transfer, three segments of --segment-s each:

  w00 big    1 req/s, prompts of exactly 2048 tokens: isolated large transfers (~268 MB)
  w01 conc   4 req/s, prompts 512 / 1024 / 2048 at random: overlapping transfers
  w02 short  8 req/s, prompts from a clipped log-normal (mean 64, cap 128): per-request
             overhead (the 283796 short mix)

Every request generates --output tokens (ignore_eos), few enough that D does not queue.
Poisson arrivals. Prompts of exactly the drawn token lengths (BOS included) are cut from
ShareGPT human turns with the helpers of scripts/make_burstgpt_dataset.py.

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

SEGMENTS = (  # name, rate (req/s), prompt lengths ("short": log-normal)
    ("big", 1.0, (2048,)),
    ("conc", 4.0, (512, 1024, 2048)),
    ("short", 8.0, "short"),
)


def clipped_lognormal(rng: random.Random, mean: float, lo: int, hi: int, sigma: float = 0.5):
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scripts", required=True, help="CanaTune scripts/ directory")
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--segment-s", type=float, default=50.0)
    ap.add_argument("--output", type=int, default=16)
    args = ap.parse_args()

    sys.path.insert(0, args.scripts)
    import make_burstgpt_dataset as mb
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    stream = mb.token_stream(tokenizer, mb.sharegpt_texts(args.sharegpt), 4 * 2048 + 200_000)
    rng = random.Random(args.seed)
    short = clipped_lognormal(rng, 64.0, 8, 128)

    def prompt_of(n: int) -> tuple[str, int]:
        best = None
        for _ in range(20):
            offset = rng.randrange(0, len(stream) - 2 * 2048)
            text, got = mb.exact_prompt(tokenizer, stream, offset, n)
            if got == n:
                return text, got
            if best is None or abs(got - n) < abs(best[1] - n):
                best = (text, got)
        return best

    rows, report = [], {"segment_s": args.segment_s, "output_tokens": args.output}
    inexact = 0
    for k, (name, rate, lengths) in enumerate(SEGMENTS):
        start, t, sent = k * args.segment_s, k * args.segment_s, []
        while True:
            t += rng.expovariate(rate)
            if t >= start + args.segment_s:
                break
            want = short() if lengths == "short" else rng.choice(lengths)
            text, got = prompt_of(want)
            inexact += got != want
            row = {"arrival_s": round(t, 3), "prompt": text, "output_tokens": args.output,
                   "request_tokens": got, "phase": name}
            rows.append(row)
            sent.append(got)
        report[f"w{k:02d}"] = {
            "name": name, "rate_rps": rate, "requests": len(sent),
            "prompt_mean": round(statistics.fmean(sent), 1), "prompt_max": max(sent),
            "kv_mb_mean": round(statistics.fmean(sent) * 131072 / 1e6, 1),  # Mistral-7B, fp16
        }
    report["inexact_prompts"] = inexact
    report["span_s"] = len(SEGMENTS) * args.segment_s
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
