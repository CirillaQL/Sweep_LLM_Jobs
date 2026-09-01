# XPYD 2P2D online binary-feedback frequency table

This job implements the test-stage online control path on two prefill and two
decode GPUs. It uses vLLM 0.15.1 and `P2pNcclConnector` throughout.

## Dedicated groups

- P0->D0 is the experiment group. It receives only background clones and is
  the only group searched by the controller.
- P1->D1 is the service group. Every original request is returned through this
  pair.
- A table miss forces P1 and D1 to their safe-HIGH frequencies (2520/1500 MHz)
  before serving the original request.
- A table hit applies the stored P/D frequencies to P1/D1 and serves the
  original request there.

The two groups have separate HTTP and KV ports and persistent vLLM servers.
There is no cross-pair routing in this job.

## Request and exploration flow

The proxy classifies requests from the exact `xpyd_input_len`,
`xpyd_output_len`, and `max_tokens` parameters. The seven accepted shapes are
`small_light`, `prefill_medium`, `prefill_heavy`, `decode_medium`,
`decode_heavy`, `balanced_medium`, and `both_heavy`.

On a table miss, the original request immediately follows the high-frequency
P1->D1 path. One deep-copied experiment request is placed in a single-worker
queue for P0->D0. Requests arriving during exploration continue on P1->D1;
new workload classes are queued, while duplicate pending classes are merged.

For each queued workload, the controller:

1. keeps D0 at HIGH and performs lower-bound binary search over 17
   hardware-supported P0 frequency levels;
2. fixes P0 at the selected level and searches 15 D0 levels;
3. runs a joint confirmation probe;
4. writes the P/D frequencies, measured average board power, TTFT, TPOT,
   endpoints, timestamp, and evidence source into the table.

A probe is feasible only when TTFT <= 500 ms and TPOT <= 200 ms. GPU frequency
changes use node-local `nvidia-smi -lgc` with immediate hardware readback.
P0+D0 energy-counter deltas are divided by probe duration to derive measured
average power. Failed exploration leaves the table key empty, so service stays
at HIGH and a later request can retry.

## Concurrency and persistence

The table uses an `RLock`, immutable read snapshots, revisions, and optional
compare-and-set writes. It retains only a lower-power SLO-safe replacement.
Every accepted write is atomically persisted with `os.replace` to
`frequency_table.json`.

The proxy also exposes atomic GET/PUT table APIs and
`GET /xpyd/controller/status`. Controller queue operations are serialized by
an asyncio lock and one background worker, so only one experiment changes
P0/D0 frequencies at a time.

## Test pacing and output

Service requests are interleaved at 0.04 requests/s, one every 25 seconds.
Experiment probes wait 20 seconds between sends. There are 16 service requests
per workload and 112 total service requests, leaving enough time for the seven
serial searches to complete during the run.

The final table and append-only `feedback_events.jsonl` are copied into the
compact Job results. Background experiment traffic is declared separately
from the original-request Phase 3C service audit.

The `READY` marker is present for broker submission. The physical 2P2D feedback
behavior remains unverified until a Slurm run passes.
