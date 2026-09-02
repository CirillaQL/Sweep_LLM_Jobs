# XPYD 2P2D windowed minimum-energy feedback r5

This job implements the test-stage online control path on two prefill and two
decode GPUs. It uses vLLM 0.15.1 and `P2pNcclConnector` throughout.

The prefill endpoints P0/P1 use GPUs 0/1 on `uranus` (L40S), while decode
endpoints D0/D1 use GPUs 0/1 on `ganymede` (L4).

This r5 keeps the corrected complete-P/D TTFT boundary, warmup behavior,
serial workload windows, and P1/D1 steady-state reporting from r4. It changes
exploration to repeated-request, sequential-axis minimum-energy selection.
It prevents
the first `small_light` request from treating lazy CUDA/P2P-NCCL startup as a
frequency failure. Before either pair's first measured request, P1/D1 and
P0/D0 each execute two full-path warmup clones at 2520/1500 MHz. Warmups use
independent request IDs and are logged but discarded: they never participate
in the service SLO audit, search decisions, or Table writes.

When a P1/D1 service frequency target changes, the request waits 10 seconds
after successful clock actuation/readback before inference begins. P0/D0
experiment probes use the same 10-second settling rule. The inference TTFT and
TPOT used by the Table and SLO audit start after settling; raw client-observed
TTFT, which includes actuation and settling, remains available separately.

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

Before the first original request is measured, the controller runs the two
one-time P1/D1 warmups. Before the first queued workload is measured, it runs
the two one-time P0/D0 warmups. For each queued workload, it then:

1. keeps D0 at HIGH and measures 7 hardware-supported P0 frequency levels;
2. selects the SLO-safe P0 level with minimum mean request energy;
3. fixes P0 at that level and applies the same selection over 5 D0 levels;
4. runs a repeated joint confirmation probe;
5. writes the P/D frequencies, measured request energy, average board power,
   TTFT P90, TPOT P90,
   endpoints, timestamp, and evidence source into the table.

A candidate and the joint confirmation each use five measured requests. They
are feasible only when TTFT P90 is strictly below 500 ms and TPOT P90 is at
most 200 ms. The separate P1/D1 service audit remains TTFT <= 500 ms and TPOT
<= 200 ms. GPU frequency changes use
node-local `nvidia-smi -lgc` with immediate hardware readback.
P0+D0 energy-counter deltas provide measured request energy; average power is
also retained for diagnostics. Failed exploration leaves the table key empty,
so service stays at HIGH and a later request can retry.

## Concurrency and persistence

The table uses an `RLock`, immutable read snapshots, revisions, and optional
compare-and-set writes. It retains only a lower-energy SLO-safe replacement.
Every accepted write is atomically persisted with `os.replace` to
`frequency_table.json`.

The proxy also exposes atomic GET/PUT table APIs and
`GET /xpyd/controller/status`. Controller queue operations are serialized by
an asyncio lock and one background worker, so only one experiment changes
P0/D0 frequencies at a time.

## Windowed service monitoring

The seven workload classes run as seven serial windows, never round-robin:
all `small_light` requests finish before `prefill_medium` begins, and so on.
Each window contains 48 requests at 0.04 requests/s (one every 25 seconds), for
336 formal P1/D1 requests and about 140 minutes of trace time. Experiment
probes continue in the background and wait 20 seconds between sends.

Every formal request is appended to `service_request_dispatch.jsonl` with its
workload, Table hit/revision, requested P1/D1 frequencies, frequency-change
flag, and settling delay. After the run, this is joined one-to-one with the
corrected inference metrics in `requests.csv`.

The steady-state analyzer divides each workload window whenever its P/D
frequency or Table state changes, discards the first three requests in every
configuration segment, and reports TTFT/TPOT mean, median, P90, P95, min, max,
standard deviation, and coefficient of variation. A segment is marked stable
only with at least eight retained samples and TTFT/TPOT CV <= 10%.

The final table, feedback events, dispatch log, joined request table, filtered
steady samples, and JSON/CSV steady summaries are copied into compact Job
results. An EXIT collector preserves every available artifact even if a later
audit or Table-completeness gate fails. The raw cache path is also returned in
`RAW_CACHE_LOCATION.txt`.

The `READY` marker is present for broker submission.
The physical windowed steady-state behavior remains unverified until a Slurm
run passes.
