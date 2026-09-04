# XPYD 2P2D fast-Canary and Production Table-hit feedback r8

This Job repairs the two infrastructure failures and the Production acceptance
gap observed in Job 257099 while preserving the requested minimum-energy
objective.

The experiment treats the system as completely unknown. The frequency Table
starts with seven empty keys. No prior Job frequency, energy, SLO boundary,
Oracle result, or workload-specific hint is loaded. Frequency candidates come
only from hardware discovery in this run, and every search decision uses only
measurements produced by the current Canary process.

## Service and Canary roles

- Canary P0-D0 receives only cloned exploration requests.
- Production P1-D1 receives all real requests.
- A Table miss serves Production immediately at safe-high 2520/1500 MHz and
  queues one deduplicated Canary exploration.
- A Table hit applies the stored P/D clocks to Production before inference.
- Failed exploration is retried automatically up to three attempts with a
  five-second backoff. If all attempts fail, a later interleaved Production
  request can queue a fresh exploration cycle.

## Faster Canary search

P retains 17 levels and D retains 15. Each axis first uses binary search to
find the lowest SLO-feasible boundary while recording latency and P0+D0 request
energy for every binary candidate. It then starts from the lowest-energy safe
binary candidate and measures adjacent levels until neither neighbor improves
energy, with a hard budget of nine unique candidates per axis. Cached binary
measurements are never repeated.

Every candidate and final joint confirmation uses exactly three requests. The
gate remains TTFT P95 < 500 ms and TPOT P95 <= 200 ms. Selection uses mean
P0+D0 energy per request in joules; average board power is diagnostic only.

The previous complete feasible-suffix scan used up to 17+15 candidates, or 99
Canary requests including confirmation. R8 uses at most 9+9 candidates, or 57
requests including confirmation: a maximum probe reduction of about 42%.
It is a bounded local energy refinement, so it can miss a separated second
energy valley; this is the explicit tradeoff for shorter Canary time. No
historical Oracle is consulted before or during the experiment.

## Slurm-step resilience

Job 257099 saw transient `srun` failures reporting that task memory was not
available, plus one opaque energy-read failure. R8 gives each node-local clock
and energy step an explicit 128 MiB memory request and retries transient Slurm
step-creation/plugin failures up to six times with bounded backoff. A terminal
energy error now preserves the last 1000 characters of Slurm output instead of
discarding the cause.

## Longer Production validation

Production changes from seven serial 48-request windows to 96 requests per
workload, interleaved across all seven classes: 672 real requests total at the
same configured 0.20 requests/s. Interleaving repeatedly revisits each Table
key while Canary works. Once a key is written, remaining requests of that class
must become Table hits and exercise its selected clocks on P1-D1.

The doubled Production trace is expected to last roughly 90--110 minutes based
on Job 257099, while the bounded Canary search is expected to finish materially
earlier. The existing 10-hour Canary drain and 12-hour Slurm wall time remain
as fail-safe limits.

Artifacts include per-probe energy/latency, retries and actuation readbacks,
the atomic frequency Table, and every Production dispatch with `table_hit`,
frequency source, and applied P/D clocks. A complete run still fails closed if
any of the seven Table keys is missing.

This directory intentionally has no `READY` marker. It has not been submitted.
