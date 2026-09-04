# XPYD 2P2D P17/D15 binary-SLO minimum-energy feedback r7b

This clean resubmission follows the service/Canary architecture of Jobs 256683 and 255414,
while correcting the optimization target and the incomplete-shutdown behavior
observed in Job 256683.

- Canary P0-D0 runs only cloned exploration requests.
- Production P1-D1 runs every real request.
- On a Table miss, Production immediately uses safe-high P=2520 MHz and
  D=1500 MHz; one deduplicated clone is queued for Canary exploration.
- On a Table hit, Production applies the stored P/D frequencies before serving
  the request.
- The persistent, atomic Table has one key for each of the seven workload
  classes.

## Search and objective

The hardware-supported P0 range 900--2520 MHz is quantized into 17 levels and
the D0 range 450--1500 MHz into 15 levels. The exact levels are selected from
the frequencies reported by each GPU and always include the safe-high endpoint.

For each workload class the single Canary worker does the following:

1. With D0 at 1500 MHz, use binary search on the 17 P levels to locate the
   lowest P index satisfying the exploration SLO.
2. Measure every level in the P feasible suffix, reusing all cached binary
   candidates, and select the SLO-safe level with minimum mean P0+D0 request
   energy in joules.
3. Hold P0 at that result and repeat the same binary-boundary plus feasible-
   suffix energy selection over the 15 D levels.
4. Run a fresh joint confirmation and write the confirmed configuration and
   evidence to the Table.

Every candidate and the joint confirmation use exactly three requests. Latency
feasibility is **TTFT P95 < 500 ms and TPOT P95 <= 200 ms**. Energy is the
NVML total-energy-counter delta of P0 plus D0 over the complete request. Board
power in watts remains diagnostic only and never participates in selection.

Binary search alone can only find the lowest SLO-feasible frequency; it cannot
prove the lowest-energy point because request energy is not monotone in clock
frequency. The feasible-suffix scan is therefore intentional. Under the
requested sequential-axis assumption, “P energy optimum + D energy optimum =
joint optimum,” this selects the measured energy minimum on each axis. The
worst case is still 17+15 unique candidates per workload; binary search saves
the infeasible prefix when the SLO boundary is above the minimum level.

## Runtime and completion behavior

The Production trace retains seven serial workload windows with 48 requests per
class at 0.20 requests/s. Exploration runs concurrently on the separate GPUs.
Unlike Job 256683, proxy shutdown waits up to 10 hours for the Canary queue to
finish instead of cancelling the last active workload. Slurm wall time is 12
hours, and per-request timeout remains 900 seconds.

Raw caches, logs, per-probe latency/energy, frequency actuation readbacks,
Production dispatches, the final Table, and summary artifacts are stored below
`/data/users/chjing/vllm_job_work/<job_id>/feedback_frequency_table_2p2d/` and
compact results are copied back under `results/<job_id>/`.

Static validation and all unit tests passed before the `READY` submission
marker was added.
