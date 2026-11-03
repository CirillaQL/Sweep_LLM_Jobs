# VoltanaLLM end to end, 4P4D on uranus / europa, aligned with CanaTune's main run

VoltanaLLM v0.1 (SGLang 0.4.7.post1) as the baseline to CanaTune. Everything that CanaTune's main run (`20261013-210335-luqia-canatune-fill-redirect-4groups-x3-slo600-100`, Slurm 275725) fixes is kept the same; only the serving system differs.

The job runs on uranus (4 × L40S, prefill) and europa (4 × L4, decode), not on 275725's neptune / ganymede. The GPU models are the same. Two CanaTune references are used:

- **275830** (`20261014-131315-luqia-canatune-4groups-slo1000-150-uranus-europa`) ran on the same nodes. Its `baseline_r1` (round robin, driver clocks) is the same-node reference. Its CanaTune run used SLO 1000/150, and it also scored 600/100 and 1000/100.
- **275725** is the main run at SLO 600/100 with three CanaTune runs, on neptune / ganymede.
- The two baselines differ by about 1 % in energy (110 vs 109 kJ at 0.5 req/s ShareGPT) and match in goodput, so the node pairs are comparable.

The EcoPred tables were profiled on neptune's L40S and ganymede's L4: the same GPU models, but other units.

Requires `20261029-105703-luqia-voltana-env-setup-v6`, the EcoPred tables of 280101 (prefill-L40S) and 280104 (decode-L4), CanaTune's cuda-env (for the client) and ShareGPT under `/data/users/chjing/datasets`.

## What matches 275725

| | 275725 (CanaTune) | This job (VoltanaLLM) |
|---|---|---|
| Nodes | neptune 4 × L40S (P), ganymede 4 × L4 (D), both exclusive | uranus 4 × L40S (P), europa 4 × L4 (D), both exclusive (as 275830) |
| Model | Mistral-7B-v0.1 | the same files (the job checks md5 of config and tokenizer) |
| Client | `vllm bench serve` of cuda-env, `/v1/completions` | the same binary and arguments, apart from paths and port |
| ShareGPT | `--sharegpt-output-len 256`, EOS stops | the same |
| BurstGPT | 275725's `burstgpt.jsonl`, trace output lengths, `--ignore-eos` | the same file, copied (regenerated at 13b6052 only if it is gone); 275830's sha256 is logged beside it |
| Rates | 0.5, 1, 2, 4, 6, 8 req/s, Poisson, 180 s each, 30 s gap, seed 20261002 | the same |
| Stage timeout | 3 × 180 + 600 s; a failed stage skips the rest of its workload | the same |
| Scoring | TTFT and vLLM's TPOT per request, goodput = good / offered | the same code (`request_outcomes`), checked on 275725's own results |
| Energy | NVML counters of all 8 GPUs over each benchmark's span | the same window, own sampler (0.2 s) |
| KV transfer | vLLM P2pNccl | NIXL, UCX over TCP: **differs** |
| Serving engine | vLLM 0.15.1 | SGLang 0.4.7.post1: **differs** |

## Configurations

There is one server launch with 8 servers and VoltanaLLM's router (`launch_lb --echo-lb`). The driver switches configurations over HTTP.

| Run | Clocks | Routing | Counterpart |
|---|---|---|---|
| `rr_default` | left to the driver | round robin | CanaTune's baseline (`baseline_r1` of 275830 and 275725) |
| `rr_max` | P 1800 / D 2040 locked | round robin | paper: SGLang at max clock |
| `voltana_s600` | EcoFreq, TTFT SLO 600 | EcoRoute (`tile_aware`) | VoltanaLLM |
| `voltana_s1000` | EcoFreq, TTFT SLO 1000 | EcoRoute (`tile_aware`) | TCP-adjusted SLO |
| `voltana_s600_rr` | EcoFreq, TTFT SLO 600 | round robin | EcoRoute ablation; runs only if time is left |

- **EcoFreq.** Frequency lists {1080, 1800} on the L40S and {1020, 2040} on the L4: two levels, as in the paper's main runs.
  - The prefill budget is the TTFT SLO minus 250 ms for the TCP KV transfer. EcoFreq's own offset covers only P queueing.
  - The decode ITL SLO is 100 ms, CanaTune's TPOT SLO. The margin is 0.
- **EcoRoute.** `tile_aware` with `TILE_SIZE=128`, the demo's setting.
  - While every decode instance has fewer than 128 requests in flight, it picks the lowest current clock and then the fewest requests.
  - The L4's KV pool holds about 120 ShareGPT requests, so the tile mode is not expected to trigger.
- **Prefill choice.** Round robin in every configuration, which is the paper's baselines; the code picks at random.
- **SLOs scored.** Every stage is scored at 600/100 (CanaTune's SLO), 1000/100, 600/150 and 1000/150, as TTFT/TPOT in ms.

## Changes to VoltanaLLM (`voltana-4p4d.patch`)

The patch is applied to a job-local copy of the sglang package; the shared checkout is untouched. The first two changes are as in the 1P1D r2 job:

- **Frequency setter.** Clocks are locked with `sudo -n nvidia-smi -lgc` (`freq_setter_nvsmi.py`), because the original root python setter is not allowed.
- **GPU index.** NVML uses the physical GPU index (`VOLTANA_NVML_INDEX`).

The router fixes are new:

- **The router's server-state loop now starts.** `run_echo_lb` builds `echo_lb_new.EchoLoadBalancer`, while the lifespan hook checks `echo_lb.EchoLoadBalancer`. The loop never started, so `tile_aware` would fail on an empty state. It now checks for the router itself.
- **A failed `/get_server_info` no longer ends that loop.**
- **The router's tokenizer is the local model** (`VOLTANA_ROUTER_TOKENIZER`). It used Llama-3.1-8B from the Hub, which is offline here. Only `fadr` uses it.
- **Round-robin prefill choice** (`VOLTANA_PREFILL_POLICY=rr`).

## Time

- Server start-up takes about 5 min.
- Each configuration has 12 stages and takes about 50–70 min, depending on how long overloaded stages take to drain.
- The wall time is 10 h. `voltana_s600_rr` and later stages are skipped when time runs short.

## Outputs (`results/<jobid>/`)

- **`<config>/bench_summary.json`.** CanaTune's schema: per stage goodput, TTFT/TPOT percentiles, energy per GPU, J per good request, `by_slo`, and clocks.
- **`summary.txt` / `summary.csv`.** One row per configuration and stage. It includes goodput at each SLO, energy, mean SM clock per role, and EcoFreq clock changes with setter latency.
- **`compare.txt` / `compare.csv`.** Goodput and energy per stage for every SLO that both sides scored. Each reference contributes `<ref>_base` (its baseline) and `<ref>_fill` (the mean of its CanaTune runs): 275830 has 600/100, 1000/100 and 1000/150; 275725 has 600/100.
- **`<config>/stages/<stage>/`.** `bench.json.gz` (vLLM's detailed result), `command.txt`, `bench.log`, and `router_state.jsonl.gz` (the router's view of all 8 servers every 50 ms).
- **Raw logs and samples.**
  - `logs/`: the server logs, including the setter lines.
  - `prefill/decode_nvml.csv.gz`: NVML samples.
  - `net_<node>.txt.gz`: NIC bytes.
  - `client.txt`: the client version, model checks and the BurstGPT sha256.
