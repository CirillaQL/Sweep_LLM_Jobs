# VoltanaLLM end to end, 1P1D (L40S prefill + L4 decode)

VoltanaLLM v0.1 (SGLang 0.4.7.post1) as a baseline on our heterogeneous testbed. Prefill runs on one L40S (neptune) and decode on one L4 (ganymede). The model is Mistral-7B-v0.1, and KV is transferred over NIXL, which uses UCX over TCP (100 GbE, no RDMA).

Requires `20261029-105703-luqia-voltana-env-setup-v6`, the EcoPred tables of 280101 (prefill-L40S) and 280104 (decode-L4), and ShareGPT under `/data/users/chjing/datasets`.

## Configurations

One server launch. The driver switches between configurations over HTTP (`/set_freq`, `/set_freq_manager_state`), the same endpoints the paper's scripts use.

| Run | Prefill (L40S) | Decode (L4) | Paper counterpart |
|---|---|---|---|
| `static_max` | 1800 MHz | 2040 MHz | SGLang at 1410 MHz (A100 max) |
| `static_low` | 1080 MHz | 1020 MHz | SGLang at 1005 MHz ("sweet point") |
| `voltana_itl100` | EcoFreq, {1080, 1800}, budget 350 ms | EcoFreq, {1020, 2040}, ITL 100 ms | VoltanaLLM |
| `voltana_itl150` | as above | ITL 150 ms | ITL sensitivity |

- **Frequency lists.** There are two levels, as in the paper's main runs ({1005, 1410}). 2520 MHz on the L40S is left out because prefill is power-capped there (R² 0.78 in 280101). The low levels are a guess at the sweet point; the static runs report energy for both levels.
- **EcoRoute.** It has nothing to choose with one decode instance, so the router is SGLang's `mini_lb`. Requests go to P and D at the same time, and P pushes KV to D over NIXL.

## SLOs

- **TTFT.** 600 ms end to end, the paper's value for LLaMA-3.1-8B, the closest model.
  - EcoFreq's prefill check is `predicted prefill latency <= slo - P queueing time`. It ignores the KV transfer, which is negligible on the paper's A100s with NVLink.
  - Over TCP, SGLang's VRAM-to-VRAM push measured about 0.6 GB/s. At 128 KB per token, 1k tokens take about 200 ms plus fixed overheads.
  - The prefill budget is therefore 600 − 250 = 350 ms. `transfer_fit.txt` reports what the transfer actually cost.
- **ITL.** The paper's 60 ms is below the L4's floor: about 57 ms at bs 1, set by memory bandwidth. Decode-L4 is scored at 100 ms and 150 ms. The analysis also scores 60/80/120 ms offline, because the static runs do not depend on the SLO.
- **Margin.** 0. The code computes `slo × (1 + margin)`, so a margin loosens the SLO. The paper does not use one.
- **Attainment.** A request counts when it completed and its TTFT and mean ITL are both within the SLO (paper §6.1).

## Workload

- **Dataset.** ShareGPT with context ≤ 3000 tokens and the dataset's output lengths, with EOS ignored (the client's default, as in the paper's scripts).
- **Arrivals.** Poisson, 180 s of arrivals per rate.
- **Rates.** 1–6 req/s. The L4's KV pool (46.6k tokens) holds about 120 ShareGPT requests, which caps throughput at around 6 req/s.
- **Seed.** The same seed is used in every configuration.
- **Smoke stage.** 20 requests at 1 req/s, as warm-up; the job stops if it fails.

## Changes to VoltanaLLM

`voltana-e2e.patch` is applied to a job-local copy of the sglang package; the shared checkout is untouched.

- **Frequency setter.** EcoFreq's setter originally runs `sudo -E python freq_setter.py`, which the cluster does not allow. It now locks clocks with `sudo -n nvidia-smi -lgc`, via the new `freq_setter_nvsmi.py`. A call takes tens to hundreds of ms rather than about 1 ms with NVML, so queued targets are coalesced. Every change is logged with its duration (`p/d_setter_ms_p50` in the summary).
- **GPU index.** EcoFreq's setter and the energy monitor use the physical NVML index (`VOLTANA_NVML_INDEX`). Without device cgroups, `base_gpu_id` 0 would point at another GPU.

The prefill and decode server settings match the profiling jobs the tables came from (memory fraction, max running requests, context length, attention backend, CUDA graphs).

## Outputs (`results/<jobid>/`)

- **`summary.txt` / `summary.csv`.** One row per configuration and rate, with:
  - TTFT p50/p90/p99 and per-request mean ITL;
  - attainment for TTFT, for ITL at 60/80/100/120/150 ms, and for both together;
  - energy per GPU and in total (J), J per output token;
  - mean SM clock, and the number of EcoFreq clock changes.
- **`requests.csv.gz`.** One row per request: input/output length, TTFT, mean ITL and p90 ITL.
- **`transfer_fit.txt`.** `TTFT = a + b × input_len` from the static_max runs at low rate, giving the KV transfer cost per token.
- **Raw logs and samples.** `prefill/decode_nvml.csv.gz` (0.2 s NVML samples), `prefill/decode.log.gz` (server logs, including the setter lines), `bench/*.jsonl`, and `tables/` (the EcoPred tables used).
