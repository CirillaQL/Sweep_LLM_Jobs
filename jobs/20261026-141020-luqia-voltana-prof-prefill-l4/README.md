# 20261026-141020-luqia-voltana-prof-prefill-l4

VoltanaLLM EcoPred **prefill** latency profile on one **L4** (ganymede), Mistral-7B-v0.1, TP 1,
SGLang 0.4.7.post1 (VoltanaLLM v0.1, 27417323 + the profiling-script patch from 20261029-105703-luqia-voltana-env-setup-v6).

Open-loop ShareGPT arrivals with output length 1 at each request rate for 60 s; per prefill iteration it records batched tokens and execution time and fits TTFT = a*tokens + b per clock (RANSAC, tokens in (2000, 8192)).

- Clocks: 2040,1455,1170,1020,735 MHz (snapped to supported clocks at run time; see `freqs.txt`).
- Request rates: 5,10,15,20,25,30,35,40 req/s, 60 s of arrivals each.
- Server: context 8192, max running 256, CUDA graphs up to bs 256,
  mem fraction 0.85; prefill uses flashinfer, decode triton attention with the fake
  KV-transfer backend (as in the upstream scripts).
- Output in `results/<jobid>/`: `latency-prefill.csv` (EcoFreq lookup table), `latency-prefill.png`,
  `clocks.csv.gz` (actual SM clock, power, throttle reasons; check it before trusting a clock),
  `freqs.txt`, `server_info.json`, log tails. Raw `*-extra.json` stays on the cluster
  (`RAW_CACHE_LOCATION.txt`); `raw_files.txt` lists it.
- Requires 20261029-105703-luqia-voltana-env-setup-v6 to have finished with `RESULT=ok`.
- Expected run time: about 1.5-2.5 h (L4 saturates near 20-25 req/s, higher rates mostly drain a queue).
