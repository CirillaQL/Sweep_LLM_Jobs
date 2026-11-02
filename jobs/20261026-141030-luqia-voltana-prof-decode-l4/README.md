# 20261026-141030-luqia-voltana-prof-decode-l4

VoltanaLLM EcoPred **decode** latency profile on one **L4** (ganymede), Mistral-7B-v0.1, TP 1,
SGLang 0.4.7.post1 (VoltanaLLM v0.1, 27417323 + the profiling-script patch from 20261029-105703-luqia-voltana-env-setup-v6).

For each batch size in 1, 8k, 8k+1, 128k, 128k+1 (<= max running) it sends that many 1-token prompts at once with outputs sized to fill the KV cache; per decode step it records batch size, KV tokens and ITL and fits ITL = a*bs + b*tokens + c per clock and batch-size bucket (RANSAC, ITL in (3, 300) ms).

- Clocks: 2040,1455,1170,1020,735 MHz (snapped to supported clocks at run time; see `freqs.txt`).
- Server: context 8192, max running 512, CUDA graphs up to bs 512,
  mem fraction 0.88; prefill uses flashinfer, decode triton attention with the fake
  KV-transfer backend (as in the upstream scripts).
- Output in `results/<jobid>/`: `latency-decode.csv` (EcoFreq lookup table), `latency-decode.png`,
  `clocks.csv.gz` (actual SM clock, power, throttle reasons; check it before trusting a clock),
  `freqs.txt`, `server_info.json`, log tails. Raw `*-extra.json` stays on the cluster
  (`RAW_CACHE_LOCATION.txt`); `raw_files.txt` lists it.
- Requires 20261029-105703-luqia-voltana-env-setup-v6 to have finished with `RESULT=ok`.
- Expected run time: about 4-8 h (KV only ~45k tokens, so outputs per batch size are short).
