# 20261026-141010-luqia-voltana-prof-decode-l40s

VoltanaLLM EcoPred **decode** latency profile on one **L40S** (neptune), Mistral-7B-v0.1, TP 1,
SGLang 0.4.7.post1 (VoltanaLLM v0.1, 27417323 + the profiling-script patch from 20261029-105703-luqia-voltana-env-setup-v6).

For each batch size in 1, 8k, 8k+1, 128k, 128k+1 (<= max running) it sends that many 1-token prompts at once with outputs sized to fill the KV cache; per decode step it records batch size, KV tokens and ITL and fits ITL = a*bs + b*tokens + c per clock and batch-size bucket (RANSAC, ITL in (3, 300) ms).

- Clocks: 2520,1800,1440,1080,720 MHz (snapped to supported clocks at run time; see `freqs.txt`).
- Server: context 16384, max running 1024, CUDA graphs up to bs 1024,
  mem fraction 0.88; prefill uses flashinfer, decode triton attention with the fake
  KV-transfer backend (as in the upstream scripts).
- Output in `results/<jobid>/`: `latency-decode.csv` (EcoFreq lookup table), `latency-decode.png`,
  `clocks.csv.gz` (actual SM clock, power, throttle reasons; check it before trusting a clock),
  `freqs.txt`, `server_info.json`, log tails. Raw `*-extra.json` stays on the cluster
  (`RAW_CACHE_LOCATION.txt`); `raw_files.txt` lists it.
- Requires 20261029-105703-luqia-voltana-env-setup-v6 to have finished with `RESULT=ok`.
- Expected run time: about 6-10 h (KV ~200k tokens, outputs up to 8176 tokens, 45 batch sizes per clock).
