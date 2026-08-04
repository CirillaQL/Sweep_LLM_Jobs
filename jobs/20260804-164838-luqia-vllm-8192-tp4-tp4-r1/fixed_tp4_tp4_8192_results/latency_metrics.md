# 8192-token fixed TP=4/TP=4 latency report

Integrity: **PASS**; all latency SLOs: **VIOLATION**.

Observed topology: 1 Prefill at TP=4 / 1 Decode at TP=4; allocation GPUs: 4 / 4.

| workload | in/out | rate | success/fail | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL | req/s | SLO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| input8192_r1_o128 | 8192/128 | 1.00 | 12/0 | 839.14 ms | 1026.71 ms | 23.52 ms | 27.27 ms | 36.90 ms | 0.78 | VIOLATION |
| input8192_r2_o128 | 8192/128 | 2.00 | 24/0 | 1070.46 ms | 2910.71 ms | 23.72 ms | 25.46 ms | 34.31 ms | 1.47 | VIOLATION |
| input8192_r4_o128 | 8192/128 | 4.00 | 48/0 | 3084.23 ms | 9610.29 ms | 34.93 ms | 44.21 ms | 97.77 ms | 1.87 | VIOLATION |
| input8192_r1_o512 | 8192/512 | 1.00 | 16/0 | 340.69 ms | 637.12 ms | 26.86 ms | 28.36 ms | 51.65 ms | 0.54 | VIOLATION |

TTFT SLO: 500 ms; TPOT SLO: 200 ms.
