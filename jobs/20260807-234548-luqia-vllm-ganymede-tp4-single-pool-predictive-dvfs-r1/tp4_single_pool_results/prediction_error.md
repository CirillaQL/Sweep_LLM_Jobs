# TP4 ganymede L4 single-pool predictive DVFS validation

Integrity: **PASS**; SLO prediction mismatches: **2**.

| workload | target/actual MHz | predicted/actual P99 TTFT | predicted/actual P99 TPOT | actual W | SLO match |
|---|---:|---:|---:|---:|---|
| short_low | 210/210.0 | 173.9/138.5 ms | 65.6/34.2 ms | 115.8 | YES |
| prefill_128 | 570/570.0 | 180.7/190.5 ms | 40.2/25.7 ms | 136.4 | YES |
| mixed_512 | 1830/1830.0 | 291.5/259.1 ms | 37.1/20.8 ms | 202.9 | NO |
| prefill_1024 | 2040/2028.8 | 372.2/211.3 ms | 45.1/21.6 ms | 182.8 | NO |
| decode_512 | 360/360.0 | 147.2/138.7 ms | 33.7/36.4 ms | 143.2 | YES |

TTFT MAE: 49.40 ms; TPOT MAE: 17.69 ms.
