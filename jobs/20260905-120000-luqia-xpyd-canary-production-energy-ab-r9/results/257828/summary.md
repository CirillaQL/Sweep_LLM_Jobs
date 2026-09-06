# Continuous Production feedback observation

| Workload | High n | Table n | High W | Table W | High J/request | Table J/request |
|---|---:|---:|---:|---:|---:|---:|
| balanced_medium | 17 | 40 | 144.332 | 114.498 | 1160.455 | 1040.428 |
| both_heavy | 27 | 30 | 145.342 | 116.189 | 2420.733 | 2258.858 |
| decode_heavy | 14 | 43 | 143.955 | 116.174 | 2262.799 | 1889.370 |
| decode_medium | 9 | 48 | 142.203 | 112.509 | 1166.947 | 1028.340 |
| prefill_heavy | 6 | 51 | 141.954 | 114.373 | 759.464 | 697.498 |
| prefill_medium | 4 | 53 | 134.030 | 121.360 | 766.907 | 494.221 |
| small_light | 2 | 55 | 146.531 | 103.868 | 547.379 | 733.198 |

Canary exploration: 2840.390 seconds, 355081.758 J, 125.012 W.
Per-request numbers include clock control; inference-only metrics are separately recorded.
No fixed-duration baseline or forced high/Table alternation. Unequal sample counts and time drift remain.
Gross Canary GPU cost includes retries and gaps; no idle-subtracted incremental claim.
