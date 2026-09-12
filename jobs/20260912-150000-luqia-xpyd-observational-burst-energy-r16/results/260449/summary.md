# Mixed variable-shape Canary/Production energy validation

Primary J/request excludes idle gaps and DVFS settling. Control-inclusive J/request includes actuation and settle.

Production J/request is allocated equally across in-flight requests per segment; it is not independent request energy. Same-category epochs hold clocks fixed and drain before switching. Canary search remains low-load and does not establish concurrent-load optimality.

| Workload | Policy | Pairs | High active J | Table active J | Active saving | High control J | Table control J | Control saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| balanced_medium | learned | 12 | 383.478 | 324.164 | 15.47% | 406.244 | 392.497 | 3.38% |
| both_heavy | high fallback (not SLO-guaranteed) | 12 | 865.126 | 866.443 | -0.15% | 865.152 | 896.883 | -3.67% |
| decode_heavy | learned | 12 | 717.856 | 594.240 | 17.22% | 764.768 | 655.092 | 14.34% |
| decode_medium | learned | 12 | 365.689 | 304.793 | 16.65% | 388.900 | 380.583 | 2.14% |
| prefill_heavy | high fallback (not SLO-guaranteed) | 12 | 285.173 | 285.006 | 0.06% | 285.205 | 307.426 | -7.79% |
| prefill_medium | learned | 12 | 229.821 | 197.215 | 14.19% | 304.878 | 257.613 | 15.50% |
| small_light | learned | 12 | 192.251 | 159.651 | 16.96% | 192.268 | 198.024 | -2.99% |

Canary wall time: 2731.552 s.
Canary active probe energy: 335769.142 J.
Canary gross attempt energy: 353521.916 J.
Canary full wall-span gross energy: 355733.438 J.
