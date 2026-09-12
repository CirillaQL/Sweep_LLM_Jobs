# Mixed variable-shape Canary/Production energy validation

Primary J/request excludes idle gaps and DVFS settling. Control-inclusive J/request includes actuation and settle.

| Workload | Policy | Pairs | High active J | Table active J | Active saving | High control J | Table control J | Control saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| balanced_medium | learned | 12 | 1118.764 | 928.628 | 17.00% | 1244.129 | 1108.286 | 10.92% |
| both_heavy | high fallback (not SLO-guaranteed) | 12 | 2436.670 | 2433.647 | 0.12% | 2456.722 | 2461.099 | -0.18% |
| decode_heavy | learned | 12 | 2273.478 | 1874.507 | 17.55% | 2430.391 | 2056.952 | 15.37% |
| decode_medium | learned | 12 | 1098.907 | 903.611 | 17.77% | 1229.716 | 1064.662 | 13.42% |
| prefill_heavy | high fallback (not SLO-guaranteed) | 12 | 652.585 | 651.549 | 0.16% | 781.076 | 785.129 | -0.52% |
| prefill_medium | learned | 12 | 597.415 | 504.632 | 15.53% | 680.209 | 773.543 | -13.72% |
| small_light | learned | 12 | 560.966 | 459.353 | 18.11% | 656.972 | 643.559 | 2.04% |

Canary wall time: 2586.434 s.
Canary active probe energy: 315487.121 J.
Canary gross attempt energy: 332227.391 J.
Canary full wall-span gross energy: 334370.143 J.
