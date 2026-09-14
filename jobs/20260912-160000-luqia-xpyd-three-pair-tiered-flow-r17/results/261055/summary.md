# Mixed variable-shape Canary/Production energy validation

Primary J/request excludes idle gaps and DVFS settling. Control-inclusive J/request includes actuation and settle.

Production J/request is allocated equally across in-flight requests per segment; it is not independent request energy. Same-category epochs hold clocks fixed and drain before switching. Canary search remains low-load and does not establish concurrent-load optimality.

| Workload | Policy | Pairs | High active J | Table active J | Active saving | High control J | Table control J | Control saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| balanced_medium | learned | 12 | 382.677 | 315.939 | 17.44% | 382.691 | 315.968 | 17.44% |
| both_heavy | learned | 12 | 861.079 | 863.109 | -0.24% | 861.104 | 863.145 | -0.24% |
| decode_heavy | learned | 12 | 712.233 | 576.415 | 19.07% | 712.270 | 576.423 | 19.07% |
| decode_medium | learned | 12 | 364.309 | 294.673 | 19.11% | 364.323 | 294.696 | 19.11% |
| prefill_heavy | learned | 12 | 285.329 | 284.659 | 0.23% | 285.343 | 284.686 | 0.23% |
| prefill_medium | learned | 12 | 228.342 | 230.173 | -0.80% | 228.372 | 230.201 | -0.80% |
| small_light | learned | 12 | 189.237 | 190.542 | -0.69% | 189.277 | 190.584 | -0.69% |

Canary wall time: 2837.955 s.
Canary active probe energy: 344147.467 J.
Canary gross attempt energy: 360467.804 J.
Canary full wall-span gross energy: 362007.276 J.
