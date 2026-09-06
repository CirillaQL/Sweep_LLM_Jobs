# Sequential workload-window Production feedback observation

| Workload | High n | Table n | High W | Table W | High J/request | Table J/request |
|---|---:|---:|---:|---:|---:|---:|
| balanced_medium | 26 | 30 | 148.360 | 121.657 | 1113.035 | 925.319 |
| both_heavy | 30 | 30 | 149.870 | 124.986 | 2310.558 | 1972.641 |
| decode_heavy | 34 | 30 | 148.278 | 121.306 | 2160.937 | 1799.995 |
| decode_medium | 26 | 30 | 147.361 | 121.000 | 1088.619 | 904.200 |
| prefill_heavy | 18 | 30 | 151.856 | 133.232 | 651.419 | 569.336 |
| prefill_medium | 24 | 30 | 148.635 | 123.330 | 600.719 | 503.300 |
| small_light | 20 | 30 | 147.058 | 118.909 | 545.746 | 462.162 |

Canary exploration: 2428.247 seconds, 301069.249 J, 123.986 W.
Per-request numbers include clock control; inference-only metrics are separately recorded.
Each workload is one contiguous window: natural safe-high while Canary searches, then consecutive Table requests.
Gross Canary GPU cost includes same-class retries and gaps, but excludes Canary-GPU idle time between workload windows.
