# Fine-grained binary-feedback DVFS result

Verdict: **FAIL**

Raw logs remain at `/data/users/chjing/vllm_job_work/255348/binary_dvfs/xpyd_binary_dvfs_255348`; this Git result contains only compact statistics and audits.

| Workload | Outcome | P-pool MHz | D-pool MHz | High J/req | Selected J/req | Savings | Min peak concurrency | Goodput req/s | Max p99 TTFT | Max p99 TPOT | Violations |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| both_heavy | CONFIRMED_SLO_SAFE | 900 | 450 | 1100.887 | 932.049 | 15.34% | 2 | 0.253 | 273.91 | 60.29 | 0 |
| decode_heavy | CONFIRMED_SLO_SAFE | 900 | 450 | 2157.152 | 1805.290 | 16.31% | 2 | 0.131 | 223.46 | 59.13 | 0 |
| prefill_heavy | CONFIRMED_SLO_SAFE | 2520 | 1050 | 639.261 | 614.352 | 3.90% | 2 | 0.472 | 492.29 | 59.68 | 0 |
| small_light | CONFIRMED_SLO_SAFE | 900 | 450 | 546.317 | 459.222 | 15.94% | 2 | 0.510 | 222.99 | 58.95 | 0 |

Hard gates: `{"binary_decision_trace_present": true, "decode_grid_has_at_least_ten_levels": true, "every_actuation_has_command_readback_and_independent_match": false, "every_feasible_outcome_is_slo_safe": true, "every_outcome_has_valid_physical_evidence": false, "every_physical_window_has_independent_target_clock_match": false, "every_physical_window_has_real_concurrency_and_balanced_routes": true, "four_workloads_have_recorded_outcomes": true, "infeasible_outcomes_use_max_frequency_baseline": true, "native_binary_dvfs_audit_valid": true, "prefill_grid_has_at_least_ten_levels": true, "raw_logs_outside_git_job_tree": true}`
