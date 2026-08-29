# XpYd Phase 4B stationary feedback-policy evaluation

Verdict: **PASS**

Energy savings use the same-job STATIC policy as the physical baseline.

| Policy | Workload | J/request | Saving vs Static | SLO safe | Violations |
|---|---|---:|---:|---|---:|
| STATIC | small_light | 917.385 | 0.00% | True | 0 |
| STATIC | prefill_heavy | 1038.734 | 0.00% | False | 1 |
| STATIC | decode_heavy | 3611.393 | 0.00% | True | 0 |
| STATIC | both_heavy | 1843.186 | 0.00% | True | 0 |
| FEEDBACK_ROUTING_ONLY | small_light | 913.137 | 0.46% | True | 0 |
| FEEDBACK_ROUTING_ONLY | prefill_heavy | 1045.076 | -0.61% | True | 0 |
| FEEDBACK_ROUTING_ONLY | decode_heavy | 3608.607 | 0.08% | True | 0 |
| FEEDBACK_ROUTING_ONLY | both_heavy | 1838.251 | 0.27% | True | 0 |
| FEEDBACK_DVFS_ONLY | small_light | 836.654 | 8.80% | True | 0 |
| FEEDBACK_DVFS_ONLY | prefill_heavy | 1048.985 | -0.99% | True | 0 |
| FEEDBACK_DVFS_ONLY | decode_heavy | 3292.605 | 8.83% | True | 0 |
| FEEDBACK_DVFS_ONLY | both_heavy | 1692.236 | 8.19% | True | 0 |
| FULL_FEEDBACK | small_light | 835.761 | 8.90% | True | 0 |
| FULL_FEEDBACK | prefill_heavy | 1047.844 | -0.88% | True | 0 |
| FULL_FEEDBACK | decode_heavy | 3297.306 | 8.70% | True | 0 |
| FULL_FEEDBACK | both_heavy | 1686.695 | 8.49% | True | 0 |

Dynamic trace status: `not_run_until_stationary_passes`.

Ready for dynamic trace evaluation: **True**.

Hard gates: `{"all_feedback_warmups_valid": true, "all_policy_workload_summaries_present": true, "all_stationary_measurements_valid": true, "complete_stationary_plan": true, "explicit_discovered_frequency_states": true, "feedback_decision_trace_complete": true, "no_models_or_future_oracle_inputs": true, "no_unresolved_error": true, "safe_high_restoration": true, "same_job_static_energy_baseline_present": true}`
