# XpYd Phase 3C real multi-endpoint substrate validation

Verdict: **FAIL**

This is infrastructure validation with deterministic baseline routing; it makes no routing-novelty or optimization claim.

| Pair | Requests |
|---|---:|
| P0->D0 | 0 |
| P1->D1 | 671 |

| Endpoint | Role | Node/GPU | Requests | Gross energy (J) | Clock valid |
|---|---|---|---:|---:|---|
| D0 | decode | ganymede/0 | 0 | 257031.06 | False |
| D1 | decode | ganymede/1 | 671 | 367313.687 | False |
| P0 | prefill | uranus/0 | 0 | 438199.449 | False |
| P1 | prefill | uranus/1 | 671 | 465272.867 | False |

Hard gates: {"endpoint_assignment": false, "endpoint_coverage": true, "explicit_compatibility": true, "fixed_clocks": true, "logical_request_id": true, "logical_workload_partition": false, "no_invalidating_thermal_or_hw_slowdown": true, "no_resource_overlap": true, "nvml_energy_windows": true, "online_service_latency_metrics_present": true, "pair_coverage": true, "phase3a_token_semantics": true, "real_sse_and_latency": true, "requested_output_tokens": true}
