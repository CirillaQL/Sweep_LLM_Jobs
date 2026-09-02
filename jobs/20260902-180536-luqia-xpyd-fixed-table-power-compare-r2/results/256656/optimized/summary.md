# XpYd Phase 3C real multi-endpoint substrate validation

Verdict: **PASS**

This is infrastructure validation with deterministic baseline routing; it makes no routing-novelty or optimization claim.

| Pair | Requests |
|---|---:|
| P0->D0 | 0 |
| P1->D1 | 288 |

| Endpoint | Role | Node/GPU | Requests | Gross energy (J) | Clock valid |
|---|---|---|---:|---:|---|
| D0 | decode | ganymede/0 | 0 | 167741.837 | True |
| D1 | decode | ganymede/1 | 288 | 223193.295 | False |
| P0 | prefill | uranus/0 | 0 | 559438.081 | True |
| P1 | prefill | uranus/1 | 288 | 449356.871 | False |

Hard gates: {"endpoint_assignment": true, "endpoint_coverage": true, "explicit_compatibility": true, "fixed_clocks": true, "logical_request_id": true, "logical_workload_partition": true, "no_invalidating_thermal_or_hw_slowdown": true, "no_resource_overlap": true, "nvml_energy_windows": true, "online_service_requests_meet_slo": true, "pair_coverage": true, "phase3a_token_semantics": true, "real_sse_and_latency": true, "requested_output_tokens": true, "service_workload_energy_windows": true}
