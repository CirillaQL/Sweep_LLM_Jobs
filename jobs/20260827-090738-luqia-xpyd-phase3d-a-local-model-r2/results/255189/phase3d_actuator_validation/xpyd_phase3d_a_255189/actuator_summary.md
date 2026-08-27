# XpYd Phase 3D-A actuator validation

Verdict: **PASS**

This validates physical per-endpoint actuation only; it makes no optimization claim.

Selected states: `{"D0": {"HIGH": 1500, "LOW": 750, "MID": 1125}, "D1": {"HIGH": 1500, "LOW": 750, "MID": 1125}, "P0": {"HIGH": 2520, "LOW": 1260, "MID": 1890}, "P1": {"HIGH": 2520, "LOW": 1260, "MID": 1890}}`.

Successful actuation/readback latency: mean 0.768481 s, max 1.162060 s.

Hard gates: `{"downward_transition_each_endpoint": true, "every_endpoint_independently_actuable": true, "fresh_readback_matches_target": true, "no_unintended_peer_change": true, "no_unresolved_actuator_error": true, "request_stream_token_accounting": true, "safe_restoration": true, "targets_hardware_supported": true, "upward_transition_each_endpoint": true, "valid_energy_telemetry": true}`
