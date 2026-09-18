"""Offline contract, inventory and syntax checks; no GPU use or submission."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def validate(root: Path) -> None:
    config_path = root / "paper/configs/xpyd_mixed_variable_canary_energy_2p2d.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = config["mixed_variable_energy_validation"]
    feedback = config["online_feedback"]

    assert config["cold_start_unknown_system"] is True
    assert config["historical_seed_enabled"] is False
    assert config["search_prior"] == "none"
    assert config["routing_policy"] == "sequential_low_load_canary_to_production_transfer"
    assert config["workload_ordering"] == "seven_contiguous_workload_transfer_windows"
    assert protocol['baseline_requests_per_workload'] == 12
    assert protocol['transfer_requests_per_workload'] == 48
    assert protocol['request_interval_s'] == 5.0
    assert protocol['canary_queue_wait_s'] >= 21600
    assert protocol['production_clock_policy'] == 'record_only_never_abort_for_mismatch'
    assert protocol['canary_clock_exhaustion_policy'] == 'publish_explicit_safe_high_fallback'
    assert config['client']['max_concurrency'] == 1
    assert protocol["mode"] == "sequential_low_load_canary_to_production_transfer"
    assert protocol["terminal_slo_failure_policy"] == "publish_explicit_safe_high_fallback"
    assert protocol["terminal_infrastructure_failure_policy"] == "record_and_continue"
    assert 0.05 <= protocol["frequency_settle_s"] <= 0.5
    assert protocol["primary_energy_boundary"] == "client_send_to_completed_stream"
    assert feedback["axis_search_levels"] == {"prefill": 17, "decode": 15}
    assert feedback["probe_samples_per_candidate"] == 3
    assert feedback["service_request_interval_s"] == 5.0
    assert feedback["probe_request_interval_s"] == 5.0
    assert "one_copy_per_class_binary_plus_greedy_coordinate_search" in feedback["algorithm"]
    assert "copies one request" in protocol["design_note"]
    assert feedback["exploration_slo"] == {"ttft_ms": 500.0, "tpot_ms": 200.0}

    workloads = config["workloads"]
    nodes = {endpoint["endpoint_id"]: endpoint["node"] for endpoint in config["endpoints"]}
    assert nodes == {"P0": "neptune", "P1": "neptune", "P2":"neptune", "P3":"neptune", "D0": "ganymede", "D1": "ganymede", "D2":"ganymede", "D3":"ganymede"}
    assert {e['endpoint_id']:e['gpu_ids'] for e in config['endpoints']} == {'P0':[0],'P1':[1],'P2':[2],'P3':[3],'D0':[0],'D1':[1],'D2':[2],'D3':[3]}
    flow = config['tiered_flow']
    assert flow['production_assignment'] == 'round_robin_logical_tier_per_drained_homogeneous_burst'
    assert flow['frequency_policy'] == 'all_production_endpoints_start_high_then_selected_endpoint_tiers_apply_per_drained_burst'
    assert flow['tiers'] == {
        'high': {'pair':['P1','D1'],'prefill_mhz':2520,'decode_mhz':1500},
        'middle': {'pair':['P2','D2'],'prefill_mhz':1710,'decode_mhz':975},
        'low': {'pair':['P3','D3'],'prefill_mhz':900,'decode_mhz':450},
    }
    assert flow['endpoint_frequency_mhz'] == {
        'P1': 2520, 'P2': 1710, 'P3': 900,
        'D1': 1500, 'D2': 975, 'D3': 450,
    }
    assert flow['route_control_path'] == '$XPYD_ROUTE_CONTROL_PATH'
    assert flow['route_control_reload'] == 'before_each_drained_production_burst'
    endpoint_clocks = {item['endpoint_id']: item['configured_frequency_mhz'] for item in config['endpoints']}
    assert {key: endpoint_clocks[key] for key in ('P1', 'P2', 'P3')} == {'P1': 2520, 'P2': 2520, 'P3': 2520}
    assert {key: endpoint_clocks[key] for key in ('D1', 'D2', 'D3')} == {'D1': 1500, 'D2': 1500, 'D3': 1500}
    assert {key: config['fixed_clocks'][key]['graphics_mhz'] for key in ('P1', 'P2', 'P3')} == {'P1': 2520, 'P2': 2520, 'P3': 2520}
    assert {key: config['fixed_clocks'][key]['graphics_mhz'] for key in ('D1', 'D2', 'D3')} == {'D1': 1500, 'D2': 1500, 'D3': 1500}
    production_pairs = {(pair['prefill_endpoint_id'], pair['decode_endpoint_id']) for pair in config['compatible_pairs'] if pair['prefill_endpoint_id'] != 'P0'}
    assert production_pairs == {(f'P{p}', f'D{d}') for p in range(1, 4) for d in range(1, 4)}
    contract = config['telemetry_contract']
    assert contract['output'] == 'nodegroup_telemetry.jsonl'
    assert contract['arrival_rate_window_s'] > 0
    assert contract['prefill_capacity_input_tokens_per_s'] is None
    assert contract['decode_capacity_output_tokens_per_s'] is None
    launcher = (root.parent / "run.sbatch").read_text()
    assert "#SBATCH --nodelist=neptune,ganymede" in launcher
    assert "L40S_NODE=neptune L4_NODE=ganymede" in launcher
    assert '#SBATCH --gpus-per-node=4' in launcher
    assert 'XPYD_ENDPOINTS_PER_ROLE=4' in launcher
    assert 'XPYD_ROUTE_CONTROL_PATH="${XPYD_ROUTE_CONTROL_PATH_OVERRIDE:-${RAW_ROOT}/production_route_control.json}"' in launcher
    assert len(workloads) == 7
    assert len({workload["id"] for workload in workloads}) == 7
    assert all(len(workload["shape_panel"]) == 3 for workload in workloads)
    small = next(workload for workload in workloads if workload["id"] == "small_light")
    assert {shape["input_len"] for shape in small["shape_panel"]} >= {128, 200}
    all_shapes = [
        (workload["id"], shape["input_len"], shape["output_len"])
        for workload in workloads for shape in workload["shape_panel"]
    ]
    assert all(input_len > 0 and output_len > 1 for _, input_len, output_len in all_shapes)

    driver = (root / "paper/scripts/xpyd/mixed_variable_canary_energy.py").read_text()
    assert "client_send_to_completed_stream" in driver
    assert "control_inclusive" in driver
    assert "matched_pair_plan" in driver
    assert "await controller.enqueue_only" in driver
    assert "nodegroup_telemetry.jsonl" in driver
    assert "workload_transfer_windows.jsonl" in driver
    assert "low_load_transfer_summary.json" in driver
    assert "production_canary_transfer" in driver
    assert "history" not in protocol["design_note"].lower()

    control_example = json.loads((root.parent / 'production_route_control.example.json').read_text(encoding='utf-8'))
    assert control_example == {'schema_version': 1, 'routes': {
        'high': {'prefill_endpoint_id': 'P1', 'decode_endpoint_id': 'D1'},
        'middle': {'prefill_endpoint_id': 'P2', 'decode_endpoint_id': 'D2'},
        'low': {'prefill_endpoint_id': 'P3', 'decode_endpoint_id': 'D3'},
    }}

    manifest = json.loads((root.parent / "source_manifest.json").read_text())
    files = {
        str(path.relative_to(root)): path for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    assert set(files) == set(manifest), "source file inventory mismatch"
    for name, path in files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[name], name
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    subprocess.run(["bash", "-n", str(root / "run_disagg_benchmark.sh")], check=True)
    subprocess.run(["bash", "-n", str(root.parent / "run.sbatch")], check=True)
    print(json.dumps({
        "valid": True,
        "source_files": len(files),
        "cold_start": True,
        "production_order": "seven_contiguous_low_load_transfer_windows",
        "evaluation": "sequential_p1d1_high_baseline_then_canary_table_transfer",
        "production_tiers": "p1d1_selected_production_group",
        "canary_grid": "17+15",
        "samples_per_candidate": 3,
        "idle_energy_excluded": True,
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    validate(parser.parse_args().source_root)
