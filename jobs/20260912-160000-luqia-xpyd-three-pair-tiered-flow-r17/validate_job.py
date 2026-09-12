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
    assert config["workload_ordering"] == "random_category_per_homogeneous_burst_epoch"
    assert protocol['burst_size_choices'] == [1,2,3,5]
    assert protocol['burst_batches_max'] == 3
    assert protocol['burst_arrival_mean_gap_s'] > 0
    assert protocol['production_clock_policy'] == 'record_only_never_abort_for_mismatch'
    assert protocol['canary_clock_exhaustion_policy'] == 'publish_explicit_safe_high_fallback'
    assert config['client']['max_concurrency'] >= 15
    assert protocol["mode"] == "online_table_immediate_with_three_slo_failure_safe_high_fallback"
    assert protocol["minimum_online_table_requests_per_class"] >= 12
    assert protocol["terminal_slo_failure_policy"] == "publish_explicit_safe_high_fallback"
    assert protocol["terminal_infrastructure_failure_policy"] == "fail_fast"
    assert protocol["matched_pairs_per_class"] >= 10
    assert 0.05 <= protocol["frequency_settle_s"] <= 0.5
    assert 0.5 <= protocol["inter_request_gap_min_s"]
    assert protocol["inter_request_gap_max_s"] > protocol["inter_request_gap_min_s"]
    assert protocol["primary_energy_boundary"] == "client_send_to_completed_stream"
    assert feedback["axis_search_levels"] == {"prefill": 17, "decode": 15}
    assert feedback["probe_samples_per_candidate"] == 3
    assert feedback["exploration_slo"] == {"ttft_ms": 500.0, "tpot_ms": 200.0}

    workloads = config["workloads"]
    nodes = {endpoint["endpoint_id"]: endpoint["node"] for endpoint in config["endpoints"]}
    assert nodes == {"P0": "neptune", "P1": "neptune", "P2":"neptune", "D0": "ganymede", "D1": "ganymede", "D2":"ganymede"}
    assert {e['endpoint_id']:e['gpu_ids'] for e in config['endpoints']} == {'P0':[0],'P1':[1],'P2':[2],'D0':[0],'D1':[1],'D2':[2]}
    assert config['tiered_flow']['selection'] == {'energy_band':.01,'headroom':.10,'ceiling':.80}
    launcher = (root.parent / "run.sbatch").read_text()
    assert "#SBATCH --nodelist=neptune,ganymede" in launcher
    assert "L40S_NODE=neptune L4_NODE=ganymede" in launcher
    assert '#SBATCH --gpus-per-node=3' in launcher
    assert 'XPYD_ENDPOINTS_PER_ROLE=3' in launcher
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
    assert "history" not in protocol["design_note"].lower()

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
        "production_order": "homogeneous_concurrent_multi_batch_bursts",
        "evaluation": "exact_cohort_matched_randomized_high_vs_table",
        "canary_grid": "17+15",
        "samples_per_candidate": 3,
        "idle_energy_excluded": True,
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    validate(parser.parse_args().source_root)
