"""Offline contract, inventory, and syntax checks; no GPU access."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


EXPECTED_LOADS = [
    0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5,
    1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.0,
]


def validate(root: Path) -> None:
    config_path = root / "paper/configs/xpyd_load_frequency_energy_2p2d_2workloads.json"
    config = json.loads(config_path.read_text())
    protocol = config["load_frequency_energy"]

    assert config["cold_start_unknown_system"] is True
    assert config["historical_seed_enabled"] is False
    assert config["search_prior"] == "none"
    assert config["background_experiment_traffic"] is False
    assert config["energy_validation_protocol"]["driver"] == "xpyd.load_frequency_energy"
    assert config["energy_validation_protocol"]["mode"] == (
        "open_loop_request_active_axis_factorized_energy_search")
    assert config["energy_validation_protocol"]["sampling_interval_seconds"] == 0.05
    assert config["energy_validation_protocol"]["slo_policy"] == (
        "report_ttft_only_no_selection_gate")
    assert config["online_feedback"]["enabled"] is False
    assert config["online_feedback"]["axis_search_levels"] == {
        "prefill": 17, "decode": 15}
    assert protocol["experiment_pair"] == ["P0", "D0"]
    assert protocol["load_rps"] == EXPECTED_LOADS
    assert len(protocol["load_rps"]) == 16
    assert protocol["minimum_requests_per_candidate"] >= 12
    assert protocol["maximum_requests_per_candidate"] >= 60
    assert protocol["target_arrival_window_s"] >= 20.0
    assert protocol["maximum_inflight_requests"] >= 64
    assert protocol["confirmation_windows"] >= 2
    assert protocol["confirmation_max_attempts"] >= protocol["confirmation_windows"]
    assert protocol["drain_timeout_s"] >= 900.0
    assert protocol["request_timeout_s"] >= 900.0
    assert protocol["experiment_timeout_s"] >= 23 * 3600
    assert protocol["reference_ttft_ms"] == 500.0
    assert protocol["selection_objective"] == (
        "minimum_joint_P0_D0_request_active_union_joules_per_completed_request")
    assert protocol["primary_energy_boundary"] == (
        "union_of_client_send_to_client_complete_intervals")
    assert protocol["idle_gap_policy"] == (
        "exclude_time_with_zero_requests_in_flight")
    assert protocol["selection_constraints"] == (
        "measurement validity only; TTFT is descriptive")
    assert [item["id"] for item in config["workloads"]] == [
        "small_light", "prefill_medium"]
    assert [(item["input_len"], item["output_len"])
            for item in config["workloads"]] == [(128, 64), (1024, 64)]
    assert config["client"]["max_concurrency"] == protocol["maximum_inflight_requests"]
    assert config["coverage_policy"] == {
        "required_endpoint_ids": ["P0", "D0"],
        "required_pairs": [["P0", "D0"]],
    }
    assert config["fixed_clocks"]["P0"]["graphics_mhz"] == 2520
    assert config["fixed_clocks"]["D0"]["graphics_mhz"] == 1500

    module = root / "paper/scripts/xpyd/load_frequency_energy.py"
    module_text = module.read_text()
    assert "binary_energy_minimize" in module_text
    assert '"slo_used_as_selection_gate": False' in module_text
    assert '"full_p_by_d_oracle_proven": False' in module_text
    assert "consume_timed" in module_text
    assert "energy_intervals" in module_text
    assert '"attribution_conservation_error_j"' in module_text

    wrapper = root.parent / "run.sbatch"
    wrapper_text = wrapper.read_text()
    assert "xpyd.load_frequency_energy" in wrapper_text
    assert "xpyd_load_frequency_energy_2p2d_2workloads.json" in wrapper_text
    assert "#SBATCH --time=24:00:00" in wrapper_text
    assert "#SBATCH --exclusive" in wrapper_text
    assert "#SBATCH --mem=0" in wrapper_text
    assert "XPYD_ENERGY_SAMPLE_INTERVAL_S" in wrapper_text

    manifest = json.loads((root.parent / "source_manifest.json").read_text())
    files = {
        str(path.relative_to(root)): path for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    assert set(files) == set(manifest), "source file inventory mismatch"
    for name, path in files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[name], name
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
    subprocess.run(["bash", "-n", str(root / "run_disagg_benchmark.sh")], check=True)
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    print(json.dumps({
        "valid": True,
        "source_files": len(files),
        "historical_prior": False,
        "arrival_process": "open_loop",
        "primary_energy_boundary": "request_active_union",
        "sampling_interval_seconds": 0.05,
        "workloads": ["small_light", "prefill_medium"],
        "load_count": len(EXPECTED_LOADS),
        "load_range_rps": [EXPECTED_LOADS[0], EXPECTED_LOADS[-1]],
        "experiment_pair": "P0-D0",
        "p_grid_levels": 17,
        "d_grid_levels": 15,
        "slo_selection_gate": False,
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    validate(parser.parse_args().source_root)
