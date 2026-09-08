"""Offline contract, inventory, and syntax checks; no GPU access."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def validate(root: Path) -> None:
    config_path = root / "paper/configs/xpyd_max_stable_throughput_2p2d_7workloads.json"
    config = json.loads(config_path.read_text())
    protocol = config["throughput_capacity"]
    assert config["cold_start_unknown_system"] is True
    assert config["historical_seed_enabled"] is False
    assert config["search_prior"] == "none"
    assert config["background_experiment_traffic"] is False
    assert config["energy_validation_protocol"]["driver"] == "xpyd.throughput_capacity"
    assert config["energy_validation_protocol"]["mode"] == "open_loop_per_workload_capacity_search"
    assert config["online_feedback"]["enabled"] is False
    assert protocol["service_pair"] == ["P1", "D1"]
    assert protocol["slo"] == {"ttft_ms": 500.0, "tpot_ms": 200.0}
    assert protocol["initial_rate_rps"] == 0.10
    assert 0 < protocol["minimum_rate_rps"] < protocol["initial_rate_rps"] < protocol["maximum_rate_rps"]
    assert protocol["bracket_growth_factor"] > 1
    assert protocol["minimum_requests_per_candidate"] >= 30
    assert protocol["maximum_requests_per_candidate"] >= protocol["minimum_requests_per_candidate"]
    assert protocol["confirmation_windows"] >= 2
    assert protocol["maximum_inflight_requests"] > 1
    assert protocol["minimum_success_ratio"] >= 0.99
    assert protocol["minimum_achieved_to_offered_ratio"] >= 0.95
    assert len(config["workloads"]) == 7
    assert len({w["id"] for w in config["workloads"]}) == 7
    assert config["client"]["max_concurrency"] == protocol["maximum_inflight_requests"]
    assert config["fixed_clocks"]["P1"]["graphics_mhz"] == 2520
    assert config["fixed_clocks"]["D1"]["graphics_mhz"] == 1500

    wrapper = root.parent / "run.sbatch"
    wrapper_text = wrapper.read_text()
    assert "xpyd.throughput_capacity" in wrapper_text
    assert "#SBATCH --time=12:00:00" in wrapper_text
    assert "#SBATCH --exclusive" in wrapper_text
    assert "#SBATCH --mem=0" in wrapper_text

    manifest = json.loads((root.parent / "source_manifest.json").read_text())
    files = {str(path.relative_to(root)): path for path in root.rglob("*")
             if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
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
        "workloads": 7,
        "capacity_pair": "P1-D1",
        "confirmation_windows": protocol["confirmation_windows"],
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    validate(parser.parse_args().source_root)
