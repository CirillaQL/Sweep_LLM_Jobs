"""Offline contract and syntax checks; no GPU access or submission."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def validate(root):
    config = json.loads((root/"paper/configs/xpyd_feedback_table_2p2d_7workloads.json").read_text())
    assert config["cold_start_unknown_system"] is True
    assert config["historical_seed_enabled"] is False and config["search_prior"] == "none"
    assert config["online_feedback"]["axis_search_levels"] == {"prefill": 17, "decode": 15}
    assert config["online_feedback"]["probe_samples_per_candidate"] == 3
    assert len(config["workloads"]) == 7
    protocol = config["energy_validation_protocol"]
    assert config["routing_policy"] == "sequential_workload_windows"
    assert config["workload_ordering"] == "windowed_by_type"
    assert protocol["mode"] == "sequential_workload_windows"
    assert protocol["minimum_table_requests_per_window"] == 30
    assert not any(key in protocol for key in ("paired_blocks_per_class", "before_after_power_window_seconds_per_class_per_arm"))
    manifest = json.loads((root.parent/"source_manifest.json").read_text())
    files = {str(p.relative_to(root)): p for p in root.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    assert set(files) == set(manifest), "source file inventory mismatch"
    for name, path in files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[name], name
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
    subprocess.run(["bash", "-n", str(root/"run_disagg_benchmark.sh")], check=True)
    subprocess.run(["bash", "-n", str(root.parent/"run.sbatch")], check=True)
    print(json.dumps({"valid": True, "source_files": len(files), "cold_start": True,
                      "production_order": "seven_contiguous_workload_windows",
                      "canary_grid": "17+15", "samples_per_candidate": 3}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    validate(parser.parse_args().source_root)
