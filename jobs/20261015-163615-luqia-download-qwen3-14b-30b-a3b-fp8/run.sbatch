#!/usr/bin/env bash
#SBATCH --job-name=dl-qwen3
# CPU only on uranus: download Qwen/Qwen3-14B (bf16, ~29.5 GB) and Qwen/Qwen3-30B-A3B-FP8 (MoE,
# fp8 weights, ~32.5 GB) into the fixed, job-independent HF cache of job 275901, which later
# jobs read offline:
#   /data/users/chjing/vllm_job_work/models/huggingface/hub/models--Qwen--<name>
# Checks free space first (needs >= 90 GB for both), then verifies each snapshot offline: every
# safetensors shard listed in the index exists and its header parses (shapes and dtypes read
# from the header, no torch), parameter count, config and tokenizer load, and the KV bytes per
# token. Writes the snapshot paths to results/<id>/model_paths.json.
#SBATCH --partition=long
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=uranus
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261015-163615-luqia-download-qwen3-14b-30b-a3b-fp8/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261015-163615-luqia-download-qwen3-14b-30b-a3b-fp8/slurm-%j.err

set -uo pipefail
JOB_NAME="20261015-163615-luqia-download-qwen3-14b-30b-a3b-fp8"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
HF_ROOT="/data/users/chjing/vllm_job_work/models/huggingface"
NEED_GB=90
# repo id, accepted parameter range in billions (header count, fp8 scales included)
MODELS=("Qwen/Qwen3-14B 14.0 15.5" "Qwen/Qwen3-30B-A3B-FP8 29.5 31.5")
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/download.log") 2>&1
fail() { echo "RESULT=failed stage=$1"; exit 1; }
echo "start=$(date -Iseconds) host=$(hostname) job=${JOB_ID}"

# ---- caches and HOME inside WORK (nothing lands in /root or ~) --------------------------
export HOME="${WORK}/home" TMPDIR="${WORK}/tmp" XDG_CACHE_HOME="${WORK}/xdg-cache"
export PYTHONPYCACHEPREFIX="${WORK}/python-pycache" PYTHONNOUSERSITE=1
export PATH="${ENV_DIR}/bin:${PATH}"
export HF_HOME="${HF_ROOT}" HF_HUB_CACHE="${HF_ROOT}/hub" HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p "${HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}" "${HF_HUB_CACHE}"

# ---- 1. environment and free space ------------------------------------------------------
"${PY}" -c 'import huggingface_hub, transformers; print("huggingface_hub", huggingface_hub.__version__, "transformers", transformers.__version__)' || fail python_env
df -h /data
free_gb=$(df -P -BG /data | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "free_gb=${free_gb}"
[ "${free_gb}" -ge "${NEED_GB}" ] || fail "disk_space (free ${free_gb} GB < ${NEED_GB} GB)"

echo "{" > "${OUT}/model_paths.json.tmp"
first=1
for entry in "${MODELS[@]}"; do
  read -r REPO_ID PMIN PMAX <<< "${entry}"
  echo "=== ${REPO_ID} $(date -Iseconds)"
  timeout 60 curl -fsSIL -o /dev/null "https://huggingface.co/${REPO_ID}/resolve/main/config.json" \
    || fail "huggingface_unreachable ${REPO_ID}"

  # ---- 2. download (resumable; retried) -------------------------------------------------
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  ok=0
  for attempt in 1 2 3; do
    echo "download attempt ${attempt} $(date -Iseconds)"
    if "${PY}" - "${REPO_ID}" > "${OUT}/snapshot_path.tmp" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt",
                                                      "LICENSE", "README.md"], max_workers=4)
print(path)
PYEOF
    then ok=1; break; fi
    sleep 30
  done
  [ "${ok}" = 1 ] || fail "download ${REPO_ID}"
  SNAP=$(tail -n 1 "${OUT}/snapshot_path.tmp"); rm -f "${OUT}/snapshot_path.tmp"
  echo "snapshot=${SNAP}"
  ls -lL "${SNAP}"
  du -shL "${SNAP}"

  # ---- 3. verify offline ------------------------------------------------------------------
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  "${PY}" - "${SNAP}" "${PMIN}" "${PMAX}" <<'PYEOF' || fail "verify ${REPO_ID}"
import json, struct, sys
from collections import Counter
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

snap, pmin, pmax = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
index = json.loads((snap / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
params, dtypes, names = 0, Counter(), set()
for s in shards:
    with open(snap / s, "rb") as f:  # safetensors: 8-byte header length, JSON header
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    for name, info in header.items():
        if name == "__metadata__":
            continue
        n = 1
        for d in info["shape"]:
            n *= d
        params += n
        dtypes[info["dtype"]] += n
        names.add(name)
    print("shard ok", s, round((snap / s).stat().st_size / 2**30, 2), "GiB")
missing = set(index["weight_map"]) - names
assert not missing, f"{len(missing)} tensors of the index missing, e.g. {sorted(missing)[:3]}"
assert len(shards) == len(list(snap.glob("*.safetensors"))), "shard count mismatch"
cfg = AutoConfig.from_pretrained(snap)
tok = AutoTokenizer.from_pretrained(snap)
head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
kv = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2
experts = getattr(cfg, "num_experts", None)
print(f"architectures={cfg.architectures} layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
      f"kv_heads={cfg.num_key_value_heads} head_dim={head_dim} hidden={cfg.hidden_size} "
      f"max_pos={cfg.max_position_embeddings} experts={experts} "
      f"experts_per_tok={getattr(cfg, 'num_experts_per_tok', None)} "
      f"quant={getattr(cfg, 'quantization_config', None)}")
print(f"params={params / 1e9:.3f}B dtypes={ {k: round(v / 1e9, 3) for k, v in dtypes.items()} } "
      f"weights_on_disk={sum((snap / s).stat().st_size for s in shards) / 2**30:.2f} GiB "
      f"kv_bytes_per_token_bf16={kv}")
print("tokenizer", type(tok).__name__, "vocab", len(tok), "sample_ids", tok("Hello, CanaTune!")["input_ids"])
assert pmin * 1e9 < params < pmax * 1e9, params
PYEOF
  [ "${first}" = 1 ] || echo "," >> "${OUT}/model_paths.json.tmp"
  printf '  "%s": "%s"' "${REPO_ID}" "${SNAP}" >> "${OUT}/model_paths.json.tmp"
  first=0
done
printf '\n}\n' >> "${OUT}/model_paths.json.tmp"
mv "${OUT}/model_paths.json.tmp" "${OUT}/model_paths.json"
cat "${OUT}/model_paths.json"
df -h /data
echo "RESULT=ok end=$(date -Iseconds)"
