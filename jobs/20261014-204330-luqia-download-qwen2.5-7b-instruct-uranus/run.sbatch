#!/usr/bin/env bash
#SBATCH --job-name=dl-qwen25-7b
# CPU only on uranus: download Qwen/Qwen2.5-7B-Instruct (safetensors, ~15.2 GB) into a fixed,
# job-independent HF cache that later jobs read offline:
#   /data/users/chjing/vllm_job_work/models/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
# Checks free space first (needs >= 25 GB), then verifies the snapshot: every safetensors shard
# listed in the index exists and parses, parameter count, config and tokenizer load offline.
# Writes the snapshot path to results/<id>/model_path.txt.
#SBATCH --partition=long
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=uranus
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261014-204330-luqia-download-qwen2.5-7b-instruct-uranus/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20261014-204330-luqia-download-qwen2.5-7b-instruct-uranus/slurm-%j.err

set -uo pipefail
JOB_NAME="20261014-204330-luqia-download-qwen2.5-7b-instruct-uranus"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/${JOB_NAME}/results/${JOB_ID}"
WORK="/data/users/chjing/vllm_job_work/${JOB_ID}"
ENV_DIR="/data/users/chjing/miniforge3/envs/cuda-env"
PY="${ENV_DIR}/bin/python"
REPO_ID="Qwen/Qwen2.5-7B-Instruct"
HF_ROOT="/data/users/chjing/vllm_job_work/models/huggingface"
NEED_GB=25
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/download.log") 2>&1
fail() { echo "RESULT=failed stage=$1"; exit 1; }
echo "start=$(date -Iseconds) host=$(hostname) job=${JOB_ID} repo=${REPO_ID}"

# ---- caches and HOME inside WORK (nothing lands in /root or ~) --------------------------
export HOME="${WORK}/home" TMPDIR="${WORK}/tmp" XDG_CACHE_HOME="${WORK}/xdg-cache"
export PYTHONPYCACHEPREFIX="${WORK}/python-pycache" PYTHONNOUSERSITE=1
export PATH="${ENV_DIR}/bin:${PATH}"
export HF_HOME="${HF_ROOT}" HF_HUB_CACHE="${HF_ROOT}/hub" HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p "${HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}" "${HF_HUB_CACHE}"

# ---- 1. environment and free space ------------------------------------------------------
"${PY}" -c 'import huggingface_hub, transformers, safetensors; print("huggingface_hub", huggingface_hub.__version__, "transformers", transformers.__version__, "safetensors", safetensors.__version__)' || fail python_env
df -h /data
free_gb=$(df -P -BG /data | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "free_gb=${free_gb}"
existing=$(du -s -BG "${HF_HUB_CACHE}/models--Qwen--Qwen2.5-7B-Instruct" 2>/dev/null | cut -f1 | tr -d G)
echo "already_cached_gb=${existing:-0}"
[ $(( free_gb + ${existing:-0} )) -ge ${NEED_GB} ] || fail "disk_space (free ${free_gb} GB < ${NEED_GB} GB)"
timeout 60 curl -fsSIL -o /dev/null "https://huggingface.co/${REPO_ID}/resolve/main/config.json" \
  || fail huggingface_unreachable

# ---- 2. download (resumable; retried) ---------------------------------------------------
for attempt in 1 2 3; do
  echo "download attempt ${attempt} $(date -Iseconds)"
  if "${PY}" - "${REPO_ID}" > "${OUT}/snapshot_path.tmp" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt",
                                                      "LICENSE", "README.md"], max_workers=4)
print(path)
PYEOF
  then ok=1; break; fi
  ok=0; sleep 30
done
[ "${ok}" = 1 ] || fail download
SNAP=$(tail -n 1 "${OUT}/snapshot_path.tmp"); rm -f "${OUT}/snapshot_path.tmp"
echo "snapshot=${SNAP}"
ls -lL "${SNAP}"
du -shL "${SNAP}"

# ---- 3. verify offline --------------------------------------------------------------------
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"${PY}" - "${SNAP}" <<'PYEOF' || fail verify
import json, sys
from pathlib import Path
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

snap = Path(sys.argv[1])
index = json.loads((snap / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
params = 0
for s in shards:
    with safe_open(snap / s, framework="pt") as f:
        for k in f.keys():
            n = 1
            for d in f.get_slice(k).get_shape():
                n *= d
            params += n
    print("shard ok", s, round((snap / s).stat().st_size / 2**30, 2), "GiB")
cfg = AutoConfig.from_pretrained(snap)
tok = AutoTokenizer.from_pretrained(snap)
kv_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2
print(f"architectures={cfg.architectures} layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
      f"kv_heads={cfg.num_key_value_heads} hidden={cfg.hidden_size} max_pos={cfg.max_position_embeddings} "
      f"dtype={cfg.torch_dtype}")
print(f"params={params / 1e9:.3f}B weights_bf16={params * 2 / 2**30:.2f} GiB kv_bytes_per_token={kv_per_token}")
print("tokenizer", type(tok).__name__, "vocab", len(tok), "sample_ids", tok("Hello, CanaTune!")["input_ids"])
assert len(shards) == len(list(snap.glob("*.safetensors"))), "shard count mismatch"
assert 7.0e9 < params < 8.0e9, params
PYEOF
echo "${SNAP}" > "${OUT}/model_path.txt"
df -h /data
echo "RESULT=ok model_path=${SNAP} end=$(date -Iseconds)"
