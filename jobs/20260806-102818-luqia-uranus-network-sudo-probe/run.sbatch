#!/usr/bin/env bash
#SBATCH --job-name=uranus-net-sudo-probe
#SBATCH --partition=long
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --nodelist=ganymede,uranus
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260806-102818-luqia-uranus-network-sudo-probe/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260806-102818-luqia-uranus-network-sudo-probe/slurm-%j.err

set -uo pipefail

JOB_DIR="/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260806-102818-luqia-uranus-network-sudo-probe"
PROBE_SCRIPT="${JOB_DIR}/probe_node.sh"
GANYMEDE_100GBE_IP="10.1.0.3"

# Prevent allocation-level TRES defaults from conflicting with exact nested
# steps that reserve the single GPU assigned on each node.
unset SLURM_CPUS_PER_TASK
unset SLURM_TRES_PER_TASK
unset SLURM_GPUS
unset SLURM_GPUS_PER_NODE
unset SLURM_GPUS_PER_SOCKET
unset SLURM_GPUS_PER_TASK
unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU
unset SLURM_MEM_PER_NODE

echo "probe_start=$(date --iso-8601=seconds)"
echo "slurm_job_id=${SLURM_JOB_ID:-unset} nodes=${SLURM_JOB_NODELIST:-unset}"
scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>&1 || true

URANUS_ROUTE_FILE="${JOB_DIR}/uranus_route_to_ganymede.txt"
srun --kill-on-bad-exit=1 --exact --nodes=1 --nodelist=uranus \
  --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 \
  --cpus-per-task=2 --mem=2G \
  ip -4 route get "$GANYMEDE_100GBE_IP" \
  > "$URANUS_ROUTE_FILE" 2>&1
ROUTE_DISCOVERY_RC=$?
cat "$URANUS_ROUTE_FILE"
echo "uranus_route_discovery_rc=${ROUTE_DISCOVERY_RC}"
if [ "$ROUTE_DISCOVERY_RC" -ne 0 ]; then
  echo "probe_result=FAIL reason=uranus_route_discovery_failed"
  exit 10
fi

URANUS_100GBE_IP="$(awk '{for (i=1; i<=NF; i++) if ($i == "src" && i < NF) {print $(i+1); exit}}' "$URANUS_ROUTE_FILE")"
URANUS_100GBE_IFACE="$(awk '{for (i=1; i<=NF; i++) if ($i == "dev" && i < NF) {print $(i+1); exit}}' "$URANUS_ROUTE_FILE")"
echo "discovered_uranus_100gbe_ip=${URANUS_100GBE_IP:-unknown}"
echo "discovered_uranus_100gbe_interface=${URANUS_100GBE_IFACE:-unknown}"
if [ -z "$URANUS_100GBE_IP" ] || [ -z "$URANUS_100GBE_IFACE" ]; then
  echo "probe_result=FAIL reason=uranus_route_fields_missing"
  exit 11
fi

URANUS_RC=0
srun --kill-on-bad-exit=1 --exact --nodes=1 --nodelist=uranus \
  --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 \
  --cpus-per-task=2 --mem=2G \
  "$PROBE_SCRIPT" uranus ganymede "$GANYMEDE_100GBE_IP" \
  > "${JOB_DIR}/uranus_probe.log" 2>&1 || URANUS_RC=$?

GANYMEDE_RC=0
srun --kill-on-bad-exit=1 --exact --nodes=1 --nodelist=ganymede \
  --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 \
  --cpus-per-task=2 --mem=2G \
  "$PROBE_SCRIPT" ganymede uranus "$URANUS_100GBE_IP" \
  > "${JOB_DIR}/ganymede_probe.log" 2>&1 || GANYMEDE_RC=$?

cat "${JOB_DIR}/uranus_probe.log"
cat "${JOB_DIR}/ganymede_probe.log"

{
  echo "uranus_100gbe_ip=${URANUS_100GBE_IP}"
  echo "uranus_100gbe_interface=${URANUS_100GBE_IFACE}"
  echo "ganymede_100gbe_ip=${GANYMEDE_100GBE_IP}"
  echo "uranus_probe_rc=${URANUS_RC}"
  echo "ganymede_probe_rc=${GANYMEDE_RC}"
  if [ "$URANUS_RC" -eq 0 ] && [ "$GANYMEDE_RC" -eq 0 ]; then
    echo "probe_result=PASS"
  else
    echo "probe_result=FAIL"
  fi
} | tee "${JOB_DIR}/probe_summary.txt"

if [ "$URANUS_RC" -ne 0 ]; then
  exit "$URANUS_RC"
fi
exit "$GANYMEDE_RC"
