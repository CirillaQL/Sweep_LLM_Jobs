#!/usr/bin/env bash

set -uo pipefail

EXPECTED_HOST="$1"
PEER_LABEL="$2"
PEER_IP="$3"
HOST="$(hostname -s)"
RC=0

section() {
  echo
  echo "===== $1 ====="
}

run_check() {
  local label="$1"
  shift
  echo "check_start=${label}"
  "$@"
  local check_rc=$?
  echo "check_rc=${check_rc} check=${label}"
  if [ "$check_rc" -ne 0 ]; then
    RC=1
  fi
}

section identity
echo "timestamp=$(date --iso-8601=seconds)"
echo "hostname=${HOST} expected_host=${EXPECTED_HOST}"
echo "peer_label=${PEER_LABEL} peer_ip=${PEER_IP}"
echo "user=$(id -un) uid=$(id -u) groups=$(id -Gn)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [ "$HOST" != "$EXPECTED_HOST" ]; then
  echo "host_mismatch=true"
  RC=1
fi

section name_resolution
getent hosts uranus 2>&1 || true
getent hosts ganymede 2>&1 || true

section network_inventory
ip -brief link 2>&1 || RC=1
ip -4 -brief address 2>&1 || RC=1
ip -4 route show table main 2>&1 || RC=1

section peer_route
ROUTE_OUTPUT="$(ip -4 route get "$PEER_IP" 2>&1)"
ROUTE_RC=$?
echo "$ROUTE_OUTPUT"
echo "route_query_rc=${ROUTE_RC}"
if [ "$ROUTE_RC" -ne 0 ]; then
  RC=1
fi
ROUTE_IFACE="$(printf '%s\n' "$ROUTE_OUTPUT" | awk '{for (i=1; i<=NF; i++) if ($i == "dev" && i < NF) {print $(i+1); exit}}')"
ROUTE_SOURCE_IP="$(printf '%s\n' "$ROUTE_OUTPUT" | awk '{for (i=1; i<=NF; i++) if ($i == "src" && i < NF) {print $(i+1); exit}}')"
echo "route_interface=${ROUTE_IFACE:-unknown}"
echo "route_source_ip=${ROUTE_SOURCE_IP:-unknown}"
if [ -z "$ROUTE_IFACE" ] || [ -z "$ROUTE_SOURCE_IP" ]; then
  RC=1
else
  ip -brief address show dev "$ROUTE_IFACE" 2>&1 || RC=1
  echo "route_link_speed_mbps=$(cat "/sys/class/net/${ROUTE_IFACE}/speed" 2>/dev/null || echo unknown)"
  echo "route_link_mtu=$(cat "/sys/class/net/${ROUTE_IFACE}/mtu" 2>/dev/null || echo unknown)"
  ethtool "$ROUTE_IFACE" 2>&1 || true
fi

section peer_connectivity
run_check ping_peer ping -c 5 -W 2 "$PEER_IP"
ip neigh show to "$PEER_IP" 2>&1 || true

section nvidia_smi_unprivileged
NVIDIA_SMI_ARGS=(
  --query-gpu=index,name,uuid,pci.bus_id,memory.total,power.draw,clocks.current.graphics
  --format=csv,noheader
)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NVIDIA_SMI_ARGS=(-i "$CUDA_VISIBLE_DEVICES" "${NVIDIA_SMI_ARGS[@]}")
fi
run_check nvidia_smi nvidia-smi "${NVIDIA_SMI_ARGS[@]}"

section nvidia_smi_sudo_read_only
echo "sudo_mode=non_interactive read_only=true"
run_check sudo_nvidia_smi sudo -n nvidia-smi "${NVIDIA_SMI_ARGS[@]}"

section result
echo "node_probe_rc=${RC} host=${HOST} peer=${PEER_LABEL}"
exit "$RC"
