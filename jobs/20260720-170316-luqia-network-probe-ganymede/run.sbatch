#!/usr/bin/env bash
#SBATCH --job-name=net-probe-ganymede
#SBATCH --partition=long
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-170316-luqia-network-probe-ganymede/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260720-170316-luqia-network-probe-ganymede/slurm-%j.err

set -uo pipefail

section() {
  echo
  echo "===== $1 ====="
}

run_if_present() {
  local command_name="$1"
  shift
  if command -v "$command_name" >/dev/null 2>&1; then
    "$command_name" "$@" 2>&1 || true
  else
    echo "command_missing=${command_name}"
  fi
}

section "job_and_host_identity"
echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "hostname_fqdn=$(hostname -f 2>/dev/null || true)"
echo "hostname_short=$(hostname -s 2>/dev/null || true)"
echo "whoami=$(whoami)"
echo "uid_gid=$(id)"
echo "kernel=$(uname -a)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
echo "SLURM_NNODES=${SLURM_NNODES:-unset}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unset}"
echo "hostname_I=$(hostname -I 2>/dev/null || true)"
echo "hostname_i=$(hostname -i 2>/dev/null || true)"
ulimit -n 2>&1 || true
run_if_present scontrol show hostnames "${SLURM_JOB_NODELIST:-ganymede}"
run_if_present scontrol show node "$(hostname -s)"

section "name_resolution"
for host in "$(hostname)" "$(hostname -s 2>/dev/null || hostname)" neptune ganymede minerva; do
  echo "--- getent hosts ${host} ---"
  getent hosts "$host" 2>&1 || true
  echo "--- getent ahostsv4 ${host} ---"
  getent ahostsv4 "$host" 2>&1 || true
  echo "--- getent ahostsv6 ${host} ---"
  getent ahostsv6 "$host" 2>&1 || true
done

section "resolver_configuration"
sed -n '1,240p' /etc/hosts 2>&1 || true
sed -n '1,240p' /etc/resolv.conf 2>&1 || true
grep -E '^[[:space:]]*hosts:' /etc/nsswitch.conf 2>&1 || true

section "ip_addresses_and_links"
run_if_present ip -brief address show
run_if_present ip -details link show
run_if_present ip -statistics link show

section "per_interface_details"
for path in /sys/class/net/*; do
  [ -e "$path" ] || continue
  iface=$(basename "$path")
  echo "--- interface=${iface} ---"
  for field in address mtu operstate speed duplex carrier type; do
    if [ -r "$path/$field" ]; then
      printf '%s=' "$field"
      sed -n '1p' "$path/$field" 2>/dev/null || true
    fi
  done
  if [ -L "$path/device/driver" ]; then
    echo "driver=$(basename "$(readlink -f "$path/device/driver")")"
  fi
  if [ -r "$path/device/numa_node" ]; then
    echo "numa_node=$(sed -n '1p' "$path/device/numa_node")"
  fi
  ip -brief address show dev "$iface" 2>&1 || true
  if command -v ethtool >/dev/null 2>&1; then
    ethtool "$iface" 2>&1 || true
    ethtool -i "$iface" 2>&1 || true
  fi
done

section "routes_and_source_address_selection"
run_if_present ip route show table all
run_if_present ip -6 route show table all
run_if_present ip rule show
for peer in neptune ganymede; do
  peer_ip=$(getent ahostsv4 "$peer" 2>/dev/null | awk 'NR==1 {print $1}')
  echo "peer=${peer} resolved_ipv4=${peer_ip:-unresolved}"
  if [ -n "${peer_ip:-}" ]; then
    ip route get "$peer_ip" 2>&1 || true
  fi
done

section "neighbor_and_arp_state"
run_if_present ip neighbor show
sed -n '1,240p' /proc/net/arp 2>&1 || true

section "peer_ping"
for peer in neptune ganymede; do
  echo "--- ping ${peer} ---"
  ping -n -c 3 -W 2 "$peer" 2>&1 || true
done

section "listening_tcp_udp_sockets"
run_if_present ss -lnt
run_if_present ss -lnu

section "tcp_and_socket_configuration"
for key in \
  net.ipv4.ip_local_port_range \
  net.ipv4.tcp_congestion_control \
  net.ipv4.tcp_available_congestion_control \
  net.ipv4.tcp_mtu_probing \
  net.core.somaxconn \
  net.core.rmem_max \
  net.core.wmem_max; do
  sysctl "$key" 2>&1 || true
done
for tool in python3 nc ncat socat curl wget iperf3; do
  echo "tool_${tool}=$(command -v "$tool" 2>/dev/null || echo missing)"
done

section "network_pci_devices"
if command -v lspci >/dev/null 2>&1; then
  lspci -nn 2>&1 | grep -Ei 'ethernet|network|infiniband' || true
else
  echo "command_missing=lspci"
fi

section "rdma_infiniband_and_ucx"
run_if_present rdma link show
run_if_present rdma dev show
run_if_present ibv_devices
run_if_present ibv_devinfo
run_if_present ibstat
run_if_present ucx_info -v
run_if_present ompi_info --version
run_if_present mpirun --version

section "bond_vlan_and_bridge_state"
for file in /proc/net/bonding/* /proc/net/vlan/*; do
  [ -f "$file" ] || continue
  echo "--- ${file} ---"
  sed -n '1,260p' "$file" 2>&1 || true
done
run_if_present bridge link show

section "communication_summary"
echo "node=$(hostname -s 2>/dev/null || hostname)"
echo "all_addresses=$(hostname -I 2>/dev/null || true)"
for peer in neptune ganymede; do
  echo "${peer}_ipv4=$(getent ahostsv4 "$peer" 2>/dev/null | awk 'NR==1 {print $1}')"
done
echo "probe_complete=true"
