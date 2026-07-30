#!/usr/bin/env bash
#SBATCH --job-name=external-https-probe
#SBATCH --partition=long
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --nodelist=ganymede
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --output=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260730-173859-luqia-external-https-probe/slurm-%j.out
#SBATCH --error=/data/users/chjing/Sweep_LLM_Jobs_broker/jobs/20260730-173859-luqia-external-https-probe/slurm-%j.err

set -uo pipefail

TARGET_URL="https://eduardoqian.com/"
TARGET_HOST="eduardoqian.com"

echo "probe_started_at=$(date --iso-8601=seconds)"
echo "compute_node=$(hostname)"
echo "target_url=${TARGET_URL}"

echo "dns_lookup_begin"
if command -v getent >/dev/null 2>&1; then
  getent ahosts "$TARGET_HOST"
  DNS_RC=$?
else
  echo "getent_not_available=true"
  DNS_RC=127
fi
echo "dns_rc=${DNS_RC}"
echo "dns_lookup_end"

echo "https_get_begin"
curl \
  --fail \
  --silent \
  --show-error \
  --location \
  --connect-timeout 10 \
  --max-time 30 \
  --output /dev/null \
  --write-out 'http_code=%{http_code}\nremote_ip=%{remote_ip}\nremote_port=%{remote_port}\nnum_redirects=%{num_redirects}\ntime_namelookup_s=%{time_namelookup}\ntime_connect_s=%{time_connect}\ntime_appconnect_s=%{time_appconnect}\ntime_starttransfer_s=%{time_starttransfer}\ntime_total_s=%{time_total}\nurl_effective=%{url_effective}\n' \
  "$TARGET_URL"
CURL_RC=$?
echo "curl_rc=${CURL_RC}"
echo "https_get_end"
echo "probe_finished_at=$(date --iso-8601=seconds)"

if [ "$DNS_RC" -ne 0 ]; then
  exit "$DNS_RC"
fi
exit "$CURL_RC"
