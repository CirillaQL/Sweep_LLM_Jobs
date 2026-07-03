#!/usr/bin/env bash
#SBATCH --job-name=inspect-f1200-csv
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -u

ROOT="/data/users/chjing/vLLM_test"

section() {
  echo
  echo "===== $1 ====="
}

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "whoami=$(whoami)"
echo "id=$(id)"
echo "root=${ROOT}"

section "find_monitor_f1200_csv"
find "$ROOT" -type f -name 'monitor_f1200*.csv' -printf '%s %TY-%Tm-%Td %TH:%TM %p\n' | sort || true

section "find_related_f1200_files"
find "$ROOT" -type f \( -name '*f1200*.csv' -o -name '*f1200*.txt' \) \
  -printf '%s %TY-%Tm-%Td %TH:%TM %p\n' | sort | head -n 300 || true

section "csv_frequency_summary"
python3 - <<'PY'
import csv
import glob
import os
import statistics

paths = sorted(glob.glob("/data/users/chjing/vLLM_test/**/monitor_f1200*.csv", recursive=True))
print(f"num_monitor_f1200_csv={len(paths)}")
if not paths:
    raise SystemExit(0)

def read_float(row, names):
    for name in names:
        if name in row and row[name] not in ("", "None", "nan", "N/A"):
            try:
                return float(row[name])
            except ValueError:
                pass
    return None

rows_summary = []
for path in paths:
    freqs = []
    utils = []
    powers = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                freq = read_float(row, ["gpu_freq_mhz", "sm_clock_mhz", "clock_mhz", "gpu_freq"])
                util = read_float(row, ["gpu_util_pct", "utilization.gpu [%]", "gpu_util"])
                power = read_float(row, ["power_w", "power.draw [W]", "power"])
                if freq is not None:
                    freqs.append(freq)
                if util is not None:
                    utils.append(util)
                if power is not None:
                    powers.append(power)
    except Exception as exc:
        print(f"ERROR reading {path}: {exc}")
        continue

    if freqs:
        active_freqs = [f for f, u in zip(freqs, utils) if u >= 50] if len(utils) == len(freqs) else freqs
        sample = freqs[:5]
        rows_summary.append({
            "path": path,
            "n": len(freqs),
            "fields": fieldnames,
            "min": min(freqs),
            "max": max(freqs),
            "mean": statistics.fmean(freqs),
            "active_n": len(active_freqs),
            "active_min": min(active_freqs) if active_freqs else None,
            "active_max": max(active_freqs) if active_freqs else None,
            "active_mean": statistics.fmean(active_freqs) if active_freqs else None,
            "util_mean": statistics.fmean(utils) if utils else None,
            "power_mean": statistics.fmean(powers) if powers else None,
            "sample": sample,
        })
    else:
        rows_summary.append({
            "path": path,
            "n": 0,
            "fields": fieldnames,
            "min": None,
            "max": None,
            "mean": None,
            "active_n": 0,
            "active_min": None,
            "active_max": None,
            "active_mean": None,
            "util_mean": statistics.fmean(utils) if utils else None,
            "power_mean": statistics.fmean(powers) if powers else None,
            "sample": [],
        })

for item in rows_summary:
    print()
    print(f"path={item['path']}")
    print(f"fields={','.join(item['fields'])}")
    print(
        "freq_summary="
        f"n={item['n']} min={item['min']} max={item['max']} mean={item['mean']} "
        f"active_n={item['active_n']} active_min={item['active_min']} "
        f"active_max={item['active_max']} active_mean={item['active_mean']} "
        f"util_mean={item['util_mean']} power_mean={item['power_mean']} "
        f"sample={item['sample']}"
    )

ok = [
    item for item in rows_summary
    if item["active_n"] and item["active_max"] is not None and item["active_max"] <= 1230
]
bad = [
    item for item in rows_summary
    if item["active_n"] and item["active_max"] is not None and item["active_max"] > 1230
]
unknown = [item for item in rows_summary if not item["active_n"]]

print()
print("===== aggregate_judgement =====")
print(f"files_total={len(rows_summary)} files_active_capped_like_1200={len(ok)} files_active_above_1230={len(bad)} files_unknown={len(unknown)}")
if bad:
    print("judgement=At least some f1200 monitor files recorded active GPU frequency above 1230 MHz, so those runs were not actually capped near 1200 MHz.")
elif ok:
    print("judgement=All active f1200 monitor files found here look capped near 1200 MHz.")
else:
    print("judgement=Could not judge because no active frequency samples were found.")
PY

section "sample_file_heads"
count=0
while IFS= read -r file; do
  echo
  echo "----- ${file} -----"
  sed -n '1,12p' "$file" || true
  count=$((count + 1))
  [ "$count" -ge 12 ] && break
done < <(find "$ROOT" -type f -name 'monitor_f1200*.csv' | sort)
