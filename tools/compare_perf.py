#!/usr/bin/env python3
"""
Compare microbenchmark results to a checked-in baseline CSV.
- Exits 0 on success or advisory regression (we print WARN, workflow stays green).
- Exits 1 only if the benchmark invocation failed (not for perf deltas while advisory).
CSV schema (baseline & output): name,metric,value
"""
import argparse, csv, math, subprocess, sys, tempfile, json
from pathlib import Path

def read_csv(path):
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["name"], row["metric"])
            data[key] = float(row["value"])
    return data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, help="Path to microbenchmark script")
    p.add_argument("--baseline", required=True, help="CSV baseline (name,metric,value)")
    p.add_argument("--threshold", type=float, default=0.05, help="Regression threshold (e.g., 0.05 = 5%)")
    args = p.parse_args()

    # Run the microbenchmark script; it should emit CSV to stdout or a temp file.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.close()
    try:
        # Convention: your benchmark accepts --csv-out
        cmd = [sys.executable, args.bench, "--csv-out", tmp.name]
        print(f"Running microbench: {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[ERROR] Benchmark failed with code {rc}")
            return 1

        base = read_csv(args.baseline)
        now = read_csv(tmp.name)

        warnings = []
        for key, base_val in base.items():
            if key not in now:
                print(f"[INFO] Missing metric in new run: {key} (skip)")
                continue
            cur_val = now[key]
            # Assume metric is latency (lower is better). Adjust if needed.
            if base_val <= 0:
                continue
            delta = (cur_val - base_val) / base_val
            if delta > args.threshold:
                warnings.append((key, base_val, cur_val, delta))

        if warnings:
            print("=== ⚠️  Performance regression(s) detected (advisory) ===")
            for (name, metric), b, c, d in warnings:
                print(f"{name}:{metric}  baseline={b:.4f}, current={c:.4f}, delta=+{d*100:.2f}%")
            print("NOTE: This is advisory; workflow remains green by design.")
        else:
            print("=== ✅ Performance within threshold ===")
        return 0
    finally:
        Path(tmp.name).unlink(missing_ok=True)

if __name__ == "__main__":
    sys.exit(main())
