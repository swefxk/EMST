import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np


def run_cmd(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


def parse_test_geo_metrics(output):
    line = None
    for raw in output.splitlines():
        if raw.startswith("test rank[det] geo_acc"):
            line = raw
            break
    if line is None:
        raise ValueError("Missing test rank[det] geo_acc line in output.")
    mrr_match = re.search(r"geo_mrr ([0-9.]+)", line)
    hits_match = re.search(r"10: ([0-9.]+)", line)
    if not mrr_match or not hits_match:
        raise ValueError(f"Cannot parse metrics from line: {line}")
    return float(mrr_match.group(1)), float(hits_match.group(1))


def mean_std(values):
    arr = np.array(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    combos = [
        ("A0", {"prev_event": "off", "band_mask": False}),
        ("A1", {"prev_event": "zero_dt", "band_mask": False}),
        ("A2", {"prev_event": "on", "band_mask": False}),
        ("B1-A0", {"prev_event": "off", "band_mask": True}),
        ("B1-A1", {"prev_event": "zero_dt", "band_mask": True}),
    ]

    results = {}
    for name, cfg in combos:
        results[name] = {"mrr": [], "hits10": []}

    python = sys.executable
    ckpt_dir = os.path.join("checkpoints", "seed_sweep")
    os.makedirs(ckpt_dir, exist_ok=True)

    for seed in seeds:
        for name, cfg in combos:
            ckpt_path = os.path.join(
                ckpt_dir, f"{name}_seed{seed}.pt"
            )
            train_cmd = [
                python,
                "train.py",
                "--data",
                args.data,
                "--save",
                ckpt_path,
                "--config",
                args.config,
                "--seed",
                str(seed),
                "--prev_event",
                cfg["prev_event"],
            ]
            if cfg["band_mask"]:
                train_cmd.append("--band_obs_mask")
            print(f"[seed {seed}] train {name}")
            run_cmd(train_cmd)

            eval_cmd = [
                python,
                "eval.py",
                "--data",
                args.data,
                "--ckpt",
                ckpt_path,
                "--config",
                args.config,
                "--seed",
                str(seed),
                "--prev_event",
                cfg["prev_event"],
            ]
            if cfg["band_mask"]:
                eval_cmd.append("--band_obs_mask")
            print(f"[seed {seed}] eval {name}")
            output = run_cmd(eval_cmd)
            mrr, hits10 = parse_test_geo_metrics(output)
            results[name]["mrr"].append(mrr)
            results[name]["hits10"].append(hits10)

    summary = {}
    for name in results:
        mrr_mean, mrr_std = mean_std(results[name]["mrr"])
        hits_mean, hits_std = mean_std(results[name]["hits10"])
        summary[name] = {
            "mrr_mean": mrr_mean,
            "mrr_std": mrr_std,
            "hits10_mean": hits_mean,
            "hits10_std": hits_std,
        }

    print("Seed sweep summary (test Geo):")
    for name in summary:
        s = summary[name]
        print(
            f"{name}: MRR {s['mrr_mean']:.4f}±{s['mrr_std']:.4f}, "
            f"Hits@10 {s['hits10_mean']:.4f}±{s['hits10_std']:.4f}"
        )

    payload = {"seeds": seeds, "raw": results, "summary": summary}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
