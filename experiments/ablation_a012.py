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


def parse_test_metrics(output):
    geo_line = None
    band_line = None
    for raw in output.splitlines():
        if raw.startswith("test rank[det] geo_acc"):
            geo_line = raw
        elif raw.startswith("test rank[det] band_acc"):
            band_line = raw
    if geo_line is None or band_line is None:
        raise ValueError("Missing test rank[det] lines in output.")

    def parse_line(line, prefix):
        mrr_match = re.search(r"mrr ([0-9.]+)", line)
        hits_match = re.search(r"10: ([0-9.]+)", line)
        if not mrr_match or not hits_match:
            raise ValueError(f"Cannot parse {prefix} metrics: {line}")
        return float(mrr_match.group(1)), float(hits_match.group(1))

    geo_mrr, geo_hits10 = parse_line(geo_line, "geo")
    band_mrr, band_hits10 = parse_line(band_line, "band")
    return geo_mrr, geo_hits10, band_mrr, band_hits10


def mean_std(values):
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", default="checkpoints/ablation_a012.json")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    settings = [
        ("A0", "off"),
        ("A1", "zero_dt"),
        ("A2", "on"),
    ]

    python = sys.executable
    ckpt_dir = os.path.join("checkpoints", "ablation_a012")
    os.makedirs(ckpt_dir, exist_ok=True)

    raw = {}
    for name, _ in settings:
        raw[name] = {
            "geo_mrr": [],
            "geo_hits10": [],
            "band_mrr": [],
            "band_hits10": [],
        }

    for seed in seeds:
        for name, prev_event in settings:
            ckpt_path = os.path.join(ckpt_dir, f"{name}_seed{seed}.pt")
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
                prev_event,
            ]
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
                prev_event,
            ]
            print(f"[seed {seed}] eval {name}")
            output = run_cmd(eval_cmd)
            geo_mrr, geo_hits10, band_mrr, band_hits10 = parse_test_metrics(output)
            raw[name]["geo_mrr"].append(geo_mrr)
            raw[name]["geo_hits10"].append(geo_hits10)
            raw[name]["band_mrr"].append(band_mrr)
            raw[name]["band_hits10"].append(band_hits10)

    summary = {}
    for name in raw:
        summary[name] = {}
        for key in ["geo_mrr", "geo_hits10", "band_mrr", "band_hits10"]:
            mean, std = mean_std(raw[name][key])
            summary[name][key] = {"mean": mean, "std": std}

    print("Ablation A0/A1/A2 (test, mean±std):")
    print("Setting\tGeo MRR\tGeo Hits@10\tBand MRR\tBand Hits@10")
    for name, _ in settings:
        s = summary[name]
        print(
            f"{name}\t"
            f"{s['geo_mrr']['mean']:.4f}±{s['geo_mrr']['std']:.4f}\t"
            f"{s['geo_hits10']['mean']:.4f}±{s['geo_hits10']['std']:.4f}\t"
            f"{s['band_mrr']['mean']:.4f}±{s['band_mrr']['std']:.4f}\t"
            f"{s['band_hits10']['mean']:.4f}±{s['band_hits10']['std']:.4f}"
        )

    payload = {"seeds": seeds, "raw": raw, "summary": summary}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
