import argparse
import json
import os
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


def load_geo_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    geo = payload["splits"]["test"]["rank"]["geo"]
    return float(geo["mrr"]), float(geo["hits10"])


def mean_std(values):
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=1))


def run_setting(label, data_path, config_path, seeds, out_root):
    python = sys.executable
    ckpt_dir = os.path.join("checkpoints", "strong_time_seed3")
    runs_dir = os.path.join("runs", "strong_time_seed3", label)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    results = {"A1": {"mrr": [], "hits10": []}, "A2": {"mrr": [], "hits10": []}}
    for seed in seeds:
        for name, prev_event in [("A1", "zero_dt"), ("A2", "on")]:
            ckpt = os.path.join(ckpt_dir, f"{label}_{name}_seed{seed}.pt")
            metrics_dir = os.path.join(runs_dir, name, f"seed_{seed}")
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_path = os.path.join(metrics_dir, "metrics.json")

            if not os.path.isfile(ckpt):
                run_cmd(
                    [
                        python,
                        "train.py",
                        "--data",
                        data_path,
                        "--save",
                        ckpt,
                        "--config",
                        config_path,
                        "--seed",
                        str(seed),
                        "--prev_event",
                        prev_event,
                    ]
                )

            run_cmd(
                [
                    python,
                    "eval.py",
                    "--data",
                    data_path,
                    "--ckpt",
                    ckpt,
                    "--config",
                    config_path,
                    "--seed",
                    str(seed),
                    "--prev_event",
                    prev_event,
                    "--out_metrics",
                    metrics_path,
                ]
            )

            mrr, hits10 = load_geo_metrics(metrics_path)
            results[name]["mrr"].append(mrr)
            results[name]["hits10"].append(hits10)

    summary = {}
    for name in ["A1", "A2"]:
        mrr_mean, mrr_std = mean_std(results[name]["mrr"])
        hits_mean, hits_std = mean_std(results[name]["hits10"])
        summary[name] = {
            "geo_mrr_mean": mrr_mean,
            "geo_mrr_std": mrr_std,
            "geo_hits10_mean": hits_mean,
            "geo_hits10_std": hits_std,
        }

    out_json = os.path.join(out_root, f"{label}_summary.json")
    os.makedirs(out_root, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"label": label, "raw": results, "summary": summary}, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--out_table",
        default="stats/strong_time_seed3.tsv",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summaries = []
    summaries.append(
        run_setting(
            "strong_time",
            "data/strong_time/pyg.pt",
            "configs/strong_time.yaml",
            seeds,
            "stats",
        )
    )
    summaries.append(
        run_setting(
            "strong_time_v2",
            "data/strong_time_v2/pyg.pt",
            "configs/strong_time_v2.yaml",
            seeds,
            "stats",
        )
    )

    with open(args.out_table, "w", encoding="utf-8") as f:
        f.write(
            "\t".join(
                [
                    "dataset",
                    "A1_geo_mrr",
                    "A1_geo_hits10",
                    "A2_geo_mrr",
                    "A2_geo_hits10",
                ]
            )
            + "\n"
        )
        for label, summary in zip(["strong_time", "strong_time_v2"], summaries):
            a1 = summary["A1"]
            a2 = summary["A2"]
            f.write(
                "\t".join(
                    [
                        label,
                        f"{a1['geo_mrr_mean']:.4f}±{a1['geo_mrr_std']:.4f}",
                        f"{a1['geo_hits10_mean']:.4f}±{a1['geo_hits10_std']:.4f}",
                        f"{a2['geo_mrr_mean']:.4f}±{a2['geo_mrr_std']:.4f}",
                        f"{a2['geo_hits10_mean']:.4f}±{a2['geo_hits10_std']:.4f}",
                    ]
                )
                + "\n"
            )

    print(f"Wrote {args.out_table}")


if __name__ == "__main__":
    main()
