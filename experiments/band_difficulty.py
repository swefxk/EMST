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


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    band = payload["splits"]["test"]["rank"]["band"]
    return float(band["mrr"]), float(band["hits10"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out_dir", default="band_difficulty")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if len(args.configs) != len(args.labels):
        raise ValueError("configs and labels must have same length.")

    python = sys.executable
    os.makedirs(args.out_dir, exist_ok=True)

    records = []
    for cfg, label in zip(args.configs, args.labels):
        data_dir = os.path.join("data", f"band_{label}")
        pyg_path = os.path.join(data_dir, "pyg.pt")
        ckpt_default = os.path.join("checkpoints", f"band_{label}_default.pt")
        ckpt_b1 = os.path.join("checkpoints", f"band_{label}_b1.pt")
        metrics_default = os.path.join(args.out_dir, f"{label}_default.json")
        metrics_b1 = os.path.join(args.out_dir, f"{label}_b1.json")

        run_cmd(
            [
                python,
                "data_gen/build_kg_files.py",
                "--config",
                cfg,
                "--out",
                data_dir,
            ]
        )
        run_cmd(
            [
                python,
                "pyg_data/build_heterodata.py",
                "--data_dir",
                data_dir,
                "--out",
                pyg_path,
                "--config",
                cfg,
            ]
        )

        run_cmd(
            [
                python,
                "train.py",
                "--data",
                pyg_path,
                "--save",
                ckpt_default,
                "--config",
                cfg,
                "--seed",
                str(args.seed),
                "--prev_event",
                "on",
            ]
        )
        run_cmd(
            [
                python,
                "eval.py",
                "--data",
                pyg_path,
                "--ckpt",
                ckpt_default,
                "--config",
                cfg,
                "--seed",
                str(args.seed),
                "--prev_event",
                "on",
                "--out_metrics",
                metrics_default,
            ]
        )

        run_cmd(
            [
                python,
                "train.py",
                "--data",
                pyg_path,
                "--save",
                ckpt_b1,
                "--config",
                cfg,
                "--seed",
                str(args.seed),
                "--prev_event",
                "on",
                "--band_obs_mask",
            ]
        )
        run_cmd(
            [
                python,
                "eval.py",
                "--data",
                pyg_path,
                "--ckpt",
                ckpt_b1,
                "--config",
                cfg,
                "--seed",
                str(args.seed),
                "--prev_event",
                "on",
                "--band_obs_mask",
                "--out_metrics",
                metrics_b1,
            ]
        )

        default_mrr, default_hits10 = load_metrics(metrics_default)
        b1_mrr, b1_hits10 = load_metrics(metrics_b1)
        records.append(
            {
                "label": label,
                "default_mrr": default_mrr,
                "default_hits10": default_hits10,
                "b1_mrr": b1_mrr,
                "b1_hits10": b1_hits10,
            }
        )

    table_path = os.path.join(args.out_dir, "band_difficulty.tsv")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(
            "\t".join(
                [
                    "label",
                    "default_mrr",
                    "default_hits10",
                    "b1_mrr",
                    "b1_hits10",
                ]
            )
            + "\n"
        )
        for row in records:
            f.write(
                "\t".join(
                    [
                        row["label"],
                        f"{row['default_mrr']:.4f}",
                        f"{row['default_hits10']:.4f}",
                        f"{row['b1_mrr']:.4f}",
                        f"{row['b1_hits10']:.4f}",
                    ]
                )
                + "\n"
            )

    with open(os.path.join(args.out_dir, "band_difficulty.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
