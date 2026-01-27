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


def load_rank_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rank = payload["splits"]["test"]["rank"]
    geo = rank["geo"]
    band = rank["band"]
    return (
        float(geo["mrr"]),
        float(geo["hits10"]),
        float(band["mrr"]),
        float(band["hits10"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--k_values", default="1,4,8,16")
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds. If set, overrides --seed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prev_event", choices=["on", "zero_dt"], default="on")
    parser.add_argument(
        "--out_table",
        default="stats/prev_event_k_sweep.tsv",
    )
    parser.add_argument(
        "--out_json",
        default="stats/prev_event",
        help="Prefix for JSON results (will add .json).",
    )
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    k_values = [int(k) for k in args.k_values.split(",") if k.strip()]
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]
    ckpt_dir = os.path.join("checkpoints", "prev_event_k")
    runs_dir = os.path.join("runs", "prev_event_k")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    raw_results = {str(seed): [] for seed in seeds}
    for seed in seeds:
        for k in k_values:
            data_out = os.path.join(args.data_dir, f"pyg_k{k}.pt")
            if args.force_rebuild or not os.path.isfile(data_out):
                run_cmd(
                    [
                        python,
                        "pyg_data/build_heterodata.py",
                        "--data_dir",
                        args.data_dir,
                        "--config",
                        args.config,
                        "--k_prev",
                        str(k),
                        "--out",
                        data_out,
                    ]
                )

            ckpt = os.path.join(ckpt_dir, f"k{k}_seed{seed}.pt")
            metrics_dir = os.path.join(runs_dir, f"k{k}", f"seed_{seed}")
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_path = os.path.join(metrics_dir, "metrics.json")

            if args.force_train or not os.path.isfile(ckpt):
                run_cmd(
                    [
                        python,
                        "train.py",
                        "--data",
                        data_out,
                        "--save",
                        ckpt,
                        "--config",
                        args.config,
                        "--seed",
                        str(seed),
                        "--prev_event",
                        args.prev_event,
                    ]
                )

            run_cmd(
                [
                    python,
                    "eval.py",
                    "--data",
                    data_out,
                    "--ckpt",
                    ckpt,
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                    "--prev_event",
                    args.prev_event,
                    "--out_metrics",
                    metrics_path,
                ]
            )

            geo_mrr, geo_hits10, band_mrr, band_hits10 = load_rank_metrics(
                metrics_path
            )
            raw_results[str(seed)].append(
                {
                    "k_prev": k,
                    "geo_mrr": geo_mrr,
                    "geo_hits10": geo_hits10,
                    "band_mrr": band_mrr,
                    "band_hits10": band_hits10,
                }
            )

    summary = []
    for k in k_values:
        geo_mrr_vals = []
        geo_hits10_vals = []
        band_mrr_vals = []
        band_hits10_vals = []
        for seed in seeds:
            seed_rows = raw_results[str(seed)]
            row = next(r for r in seed_rows if r["k_prev"] == k)
            geo_mrr_vals.append(row["geo_mrr"])
            geo_hits10_vals.append(row["geo_hits10"])
            band_mrr_vals.append(row["band_mrr"])
            band_hits10_vals.append(row["band_hits10"])
        summary.append(
            {
                "k_prev": k,
                "geo_mrr": geo_mrr_vals,
                "geo_hits10": geo_hits10_vals,
                "band_mrr": band_mrr_vals,
                "band_hits10": band_hits10_vals,
            }
        )

    out_table = args.out_table
    os.makedirs(os.path.dirname(out_table) or ".", exist_ok=True)
    with open(out_table, "w", encoding="utf-8") as f:
        f.write("k_prev\tgeo_mrr\tgeo_hits10\tband_mrr\tband_hits10\n")
        for row in summary:
            if len(seeds) == 1:
                f.write(
                    f"{row['k_prev']}\t{row['geo_mrr'][0]:.4f}\t{row['geo_hits10'][0]:.4f}\t"
                    f"{row['band_mrr'][0]:.4f}\t{row['band_hits10'][0]:.4f}\n"
                )
            else:
                geo_mrr_mean = float(np.mean(row["geo_mrr"]))
                geo_mrr_std = float(np.std(row["geo_mrr"], ddof=1))
                geo_hits_mean = float(np.mean(row["geo_hits10"]))
                geo_hits_std = float(np.std(row["geo_hits10"], ddof=1))
                band_mrr_mean = float(np.mean(row["band_mrr"]))
                band_mrr_std = float(np.std(row["band_mrr"], ddof=1))
                band_hits_mean = float(np.mean(row["band_hits10"]))
                band_hits_std = float(np.std(row["band_hits10"], ddof=1))
                f.write(
                    f"{row['k_prev']}\t{geo_mrr_mean:.4f}±{geo_mrr_std:.4f}\t"
                    f"{geo_hits_mean:.4f}±{geo_hits_std:.4f}\t"
                    f"{band_mrr_mean:.4f}±{band_mrr_std:.4f}\t"
                    f"{band_hits_mean:.4f}±{band_hits_std:.4f}\n"
                )
    out_json = f"{args.out_json}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": seeds,
                "prev_event": args.prev_event,
                "raw": raw_results,
                "summary": summary,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_table}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
