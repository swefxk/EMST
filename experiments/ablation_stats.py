import argparse
import json
import math
import os

import numpy as np


def paired_bootstrap(diffs, num_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=np.float64)
    n = diffs.shape[0]
    if n == 0:
        return {"mean_diff": 0.0, "ci95": (0.0, 0.0), "p_le_zero": 1.0}
    samples = rng.choice(diffs, size=(num_boot, n), replace=True)
    boot_means = samples.mean(axis=1)
    lower, upper = np.percentile(boot_means, [2.5, 97.5]).tolist()
    p_le_zero = float(np.mean(boot_means <= 0.0))
    return {
        "mean_diff": float(diffs.mean()),
        "ci95": (float(lower), float(upper)),
        "p_le_zero": p_le_zero,
    }


def paired_t_test(diffs):
    diffs = np.asarray(diffs, dtype=np.float64)
    n = diffs.shape[0]
    if n <= 1:
        return {"t_stat": 0.0, "p_value": 1.0}
    mean = diffs.mean()
    std = diffs.std(ddof=1)
    if std == 0.0:
        return {"t_stat": 0.0, "p_value": 1.0 if mean == 0.0 else 0.0}
    t_stat = mean / (std / math.sqrt(n))
    try:
        import scipy.stats as stats

        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    except Exception:
        from math import erf, sqrt

        p_value = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))
    return {"t_stat": float(t_stat), "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="checkpoints/ablation_a012.json",
        help="Ablation raw json from experiments/ablation_a012.py",
    )
    parser.add_argument(
        "--out_json",
        default="checkpoints/ablation_a012_stats.json",
        help="Output stats json path.",
    )
    parser.add_argument(
        "--out_table",
        default="stats/ablation_a012_table.tsv",
        help="Output summary table (tsv).",
    )
    parser.add_argument("--num_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw = payload["raw"]
    metrics = ["geo_mrr", "geo_hits10", "band_mrr", "band_hits10"]
    comparisons = [("A1", "A0"), ("A2", "A0"), ("A2", "A1")]

    stats = {"comparisons": {}}
    table_rows = []
    for a, b in comparisons:
        comp_key = f"{a}_vs_{b}"
        stats["comparisons"][comp_key] = {}
        for metric in metrics:
            diffs = [va - vb for va, vb in zip(raw[a][metric], raw[b][metric])]
            boot = paired_bootstrap(diffs, num_boot=args.num_boot, seed=args.seed)
            ttest = paired_t_test(diffs)
            stats["comparisons"][comp_key][metric] = {
                "mean_diff": boot["mean_diff"],
                "ci95": boot["ci95"],
                "p_le_zero": boot["p_le_zero"],
                "t_stat": ttest["t_stat"],
                "p_value": ttest["p_value"],
                "n": len(diffs),
            }
            table_rows.append(
                {
                    "comparison": comp_key,
                    "metric": metric,
                    "mean_diff": boot["mean_diff"],
                    "ci95_low": boot["ci95"][0],
                    "ci95_high": boot["ci95"][1],
                    "p_le_zero": boot["p_le_zero"],
                    "p_value": ttest["p_value"],
                }
            )

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    table_dir = os.path.dirname(args.out_table)
    if table_dir:
        os.makedirs(table_dir, exist_ok=True)
    with open(args.out_table, "w", encoding="utf-8") as f:
        header = [
            "comparison",
            "metric",
            "mean_diff",
            "ci95_low",
            "ci95_high",
            "p_le_zero",
            "p_value",
        ]
        f.write("\t".join(header) + "\n")
        for row in table_rows:
            f.write(
                "\t".join(
                    [
                        row["comparison"],
                        row["metric"],
                        f"{row['mean_diff']:.6f}",
                        f"{row['ci95_low']:.6f}",
                        f"{row['ci95_high']:.6f}",
                        f"{row['p_le_zero']:.6f}",
                        f"{row['p_value']:.6f}",
                    ]
                )
                + "\n"
            )

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_table}")


if __name__ == "__main__":
    main()
