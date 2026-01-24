import argparse
import json
import os
import re
from collections import defaultdict

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
    t_stat = mean / (std / np.sqrt(n))
    try:
        import scipy.stats as stats

        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    except Exception:
        from math import erf, sqrt

        p_value = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))
    return {"t_stat": float(t_stat), "p_value": p_value}


def _get_nested(data, path):
    cur = data
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def extract_metric(metrics, metric):
    paths = [
        ("geo_mrr",) if metric == "geo_mrr" else None,
        ("geo_hits10",) if metric == "geo_hits10" else None,
        ("band_mrr",) if metric == "band_mrr" else None,
        ("band_hits10",) if metric == "band_hits10" else None,
    ]
    paths = [p for p in paths if p is not None]

    # Nested common format
    split_paths = []
    if metric.startswith("geo_"):
        field = "mrr" if metric.endswith("mrr") else "hits10"
        split_paths = [
            ("splits", "test", "rank", "geo", field),
            ("test", "rank", "geo", field),
            ("splits", "test", "geo", field),
            ("test", "geo", field),
        ]
    elif metric.startswith("band_"):
        field = "mrr" if metric.endswith("mrr") else "hits10"
        split_paths = [
            ("splits", "test", "rank", "band", field),
            ("test", "rank", "band", field),
            ("splits", "test", "band", field),
            ("test", "band", field),
        ]
    for p in paths + split_paths:
        value = _get_nested(metrics, p)
        if value is not None:
            return float(value)

    # Handle hits dict
    if metric.endswith("hits10"):
        if metric.startswith("geo_"):
            hits = _get_nested(metrics, ("splits", "test", "rank", "geo", "hits"))
        else:
            hits = _get_nested(metrics, ("splits", "test", "rank", "band", "hits"))
        if isinstance(hits, dict):
            for key in ("10", 10):
                if key in hits:
                    return float(hits[key])

    raise ValueError(f"Cannot find metric {metric} in metrics.json.")


def parse_seed_from_dir(name):
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def load_runs(runs_root, metrics):
    data = defaultdict(dict)
    for setting in os.listdir(runs_root):
        setting_dir = os.path.join(runs_root, setting)
        if not os.path.isdir(setting_dir):
            continue
        for seed_dir in os.listdir(setting_dir):
            metrics_path = os.path.join(setting_dir, seed_dir, "metrics.json")
            if not os.path.isfile(metrics_path):
                continue
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_json = json.load(f)
            seed = metrics_json.get("seed")
            if seed is None:
                seed = parse_seed_from_dir(seed_dir)
            if seed is None:
                raise ValueError(f"Cannot infer seed for {metrics_path}.")
            data[setting][seed] = {
                m: extract_metric(metrics_json, m) for m in metrics
            }
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs_root",
        default="runs",
        help="Root directory containing runs/*/*/metrics.json",
    )
    parser.add_argument("--out_dir", default="stats")
    parser.add_argument(
        "--comparisons",
        default="A1_vs_A0,A2_vs_A0",
        help="Comma separated comparisons.",
    )
    parser.add_argument(
        "--metrics",
        default="geo_mrr,geo_hits10",
        help="Comma separated metrics.",
    )
    parser.add_argument("--num_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    comparisons = [c.strip() for c in args.comparisons.split(",") if c.strip()]

    runs = load_runs(args.runs_root, metrics)
    os.makedirs(args.out_dir, exist_ok=True)

    summary_rows = []
    for comp in comparisons:
        a, b = comp.split("_vs_")
        if a not in runs or b not in runs:
            raise ValueError(f"Missing runs for {a} or {b}.")
        seeds = sorted(set(runs[a].keys()) & set(runs[b].keys()))
        if not seeds:
            raise ValueError(f"No overlapping seeds for {a} vs {b}.")

        comp_payload = {"comparison": comp, "seeds": seeds, "metrics": {}}
        for metric in metrics:
            diffs = [runs[a][s][metric] - runs[b][s][metric] for s in seeds]
            boot = paired_bootstrap(diffs, num_boot=args.num_boot, seed=args.seed)
            ttest = paired_t_test(diffs)
            comp_payload["metrics"][metric] = {
                "mean_diff": boot["mean_diff"],
                "ci95": boot["ci95"],
                "p_le_zero": boot["p_le_zero"],
                "t_stat": ttest["t_stat"],
                "p_value": ttest["p_value"],
                "n": len(seeds),
            }
            summary_rows.append(
                {
                    "comparison": comp,
                    "metric": metric,
                    "mean_diff": boot["mean_diff"],
                    "ci95_low": boot["ci95"][0],
                    "ci95_high": boot["ci95"][1],
                    "p_le_zero": boot["p_le_zero"],
                    "t_stat": ttest["t_stat"],
                    "p_value": ttest["p_value"],
                    "n": len(seeds),
                }
            )

        out_path = os.path.join(args.out_dir, f"{comp}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(comp_payload, f, indent=2)

    summary_path = os.path.join(args.out_dir, "summary.tsv")
    with open(summary_path, "w", encoding="utf-8") as f:
        header = [
            "comparison",
            "metric",
            "mean_diff",
            "ci95_low",
            "ci95_high",
            "p_le_zero",
            "t_stat",
            "p_value",
            "n",
        ]
        f.write("\t".join(header) + "\n")
        for row in summary_rows:
            f.write(
                "\t".join(
                    [
                        row["comparison"],
                        row["metric"],
                        f"{row['mean_diff']:.6f}",
                        f"{row['ci95_low']:.6f}",
                        f"{row['ci95_high']:.6f}",
                        f"{row['p_le_zero']:.6f}",
                        f"{row['t_stat']:.4f}",
                        f"{row['p_value']:.6f}",
                        str(row["n"]),
                    ]
                )
                + "\n"
            )

    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
