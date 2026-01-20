import argparse
import json
import math
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
        # Fallback: normal approximation
        from math import erf, sqrt

        p_value = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))
    return {"t_stat": float(t_stat), "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="checkpoints/seed_sweep_geo.json",
        help="Seed sweep json path.",
    )
    parser.add_argument("--num_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw = payload["raw"]
    comparisons = [("A1", "A0"), ("A2", "A0")]
    metrics = ["mrr", "hits10"]

    print("Paired significance (Geo, test, seed-level):")
    for metric in metrics:
        print(f"Metric: {metric}")
        for a, b in comparisons:
            vals_a = raw[a][metric]
            vals_b = raw[b][metric]
            if len(vals_a) != len(vals_b):
                raise ValueError(f"Seed count mismatch for {a} vs {b}.")
            diffs = [va - vb for va, vb in zip(vals_a, vals_b)]
            boot = paired_bootstrap(diffs, num_boot=args.num_boot, seed=args.seed)
            ttest = paired_t_test(diffs)
            print(
                f"  {a} - {b}: mean_diff {boot['mean_diff']:.4f}, "
                f"ci95 [{boot['ci95'][0]:.4f}, {boot['ci95'][1]:.4f}], "
                f"p_le_zero {boot['p_le_zero']:.4f}, "
                f"t={ttest['t_stat']:.3f}, p={ttest['p_value']:.4f}"
            )


if __name__ == "__main__":
    main()
