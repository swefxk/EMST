import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="checkpoints/ablation_a012.json",
        help="Ablation raw json.",
    )
    parser.add_argument(
        "--out",
        default="figures/geo_ablation_mrr.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload["raw"]

    settings = ["A0", "A1", "A2"]
    means = []
    stds = []
    for s in settings:
        vals = np.array(raw[s]["geo_mrr"], dtype=np.float32)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure(figsize=(5.0, 3.2))
    x = np.arange(len(settings))
    plt.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o-",
        capsize=4,
        color="#1f77b4",
        ecolor="#1f77b4",
    )
    plt.xticks(x, settings)
    plt.ylabel("Geo MRR (test)")
    plt.title("Geo MRR vs A0/A1/A2")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
