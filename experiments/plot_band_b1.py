import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    band = payload["splits"]["test"]["rank"]["band"]
    return band["mrr"], band["hits10"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_metrics", required=True)
    parser.add_argument("--b1_metrics", required=True)
    parser.add_argument(
        "--out",
        default="figures/band_b1_compare.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    default_mrr, default_hits10 = load_metrics(args.default_metrics)
    b1_mrr, b1_hits10 = load_metrics(args.b1_metrics)

    labels = ["MRR", "Hits@10"]
    default_vals = [default_mrr, default_hits10]
    b1_vals = [b1_mrr, b1_hits10]

    x = np.arange(len(labels))
    width = 0.35
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure(figsize=(5.5, 3.2))
    plt.bar(x - width / 2, default_vals, width, label="Default")
    plt.bar(x + width / 2, b1_vals, width, label="B1 (mask band_obs)")
    plt.xticks(x, labels)
    plt.ylabel("Band (test)")
    plt.title("Band performance: Default vs B1")
    plt.legend(frameon=False)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
