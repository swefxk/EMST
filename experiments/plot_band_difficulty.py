import argparse
import json
import os

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--out",
        default="figures/band_difficulty_curve.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    labels = [r["label"] for r in records]
    default_mrr = [r["default_mrr"] for r in records]
    b1_mrr = [r["b1_mrr"] for r in records]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure(figsize=(5.2, 3.2))
    plt.plot(labels, default_mrr, marker="o", label="Default")
    plt.plot(labels, b1_mrr, marker="o", label="B1 (mask band_obs)")
    plt.xlabel("Band drift difficulty")
    plt.ylabel("Band MRR (test)")
    plt.title("Band difficulty curve")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
