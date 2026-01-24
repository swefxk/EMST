import argparse
import json
import os


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    band = payload["splits"]["test"]["calib"]["band"]
    return float(band["nll"]), float(band["ece"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="d1,d3,d5")
    parser.add_argument("--out", default="band_difficulty/band_calib.tsv")
    args = parser.parse_args()

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    rows = []
    for label in labels:
        unc = load_metrics(os.path.join("band_difficulty", f"{label}_default.json"))
        cal = load_metrics(
            os.path.join("band_difficulty", f"{label}_calib_metrics.json")
        )
        rows.append(
            {
                "label": label,
                "nll_uncal": unc[0],
                "ece_uncal": unc[1],
                "nll_cal": cal[0],
                "ece_cal": cal[1],
                "nll_delta": cal[0] - unc[0],
                "ece_delta": cal[1] - unc[1],
            }
        )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(
            "\t".join(
                [
                    "label",
                    "nll_uncal",
                    "nll_cal",
                    "nll_delta",
                    "ece_uncal",
                    "ece_cal",
                    "ece_delta",
                ]
            )
            + "\n"
        )
        for row in rows:
            f.write(
                "\t".join(
                    [
                        row["label"],
                        f"{row['nll_uncal']:.4f}",
                        f"{row['nll_cal']:.4f}",
                        f"{row['nll_delta']:.4f}",
                        f"{row['ece_uncal']:.4f}",
                        f"{row['ece_cal']:.4f}",
                        f"{row['ece_delta']:.4f}",
                    ]
                )
                + "\n"
            )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
