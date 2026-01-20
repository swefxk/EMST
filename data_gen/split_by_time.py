import argparse
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import ensure_dir, load_config, seed_everything


def load_events(event_path):
    events = []
    with open(event_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            events.append(
                {
                    "event_id": int(row["event_id"]),
                    "t_center": float(row["t_center"]),
                }
            )
    return events


def split_by_time(events, splits):
    events_sorted = sorted(events, key=lambda x: (x["t_center"], x["event_id"]))
    n = len(events_sorted)
    n_train = int(n * splits["train"])
    n_valid = int(n * splits["valid"])

    train_ids = {e["event_id"] for e in events_sorted[:n_train]}
    valid_ids = {
        e["event_id"] for e in events_sorted[n_train : n_train + n_valid]
    }
    test_ids = {e["event_id"] for e in events_sorted[n_train + n_valid :]}
    return train_ids, valid_ids, test_ids


def save_split(out_dir, train_ids, valid_ids, test_ids):
    ensure_dir(out_dir)
    split_path = os.path.join(out_dir, "event_split.tsv")
    with open(split_path, "w", encoding="utf-8") as f:
        f.write("event_id\tsplit\n")
        for event_id in sorted(train_ids):
            f.write(f"{event_id}\ttrain\n")
        for event_id in sorted(valid_ids):
            f.write(f"{event_id}\tvalid\n")
        for event_id in sorted(test_ids):
            f.write(f"{event_id}\ttest\n")
    return split_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--events", required=True, help="Event TSV path.")
    parser.add_argument("--out", required=True, help="Output directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])

    events = load_events(args.events)
    train_ids, valid_ids, test_ids = split_by_time(events, config["splits"])
    split_path = save_split(args.out, train_ids, valid_ids, test_ids)

    print(f"Split file: {split_path}")
    print(
        f"train: {len(train_ids)} valid: {len(valid_ids)} test: {len(test_ids)}"
    )


if __name__ == "__main__":
    main()
