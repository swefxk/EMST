import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch_geometric.data import HeteroData

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import ensure_dir, load_config, seed_everything


def load_table(path, dtype_map):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            parsed = {}
            for key, dtype in dtype_map.items():
                parsed[key] = dtype(row[key])
            rows.append(parsed)
    return rows


def load_splits(path):
    split_map = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            split_map[int(row["event_id"])] = row["split"]
    return split_map


def build_prev_event_edges(events, k):
    events_sorted = sorted(events, key=lambda x: (x["t_center"], x["event_id"]))
    history_sensor = {}
    history_geo = {}
    edge_src = []
    edge_dst = []
    edge_dt = []

    for event in events_sorted:
        e_id = event["event_id"]
        t_center = event["t_center"]
        geo_id = event.get("geocell_id_true", -1)

        sensor_hist = history_sensor.get(event["sensor_id"], [])
        geo_hist = history_geo.get(geo_id, []) if geo_id >= 0 else []
        candidates = sensor_hist[-k:] + geo_hist[-k:]

        seen = set()
        for prev_id, prev_t in reversed(candidates):
            if prev_id in seen:
                continue
            seen.add(prev_id)
            edge_src.append(prev_id)
            edge_dst.append(e_id)
            edge_dt.append(max(0.0, t_center - prev_t))

        history_sensor.setdefault(event["sensor_id"], []).append((e_id, t_center))
        if geo_id >= 0:
            history_geo.setdefault(geo_id, []).append((e_id, t_center))

    return edge_src, edge_dst, edge_dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory of TSV files.")
    parser.add_argument("--config", default="configs/default.yaml", help="Config path.")
    parser.add_argument(
        "--k_prev",
        type=int,
        default=None,
        help="Override prev_event k when building edges.",
    )
    parser.add_argument("--out", required=True, help="Output .pt path.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])

    event_path = os.path.join(args.data_dir, "event.tsv")
    sensor_path = os.path.join(args.data_dir, "sensor.tsv")
    geocell_path = os.path.join(args.data_dir, "geocell.tsv")
    band_path = os.path.join(args.data_dir, "band.tsv")
    split_path = os.path.join(args.data_dir, "event_split.tsv")

    events = load_table(
        event_path,
        {
            "event_id": int,
            "t_center": float,
            "band_id_obs": int,
            "geocell_id_true": int,
            "sensor_id": int,
            "power_obs": float,
            "bw_obs": float,
            "conf": float,
            "is_true": int,
            "band_id_true": int,
        },
    )
    sensors = load_table(
        sensor_path,
        {"sensor_id": int, "geocell_id": int, "reliability": float},
    )
    geocells = load_table(
        geocell_path,
        {"geocell_id": int, "x": int, "y": int, "x_norm": float, "y_norm": float},
    )
    bands = load_table(band_path, {"band_id": int, "f_center_norm": float})
    split_map = load_splits(split_path)

    events = sorted(events, key=lambda x: x["event_id"])
    sensors = sorted(sensors, key=lambda x: x["sensor_id"])
    geocells = sorted(geocells, key=lambda x: x["geocell_id"])
    bands = sorted(bands, key=lambda x: x["band_id"])

    num_events = len(events)
    num_sensors = len(sensors)
    num_geocell = len(geocells)
    num_band = len(bands)

    event_id_map = {e["event_id"]: idx for idx, e in enumerate(events)}
    sensor_id_map = {s["sensor_id"]: idx for idx, s in enumerate(sensors)}
    geocell_id_map = {g["geocell_id"]: idx for idx, g in enumerate(geocells)}
    band_id_map = {b["band_id"]: idx for idx, b in enumerate(bands)}

    split_lookup = {"train": 0, "valid": 1, "test": 2}
    events_by_split = {"train": [], "valid": [], "test": []}
    for event in events:
        split = split_map.get(event["event_id"], "train")
        if split not in events_by_split:
            split = "train"
        events_by_split[split].append(event)

    t_vals = np.array([e["t_center"] for e in events], dtype=np.float32)
    max_t = float(t_vals.max()) if num_events else 1.0
    power_vals = np.array([e["power_obs"] for e in events], dtype=np.float32)
    bw_vals = np.array([e["bw_obs"] for e in events], dtype=np.float32)
    power_mean, power_std = float(power_vals.mean()), float(power_vals.std() or 1.0)
    bw_mean, bw_std = float(bw_vals.mean()), float(bw_vals.std() or 1.0)

    event_x = []
    y_geo = []
    y_band = []
    train_mask = []
    valid_mask = []
    test_mask = []
    is_true_flags = []
    split_ids = []
    raw_y_geo = []
    raw_y_band = []

    for event in events:
        t_norm = event["t_center"] / max_t if max_t > 0 else 0.0
        band_obs_idx = band_id_map.get(event["band_id_obs"], 0)
        band_norm = band_obs_idx / max(1, num_band - 1) if num_band > 1 else 0.0
        power_norm = (event["power_obs"] - power_mean) / power_std
        bw_norm = (event["bw_obs"] - bw_mean) / bw_std
        event_x.append([t_norm, band_norm, bw_norm, power_norm, event["conf"]])
        y_geo.append(geocell_id_map.get(event["geocell_id_true"], 0))
        y_band.append(band_id_map.get(event["band_id_true"], 0))
        raw_y_geo.append(event["geocell_id_true"])
        raw_y_band.append(event["band_id_true"])

        is_true = event["is_true"] == 1
        is_true_flags.append(is_true)

        split = split_map.get(event["event_id"], "train")
        split_id = split_lookup.get(split, 0)
        split_ids.append(split_id)
        train_mask.append(split == "train" and is_true)
        valid_mask.append(split == "valid" and is_true)
        test_mask.append(split == "test" and is_true)

    sensor_x = [[s["reliability"]] for s in sensors]
    geocell_x = [[g["x_norm"], g["y_norm"]] for g in geocells]
    band_x = [[b["f_center_norm"]] for b in bands]

    data = HeteroData()
    data["event"].x = torch.tensor(event_x, dtype=torch.float32)
    data["event"].y_geo = torch.tensor(y_geo, dtype=torch.long)
    data["event"].y_band = torch.tensor(y_band, dtype=torch.long)
    data["event"].train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data["event"].valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
    data["event"].test_mask = torch.tensor(test_mask, dtype=torch.bool)
    data["event"].is_true = torch.tensor(is_true_flags, dtype=torch.bool)
    data["event"].split_id = torch.tensor(split_ids, dtype=torch.long)
    data["event"].raw_y_geo = torch.tensor(raw_y_geo, dtype=torch.long)
    data["event"].raw_y_band = torch.tensor(raw_y_band, dtype=torch.long)

    data["sensor"].x = torch.tensor(sensor_x, dtype=torch.float32)
    data["geocell"].x = torch.tensor(geocell_x, dtype=torch.float32)
    data["band"].x = torch.tensor(band_x, dtype=torch.float32)

    event_ids = [event_id_map[e["event_id"]] for e in events]
    sensor_ids = [sensor_id_map[e["sensor_id"]] for e in events]
    edge_event_sensor = torch.tensor(
        [event_ids, sensor_ids], dtype=torch.long
    )
    data["event", "observed_by", "sensor"].edge_index = edge_event_sensor
    data["sensor", "rev_observed_by", "event"].edge_index = edge_event_sensor.flip(0)

    sensor_src = [sensor_id_map[s["sensor_id"]] for s in sensors]
    sensor_dst = [geocell_id_map[s["geocell_id"]] for s in sensors]
    edge_sensor_geo = torch.tensor(
        [sensor_src, sensor_dst], dtype=torch.long
    )
    data["sensor", "located_in", "geocell"].edge_index = edge_sensor_geo
    data["geocell", "rev_located_in", "sensor"].edge_index = edge_sensor_geo.flip(0)

    if args.k_prev is not None:
        k_prev = args.k_prev
    else:
        k_prev = config["data"]["prev_event"]["k"]

    def build_prev_tensors(split_events):
        edge_src, edge_dst, edge_dt = build_prev_event_edges(split_events, k_prev)
        edge_src = [event_id_map[idx] for idx in edge_src]
        edge_dst = [event_id_map[idx] for idx in edge_dst]
        if len(edge_src) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
            log_mean = torch.tensor(0.0, dtype=torch.float32)
            log_std = torch.tensor(1.0, dtype=torch.float32)
        else:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            dt_norm = [dt / max_t if max_t > 0 else 0.0 for dt in edge_dt]
            edge_attr = torch.tensor(dt_norm, dtype=torch.float32).view(-1, 1)
            dt_log = torch.log1p(edge_attr)
            log_mean = dt_log.mean()
            log_std = dt_log.std(unbiased=False)
            if log_std.item() == 0.0:
                log_std = torch.tensor(1.0, dtype=torch.float32)
        return edge_index, edge_attr, log_mean, log_std

    prev_edges = {}
    for split_name, split_events in events_by_split.items():
        edge_index, edge_attr, log_mean, log_std = build_prev_tensors(split_events)
        prev_edges[split_name] = (edge_index, edge_attr)
        setattr(data["event"], f"prev_event_{split_name}_edge_index", edge_index)
        setattr(data["event"], f"prev_event_{split_name}_edge_attr", edge_attr)
        setattr(data["event"], f"prev_event_{split_name}_dt_log_mean", log_mean)
        setattr(data["event"], f"prev_event_{split_name}_dt_log_std", log_std)

    data["event", "prev_event", "event"].edge_index = prev_edges["train"][0]
    data["event", "prev_event", "event"].edge_attr = prev_edges["train"][1]

    ensure_dir(os.path.dirname(args.out))
    torch.save(data, args.out)

    print(f"Events: {num_events} Sensors: {num_sensors}")
    print(f"Geocells: {num_geocell} Bands: {num_band}")
    print(
        f"Edges observed_by: {edge_event_sensor.size(1)} "
        f"located_in: {edge_sensor_geo.size(1)}"
    )
    print(
        "Prev_event edges -> "
        f"train {prev_edges['train'][0].size(1)} "
        f"valid {prev_edges['valid'][0].size(1)} "
        f"test {prev_edges['test'][0].size(1)}"
    )
    print(
        f"Split (true only) -> train {sum(train_mask)} "
        f"valid {sum(valid_mask)} test {sum(test_mask)}"
    )
    print(f"Saved HeteroData to {args.out}")


if __name__ == "__main__":
    main()
