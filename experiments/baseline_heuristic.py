import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import load_config, save_json


def build_mapping(edge_index, size):
    mapping = torch.full((size,), -1, dtype=torch.long)
    mapping[edge_index[0]] = edge_index[1]
    if (mapping < 0).any():
        missing = int((mapping < 0).sum().item())
        raise ValueError(f"Mapping incomplete: {missing} entries missing.")
    return mapping


def init_stats(ks):
    return {
        "count": 0,
        "acc_sum": 0.0,
        "mrr_sum": 0.0,
        "hits_sum": {k: 0.0 for k in ks},
    }


def update_stats(stats, logits, labels, ks):
    if labels.numel() == 0:
        return
    sorted_idx = torch.argsort(logits, dim=-1, descending=True)
    preds = sorted_idx[:, 0]
    stats["acc_sum"] += (preds == labels).float().sum().item()
    ranks = (sorted_idx == labels.unsqueeze(-1)).nonzero(as_tuple=False)
    rank_pos = ranks[:, 1].float() + 1.0
    stats["mrr_sum"] += (1.0 / rank_pos).sum().item()
    for k in ks:
        stats["hits_sum"][k] += (rank_pos <= k).float().sum().item()
    stats["count"] += labels.numel()


def finalize_stats(stats):
    count = stats["count"]
    if count == 0:
        return 0.0, 0.0, {k: 0.0 for k in stats["hits_sum"]}
    acc = stats["acc_sum"] / count
    mrr = stats["mrr_sum"] / count
    hits = {k: stats["hits_sum"][k] / count for k in stats["hits_sum"]}
    return acc, mrr, hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to pyg .pt file.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--distance",
        choices=["manhattan", "euclidean"],
        default=None,
        help="Distance metric for ranking geocells.",
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument(
        "--out_metrics",
        default="stats/baseline_heuristic_metrics.json",
        help="Optional path to save metrics JSON.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    hits_ks = config["eval"]["hits_ks"]
    distance = args.distance or config["data"]["obs"].get(
        "distance_metric", "manhattan"
    )

    data = torch.load(args.data)
    num_events = data["event"].num_nodes
    num_sensors = data["sensor"].num_nodes

    edge_event_sensor = data["event", "observed_by", "sensor"].edge_index
    edge_sensor_geo = data["sensor", "located_in", "geocell"].edge_index
    event2sensor = build_mapping(edge_event_sensor, num_events)
    sensor2geo = build_mapping(edge_sensor_geo, num_sensors)

    geocell_xy = data["geocell"].x[:, :2].cpu()
    p_norm = 1 if distance == "manhattan" else 2

    metrics_payload = {
        "baseline": "sensor_geocell_distance_rank",
        "distance": distance,
        "splits": {},
    }

    for split_name, mask in [
        ("valid", data["event"].valid_mask),
        ("test", data["event"].test_mask),
    ]:
        indices = mask.nonzero(as_tuple=False).view(-1)
        if indices.numel() == 0:
            metrics_payload["splits"][split_name] = {"rank": {"geo": {}}}
            continue

        labels = data["event"].y_geo[indices].cpu()
        sensor_idx = event2sensor[indices].cpu()
        sensor_geo = sensor2geo[sensor_idx]
        sensor_xy = geocell_xy[sensor_geo]

        stats = init_stats(hits_ks)
        for start in range(0, indices.numel(), args.batch_size):
            end = min(indices.numel(), start + args.batch_size)
            batch_xy = sensor_xy[start:end]
            batch_labels = labels[start:end]
            dists = torch.cdist(batch_xy, geocell_xy, p=p_norm)
            logits = -dists
            update_stats(stats, logits, batch_labels, hits_ks)

        acc, mrr, hits = finalize_stats(stats)
        print(
            f"{split_name} heuristic geo_acc {acc:.3f} "
            f"geo_mrr {mrr:.3f} geo_hits {hits}"
        )
        metrics_payload["splits"][split_name] = {
            "rank": {
                "tag": "heuristic",
                "geo": {
                    "acc": acc,
                    "mrr": mrr,
                    "hits": hits,
                    "hits10": hits.get(10, 0.0),
                }
            }
        }

    if args.out_metrics:
        out_dir = os.path.dirname(args.out_metrics)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_json(args.out_metrics, metrics_payload)
        print(f"Wrote {args.out_metrics}")


if __name__ == "__main__":
    main()
