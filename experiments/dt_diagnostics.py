import argparse
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.st_event_kgc import STEventKGC
from utils import (
    classification_metrics,
    filter_edge_types,
    load_config,
    seed_everything,
    set_prev_event_edges,
)


def build_neighbor_loader(data, mask, num_neighbors, batch_size):
    num_neighbors_dict = {edge: num_neighbors for edge in data.edge_types}
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors_dict,
        input_nodes=("event", mask),
        batch_size=batch_size,
        shuffle=False,
    )


def collect_geo_logits(data, model, mask, num_neighbors, batch_size, device):
    loader = build_neighbor_loader(data, mask, num_neighbors, batch_size)
    model.eval()
    logits_list = []
    labels_list = []
    nids_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            seed_size = batch["event"].batch_size
            geo_logits, _ = model(batch)
            logits_list.append(geo_logits[:seed_size].cpu())
            labels_list.append(batch["event"].y_geo[:seed_size].cpu())
            nids_list.append(batch["event"].n_id[:seed_size].cpu())
    return (
        torch.cat(logits_list, dim=0),
        torch.cat(labels_list, dim=0),
        torch.cat(nids_list, dim=0),
    )


def compute_edge_dt_disp(data, config, split):
    set_prev_event_edges(data, split, mode="on")
    edge_index = data["event", "prev_event", "event"].edge_index
    edge_attr = data["event", "prev_event", "event"].edge_attr
    if edge_attr.numel() == 0:
        return np.array([]), np.array([])

    event = data["event"]
    is_true = event.is_true.cpu().numpy()
    raw_geo = event.raw_y_geo.cpu().numpy()
    split_id = event.split_id.cpu().numpy()
    split_map = {"train": 0, "valid": 1, "test": 2}
    split_val = split_map[split]

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    valid = (split_id[src] == split_val) & (split_id[dst] == split_val)
    valid &= is_true[src] & is_true[dst]
    valid &= raw_geo[src] >= 0
    valid &= raw_geo[dst] >= 0
    if not np.any(valid):
        return np.array([]), np.array([])

    dt_scale = float(
        config["data"].get("dt_scale", config["data"]["time_steps"])
    )
    dt_vals = edge_attr.view(-1).cpu().numpy()[valid] * dt_scale

    nx = config["data"]["nx"]
    ny = config["data"]["ny"]
    geo_xy = data["geocell"].x.cpu().numpy()
    geo_x = geo_xy[:, 0] * max(1, nx - 1)
    geo_y = geo_xy[:, 1] * max(1, ny - 1)
    src_geo = raw_geo[src[valid]]
    dst_geo = raw_geo[dst[valid]]
    dx = np.abs(geo_x[src_geo] - geo_x[dst_geo])
    dy = np.abs(geo_y[src_geo] - geo_y[dst_geo])
    dist = dx + dy
    return dt_vals, dist


def compute_event_dt(data, config, split):
    set_prev_event_edges(data, split, mode="on")
    edge_index = data["event", "prev_event", "event"].edge_index
    edge_attr = data["event", "prev_event", "event"].edge_attr
    if edge_attr.numel() == 0:
        return np.array([])

    dt_scale = float(
        config["data"].get("dt_scale", config["data"]["time_steps"])
    )
    dt_vals = edge_attr.view(-1).cpu().numpy() * dt_scale
    dst = edge_index[1].cpu().numpy()
    max_dt = np.full((data["event"].num_nodes,), -1.0, dtype=np.float32)
    for d, dt in zip(dst, dt_vals):
        if dt > max_dt[d]:
            max_dt[d] = dt
    return max_dt


def mask_from_split(data, split):
    if split == "train":
        return data["event"].train_mask
    if split == "valid":
        return data["event"].valid_mask
    return data["event"].test_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument(
        "--prev_event",
        default="on",
        choices=["on", "zero_dt", "off", "shuffle_dt"],
    )
    parser.add_argument("--dt_percentile", type=float, default=0.8)
    parser.add_argument("--dt_threshold", type=float, default=None)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])
    data = torch.load(args.data)

    dt_vals, dist = compute_edge_dt_disp(data, config, args.split)
    diag = {"split": args.split, "prev_event": args.prev_event}
    if dt_vals.size > 0:
        corr = float(np.corrcoef(dt_vals, dist)[0, 1])
        if args.dt_threshold is None:
            dt_thr = float(np.quantile(dt_vals, args.dt_percentile))
        else:
            dt_thr = float(args.dt_threshold)
        long_mask = dt_vals >= dt_thr
        short_mask = dt_vals < dt_thr
        diag.update(
            {
                "edge_count": int(dt_vals.size),
                "dt_corr_disp": corr,
                "dt_threshold": dt_thr,
                "dt_long_ratio": float(long_mask.mean()),
                "disp_mean_long": float(dist[long_mask].mean()) if long_mask.any() else 0.0,
                "disp_mean_short": float(dist[short_mask].mean()) if short_mask.any() else 0.0,
            }
        )
        print(
            f"D1[{args.split}] corr(dt,disp)={corr:.4f} "
            f"thr={dt_thr:.3f} long_ratio={long_mask.mean():.3f} "
            f"E[d|long]={diag['disp_mean_long']:.3f} "
            f"E[d|short]={diag['disp_mean_short']:.3f}"
        )
    else:
        print("D1: no valid edges for correlation.")

    max_dt = compute_event_dt(data, config, args.split)
    split_mask = mask_from_split(data, args.split).cpu().numpy()
    is_true = data["event"].is_true.cpu().numpy()
    valid_event = split_mask & is_true & (max_dt >= 0)
    if valid_event.any():
        if args.dt_threshold is None:
            dt_thr_event = float(np.quantile(max_dt[valid_event], args.dt_percentile))
        else:
            dt_thr_event = float(args.dt_threshold)
        dt_long_mask = valid_event & (max_dt >= dt_thr_event)
        diag["dt_event_threshold"] = dt_thr_event
        diag["dt_event_long_ratio"] = float(dt_long_mask[valid_event].mean())
        print(
            f"D2[{args.split}] event dt_thr={dt_thr_event:.3f} "
            f"long_ratio={diag['dt_event_long_ratio']:.3f}"
        )
    else:
        dt_long_mask = None
        print("D2: no valid events with dt.")

    if args.ckpt:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_prev_event_edges(data, args.split, mode=args.prev_event)
        edge_types = filter_edge_types(
            data.edge_types, config["model"].get("source_message", "bi")
        )
        in_dims = {k: data[k].x.size(-1) for k in data.node_types}
        dt_encoding = config["model"].get("dt_encoding", "raw")
        dt_freqs = config["model"].get("dt_freqs", [1, 2, 4, 8])
        model = STEventKGC(
            in_dims=in_dims,
            num_geocell=data["geocell"].num_nodes,
            num_band=data["band"].num_nodes,
            hidden_dim=config["model"]["hidden_dim"],
            te_dim=config["model"]["te_dim"],
            heads=config["model"]["heads"],
            dropout=config["model"]["dropout"],
            dt_encoding=dt_encoding,
            dt_freqs=dt_freqs,
            edge_types=edge_types,
            motion_gate=config["model"].get("motion_gate", False),
        ).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        logits, labels, nids = collect_geo_logits(
            data,
            model,
            mask_from_split(data, args.split),
            config["train"]["num_neighbors"],
            config["train"]["batch_size"],
            device,
        )
        acc, mrr, hits = classification_metrics(
            logits, labels, ks=config["eval"]["hits_ks"]
        )
        diag["geo_all"] = {"acc": acc, "mrr": mrr, "hits": hits}
        print(
            f"Geo[{args.split} {args.prev_event}] acc {acc:.3f} "
            f"mrr {mrr:.3f} hits10 {hits.get(10, 0):.3f}"
        )
        if dt_long_mask is not None:
            dt_long_mask_t = torch.from_numpy(dt_long_mask)
            subset = dt_long_mask_t[nids]
            if subset.any():
                acc_l, mrr_l, hits_l = classification_metrics(
                    logits[subset], labels[subset], ks=config["eval"]["hits_ks"]
                )
            else:
                acc_l, mrr_l, hits_l = 0.0, 0.0, {}
            diag["geo_dt_long"] = {"acc": acc_l, "mrr": mrr_l, "hits": hits_l}
            print(
                f"Geo[{args.split} dt_long] acc {acc_l:.3f} "
                f"mrr {mrr_l:.3f} hits10 {hits_l.get(10, 0):.3f}"
            )

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
