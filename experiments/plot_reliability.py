import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.st_event_kgc import STEventKGC
from utils import (
    expected_calibration_error,
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


def reliability_curve(probs, labels, n_bins=15):
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    bin_acc = []
    bin_conf = []
    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.sum() == 0:
            bin_acc.append(float("nan"))
            bin_conf.append(float("nan"))
        else:
            bin_acc.append(accuracies[mask].float().mean().item())
            bin_conf.append(confidences[mask].mean().item())
    return bin_conf, bin_acc


def collect_probs(
    data,
    model,
    mask,
    num_neighbors,
    batch_size,
    device,
    temperature=None,
    mc_dropout=0,
):
    loader = build_neighbor_loader(data, mask, num_neighbors, batch_size)
    if mc_dropout > 0:
        model.train()
    else:
        model.eval()

    probs_geo = []
    probs_band = []
    labels_geo = []
    labels_band = []

    t_geo = temperature["geo"] if temperature else 1.0
    t_band = temperature["band"] if temperature else 1.0

    for batch in loader:
        batch = batch.to(device)
        seed_size = batch["event"].batch_size
        y_geo = batch["event"].y_geo[:seed_size]
        y_band = batch["event"].y_band[:seed_size]
        with torch.no_grad():
            if mc_dropout > 0:
                mc_logits_geo = []
                mc_logits_band = []
                for _ in range(mc_dropout):
                    geo_logits, band_logits = model(batch)
                    mc_logits_geo.append(geo_logits[:seed_size])
                    mc_logits_band.append(band_logits[:seed_size])
                mc_logits_geo = torch.stack(mc_logits_geo, dim=0)
                mc_logits_band = torch.stack(mc_logits_band, dim=0)
                mean_logits_geo = mc_logits_geo.mean(dim=0) / t_geo
                mean_logits_band = mc_logits_band.mean(dim=0) / t_band
                p_geo = torch.softmax(mean_logits_geo, dim=-1)
                p_band = torch.softmax(mean_logits_band, dim=-1)
            else:
                geo_logits, band_logits = model(batch)
                p_geo = torch.softmax(geo_logits[:seed_size] / t_geo, dim=-1)
                p_band = torch.softmax(band_logits[:seed_size] / t_band, dim=-1)

        probs_geo.append(p_geo.cpu())
        probs_band.append(p_band.cpu())
        labels_geo.append(y_geo.cpu())
        labels_band.append(y_band.cpu())

    return (
        torch.cat(probs_geo, dim=0),
        torch.cat(labels_geo, dim=0),
        torch.cat(probs_band, dim=0),
        torch.cat(labels_band, dim=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--calib", default=None)
    parser.add_argument("--mc_dropout", type=int, default=0)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument(
        "--prev_event",
        default="on",
        choices=["on", "zero_dt", "off", "shuffle_dt"],
    )
    parser.add_argument(
        "--out",
        default="figures/reliability_diagram.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.data)
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
        prev_event_scale=config["model"].get("prev_event_scale", 1.0),
        prev_event_residual_scale=config["model"].get("prev_event_residual_scale", 0.0),
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    temperature = None
    if args.calib:
        with open(args.calib, "r", encoding="utf-8") as f:
            calib = json.load(f)
        temperature = {"geo": calib["geo_temperature"], "band": calib["band_temperature"]}

    mask = data["event"].valid_mask if args.split == "valid" else data["event"].test_mask
    probs_geo, labels_geo, probs_band, labels_band = collect_probs(
        data,
        model,
        mask,
        config["train"]["num_neighbors"],
        config["train"]["batch_size"],
        device,
        temperature=temperature,
        mc_dropout=args.mc_dropout,
    )

    geo_ece = expected_calibration_error(probs_geo, labels_geo)
    band_ece = expected_calibration_error(probs_band, labels_band)
    geo_conf, geo_acc = reliability_curve(probs_geo, labels_geo)
    band_conf, band_acc = reliability_curve(probs_band, labels_band)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for ax, conf, acc, title, ece in [
        (axes[0], geo_conf, geo_acc, "Geo", geo_ece),
        (axes[1], band_conf, band_acc, "Band", band_ece),
    ]:
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.plot(conf, acc, marker="o", color="#1f77b4")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title} Reliability (ECE={ece:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
