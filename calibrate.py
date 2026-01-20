import argparse
import json
import os

import torch
from torch.nn import functional as F
from torch_geometric.loader import NeighborLoader

from models.st_event_kgc import STEventKGC
from utils import load_config, seed_everything, set_prev_event_edges


def build_neighbor_loader(data, mask, num_neighbors, batch_size):
    num_neighbors_dict = {edge: num_neighbors for edge in data.edge_types}
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors_dict,
        input_nodes=("event", mask),
        batch_size=batch_size,
        shuffle=False,
    )


def mask_band_feature(event_x, band_feat_idx):
    masked = event_x.clone()
    masked[:, band_feat_idx] = 0.0
    return masked


def collect_logits(
    data,
    model,
    mask,
    num_neighbors,
    batch_size,
    device,
    mask_band=False,
    band_feat_idx=1,
):
    loader = build_neighbor_loader(data, mask, num_neighbors, batch_size)
    model.eval()
    logits_geo = []
    logits_band = []
    labels_geo = []
    labels_band = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            seed_size = batch["event"].batch_size
            if mask_band:
                masked_x = mask_band_feature(
                    batch["event"].x, band_feat_idx
                )
                geo_logits, band_logits = model(
                    batch, event_x_override=masked_x
                )
            else:
                geo_logits, band_logits = model(batch)
            logits_geo.append(geo_logits[:seed_size].detach().cpu())
            logits_band.append(band_logits[:seed_size].detach().cpu())
            labels_geo.append(batch["event"].y_geo[:seed_size].cpu())
            labels_band.append(batch["event"].y_band[:seed_size].cpu())

    return (
        torch.cat(logits_geo, dim=0),
        torch.cat(labels_geo, dim=0),
        torch.cat(logits_band, dim=0),
        torch.cat(labels_band, dim=0),
    )


def optimize_temperature(logits, labels, max_iter=200, lr=0.05, device="cpu"):
    logits = logits.to(device)
    labels = labels.to(device)
    log_t = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([log_t], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        temperature = torch.exp(log_t)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        optimizer.step()

    return torch.exp(log_t).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to pyg .pt file.")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint.")
    parser.add_argument("--out", required=True, help="Output calibration json.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument(
        "--band_obs_mask",
        action="store_true",
        help="Mask band_obs feature when calibrating band logits.",
    )
    parser.add_argument(
        "--band_feat_idx",
        type=int,
        default=1,
        help="Index of band_obs feature in event.x.",
    )
    parser.add_argument(
        "--prev_event",
        choices=["on", "zero_dt", "off"],
        default="on",
        help="Prev_event usage: on, zero_dt, or off.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["seed"] if args.seed is None else args.seed
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.data)
    set_prev_event_edges(data, "valid", mode=args.prev_event)

    in_dims = {k: data[k].x.size(-1) for k in data.node_types}
    model = STEventKGC(
        in_dims=in_dims,
        num_geocell=data["geocell"].num_nodes,
        num_band=data["band"].num_nodes,
        hidden_dim=config["model"]["hidden_dim"],
        te_dim=config["model"]["te_dim"],
        heads=config["model"]["heads"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    logits_geo, labels_geo, logits_band, labels_band = collect_logits(
        data,
        model,
        data["event"].valid_mask,
        config["train"]["num_neighbors"],
        config["train"]["batch_size"],
        device,
        mask_band=False,
    )
    if args.band_obs_mask:
        _, _, logits_band, labels_band = collect_logits(
            data,
            model,
            data["event"].valid_mask,
            config["train"]["num_neighbors"],
            config["train"]["batch_size"],
            device,
            mask_band=True,
            band_feat_idx=args.band_feat_idx,
        )

    geo_temp = optimize_temperature(
        logits_geo,
        labels_geo,
        max_iter=config["calib"]["max_iter"],
        lr=config["calib"]["lr"],
        device=device,
    )
    band_temp = optimize_temperature(
        logits_band,
        labels_band,
        max_iter=config["calib"]["max_iter"],
        lr=config["calib"]["lr"],
        device=device,
    )

    payload = {"geo_temperature": geo_temp, "band_temperature": band_temp}
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved calibration to {args.out}")
    print(f"geo_temperature {geo_temp:.4f} band_temperature {band_temp:.4f}")


if __name__ == "__main__":
    main()
