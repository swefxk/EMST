import argparse
import os

import torch
from torch.nn import functional as F
from torch_geometric.loader import NeighborLoader

from models.st_event_kgc import STEventKGC
from utils import (
    classification_metrics,
    load_config,
    seed_everything,
    set_prev_event_edges,
)


def build_neighbor_loader(data, mask, num_neighbors, batch_size, shuffle=True):
    num_neighbors_dict = {edge: num_neighbors for edge in data.edge_types}
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors_dict,
        input_nodes=("event", mask),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def mask_band_feature(event_x, band_feat_idx):
    masked = event_x.clone()
    masked[:, band_feat_idx] = 0.0
    return masked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to pyg .pt file.")
    parser.add_argument("--save", required=True, help="Model checkpoint path.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--band_obs_mask",
        action="store_true",
        help="Mask band_obs feature for the band head (B1 setting).",
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
    seed_everything(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.data)
    set_prev_event_edges(data, "train", mode=args.prev_event)

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    loader = build_neighbor_loader(
        data,
        data["event"].train_mask,
        config["train"]["num_neighbors"],
        config["train"]["batch_size"],
        shuffle=True,
    )

    log_every = config["train"]["log_every"]
    geo_weight = config["train"]["geo_weight"]
    band_weight = config["train"]["band_weight"]
    hits_ks = config["eval"]["hits_ks"]

    step = 0
    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            geo_logits, band_logits = model(batch)
            seed_size = batch["event"].batch_size

            if args.band_obs_mask:
                masked_x = mask_band_feature(
                    batch["event"].x, args.band_feat_idx
                )
                _, band_logits = model(batch, event_x_override=masked_x)

            geo_labels = batch["event"].y_geo[:seed_size]
            band_labels = batch["event"].y_band[:seed_size]

            loss_geo = F.cross_entropy(geo_logits[:seed_size], geo_labels)
            loss_band = F.cross_entropy(band_logits[:seed_size], band_labels)
            loss = geo_weight * loss_geo + band_weight * loss_band

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                geo_acc, geo_mrr, geo_hits = classification_metrics(
                    geo_logits[:seed_size].detach().cpu(),
                    geo_labels.detach().cpu(),
                    ks=hits_ks,
                )
                band_acc, band_mrr, band_hits = classification_metrics(
                    band_logits[:seed_size].detach().cpu(),
                    band_labels.detach().cpu(),
                    ks=hits_ks,
                )
                print(
                    f"epoch {epoch} step {step} "
                    f"loss {loss.item():.4f} "
                    f"geo_acc {geo_acc:.3f} geo_mrr {geo_mrr:.3f} "
                    f"geo_hits {geo_hits} "
                    f"band_acc {band_acc:.3f} band_mrr {band_mrr:.3f} "
                    f"band_hits {band_hits}"
                )
            step += 1

        print(f"Epoch {epoch} completed.")

    save_dir = os.path.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), args.save)
    print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
