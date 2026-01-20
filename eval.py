import argparse
import json

import torch
from torch_geometric.loader import NeighborLoader

from models.st_event_kgc import STEventKGC
from utils import (
    brier_score,
    expected_calibration_error,
    load_config,
    risk_coverage_curve,
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


def metrics_from_probs(probs, labels, ks):
    preds = probs.argmax(dim=1)
    acc = (preds == labels).float().mean().item() if labels.numel() else 0.0
    sorted_idx = torch.argsort(probs, dim=-1, descending=True)
    ranks = (sorted_idx == labels.unsqueeze(-1)).nonzero(as_tuple=False)
    rank_pos = ranks[:, 1].float() + 1.0 if ranks.numel() else torch.tensor([])
    mrr = (1.0 / rank_pos).mean().item() if rank_pos.numel() else 0.0
    hits = {}
    for k in ks:
        hits[k] = (rank_pos <= k).float().mean().item() if rank_pos.numel() else 0.0
    return acc, mrr, hits


def predict_probs(
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
    probs_geo_list = []
    probs_band_list = []
    labels_geo_list = []
    labels_band_list = []
    uncertainty_geo_list = []
    uncertainty_band_list = []
    var_geo_list = []
    var_band_list = []

    if mc_dropout > 0:
        model.train()
    else:
        model.eval()

    for batch in loader:
        batch = batch.to(device)
        seed_size = batch["event"].batch_size
        labels_geo = batch["event"].y_geo[:seed_size]
        labels_band = batch["event"].y_band[:seed_size]

        if mc_dropout > 0:
            mc_probs_geo = []
            mc_probs_band = []
            with torch.no_grad():
                for _ in range(mc_dropout):
                    geo_logits, band_logits = model(batch)
                    geo_logits = geo_logits[:seed_size]
                    band_logits = band_logits[:seed_size]
                    if temperature:
                        geo_logits = geo_logits / temperature["geo"]
                        band_logits = band_logits / temperature["band"]
                    mc_probs_geo.append(torch.softmax(geo_logits, dim=-1))
                    mc_probs_band.append(torch.softmax(band_logits, dim=-1))

            probs_geo = torch.stack(mc_probs_geo, dim=0).mean(dim=0)
            probs_band = torch.stack(mc_probs_band, dim=0).mean(dim=0)
            var_geo = torch.stack(mc_probs_geo, dim=0).var(dim=0).mean(dim=1)
            var_band = torch.stack(mc_probs_band, dim=0).var(dim=0).mean(dim=1)
        else:
            with torch.no_grad():
                geo_logits, band_logits = model(batch)
                geo_logits = geo_logits[:seed_size]
                band_logits = band_logits[:seed_size]
                if temperature:
                    geo_logits = geo_logits / temperature["geo"]
                    band_logits = band_logits / temperature["band"]
                probs_geo = torch.softmax(geo_logits, dim=-1)
                probs_band = torch.softmax(band_logits, dim=-1)
                var_geo = torch.zeros(probs_geo.size(0), device=probs_geo.device)
                var_band = torch.zeros(probs_band.size(0), device=probs_band.device)

        entropy_geo = -(probs_geo * (probs_geo + 1e-9).log()).sum(dim=1)
        entropy_band = -(probs_band * (probs_band + 1e-9).log()).sum(dim=1)

        probs_geo_list.append(probs_geo.cpu())
        probs_band_list.append(probs_band.cpu())
        labels_geo_list.append(labels_geo.cpu())
        labels_band_list.append(labels_band.cpu())
        uncertainty_geo_list.append(entropy_geo.cpu())
        uncertainty_band_list.append(entropy_band.cpu())
        var_geo_list.append(var_geo.cpu())
        var_band_list.append(var_band.cpu())

    return {
        "probs_geo": torch.cat(probs_geo_list, dim=0),
        "probs_band": torch.cat(probs_band_list, dim=0),
        "labels_geo": torch.cat(labels_geo_list, dim=0),
        "labels_band": torch.cat(labels_band_list, dim=0),
        "unc_geo": torch.cat(uncertainty_geo_list, dim=0),
        "unc_band": torch.cat(uncertainty_band_list, dim=0),
        "var_geo": torch.cat(var_geo_list, dim=0),
        "var_band": torch.cat(var_band_list, dim=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to pyg .pt file.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--calib", default=None, help="Calibration json path.")
    parser.add_argument("--mc_dropout", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.data)

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

    temperature = None
    if args.calib:
        with open(args.calib, "r", encoding="utf-8") as f:
            calib = json.load(f)
        temperature = {"geo": calib["geo_temperature"], "band": calib["band_temperature"]}

    num_neighbors = config["train"]["num_neighbors"]
    batch_size = args.batch_size or config["train"]["batch_size"]
    hits_ks = config["eval"]["hits_ks"]

    for split_name, mask in [
        ("valid", data["event"].valid_mask),
        ("test", data["event"].test_mask),
    ]:
        set_prev_event_edges(data, split_name)
        outputs = predict_probs(
            data,
            model,
            mask,
            num_neighbors,
            batch_size,
            device,
            temperature=temperature,
            mc_dropout=args.mc_dropout,
        )

        geo_acc, geo_mrr, geo_hits = metrics_from_probs(
            outputs["probs_geo"], outputs["labels_geo"], hits_ks
        )
        band_acc, band_mrr, band_hits = metrics_from_probs(
            outputs["probs_band"], outputs["labels_band"], hits_ks
        )

        geo_ece = expected_calibration_error(
            outputs["probs_geo"], outputs["labels_geo"]
        )
        band_ece = expected_calibration_error(
            outputs["probs_band"], outputs["labels_band"]
        )

        geo_brier = brier_score(
            outputs["probs_geo"], outputs["labels_geo"], data["geocell"].num_nodes
        )
        band_brier = brier_score(
            outputs["probs_band"], outputs["labels_band"], data["band"].num_nodes
        )

        geo_rc = risk_coverage_curve(
            outputs["probs_geo"], outputs["labels_geo"], outputs["unc_geo"]
        )
        band_rc = risk_coverage_curve(
            outputs["probs_band"], outputs["labels_band"], outputs["unc_band"]
        )

        print(f"{split_name} geo_acc {geo_acc:.3f} geo_mrr {geo_mrr:.3f} geo_hits {geo_hits}")
        print(
            f"{split_name} band_acc {band_acc:.3f} band_mrr {band_mrr:.3f} band_hits {band_hits}"
        )
        print(f"{split_name} geo_ece {geo_ece:.4f} geo_brier {geo_brier:.4f}")
        print(f"{split_name} band_ece {band_ece:.4f} band_brier {band_brier:.4f}")
        print(f"{split_name} geo_unc_var_mean {outputs['var_geo'].mean().item():.6f}")
        print(f"{split_name} band_unc_var_mean {outputs['var_band'].mean().item():.6f}")
        print(f"{split_name} geo_risk_coverage {geo_rc}")
        print(f"{split_name} band_risk_coverage {band_rc}")


if __name__ == "__main__":
    main()
