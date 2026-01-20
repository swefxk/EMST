import argparse
import json

import torch
from torch.nn import functional as F
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


def mask_band_feature(event_x, band_feat_idx):
    masked = event_x.clone()
    masked[:, band_feat_idx] = 0.0
    return masked


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
    logits_geo_list = []
    logits_band_list = []
    labels_geo_list = []
    labels_band_list = []
    model.eval()

    for batch in loader:
        batch = batch.to(device)
        seed_size = batch["event"].batch_size
        labels_geo = batch["event"].y_geo[:seed_size]
        labels_band = batch["event"].y_band[:seed_size]

        with torch.no_grad():
            if mask_band:
                masked_x = mask_band_feature(
                    batch["event"].x, band_feat_idx
                )
                geo_logits, band_logits = model(
                    batch, event_x_override=masked_x
                )
            else:
                geo_logits, band_logits = model(batch)
            geo_logits = geo_logits[:seed_size]
            band_logits = band_logits[:seed_size]

        logits_geo_list.append(geo_logits.cpu())
        logits_band_list.append(band_logits.cpu())
        labels_geo_list.append(labels_geo.cpu())
        labels_band_list.append(labels_band.cpu())

    return {
        "logits_geo": torch.cat(logits_geo_list, dim=0),
        "logits_band": torch.cat(logits_band_list, dim=0),
        "labels_geo": torch.cat(labels_geo_list, dim=0),
        "labels_band": torch.cat(labels_band_list, dim=0),
    }


def predict_mc_probs(
    data,
    model,
    mask,
    num_neighbors,
    batch_size,
    device,
    temperature=None,
    mc_dropout=20,
    mask_band=False,
    band_feat_idx=1,
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

    model.train()

    t_geo = temperature["geo"] if temperature else 1.0
    t_band = temperature["band"] if temperature else 1.0

    for batch in loader:
        batch = batch.to(device)
        seed_size = batch["event"].batch_size
        labels_geo = batch["event"].y_geo[:seed_size]
        labels_band = batch["event"].y_band[:seed_size]

        mc_logits_geo = []
        mc_logits_band = []
        with torch.no_grad():
            for _ in range(mc_dropout):
                if mask_band:
                    masked_x = mask_band_feature(
                        batch["event"].x, band_feat_idx
                    )
                    geo_logits, band_logits = model(
                        batch, event_x_override=masked_x
                    )
                else:
                    geo_logits, band_logits = model(batch)
                mc_logits_geo.append(geo_logits[:seed_size])
                mc_logits_band.append(band_logits[:seed_size])

        mc_logits_geo = torch.stack(mc_logits_geo, dim=0)
        mc_logits_band = torch.stack(mc_logits_band, dim=0)

        mean_logits_geo = mc_logits_geo.mean(dim=0) / t_geo
        mean_logits_band = mc_logits_band.mean(dim=0) / t_band
        probs_geo = torch.softmax(mean_logits_geo, dim=-1)
        probs_band = torch.softmax(mean_logits_band, dim=-1)

        mc_probs_geo = torch.softmax(mc_logits_geo / t_geo, dim=-1)
        mc_probs_band = torch.softmax(mc_logits_band / t_band, dim=-1)
        var_geo = mc_probs_geo.var(dim=0).mean(dim=1)
        var_band = mc_probs_band.var(dim=0).mean(dim=1)

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


def probs_from_logits(logits, temperature=None):
    if temperature:
        logits = logits / temperature
    return torch.softmax(logits, dim=-1)


def nll_from_probs(probs, labels):
    if labels.numel() == 0:
        return 0.0
    log_probs = torch.log(probs + 1e-9)
    return F.nll_loss(log_probs, labels).item()


def summarize_risk_coverage(rc_list, targets=(1.0, 0.8, 0.6, 0.4)):
    if not rc_list:
        return {}
    summary = {}
    for target in targets:
        closest = min(rc_list, key=lambda r: abs(r["coverage"] - target))
        summary[target] = closest["accuracy"]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to pyg .pt file.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument("--calib", default=None, help="Calibration json path.")
    parser.add_argument("--mc_dropout", type=int, default=0)
    parser.add_argument(
        "--rank_from_mc",
        action="store_true",
        help="Use MC-averaged probabilities for ranking metrics.",
    )
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
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["seed"] if args.seed is None else args.seed
    seed_everything(seed)

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
        set_prev_event_edges(data, split_name, mode=args.prev_event)

        det_outputs = collect_logits(
            data, model, mask, num_neighbors, batch_size, device
        )
        if args.band_obs_mask:
            det_band_outputs = collect_logits(
                data,
                model,
                mask,
                num_neighbors,
                batch_size,
                device,
                mask_band=True,
                band_feat_idx=args.band_feat_idx,
            )
        else:
            det_band_outputs = det_outputs
        det_probs_geo = probs_from_logits(
            det_outputs["logits_geo"],
            temperature["geo"] if temperature else None,
        )
        det_probs_band = probs_from_logits(
            det_band_outputs["logits_band"],
            temperature["band"] if temperature else None,
        )

        if args.mc_dropout > 0:
            mc_outputs = predict_mc_probs(
                data,
                model,
                mask,
                num_neighbors,
                batch_size,
                device,
                temperature=temperature,
                mc_dropout=args.mc_dropout,
            )
            if args.band_obs_mask:
                mc_band_outputs = predict_mc_probs(
                    data,
                    model,
                    mask,
                    num_neighbors,
                    batch_size,
                    device,
                    temperature=temperature,
                    mc_dropout=args.mc_dropout,
                    mask_band=True,
                    band_feat_idx=args.band_feat_idx,
                )
            else:
                mc_band_outputs = mc_outputs
            calib_probs_geo = mc_outputs["probs_geo"]
            calib_probs_band = mc_band_outputs["probs_band"]
            calib_labels_geo = mc_outputs["labels_geo"]
            calib_labels_band = mc_band_outputs["labels_band"]
            unc_geo = mc_outputs["unc_geo"]
            unc_band = mc_band_outputs["unc_band"]
            var_geo = mc_outputs["var_geo"]
            var_band = mc_band_outputs["var_band"]
            calib_tag = "mc"
        else:
            calib_probs_geo = det_probs_geo
            calib_probs_band = det_probs_band
            calib_labels_geo = det_outputs["labels_geo"]
            calib_labels_band = det_band_outputs["labels_band"]
            unc_geo = torch.zeros(calib_probs_geo.size(0))
            unc_band = torch.zeros(calib_probs_band.size(0))
            var_geo = torch.zeros(calib_probs_geo.size(0))
            var_band = torch.zeros(calib_probs_band.size(0))
            calib_tag = "det"

        rank_probs_geo = calib_probs_geo if args.rank_from_mc else det_probs_geo
        rank_probs_band = calib_probs_band if args.rank_from_mc else det_probs_band
        rank_labels_geo = det_outputs["labels_geo"]
        rank_labels_band = det_band_outputs["labels_band"]
        rank_tag = "mc" if args.rank_from_mc else "det"

        geo_acc, geo_mrr, geo_hits = metrics_from_probs(
            rank_probs_geo, rank_labels_geo, hits_ks
        )
        band_acc, band_mrr, band_hits = metrics_from_probs(
            rank_probs_band, rank_labels_band, hits_ks
        )

        geo_nll = nll_from_probs(calib_probs_geo, calib_labels_geo)
        band_nll = nll_from_probs(calib_probs_band, calib_labels_band)

        geo_ece = expected_calibration_error(calib_probs_geo, calib_labels_geo)
        band_ece = expected_calibration_error(calib_probs_band, calib_labels_band)

        geo_brier = brier_score(
            calib_probs_geo, calib_labels_geo, data["geocell"].num_nodes
        )
        band_brier = brier_score(
            calib_probs_band, calib_labels_band, data["band"].num_nodes
        )

        geo_rc = risk_coverage_curve(calib_probs_geo, calib_labels_geo, unc_geo)
        band_rc = risk_coverage_curve(calib_probs_band, calib_labels_band, unc_band)
        geo_rc_summary = summarize_risk_coverage(geo_rc)
        band_rc_summary = summarize_risk_coverage(band_rc)

        print(
            f"{split_name} rank[{rank_tag}] "
            f"geo_acc {geo_acc:.3f} geo_mrr {geo_mrr:.3f} geo_hits {geo_hits}"
        )
        print(
            f"{split_name} rank[{rank_tag}] "
            f"band_acc {band_acc:.3f} band_mrr {band_mrr:.3f} band_hits {band_hits}"
        )
        print(
            f"{split_name} calib[{calib_tag}] "
            f"geo_nll {geo_nll:.4f} geo_ece {geo_ece:.4f} geo_brier {geo_brier:.4f}"
        )
        print(
            f"{split_name} calib[{calib_tag}] "
            f"band_nll {band_nll:.4f} band_ece {band_ece:.4f} band_brier {band_brier:.4f}"
        )
        print(f"{split_name} geo_unc_var_mean {var_geo.mean().item():.6f}")
        print(f"{split_name} band_unc_var_mean {var_band.mean().item():.6f}")
        print(f"{split_name} geo_risk_coverage {geo_rc}")
        print(f"{split_name} band_risk_coverage {band_rc}")
        print(f"{split_name} geo_rc_summary {geo_rc_summary}")
        print(f"{split_name} band_rc_summary {band_rc_summary}")


if __name__ == "__main__":
    main()
