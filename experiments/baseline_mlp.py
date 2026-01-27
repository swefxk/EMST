import argparse
import os
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import classification_metrics, load_config, seed_everything, save_json


class EventMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_geocell, num_band, dropout):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.geo_head = nn.Linear(hidden_dim, num_geocell)
        self.band_head = nn.Linear(hidden_dim, num_band)

    def forward(self, x):
        h = self.backbone(x)
        return self.geo_head(h), self.band_head(h)


def build_loader(event_x, y_geo, y_band, batch_size, shuffle=True):
    dataset = TensorDataset(event_x, y_geo, y_band)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def eval_split(model, event_x, y_geo, y_band, mask, ks, device):
    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return {"geo": {}, "band": {}}
    x = event_x[idx].to(device)
    labels_geo = y_geo[idx].to(device)
    labels_band = y_band[idx].to(device)
    with torch.no_grad():
        geo_logits, band_logits = model(x)
    geo_acc, geo_mrr, geo_hits = classification_metrics(
        geo_logits.cpu(), labels_geo.cpu(), ks=ks
    )
    band_acc, band_mrr, band_hits = classification_metrics(
        band_logits.cpu(), labels_band.cpu(), ks=ks
    )
    return {
        "geo": {
            "acc": geo_acc,
            "mrr": geo_mrr,
            "hits": geo_hits,
            "hits10": geo_hits.get(10, 0.0),
        },
        "band": {
            "acc": band_acc,
            "mrr": band_mrr,
            "hits": band_hits,
            "hits10": band_hits.get(10, 0.0),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to pyg .pt file.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--ckpt",
        default="checkpoints/baseline_mlp.pt",
        help="Checkpoint path (load if exists, else train and save).",
    )
    parser.add_argument(
        "--force_train",
        action="store_true",
        help="Force training even if checkpoint exists.",
    )
    parser.add_argument(
        "--out_metrics",
        default="stats/baseline_mlp_metrics.json",
        help="Path to save metrics JSON.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["seed"] if args.seed is None else args.seed
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.data)
    event_x = data["event"].x
    y_geo = data["event"].y_geo
    y_band = data["event"].y_band

    model = EventMLP(
        in_dim=event_x.size(-1),
        hidden_dim=config["model"]["hidden_dim"],
        num_geocell=data["geocell"].num_nodes,
        num_band=data["band"].num_nodes,
        dropout=config["model"]["dropout"],
    ).to(device)

    ckpt_exists = os.path.isfile(args.ckpt)
    if ckpt_exists and not args.force_train:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded checkpoint from {args.ckpt}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"],
        )
        loader = build_loader(
            event_x[data["event"].train_mask],
            y_geo[data["event"].train_mask],
            y_band[data["event"].train_mask],
            batch_size=config["train"]["batch_size"],
            shuffle=True,
        )
        geo_weight = config["train"]["geo_weight"]
        band_weight = config["train"]["band_weight"]
        epochs = config["train"]["epochs"]
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for batch_x, batch_geo, batch_band in loader:
                batch_x = batch_x.to(device)
                batch_geo = batch_geo.to(device)
                batch_band = batch_band.to(device)
                geo_logits, band_logits = model(batch_x)
                loss_geo = F.cross_entropy(geo_logits, batch_geo)
                loss_band = F.cross_entropy(band_logits, batch_band)
                loss = geo_weight * loss_geo + band_weight * loss_band
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(1, len(loader))
            print(f"epoch {epoch} loss {avg_loss:.4f}")

        ckpt_dir = os.path.dirname(args.ckpt)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), args.ckpt)
        print(f"Saved checkpoint to {args.ckpt}")

    hits_ks = config["eval"]["hits_ks"]
    metrics_payload = {"baseline": "event_x_mlp", "splits": {}}
    model.eval()
    for split_name, mask in [
        ("valid", data["event"].valid_mask),
        ("test", data["event"].test_mask),
    ]:
        result = eval_split(model, event_x, y_geo, y_band, mask, hits_ks, device)
        print(
            f"{split_name} mlp geo_acc {result['geo'].get('acc', 0):.3f} "
            f"geo_mrr {result['geo'].get('mrr', 0):.3f} geo_hits {result['geo'].get('hits', {})}"
        )
        print(
            f"{split_name} mlp band_acc {result['band'].get('acc', 0):.3f} "
            f"band_mrr {result['band'].get('mrr', 0):.3f} band_hits {result['band'].get('hits', {})}"
        )
        metrics_payload["splits"][split_name] = {"rank": {"tag": "mlp", **result}}

    if args.out_metrics:
        out_dir = os.path.dirname(args.out_metrics)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_json(args.out_metrics, metrics_payload)
        print(f"Wrote {args.out_metrics}")


if __name__ == "__main__":
    main()
