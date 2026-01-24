import json
import os
import random

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def classification_metrics(logits, labels, ks=(1, 3, 10)):
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        acc = (pred == labels).float().mean().item() if labels.numel() else 0.0

        sorted_idx = torch.argsort(probs, dim=-1, descending=True)
        ranks = (sorted_idx == labels.unsqueeze(-1)).nonzero(as_tuple=False)
        rank_pos = ranks[:, 1].float() + 1.0 if ranks.numel() else torch.tensor([])
        mrr = (1.0 / rank_pos).mean().item() if rank_pos.numel() else 0.0

        hits = {}
        for k in ks:
            if rank_pos.numel():
                hits[k] = (rank_pos <= k).float().mean().item()
            else:
                hits[k] = 0.0
        return acc, mrr, hits


def expected_calibration_error(probs, labels, n_bins=15):
    if probs.numel() == 0:
        return 0.0
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.sum() > 0:
            acc_bin = accuracies[mask].float().mean()
            conf_bin = confidences[mask].mean()
            ece += (mask.float().mean() * torch.abs(acc_bin - conf_bin))
    return ece.item()


def brier_score(probs, labels, num_classes):
    if probs.numel() == 0:
        return 0.0
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    return torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item()


def risk_coverage_curve(probs, labels, uncertainty, num_points=10):
    if probs.numel() == 0:
        return []
    order = torch.argsort(uncertainty, dim=0)
    sorted_labels = labels[order]
    sorted_probs = probs[order]
    confidences, preds = torch.max(sorted_probs, dim=1)
    correct = preds.eq(sorted_labels).float()

    results = []
    n = labels.numel()
    for i in range(1, num_points + 1):
        cutoff = int(n * i / num_points)
        if cutoff == 0:
            results.append({"coverage": i / num_points, "accuracy": 0.0})
            continue
        acc = correct[:cutoff].mean().item()
        results.append({"coverage": cutoff / n, "accuracy": acc})
    return results


def set_prev_event_edges(data, split, mode="on"):
    key_index = f"prev_event_{split}_edge_index"
    key_attr = f"prev_event_{split}_edge_attr"
    key_log_mean = f"prev_event_{split}_dt_log_mean"
    key_log_std = f"prev_event_{split}_dt_log_std"
    if hasattr(data["event"], key_index) and hasattr(data["event"], key_attr):
        data["event", "prev_event", "event"].edge_index = getattr(
            data["event"], key_index
        )
        data["event", "prev_event", "event"].edge_attr = getattr(
            data["event"], key_attr
        )
        if hasattr(data["event"], key_log_mean) and hasattr(data["event"], key_log_std):
            data["event"].prev_event_dt_log_mean = getattr(data["event"], key_log_mean)
            data["event"].prev_event_dt_log_std = getattr(data["event"], key_log_std)
        if mode == "off":
            data["event", "prev_event", "event"].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )
            data["event", "prev_event", "event"].edge_attr = torch.empty(
                (0, 1), dtype=torch.float32
            )
        elif mode == "zero_dt":
            edge_attr = data["event", "prev_event", "event"].edge_attr
            data["event", "prev_event", "event"].edge_attr = torch.zeros_like(edge_attr)
        return True
    return False
