import argparse

import numpy as np
import torch


def _edge_stats(edge_index, edge_attr, split_nodes):
    num_edges = edge_index.size(1)
    avg_out = num_edges / max(1, split_nodes)
    if num_edges == 0:
        return {"edges": 0, "avg_out": avg_out, "dt_mean": 0.0, "dt_q": []}
    dt_vals = edge_attr.view(-1).cpu().numpy()
    dt_mean = float(dt_vals.mean())
    dt_q = [float(q) for q in np.quantile(dt_vals, [0.5, 0.9, 0.99])]
    return {"edges": num_edges, "avg_out": avg_out, "dt_mean": dt_mean, "dt_q": dt_q}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to pyg .pt file.")
    args = parser.parse_args()

    data = torch.load(args.data)
    event = data["event"]

    num_events = event.num_nodes
    is_true = event.is_true
    num_true = int(is_true.sum().item())
    num_fp = num_events - num_true

    train_mask = event.train_mask
    valid_mask = event.valid_mask
    test_mask = event.test_mask

    print(
        f"Events total {num_events} true {num_true} fp {num_fp} "
        f"train_true {int(train_mask.sum())} valid_true {int(valid_mask.sum())} "
        f"test_true {int(test_mask.sum())}"
    )

    raw_geo = event.raw_y_geo
    raw_band = event.raw_y_band

    assert torch.all(~is_true | (raw_geo >= 0)), "True events must have geo label."
    assert torch.all(~is_true | (raw_band >= 0)), "True events must have band label."

    for mask_name in ["train_mask", "valid_mask", "test_mask"]:
        mask = getattr(event, mask_name)
        assert torch.all(~mask | is_true), f"{mask_name} includes false events."
        assert torch.all(
            ~mask | (raw_geo >= 0)
        ), f"{mask_name} includes geo=-1."
        assert torch.all(
            ~mask | (raw_band >= 0)
        ), f"{mask_name} includes band=-1."

    for relation in data.edge_types:
        assert relation[1] not in (
            "occurs_in",
            "overlaps_band",
        ), "Leakage edge found in graph."

    split_id = event.split_id
    split_map = {"train": 0, "valid": 1, "test": 2}
    for split_name, split_val in split_map.items():
        edge_index = getattr(event, f"prev_event_{split_name}_edge_index")
        edge_attr = getattr(event, f"prev_event_{split_name}_edge_attr")
        if edge_index.numel() > 0:
            src = edge_index[0]
            dst = edge_index[1]
            assert torch.all(
                split_id[src] == split_val
            ), f"{split_name} prev_event crosses split (src)."
            assert torch.all(
                split_id[dst] == split_val
            ), f"{split_name} prev_event crosses split (dst)."

        split_nodes = int((split_id == split_val).sum().item())
        stats = _edge_stats(edge_index, edge_attr, split_nodes)
        print(
            f"{split_name} prev_event edges {stats['edges']} avg_out "
            f"{stats['avg_out']:.3f} dt_mean {stats['dt_mean']:.4f} "
            f"dt_q {stats['dt_q']}"
        )

    num_geocell = data["geocell"].num_nodes
    num_band = data["band"].num_nodes
    assert torch.all(
        event.y_geo[train_mask] < num_geocell
    ), "Geo label out of range."
    assert torch.all(
        event.y_band[train_mask] < num_band
    ), "Band label out of range."
    print(f"Label ranges ok: geocell {num_geocell} band {num_band}")


if __name__ == "__main__":
    main()
