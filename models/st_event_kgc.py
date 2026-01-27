import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, TransformerConv


class STEventKGC(nn.Module):
    def __init__(
        self,
        in_dims,
        num_geocell,
        num_band,
        hidden_dim=128,
        te_dim=16,
        heads=2,
        dropout=0.2,
        dt_encoding="raw",
        dt_freqs=None,
        edge_types=None,
        motion_gate=False,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")

        self.hidden_dim = hidden_dim
        self.te_dim = te_dim
        self.heads = heads
        self.dropout = dropout

        self.node_lin = nn.ModuleDict()
        for node_type, in_dim in in_dims.items():
            self.node_lin[node_type] = nn.Linear(in_dim, hidden_dim)

        self.dt_encoding = dt_encoding
        self.motion_gate = motion_gate
        self.edge_types = set(edge_types) if edge_types is not None else None
        if dt_freqs is None:
            dt_freqs = [1, 2, 4, 8]
        self.register_buffer(
            "dt_freqs",
            torch.tensor(dt_freqs, dtype=torch.float32).view(1, -1),
            persistent=False,
        )
        dt_in_dim = 1 if dt_encoding in ("raw", "log") else 2 * len(dt_freqs)
        self.time_mlp = nn.Sequential(
            nn.Linear(dt_in_dim, te_dim),
            nn.ReLU(),
            nn.Linear(te_dim, te_dim),
        )
        if self.motion_gate:
            self.dt_gate = nn.Sequential(nn.Linear(dt_in_dim, 1), nn.Sigmoid())
        else:
            self.dt_gate = None

        self.convs = nn.ModuleList()
        for _ in range(2):
            rel_convs = {}
            if self.edge_types is None or ("sensor", "rev_observed_by", "event") in self.edge_types:
                rel_convs[("sensor", "rev_observed_by", "event")] = SAGEConv(
                    (hidden_dim, hidden_dim), hidden_dim
                )
            if self.edge_types is None or ("sensor", "located_in", "geocell") in self.edge_types:
                rel_convs[("sensor", "located_in", "geocell")] = SAGEConv(
                    (hidden_dim, hidden_dim), hidden_dim
                )
            if self.edge_types is None or ("geocell", "rev_located_in", "sensor") in self.edge_types:
                rel_convs[("geocell", "rev_located_in", "sensor")] = SAGEConv(
                    (hidden_dim, hidden_dim), hidden_dim
                )
            if self.edge_types is None or ("source", "rev_from_source", "event") in self.edge_types:
                rel_convs[("source", "rev_from_source", "event")] = SAGEConv(
                    (hidden_dim, hidden_dim), hidden_dim
                )
            if self.edge_types is None or ("event", "from_source", "source") in self.edge_types:
                rel_convs[("event", "from_source", "source")] = SAGEConv(
                    (hidden_dim, hidden_dim), hidden_dim
                )
            if self.edge_types is None or ("event", "prev_event", "event") in self.edge_types:
                rel_convs[("event", "prev_event", "event")] = TransformerConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    edge_dim=te_dim,
                    dropout=dropout,
                )
            self.convs.append(
                HeteroConv(
                    rel_convs,
                    aggr="sum",
                )
            )

        self.geo_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_geocell),
        )
        self.band_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_band),
        )

    def forward(self, data, event_x_override=None):
        x_dict = {}
        for node_type in self.node_lin:
            if node_type == "event" and event_x_override is not None:
                x_in = event_x_override
            else:
                x_in = data[node_type].x
            x_dict[node_type] = self.node_lin[node_type](x_in)

        edge_index_dict = data.edge_index_dict
        edge_attr_dict = {}
        if ("event", "prev_event", "event") in data.edge_types:
            edge_attr = data["event", "prev_event", "event"].edge_attr
            if self.dt_encoding == "log":
                dt_feat = torch.log1p(edge_attr)
                mean = getattr(data["event"], "prev_event_dt_log_mean", None)
                std = getattr(data["event"], "prev_event_dt_log_std", None)
                if mean is not None and std is not None:
                    mean = mean.to(edge_attr.device)
                    std = std.to(edge_attr.device)
                    dt_feat = (dt_feat - mean) / std
            elif self.dt_encoding == "sincos":
                angles = edge_attr * (2 * torch.pi) * self.dt_freqs.to(edge_attr.device)
                dt_feat = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
            else:
                dt_feat = edge_attr
            encoded = self.time_mlp(dt_feat)
            if self.dt_gate is not None:
                encoded = encoded * self.dt_gate(dt_feat)
            edge_attr_dict[("event", "prev_event", "event")] = encoded

        for conv in self.convs:
            out_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            for node_type, out in out_dict.items():
                out = F.relu(out)
                out = F.dropout(out, p=self.dropout, training=self.training)
                out_dict[node_type] = out
            for node_type in x_dict:
                if node_type not in out_dict:
                    out_dict[node_type] = x_dict[node_type]
            x_dict = out_dict

        z_event = x_dict["event"]
        return self.geo_head(z_event), self.band_head(z_event)
