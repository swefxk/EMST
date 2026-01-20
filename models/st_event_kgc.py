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

        self.time_mlp = nn.Sequential(
            nn.Linear(1, te_dim),
            nn.ReLU(),
            nn.Linear(te_dim, te_dim),
        )

        self.convs = nn.ModuleList()
        for _ in range(2):
            self.convs.append(
                HeteroConv(
                    {
                        ("sensor", "rev_observed_by", "event"): SAGEConv(
                            (hidden_dim, hidden_dim), hidden_dim
                        ),
                        ("sensor", "located_in", "geocell"): SAGEConv(
                            (hidden_dim, hidden_dim), hidden_dim
                        ),
                        ("geocell", "rev_located_in", "sensor"): SAGEConv(
                            (hidden_dim, hidden_dim), hidden_dim
                        ),
                        ("event", "prev_event", "event"): TransformerConv(
                            hidden_dim,
                            hidden_dim // heads,
                            heads=heads,
                            edge_dim=te_dim,
                            dropout=dropout,
                        ),
                    },
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

    def forward(self, data):
        x_dict = {}
        for node_type in self.node_lin:
            x_dict[node_type] = self.node_lin[node_type](data[node_type].x)

        edge_index_dict = data.edge_index_dict
        edge_attr_dict = {}
        if ("event", "prev_event", "event") in data.edge_types:
            edge_attr = data["event", "prev_event", "event"].edge_attr
            edge_attr_dict[("event", "prev_event", "event")] = self.time_mlp(edge_attr)

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
