import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear

class AMLGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
            add_self_loops=False,
            normalize=False
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels // 2,
            add_self_loops=False,
            normalize=False
        )

        self.lin = Linear(hidden_channels // 2, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        x = self.lin(x)
        return x
