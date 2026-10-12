"""Stoeckl-inspired edge/node blocks with mean or learned attention aggregation."""
import torch
from torch import nn
from torch_geometric.utils import scatter, softmax


def _mlp(inputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.ELU(), nn.Linear(width, width),
                         nn.ELU(), nn.Linear(width, width))


class RelationshipConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim, heads, *, attention=False):
        super().__init__()
        self.heads = heads
        self.width = out_channels
        self.edge_mlp = _mlp(2 * in_channels + edge_dim, out_channels)
        self.node_mlp = _mlp(in_channels + 2 * out_channels, out_channels)
        self.in_attention = nn.Linear(out_channels, heads) if attention else None
        self.out_attention = nn.Linear(out_channels, heads) if attention else None

    def aggregate(self, messages, index, count, scorer):
        values = messages.reshape(-1, self.heads, self.width // self.heads)
        if scorer is None:
            result = scatter(values, index, dim=0, dim_size=count, reduce="mean")
        else:
            weights = softmax(scorer(messages), index, num_nodes=count)
            result = scatter(values * weights.unsqueeze(-1), index, dim=0,
                             dim_size=count, reduce="sum")
        return result.reshape(count, self.width)

    def forward(self, x, edge_index, edge_attr):
        sender, receiver = edge_index
        messages = self.edge_mlp(torch.cat((x[sender], x[receiver], edge_attr), dim=-1))
        incoming = self.aggregate(messages, receiver, len(x), self.in_attention)
        outgoing = self.aggregate(messages, sender, len(x), self.out_attention)
        return self.node_mlp(torch.cat((x, incoming, outgoing), dim=-1))
