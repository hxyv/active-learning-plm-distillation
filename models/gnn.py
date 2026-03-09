"""GNN student model definitions."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, SAGEConv


class NodeFeatureEncoder(nn.Module):
    def __init__(
        self,
        aa_vocab_size: int,
        atom_vocab_size: int,
        aa_emb_dim: int,
        atom_emb_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.aa_emb = nn.Embedding(aa_vocab_size, aa_emb_dim)
        self.atom_emb = nn.Embedding(atom_vocab_size, atom_emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(aa_emb_dim + atom_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, aa_idx: torch.Tensor, atom_idx: torch.Tensor) -> torch.Tensor:
        aa = self.aa_emb(aa_idx)
        atom = self.atom_emb(atom_idx)
        x = torch.cat([aa, atom], dim=-1)
        return self.proj(x)


class SimpleDistillGNN(nn.Module):
    def __init__(
        self,
        aa_vocab_size: int = 21,
        atom_vocab_size: int = 3,
        aa_emb_dim: int = 32,
        atom_emb_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        out_dim: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = NodeFeatureEncoder(
            aa_vocab_size=aa_vocab_size,
            atom_vocab_size=atom_vocab_size,
            aa_emb_dim=aa_emb_dim,
            atom_emb_dim=atom_emb_dim,
            hidden_dim=hidden_dim,
        )

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, normalize=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, data) -> torch.Tensor:
        h = self.encoder(data.aa_idx, data.atom_idx)
        for conv, norm in zip(self.layers, self.norms):
            h_new = conv(h, data.edge_index)
            h = norm(h + h_new)
            h = torch.relu(h)
            h = self.dropout(h)
        return self.head(h)


class GeoDistillGNN(nn.Module):
    def __init__(
        self,
        aa_vocab_size: int = 21,
        atom_vocab_size: int = 3,
        aa_emb_dim: int = 32,
        atom_emb_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        out_dim: int = 8,
        edge_dim: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = NodeFeatureEncoder(
            aa_vocab_size=aa_vocab_size,
            atom_vocab_size=atom_vocab_size,
            aa_emb_dim=aa_emb_dim,
            atom_emb_dim=atom_emb_dim,
            hidden_dim=hidden_dim,
        )

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINEConv(nn=mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, data) -> torch.Tensor:
        h = self.encoder(data.aa_idx, data.atom_idx)
        edge_attr = data.edge_attr
        for conv, norm in zip(self.layers, self.norms):
            h_new = conv(h, data.edge_index, edge_attr=edge_attr)
            h = norm(h + h_new)
            h = torch.relu(h)
            h = self.dropout(h)
        return self.head(h)
