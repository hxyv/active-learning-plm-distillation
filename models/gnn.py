"""GNN student model definitions."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, SAGEConv

from data.constants import AA1_TO_INDEX


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


class DualRangeDistillGNN(nn.Module):
    """
    Paper-aligned approximation:
    - residue embedding + backbone-atom embedding (16 + 16 by default)
    - two message passing blocks
    - short-range all-atom branch and long-range CA-only branch
    - three-layer output MLP producing 8 SS8 logits

    This is not a full SAKE+SchNet reimplementation, but the settings and
    data flow match the paper protocol more closely than generic baselines.
    """

    def __init__(
        self,
        aa_vocab_size: int = 21,
        atom_vocab_size: int = 3,
        aa_emb_dim: int = 16,
        atom_emb_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_dim: int = 8,
        edge_dim: int = 4,
        short_cut_min: float = 0.0,
        short_cut_max: float = 10.0,
        long_cut_min: float = 0.0,
        long_cut_max: float = 25.0,
        ca_atom_index: int = 1,
        head_hidden_dims: Sequence[int] = (16, 8),
    ) -> None:
        super().__init__()
        self.encoder = NodeFeatureEncoder(
            aa_vocab_size=aa_vocab_size,
            atom_vocab_size=atom_vocab_size,
            aa_emb_dim=aa_emb_dim,
            atom_emb_dim=atom_emb_dim,
            hidden_dim=hidden_dim,
        )

        self.short_cut_min = float(short_cut_min)
        self.short_cut_max = float(short_cut_max)
        self.long_cut_min = float(long_cut_min)
        self.long_cut_max = float(long_cut_max)
        self.ca_atom_index = int(ca_atom_index)

        self.short_layers = nn.ModuleList()
        self.long_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            short_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            long_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.short_layers.append(GINEConv(nn=short_mlp, edge_dim=edge_dim, train_eps=True))
            self.long_layers.append(GINEConv(nn=long_mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        mlp_dims = [hidden_dim] + [int(d) for d in head_hidden_dims] + [out_dim]
        layers = []
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                layers.append(nn.ReLU())
        self.head = nn.Sequential(*layers)

    def _select_edges(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, mask: torch.Tensor):
        if mask.numel() == 0 or not bool(mask.any()):
            return edge_index[:, :0], edge_attr[:0]
        return edge_index[:, mask], edge_attr[mask]

    def forward(self, data) -> torch.Tensor:
        h = self.encoder(data.aa_idx, data.atom_idx)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        if edge_attr.numel() == 0:
            return self.head(h)

        dist = edge_attr[:, 0]
        src, dst = edge_index
        ca_mask = data.atom_idx == self.ca_atom_index

        short_mask = (dist >= self.short_cut_min) & (dist < self.short_cut_max)
        long_mask = (
            (dist >= self.long_cut_min)
            & (dist <= self.long_cut_max)
            & ca_mask[src]
            & ca_mask[dst]
        )

        short_edge_index, short_edge_attr = self._select_edges(edge_index, edge_attr, short_mask)
        long_edge_index, long_edge_attr = self._select_edges(edge_index, edge_attr, long_mask)

        for short_conv, long_conv, norm in zip(self.short_layers, self.long_layers, self.norms):
            h_short = (
                short_conv(h, short_edge_index, edge_attr=short_edge_attr)
                if short_edge_index.numel() > 0
                else torch.zeros_like(h)
            )
            h_long = (
                long_conv(h, long_edge_index, edge_attr=long_edge_attr)
                if long_edge_index.numel() > 0
                else torch.zeros_like(h)
            )
            h = norm(h + h_short + h_long)
            h = torch.relu(h)
            h = self.dropout(h)
        return self.head(h)


class SchakeDistillModel(nn.Module):
    """Schake v2 architecture wrapper used for distillation."""

    def __init__(
        self,
        hidden_channels: int = 32,
        num_layers: int = 2,
        kernel_size: int = 18,
        sake_low_cut: float = 0.25,
        sake_high_cut: float = 1.0,
        schnet_low_cut: float = 1.0,
        schnet_high_cut: float = 2.5,
        max_num_neigh: int = 10000,
        num_heads: int = 4,
        num_out_layers: int = 3,
    ) -> None:
        super().__init__()
        try:
            from models.vendor.schake_model_v2 import create_Schake
        except Exception as exc:
            raise RuntimeError(
                "Failed to import official Schake v2 module. "
                "Ensure torch-scatter/torch-geometric binaries match your torch+cuda build."
            ) from exc

        self.model = create_Schake(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            neighbor_embed="resid_bb3",
            sake_low_cut=sake_low_cut,
            sake_high_cut=sake_high_cut,
            schnet_low_cut=schnet_low_cut,
            schnet_high_cut=schnet_high_cut,
            schnet_act=torch.nn.CELU(2.0),
            sake_act=torch.nn.CELU(2.0),
            out_act=torch.nn.Tanh(),
            max_num_neigh=max_num_neigh,
            schnet_sel=1,
            trainable_sake_kernel=False,
            trainable_schnet_kernel=False,
            num_heads=num_heads,
            embed_type="elements",
            num_out_layers=num_out_layers,
            device="cpu",
            single_pro=False,
        )

        # Atom embedding IDs expected by official Schake helpers for bb3:
        # C -> 0, CA -> 1, N -> 63.
        # Project this project's atom_idx order [N, CA, C] -> [63, 1, 0].
        self.register_buffer("_atom_map", torch.tensor([63, 1, 0], dtype=torch.long), persistent=False)

        # Official 20-AA index order used by Schake helpers.
        schake_aa_order = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]
        remap = torch.zeros(21, dtype=torch.long)
        for schake_i, aa in enumerate(schake_aa_order):
            project_i = AA1_TO_INDEX[aa]
            remap[project_i] = schake_i
        # Unknown token "X" should never appear in DISPEF-M preprocessing;
        # keep a safe default mapping to alanine index 0.
        remap[AA1_TO_INDEX["X"]] = 0
        self.register_buffer("_aa_map", remap, persistent=False)

    def forward(self, data) -> torch.Tensor:
        pos = data.pos
        device = pos.device
        device_s = str(device)

        # Schake's forward explicitly moves tensors to self.device.
        if getattr(self.model, "device", None) != device_s:
            self.model.device = device_s
            self.model.to(device)

        atom_ids = self._atom_map[data.atom_idx]
        aa_ids = self._aa_map[data.aa_idx]

        if hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=device)

        return self.model(atom_ids, aa_ids, pos, batch)
