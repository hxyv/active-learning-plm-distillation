"""Model factory."""

from __future__ import annotations

from models.gnn import GeoDistillGNN, SchakeDistillModel, SimpleDistillGNN


def build_model(cfg: dict):
    model_cfg = cfg["model"]
    model_type = model_cfg.get("name", "simple_gnn").lower()

    common_kwargs = {
        "aa_vocab_size": model_cfg.get("aa_vocab_size", 21),
        "atom_vocab_size": model_cfg.get("atom_vocab_size", 3),
        "aa_emb_dim": model_cfg.get("aa_emb_dim", 32),
        "atom_emb_dim": model_cfg.get("atom_emb_dim", 8),
        "hidden_dim": model_cfg.get("hidden_dim", 128),
        "num_layers": model_cfg.get("num_layers", 4),
        "dropout": model_cfg.get("dropout", 0.1),
        "out_dim": model_cfg.get("out_dim", 8),
    }

    if model_type == "simple_gnn":
        return SimpleDistillGNN(**common_kwargs)
    if model_type == "geo_gnn":
        return GeoDistillGNN(edge_dim=model_cfg.get("edge_dim", 4), **common_kwargs)
    if model_type == "schake":
        return SchakeDistillModel(
            hidden_channels=int(model_cfg.get("hidden_channels", 32)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            kernel_size=int(model_cfg.get("kernel_size", 18)),
            sake_low_cut=float(model_cfg.get("sake_low_cut", 0.25)),
            sake_high_cut=float(model_cfg.get("sake_high_cut", 1.0)),
            schnet_low_cut=float(model_cfg.get("schnet_low_cut", 1.0)),
            schnet_high_cut=float(model_cfg.get("schnet_high_cut", 2.5)),
            max_num_neigh=int(model_cfg.get("max_num_neigh", 10000)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            num_out_layers=int(model_cfg.get("num_out_layers", 3)),
            mc_dropout_p=float(model_cfg.get("mc_dropout_p", 0.0)),
        )

    raise ValueError(f"Unknown model type: {model_type}")
