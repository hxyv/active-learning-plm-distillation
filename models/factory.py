"""Model factory."""

from __future__ import annotations

from models.gnn import GeoDistillGNN, SimpleDistillGNN


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

    raise ValueError(f"Unknown model type: {model_type}")
