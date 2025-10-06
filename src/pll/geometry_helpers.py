from __future__ import annotations

from typing import Tuple


def _get_attn_submodule(model, layer_idx: int):
    # Support common transformer naming patterns
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[int(layer_idx)].self_attn
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[int(layer_idx)].attn
    raise AttributeError("Unsupported model structure for attention access")


def _get_heads_info(model) -> Tuple[int, int, int]:
    # returns (head_dim, n_heads, n_kv)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        head_dim = getattr(cfg, "hidden_size", 0) // max(1, getattr(cfg, "num_attention_heads", 1))
        n_heads = getattr(cfg, "num_attention_heads", 0)
        n_kv = getattr(cfg, "num_key_value_heads", n_heads)
        return int(head_dim), int(n_heads), int(n_kv)
    raise AttributeError("Model config not found")


