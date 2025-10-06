#!/usr/bin/env python
"""eval_pla_apc.py
Evaluate APC for a selected (layer, head) with PLA adapter attached (o_proj).

This script attaches a PLAAdapter with given λ to the model and reports APC
on a provided text file using the trained decoder. Designed for quick
post-training evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from pll.runtime import get_model, get_tokenizer, _resolve_device
from pll.geometry_helpers import (
    _get_attn_submodule,
    _get_heads_info,
)
from pll.train_pla import (
    PLAAdapter,
    _phi_for_weight_sub,
    _compute_token_phases_with_decoder,
    _angular_diff,
)


def _read_texts_file(path: Path, max_texts: int = 0) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"texts file not found: {path}")
    texts: List[str] = []
    if path.suffix.lower() == ".jsonl":
        import json as _json
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                    t = obj.get("text") if isinstance(obj, dict) else None
                    if isinstance(t, str) and t:
                        texts.append(t)
                except Exception:
                    pass
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    texts.append(t)
    if max_texts and max_texts > 0:
        texts = texts[:max_texts]
    return texts


def _compute_apc_with_adapter(
    model: torch.nn.Module,
    texts: Sequence[str],
    *,
    layer: int,
    head: int,
    n: int,
    beta: float,
    decoder_dir: Path,
    phase_layers: Optional[Sequence[int]],
) -> dict:
    # Geometry for the head
    attn = _get_attn_submodule(model, layer)
    w_tensor = None
    if hasattr(attn, "o_proj") and hasattr(attn.o_proj, "weight"):
        w_tensor = attn.o_proj.weight.detach().to("cpu")
    elif hasattr(attn, "out_proj") and hasattr(attn.out_proj, "weight"):
        w_tensor = attn.out_proj.weight.detach().to("cpu")
    if w_tensor is None:
        return {"apc": float("nan"), "phi": float("nan"), "var2": float("nan"), "var3": float("nan")}
    head_dim, n_heads, _ = _get_heads_info(model)
    c0 = head * int(head_dim)
    c1 = c0 + int(head_dim)
    sub = w_tensor[:, c0:c1]
    phi, var2, var3 = _phi_for_weight_sub(sub, n)

    # Token phases via decoder (with adapter already attached to model)
    phases = _compute_token_phases_with_decoder(texts, decoder_dir, device=_resolve_device("auto"), layers=phase_layers, n=n)
    tol = beta * math.pi / max(1, n)
    total = 0
    ok = 0
    rays = (2 * math.pi / n) * np.arange(n) + phi
    for seq_ph in phases:
        for ph in seq_ph:
            if ph is None:
                continue
            d = float(np.min(np.abs(_angular_diff(np.array([ph]), rays))))
            total += 1
            if d <= tol:
                ok += 1
    apc = (ok / total) if total > 0 else float("nan")
    return {"apc": apc, "phi": phi, "var2": var2, "var3": var3, "total": total, "ok": ok}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate APC with PLA adapter (o_proj)")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--texts_file", type=Path, required=True)
    ap.add_argument("--max_texts", type=int, default=500)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=8)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--beta", type=float, default=0.35)
    ap.add_argument("--lambda_val", type=float, default=0.1)
    ap.add_argument("--outside_only", type=int, default=0)
    ap.add_argument("--decoder", type=Path, required=True)
    ap.add_argument("--phase_layers", type=int, nargs="+", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = _resolve_device(args.device)
    torch.manual_seed(42)

    model = get_model(device=dev)
    tok = get_tokenizer(device=dev)

    # Freeze base model
    for p in model.parameters():
        p.requires_grad_(False)

    # Attach adapter (per-head, o_proj)
    adapter = PLAAdapter(
        model,
        layer_idx=int(args.layer),
        heads=[int(args.head)],
        n=int(args.n),
        beta=float(args.beta),
        lambda_init=float(args.lambda_val),
        outside_only=bool(args.outside_only),
        device=dev,
    )
    adapter.to(dev)

    # Prepare data
    texts = _read_texts_file(Path(args.texts_file), max_texts=int(args.max_texts))
    report = _compute_apc_with_adapter(
        model,
        texts,
        layer=int(args.layer),
        head=int(args.head),
        n=int(args.n),
        beta=float(args.beta),
        decoder_dir=Path(args.decoder),
        phase_layers=args.phase_layers,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


