#!/usr/bin/env python
"""train_pla.py
Phase–Lattice Adapter (PLA): lightweight, per-head, soft Cn-projection on activations.

Goal
----
- Freeze the base LM. Insert a tiny adapter on a selected attention part (o_proj for now).
- For chosen heads, softly mix the pre-o_proj activation x with its hard projection Π(x)
  onto the nearest Cn ray in that head subspace: x' = (1-λ) x + λ Π(x), λ ∈ [0,1].
- Optimise λ (per-head scalars) using LM loss plus off-ray penalty L_off = E||x' - Π(x)||^2.

Notes
-----
- Supports `o_proj` (pre-hook on input, per-head columns) и `v_proj` (post-hook на выход, per-head строки).
- Rays are built in the per-head subspace of the selected part (columns for o_proj, rows for v_proj).
- Token phases/APC evaluation can be reported before/after if a decoder is provided.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

from pll.runtime import get_model, get_tokenizer, _resolve_device
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # fallback to plain prints
from pll.geometry_helpers import (
    _get_attn_submodule,
    _get_heads_info,
)


# ------------------------------- Utilities ---------------------------------

def _row_normalize(w: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
    return w / norms


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:2].T
    return x @ comps


def _angles_2d(coords: np.ndarray) -> np.ndarray:
    return np.arctan2(coords[:, 1], coords[:, 0]) % (2 * math.pi)


def _angular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


def _phi_for_weight_sub(w_sub: torch.Tensor, n: int) -> Tuple[float, float, float]:
    x = w_sub.float().cpu().numpy()
    x = _row_normalize(x)
    coords2d = _pca_2d(x)
    angles = _angles_2d(coords2d)
    # fixed-n grid-search for phi
    bins = 360
    phis = np.linspace(0.0, 2 * math.pi, bins, endpoint=False)
    best_phi = 0.0
    best_mean = 1e9
    grid = (2 * math.pi / n) * np.arange(n)
    for phi in phis:
        diffs = np.min(np.abs(_angular_diff(angles[:, None], (grid[None, :] + phi))), axis=1)
        mean_abs = float(diffs.mean())
        if mean_abs < best_mean:
            best_mean = mean_abs
            best_phi = float(phi)
    # var explained
    x_center = x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x_center, full_matrices=False)
    power = (s ** 2)
    denom = float(power.sum()) + 1e-12
    var2 = float(power[: min(2, power.shape[0])].sum()) / denom
    var3 = float(power[: min(3, power.shape[0])].sum()) / denom
    return best_phi, var2, var3


def _cluster_unit_dirs(W: np.ndarray, nearest_idx: np.ndarray, n: int) -> List[np.ndarray]:
    Wn = _row_normalize(W)
    units: List[np.ndarray] = []
    for k in range(n):
        sel = Wn[nearest_idx == k]
        if sel.shape[0] == 0:
            X = W - W.mean(axis=0, keepdims=True)
            u, s, vt = np.linalg.svd(X, full_matrices=False)
            d = vt[0]
        else:
            d = sel.mean(axis=0)
        norm = np.linalg.norm(d) + 1e-8
        units.append(d / norm)
    return units


def _build_units_for_o_proj(model: nn.Module, layer_idx: int, head_idx: int, n: int, beta: float) -> Tuple[List[np.ndarray], float, float, float, float]:
    """Return (units, phi, var2, var3, tol) for a specific layer/head in o_proj.

    units: list of length n with vectors in R^{head_dim} (numpy, unit norm).
    tol: beta * pi / n.
    """
    attn = _get_attn_submodule(model, layer_idx)
    w_tensor = None
    if hasattr(attn, "o_proj") and hasattr(attn.o_proj, "weight"):
        w_tensor = attn.o_proj.weight.detach().to("cpu")
    elif hasattr(attn, "out_proj") and hasattr(attn.out_proj, "weight"):
        w_tensor = attn.out_proj.weight.detach().to("cpu")
    if w_tensor is None:
        raise RuntimeError("o_proj weight not found for the specified layer")

    head_dim, n_heads, _ = _get_heads_info(model)
    if not head_dim or not n_heads:
        raise RuntimeError("Invalid heads/head_dim in model config")
    c0 = head_idx * int(head_dim)
    c1 = c0 + int(head_dim)
    if c0 < 0 or c1 > w_tensor.shape[1]:
        raise ValueError(f"Head index {head_idx} out of bounds for o_proj columns")
    sub = w_tensor[:, c0:c1]  # [rows=hidden, cols=head_dim]

    phi, var2, var3 = _phi_for_weight_sub(sub, n)

    W = sub.float().cpu().numpy()
    rows = W.shape[0]
    Wn = _row_normalize(W)
    coords = _pca_2d(Wn)
    angles = _angles_2d(coords)
    grid = (2 * math.pi / n) * np.arange(n) + phi
    diffs = np.abs(_angular_diff(angles[:, None], grid[None, :]))
    nearest = diffs.argmin(axis=1)

    units = _cluster_unit_dirs(W, nearest, n)
    tol = beta * math.pi / max(1, n)
    return units, float(phi), float(var2), float(var3), float(tol)


def _build_units_for_v_proj(model: nn.Module, layer_idx: int, head_idx: int, n: int, beta: float) -> Tuple[List[np.ndarray], float, float, float, float]:
    """Return (units, phi, var2, var3, tol) for a specific layer/head in v_proj.

    For v_proj we split by rows (per-head), sub is [head_dim, hidden]. We transpose to
    get W_t = [hidden, head_dim] so that rows live in the head subspace R^{head_dim}.
    Units are built in R^{head_dim} to match per-head activation slices.
    """
    attn = _get_attn_submodule(model, layer_idx)
    if not hasattr(attn, "v_proj") or not hasattr(attn.v_proj, "weight"):
        raise RuntimeError("v_proj weight not found for the specified layer")
    w_tensor = attn.v_proj.weight.detach().to("cpu")  # [hidden, hidden]

    head_dim, n_heads, n_kv = _get_heads_info(model)
    h_count = int(n_kv or n_heads or 0)
    if not head_dim or h_count <= 0:
        raise RuntimeError("Invalid heads/head_dim for v_proj in model config")
    r0 = head_idx * int(head_dim)
    r1 = r0 + int(head_dim)
    if r0 < 0 or r1 > w_tensor.shape[0]:
        raise ValueError(f"Head index {head_idx} out of bounds for v_proj rows")
    sub = w_tensor[r0:r1, :]  # [head_dim, hidden]

    # Transpose: rows as samples in R^{head_dim}
    W = sub.float().cpu().numpy().T  # [hidden, head_dim]
    X = _row_normalize(W)
    coords2d = _pca_2d(X)
    angles = _angles_2d(coords2d)
    # fit phi for fixed n
    bins = 360
    phis = np.linspace(0.0, 2 * math.pi, bins, endpoint=False)
    best_phi = 0.0
    best_mean = 1e9
    grid0 = (2 * math.pi / n) * np.arange(n)
    for phi in phis:
        diffs = np.min(np.abs(_angular_diff(angles[:, None], (grid0[None, :] + phi))), axis=1)
        mean_abs = float(diffs.mean())
        if mean_abs < best_mean:
            best_mean = mean_abs
            best_phi = float(phi)
    phi = best_phi
    grid = grid0 + phi
    diffs = np.abs(_angular_diff(angles[:, None], grid[None, :]))
    nearest = diffs.argmin(axis=1)

    units = _cluster_unit_dirs(W, nearest, n)  # unit vectors in R^{head_dim}
    # var explained for sanity
    X_center = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X_center, full_matrices=False)
    power = (s ** 2)
    denom = float(power.sum()) + 1e-12
    var2 = float(power[: min(2, power.shape[0])].sum()) / denom
    var3 = float(power[: min(3, power.shape[0])].sum()) / denom
    tol = beta * math.pi / max(1, n)
    return units, float(phi), float(var2), float(var3), float(tol)


def _hard_projection_preserve_norm(x: torch.Tensor, units: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Π(x) per-token preserving vector norm.

    x: [B, T, D]; units: [n, D] (unit norm).
    Returns (proj, k_idx) with proj same shape as x, and k_idx in [0..n-1].
    """
    # [B,T,D] · [n,D]^T → [B,T,n]
    dots = torch.einsum("btd,nd->btn", x, units)
    abs_dots = torch.abs(dots)
    k_idx = torch.argmax(abs_dots, dim=-1)  # [B,T]
    # gather best unit per position
    best_units = units[k_idx]  # [B,T,D]
    # sign for alignment
    signs = torch.sign(torch.sum(x * best_units, dim=-1, keepdim=True))  # [B,T,1]
    # preserve original norm
    x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-8
    proj = x_norm * signs * best_units
    return proj, k_idx


def _angle_to_unit(x: torch.Tensor, units: torch.Tensor, k_idx: torch.Tensor) -> torch.Tensor:
    """Return angular deviation (radians) between x and selected unit (by k_idx)."""
    best_units = units[k_idx]
    x_norm = torch.norm(x, dim=-1) + 1e-8
    cos = torch.sum(x * best_units, dim=-1) / x_norm
    cos = torch.clamp(cos, -1.0, 1.0)
    return torch.arccos(torch.abs(cos))


class PLAAdapter(nn.Module):
    """Per-head soft projector module for pre-o_proj activation.

    Registers a pre-forward hook on o_proj to modify its input. Keeps a small set of
    trainable λ parameters (one per targeted head). Computes off-ray penalty over
    the modified activations.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        heads: Sequence[int],
        n: int,
        beta: float,
        lambda_init: float = 0.1,
        outside_only: bool = False,
        device: str = "cpu",
        part: str = "o_proj",
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.heads = [int(h) for h in heads]
        self.n = int(n)
        self.beta = float(beta)
        self.outside_only = bool(outside_only)
        self.resolved_device = _resolve_device(device)
        self.part = str(part)

        # Build per-head units
        self.units_per_head: Dict[int, torch.Tensor] = {}
        self.tol: float = 0.0
        head_dim, n_heads, _ = _get_heads_info(model)
        if not head_dim:
            raise RuntimeError("head_dim is 0; cannot split per-head")
        self.head_dim = int(head_dim)

        tol_ref: Optional[float] = None
        if self.part == "o_proj":
            for h in self.heads:
                units_np, phi, var2, var3, tol = _build_units_for_o_proj(model, layer_idx, h, n, beta)
                units_t = torch.tensor(np.asarray(units_np, dtype=np.float32), device=self.resolved_device)
                self.units_per_head[h] = F.normalize(units_t, dim=-1)  # [n, head_dim]
                tol_ref = tol if tol_ref is None else tol_ref
        elif self.part == "v_proj":
            for h in self.heads:
                units_np, phi, var2, var3, tol = _build_units_for_v_proj(model, layer_idx, h, n, beta)
                units_t = torch.tensor(np.asarray(units_np, dtype=np.float32), device=self.resolved_device)
                self.units_per_head[h] = F.normalize(units_t, dim=-1)
                tol_ref = tol if tol_ref is None else tol_ref
        else:
            raise ValueError(f"Unsupported part for PLAAdapter: {self.part}")
        self.tol = float(tol_ref or (beta * math.pi / max(1, n)))

        # Trainable alphas → lambdas via sigmoid
        self.alpha = nn.Parameter(torch.full((len(self.heads),), float(math.log(lambda_init / (1 - lambda_init + 1e-8) + 1e-8)), dtype=torch.float32))

        # Runtime penalty terms collected per forward (kept with graph)
        self._penalty_terms: List[torch.Tensor] = []

        # Register hook depending on part
        self._hook_handle = None
        attn = _get_attn_submodule(model, layer_idx)
        if self.part == "o_proj":
            mod = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
            if mod is None:
                raise RuntimeError("o_proj/out_proj module not found")
            self._hook_handle = mod.register_forward_pre_hook(self._pre_hook)  # type: ignore[arg-type]
        elif self.part == "v_proj":
            if not hasattr(attn, "v_proj"):
                raise RuntimeError("v_proj module not found")
            mod = getattr(attn, "v_proj")
            self._hook_handle = mod.register_forward_hook(self._post_hook)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unsupported part: {self.part}")

    @property
    def lambdas(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha)  # [H]

    def _pre_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        x = inputs[0]
        if x is None or x.ndim != 3:
            return inputs
        # x shape: [B, T, hidden] with hidden = heads * head_dim
        B, T, hidden = x.shape
        if hidden % self.head_dim != 0:
            return inputs
        device = x.device
        x_view = x.view(B, T, hidden // self.head_dim, self.head_dim)
        x_new = x_view.clone()

        # Apply per-head where configured
        for idx, h in enumerate(self.heads):
            if h < 0 or h >= x_view.shape[2]:  # head axis
                continue
            v = x_view[:, :, h, :]  # [B, T, D]
            units = self.units_per_head[h].to(device)  # [n, D]
            proj, k_idx = _hard_projection_preserve_norm(v, units)
            # Optional gating by angle tolerance
            if self.outside_only:
                ang = _angle_to_unit(v, units, k_idx)  # [B, T]
                mask = (ang > self.tol).float().unsqueeze(-1)  # [B, T, 1]
            else:
                mask = torch.ones_like(proj[..., :1])

            lam = self.lambdas[idx].clamp(0.0, 1.0)
            v_prime = (1.0 - lam) * v + lam * proj
            # mix only where mask==1
            v_mixed = mask * v_prime + (1.0 - mask) * v
            x_new[:, :, h, :] = v_mixed

            # differentiable penalty only via λ; stop-grad through activations to avoid
            # backward over the whole transformer graph (keeps runtime low)
            v_det = v.detach()
            proj_det = proj.detach()
            diff0 = (v_det - proj_det)
            # penalty ≈ E[(1-λ)^2 ||v - Π(v)||^2] over masked positions (mask detached)
            m_det = mask.detach().squeeze(-1)
            pen = ((1.0 - lam) ** 2) * torch.mean(torch.sum(diff0 * diff0, dim=-1) * m_det)
            self._penalty_terms.append(pen)

        return (x_new.view(B, T, hidden),)

    def pull_and_reset_penalty(self) -> float:
        if not self._penalty_terms:
            return torch.tensor(0.0, device=self.alpha.device)
        pen = torch.stack(self._penalty_terms).mean()
        # reset list for next accumulation
        self._penalty_terms = []
        return pen

    def remove(self) -> None:
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
            self._hook_handle = None

    def _post_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        # Apply on output of v_proj: shape [B, T, hidden]
        if output is None or output.ndim != 3:
            return output
        x = output
        B, T, hidden = x.shape
        if hidden % self.head_dim != 0:
            return output
        x_view = x.view(B, T, hidden // self.head_dim, self.head_dim)
        x_new = x_view.clone()
        device = x.device
        for idx, h in enumerate(self.heads):
            if h < 0 or h >= x_view.shape[2]:
                continue
            v = x_view[:, :, h, :]  # [B, T, D]
            units = self.units_per_head[h].to(device)
            proj, k_idx = _hard_projection_preserve_norm(v, units)
            if self.outside_only:
                ang = _angle_to_unit(v, units, k_idx)
                mask = (ang > self.tol).float().unsqueeze(-1)
            else:
                mask = torch.ones_like(proj[..., :1])
            lam = self.lambdas[idx].clamp(0.0, 1.0)
            v_prime = (1.0 - lam) * v + lam * proj
            v_mixed = mask * v_prime + (1.0 - mask) * v
            x_new[:, :, h, :] = v_mixed

            # penalty only via λ
            v_det = v.detach()
            proj_det = proj.detach()
            diff0 = (v_det - proj_det)
            m_det = mask.detach().squeeze(-1)
            pen = ((1.0 - lam) ** 2) * torch.mean(torch.sum(diff0 * diff0, dim=-1) * m_det)
            self._penalty_terms.append(pen)

        return x_new.view(B, T, hidden)


# ------------------------------- Dataset ------------------------------------

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


def _make_batches(tokenizer, texts: Sequence[str], seq_len: int) -> List[torch.Tensor]:
    """Naive batching: concatenate texts with EOS and split into seq_len chunks."""
    ids: List[int] = []
    for t in texts:
        enc = tokenizer(t, add_special_tokens=False, return_tensors=None)
        seq = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids  # type: ignore
        if isinstance(seq, list):
            ids.extend(int(x) for x in (seq if isinstance(seq[0], int) else seq[0]))
        else:
            try:
                ids.extend(int(x) for x in seq[0])  # type: ignore[index]
            except Exception:
                pass
        # EOS if available
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            ids.append(int(tokenizer.eos_token_id))
    # split
    chunks: List[torch.Tensor] = []
    i = 0
    while i + seq_len + 1 <= len(ids):
        arr = torch.tensor(ids[i:i + seq_len + 1], dtype=torch.long)
        chunks.append(arr)
        i += seq_len
    return chunks


# ------------------------------- APC metric ---------------------------------

def _phase_from_probs(probs: np.ndarray, n: int) -> Optional[float]:
    if probs.ndim != 1 or probs.shape[0] <= 0:
        return None
    total = float(probs.sum())
    if total <= 1e-12:
        return None
    p = probs / total
    ks = np.arange(len(p))
    theta = 2 * math.pi * ks / n
    z = np.sum(p * np.exp(1j * theta))
    if abs(z) <= 1e-6:
        return None
    ang = math.atan2(z.imag, z.real) % (2 * math.pi)
    return float(ang)


def _compute_token_phases_with_decoder(texts: Sequence[str], decoder_dir: Path, *, device: str, layers: Optional[Sequence[int]], n: int) -> List[List[Optional[float]]]:
    model = get_model(device=device)
    tok = get_tokenizer(device=device)
    # Load decoder
    with open(decoder_dir / "decoder_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    rank = int(cfg.get("rank", n))
    dec_layers = cfg.get("layers", None)
    if layers is None and isinstance(dec_layers, list):
        layers = dec_layers
    linear = torch.nn.Sequential(torch.nn.Linear(model.config.hidden_size, rank)).to(device)  # type: ignore[attr-defined]
    state = torch.load(decoder_dir / "decoder.pt", map_location=device)
    linear.load_state_dict(state)
    linear.eval()
    enc = tok(list(texts), return_tensors="pt", padding=True, truncation=False)
    input_ids = enc["input_ids"].to(device)
    attention = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
        hs = out.hidden_states
        if layers is None:
            h = hs[-1]
        else:
            total = len(hs) - 1
            norm_idx: List[int] = []
            for i in layers:
                idx = int(i)
                if idx < 0:
                    idx = total + idx
                if 0 <= idx < total:
                    norm_idx.append(idx)
            sel = [hs[j + 1] for j in norm_idx] if norm_idx else [hs[-1]]
            h = torch.stack(sel).mean(dim=0)
        logits = linear(h)  # [batch, seq, rank]
        probs = torch.softmax(logits, dim=-1)
    phases: List[List[Optional[float]]] = []
    for i in range(probs.shape[0]):
        valid = int(attention[i].sum().item())
        seq_phases: List[Optional[float]] = []
        for j in range(valid):
            ph = _phase_from_probs(probs[i, j].detach().cpu().numpy(), n=n)
            seq_phases.append(ph)
        phases.append(seq_phases)
    return phases


def _compute_apc_for_head(model: nn.Module, texts: Sequence[str], *, layer: int, head: int, n: int, beta: float, decoder_dir: Optional[Path], phase_layers: Optional[Sequence[int]], part: str = "o_proj") -> Dict[str, float]:
    # Build phi for selected part/head
    attn = _get_attn_submodule(model, layer)
    w_tensor = None
    if part == "o_proj" and hasattr(attn, "o_proj") and hasattr(attn.o_proj, "weight"):
        w_tensor = attn.o_proj.weight.detach().to("cpu")
    elif part == "o_proj" and hasattr(attn, "out_proj") and hasattr(attn.out_proj, "weight"):
        w_tensor = attn.out_proj.weight.detach().to("cpu")
    elif part == "v_proj" and hasattr(attn, "v_proj") and hasattr(attn.v_proj, "weight"):
        w_tensor = attn.v_proj.weight.detach().to("cpu")
    if w_tensor is None:
        return {"apc": float("nan"), "phi": float("nan"), "var2": float("nan"), "var3": float("nan")}
    head_dim, n_heads, _ = _get_heads_info(model)
    if part == "o_proj":
        c0 = head * int(head_dim)
        c1 = c0 + int(head_dim)
        sub = w_tensor[:, c0:c1]
    else:  # v_proj split by rows
        r0 = head * int(head_dim)
        r1 = r0 + int(head_dim)
        sub = w_tensor[r0:r1, :]
    phi, var2, var3 = _phi_for_weight_sub(sub, n)

    # Token phases via decoder (if provided)
    if decoder_dir is None:
        return {"apc": float("nan"), "phi": phi, "var2": var2, "var3": var3}
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
    return {"apc": apc, "phi": phi, "var2": var2, "var3": var3}


# ------------------------------- Training -----------------------------------

@dataclass
class TrainSummary:
    steps: int
    seq_len: int
    batch_size: int
    accum: int
    lr: float
    mu: float
    lambda_init: float
    final_lambda: List[float]
    mean_loss: float
    mean_lm: float
    mean_off: float
    time_s: float


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PLA adapter (per-head, o_proj|v_proj)")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--no_cache", type=int, default=1)
    ap.add_argument("--texts_file", type=Path, default=Path("data/sem_mask_train.jsonl"))
    ap.add_argument("--max_texts", type=int, default=500)
    ap.add_argument("--outdir", type=Path, default=Path("runs/pla_run"))
    ap.add_argument("--log_every", type=int, default=10)
    # PLA params
    ap.add_argument("--pla_layer", type=int, default=0)
    ap.add_argument("--pla_part", type=str, default="o_proj", choices=["o_proj", "v_proj"])  # supports v_proj too
    ap.add_argument("--pla_heads", type=str, default="8")  # comma-separated
    ap.add_argument("--pla_n", type=int, default=7)
    ap.add_argument("--pla_beta", type=float, default=0.35)
    ap.add_argument("--pla_lambda", type=float, default=0.1)
    ap.add_argument("--pla_mu", type=float, default=0.01)
    ap.add_argument("--outside_only", type=int, default=0)
    # APC eval
    ap.add_argument("--decoder", type=Path, default=None)
    ap.add_argument("--phase_layers", type=int, nargs="+", default=None)
    args = ap.parse_args()

    dev = _resolve_device(args.device)
    torch.manual_seed(42)

    model = get_model(device=dev)
    tok = get_tokenizer(device=dev)

    # Freeze base model
    for p in model.parameters():
        p.requires_grad_(False)

    if args.grad_ckpt:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    heads = [int(x) for x in str(args.pla_heads).split(",") if str(x).strip() != ""]
    adapter = PLAAdapter(
        model,
        layer_idx=int(args.pla_layer),
        heads=heads,
        n=int(args.pla_n),
        beta=float(args.pla_beta),
        lambda_init=float(args.pla_lambda),
        outside_only=bool(args.outside_only),
        device=dev,
        part=str(args.pla_part),
    )
    adapter.to(dev)

    opt = AdamW([p for p in adapter.parameters() if p.requires_grad], lr=float(args.lr))

    # Data
    texts = _read_texts_file(Path(args.texts_file), max_texts=int(args.max_texts))
    batches = _make_batches(tok, texts, int(args.seq_len))
    if not batches:
        raise SystemExit("no batches prepared; reduce seq_len or provide more data")

    # Prepare outdir and progress log
    outdir: Path = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    progress_path = outdir / "train_progress.jsonl"

    # APC before
    apc_before = None
    if args.decoder is not None and Path(args.decoder).exists():
        try:
            apc_before = _compute_apc_for_head(model, texts, layer=int(args.pla_layer), head=heads[0], n=int(args.pla_n), beta=float(args.pla_beta), decoder_dir=Path(args.decoder), phase_layers=args.phase_layers, part=str(args.pla_part))
        except Exception:
            apc_before = None

    # Train loop
    model.train(False)  # keep in eval mode; we only optimise adapter
    steps = int(args.steps)
    accum = max(1, int(args.accum))
    bs = int(args.batch_size)
    start = time.time()
    loss_sum = 0.0
    lm_sum = 0.0
    off_sum = 0.0
    count = 0

    # simple pointer over batches
    ptr = 0
    iterator = range(steps)
    pbar = None
    if tqdm is not None:
        desc = f"PLA l{int(args.pla_layer)}.{str(args.pla_part)} h{','.join(map(str, heads))}"
        pbar = tqdm(iterator, total=steps, desc=desc)
        iterator = pbar

    for step in iterator:  # type: ignore[assignment]
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        step_lm = 0.0
        step_off = 0.0
        for a in range(accum):
            arr = batches[ptr % len(batches)]
            ptr += 1
            # build mini-batch by repeating sample bs times for simplicity
            data = arr.unsqueeze(0).repeat(bs, 1).to(dev)
            input_ids = data[:, :-1]
            labels = data[:, 1:]
            out = model(input_ids=input_ids, labels=labels, use_cache=(not bool(args.no_cache) and True), return_dict=True)
            lm_loss = out.loss
            off = adapter.pull_and_reset_penalty()
            loss = lm_loss + float(args.pla_mu) * off
            # as adapter has few params, backward on scalar
            loss.backward()
            total_loss += float(loss.detach().cpu())
            lm_val = float(lm_loss.detach().cpu())
            off_val = float(off.detach().cpu()) if isinstance(off, torch.Tensor) else float(off)
            lm_sum += lm_val
            off_sum += off_val
            step_lm += lm_val
            step_off += off_val
            count += 1
        opt.step()
        loss_sum += total_loss / float(accum)
        # Step stats
        step_loss_avg = total_loss / float(accum)
        step_lm_avg = step_lm / float(accum)
        step_off_avg = step_off / float(accum)
        lam_list = [float(x) for x in adapter.lambdas.detach().cpu().tolist()]

        # Progress outputs
        if pbar is not None:
            try:
                pbar.set_postfix({"loss": f"{step_loss_avg:.4f}", "lm": f"{step_lm_avg:.4f}", "off": f"{step_off_avg:.6f}", "lam": f"{sum(lam_list)/len(lam_list):.4f}"})
            except Exception:
                pass
        elif ((step + 1) % max(1, int(args.log_every)) == 0) or step == 0:
            print(f"step {step+1}/{steps} | loss={step_loss_avg:.4f} | lm={step_lm_avg:.4f} | off={step_off_avg:.6f} | lambda={lam_list}")

        # Append to JSONL progress log
        try:
            with progress_path.open("a", encoding="utf-8") as pf:
                pf.write(json.dumps({
                    "step": int(step + 1),
                    "loss": float(step_loss_avg),
                    "lm": float(step_lm_avg),
                    "off": float(step_off_avg),
                    "lambda": lam_list,
                    "time_s": float(time.time() - start),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    if pbar is not None:
        try:
            pbar.close()
        except Exception:
            pass
    elapsed = time.time() - start

    # APC after
    apc_after = None
    if args.decoder is not None and Path(args.decoder).exists():
        try:
            apc_after = _compute_apc_for_head(model, texts, layer=int(args.pla_layer), head=heads[0], n=int(args.pla_n), beta=float(args.pla_beta), decoder_dir=Path(args.decoder), phase_layers=args.phase_layers, part=str(args.pla_part))
        except Exception:
            apc_after = None

    outdir: Path = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save train summary
    ts = TrainSummary(
        steps=steps,
        seq_len=int(args.seq_len),
        batch_size=bs,
        accum=accum,
        lr=float(args.lr),
        mu=float(args.pla_mu),
        lambda_init=float(args.pla_lambda),
        final_lambda=[float(x) for x in adapter.lambdas.detach().cpu().tolist()],
        mean_loss=float(loss_sum / max(1, steps)),
        mean_lm=float(lm_sum / max(1, count)),
        mean_off=float(off_sum / max(1, count)),
        time_s=float(elapsed),
    )
    with (outdir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(ts), f, ensure_ascii=False, indent=2)

    # Save APC before/after
    if apc_before is not None or apc_after is not None:
        with (outdir / "apc_before_after.json").open("w", encoding="utf-8") as f:
            json.dump({"before": apc_before, "after": apc_after}, f, ensure_ascii=False, indent=2)

    # Geometry snapshot before/after (var2/var3 only)
    try:
        attn = _get_attn_submodule(model, int(args.pla_layer))
        if str(args.pla_part) == "o_proj":
            part_name = "o_proj" if hasattr(attn, "o_proj") else "out_proj"
            w_tensor = getattr(attn, part_name).weight.detach().to("cpu")
            head_dim, _, _ = _get_heads_info(model)
            c0 = heads[0] * int(head_dim)
            c1 = c0 + int(head_dim)
            sub = w_tensor[:, c0:c1]
        else:
            w_tensor = attn.v_proj.weight.detach().to("cpu")
            head_dim, _, _ = _get_heads_info(model)
            r0 = heads[0] * int(head_dim)
            r1 = r0 + int(head_dim)
            sub = w_tensor[r0:r1, :]
        phi_b, var2_b, var3_b = _phi_for_weight_sub(sub, int(args.pla_n))
        geom = {"before": {"phi": phi_b, "var2": var2_b, "var3": var3_b}}
    except Exception:
        geom = None
    if geom is not None:
        with (outdir / "geometry_before_after.json").open("w", encoding="utf-8") as f:
            json.dump(geom, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
