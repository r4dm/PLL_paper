"""Piecewise SPS utilities: regionizer, linear fit, and PSPS export.

This module is a minimal, dependency-free port of the internal tooling used to
extract Piecewise-SPS summaries.  It intentionally mirrors the APIs from the
original research code but replaces the multipolar terminology with PLL names.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import json
import math
import numpy as np
import torch
from torch import Tensor, nn

from .geometry_helpers import _get_attn_submodule
from .phase_group import PhaseGroup
from .phase_sum import PhaseSum
from .runtime import get_model, get_tokenizer, _resolve_device

# ---------------------------------------------------------------------------
# Generic data containers
# ---------------------------------------------------------------------------


@dataclass
class RegionInfo:
    mask_bits: str
    count: int
    coverage: float
    sample_locs: List[Tuple[int, int]]


@dataclass
class RegionizerResult:
    layer: int
    part: str
    topk: int
    regions_limit: int
    total_tokens: int
    coverage_kept: float
    selected_output_neurons: List[int]
    selected_input_dims: List[int]
    regions: List[RegionInfo]


@dataclass
class FitSpec:
    layer: int
    part: str
    head_slice: Optional[Tuple[int, int]] = None
    neuron_indices: Optional[Sequence[int]] = None


@dataclass
class RegionFitResult:
    region_mask: str
    num_samples: int
    in_dim: int
    out_dim: int
    W: Tensor
    b: Tensor
    mse: float
    nmse: float
    r2: float


@dataclass
class PspsRegionExport:
    region_mask: str
    coverage: float
    mse: float
    formula_str: str
    terms: Dict[str, float]
    bias: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_texts(dataset: str | Path, max_texts: int = 0) -> List[str]:
    path = Path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    texts: List[str] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    t = obj.get("text")
                    if isinstance(t, str) and t:
                        texts.append(t)
                elif isinstance(obj, str) and obj:
                    texts.append(obj)
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    if max_texts and max_texts > 0:
        texts = texts[: max_texts]
    return texts


def _get_part_module(model: nn.Module, layer: int, part: str) -> nn.Module:
    if part in {"mlp_gate", "mlp_up", "mlp_down"}:
        mlp = model.get_submodule(f"model.layers.{layer}.mlp")
        attr = {
            "mlp_gate": "gate_proj",
            "mlp_up": "up_proj",
            "mlp_down": "down_proj",
        }[part]
        mod = getattr(mlp, attr, None)
        if mod is None:
            # tolerate alternate names used by some HF checkpoints
            if part == "mlp_up":
                mod = getattr(mlp, "fc1", None)
            elif part == "mlp_down":
                mod = getattr(mlp, "fc2", None)
        if mod is None:
            raise AttributeError(f"Cannot resolve module for part={part} layer={layer}")
        return mod
    if part in {"o_proj", "out_proj"}:
        attn = _get_attn_submodule(model, layer)
        mod = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
        if mod is None:
            raise AttributeError(f"Cannot resolve attention output projection for layer={layer}")
        return mod
    raise ValueError(f"Unsupported part: {part}")


def _row_normalize(w: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
    return w / norms


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:2].T
    return x @ comps


def _pca_var_explained(x: np.ndarray) -> Tuple[float, float]:
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    power = (s ** 2)
    denom = float(power.sum()) + 1e-12
    var2 = float(power[: min(2, power.shape[0])].sum()) / denom
    var3 = float(power[: min(3, power.shape[0])].sum()) / denom
    return var2, var3


def _angular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _angles_2d(coords: np.ndarray) -> np.ndarray:
    return np.arctan2(coords[:, 1], coords[:, 0]) % (2 * math.pi)


# ---------------------------------------------------------------------------
# Regionizer
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_piecewise_regions(
    *,
    layer: int,
    part: str,
    topk_neurons: int,
    regions: int,
    dataset: str | Path,
    max_texts: int = 0,
    device: str | torch.device | None = "auto",
    batch_size: int = 1,
    sample_per_region: int = 64,
    head_slice: Optional[Tuple[int, int]] = None,
) -> RegionizerResult:
    dev = _resolve_device(device)
    model = get_model(device=dev)
    tok = get_tokenizer(device=dev)
    texts = _load_texts(dataset, max_texts=max_texts)
    if not texts:
        raise ValueError("No texts provided for regionizer")

    module = _get_part_module(model, layer, part)
    out_dim: Optional[int] = None
    sum_abs: Optional[Tensor] = None
    in_dim: Optional[int] = None
    sum_abs_in: Optional[Tensor] = None
    total_tokens = 0

    def _hook_collect(_mod, _inp, out):
        nonlocal out_dim, sum_abs, total_tokens, in_dim, sum_abs_in
        y = out[0] if isinstance(out, tuple) else out
        if y is None:
            return
        y = y.detach()
        if y.dim() == 3:
            bt, sl, hid = y.shape
            Y_full = y.reshape(bt * sl, hid)
        else:
            Y_full = y
        Y_focus = Y_full
        slice_offset = 0
        if head_slice is not None and part in {"o_proj", "out_proj"}:
            c0, c1 = head_slice
            slice_offset = int(c0)
            Y_focus = Y_full[:, c0:c1]
        if out_dim is None:
            out_dim = int(Y_focus.shape[-1])
            sum_abs = torch.zeros(out_dim, device=Y_focus.device, dtype=torch.float32)
        sum_abs += Y_focus.abs().sum(dim=0)
        total_tokens += int(Y_focus.shape[0])
        xin = _inp[0] if isinstance(_inp, (tuple, list)) and _inp else None
        if isinstance(xin, torch.Tensor):
            if xin.dim() == 3:
                bt, sl, hid = xin.shape
                Xin = xin.reshape(bt * sl, hid)
            else:
                Xin = xin
            if in_dim is None:
                in_dim = int(Xin.shape[-1])
                sum_abs_in = torch.zeros(in_dim, device=Xin.device, dtype=torch.float32)
            sum_abs_in += Xin.abs().sum(dim=0)

    handle = module.register_forward_hook(lambda m, i, o: _hook_collect(m, i, o))
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=False)
            input_ids = enc["input_ids"].to(dev)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(dev)
            model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
    finally:
        handle.remove()

    if sum_abs is None or out_dim is None:
        raise RuntimeError("Failed to capture activations for regionizer")
    k = min(int(topk_neurons), int(out_dim))
    topk_vals, topk_idx = torch.topk(sum_abs, k=k, largest=True)
    offset = int(head_slice[0]) if head_slice is not None and part in {"o_proj", "out_proj"} else 0
    selected_out = [offset + int(i) for i in topk_idx.detach().cpu().tolist()]

    # Re-run with mask counting limited to top-K
    mask_counts: Dict[str, int] = {}
    sample_store: Dict[str, List[Tuple[int, int]]] = {}

    def _hook_masks(_mod, _inp, out):
        y = out[0] if isinstance(out, tuple) else out
        if y is None:
            return
        if y.dim() == 3:
            bt, sl, hid = y.shape
            Y_full = y.reshape(bt * sl, hid)
        else:
            Y_full = y
        if head_slice is not None and part in {"o_proj", "out_proj"}:
            c0, c1 = head_slice
            Y_focus = Y_full[:, c0:c1]
        else:
            Y_focus = Y_full
        if Y_focus.shape[1] < len(selected_out):
            raise RuntimeError("Head slice produced fewer dims than requested top-K")
        Y_sel = Y_focus[:, [idx - (head_slice[0] if head_slice is not None else 0) for idx in selected_out]]
        mask = (Y_sel > 0).to(torch.uint8)
        for row_idx in range(mask.shape[0]):
            bits = ''.join('1' if int(v) else '0' for v in mask[row_idx].tolist())
            mask_counts[bits] = mask_counts.get(bits, 0) + 1
            if len(sample_store.setdefault(bits, [])) < sample_per_region:
                sample_store[bits].append((0, row_idx))

    handle = module.register_forward_hook(lambda m, i, o: _hook_masks(m, i, o))
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=False)
            input_ids = enc["input_ids"].to(dev)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(dev)
            model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
    finally:
        handle.remove()

    total_tokens = sum(mask_counts.values())
    items = sorted(mask_counts.items(), key=lambda kv: -kv[1])[: int(regions)]
    region_list: List[RegionInfo] = []
    coverage_kept = 0.0
    for bits, count in items:
        cov = float(count) / max(1, total_tokens)
        coverage_kept += cov
        region_list.append(RegionInfo(mask_bits=bits, count=int(count), coverage=cov, sample_locs=sample_store.get(bits, [])))

    if in_dim is None or sum_abs_in is None:
        selected_in = selected_out
    else:
        kin = min(len(selected_out), int(in_dim))
        topk_in = torch.topk(sum_abs_in, k=kin, largest=True).indices.detach().cpu().tolist()
        selected_in = topk_in if topk_in else selected_out

    return RegionizerResult(
        layer=int(layer),
        part=str(part),
        topk=int(k),
        regions_limit=int(regions),
        total_tokens=int(total_tokens),
        coverage_kept=float(coverage_kept),
        selected_output_neurons=[int(x) for x in selected_out],
        selected_input_dims=[int(x) for x in selected_in],
        regions=region_list,
    )


# ---------------------------------------------------------------------------
# Linear fit
# ---------------------------------------------------------------------------


def _resolve_module_and_slices(model: nn.Module, spec: FitSpec) -> Tuple[nn.Module, Optional[slice], Optional[Sequence[int]]]:
    if spec.part in {"mlp_gate", "mlp_up", "mlp_down"}:
        mlp = model.get_submodule(f"model.layers.{spec.layer}.mlp")
        attr = {
            "mlp_gate": "gate_proj",
            "mlp_up": "up_proj",
            "mlp_down": "down_proj",
        }[spec.part]
        mod = getattr(mlp, attr, None)
        if mod is None:
            if spec.part == "mlp_up":
                mod = getattr(mlp, "fc1", None)
            elif spec.part == "mlp_down":
                mod = getattr(mlp, "fc2", None)
        if mod is None:
            raise AttributeError(f"Cannot resolve module for part={spec.part} layer={spec.layer}")
        return mod, None, spec.neuron_indices
    attn = _get_attn_submodule(model, spec.layer)
    mod = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
    if mod is None:
        raise AttributeError(f"Cannot resolve attention projection for layer={spec.layer}")
    col_slice = None
    if spec.head_slice is not None:
        c0, c1 = spec.head_slice
        col_slice = slice(int(c0), int(c1))
    return mod, col_slice, spec.neuron_indices


def _ridge_fit(X: Tensor, Y: Tensor, *, l2: float = 1e-3) -> Tuple[Tensor, Tensor, float]:
    device = X.device
    n, in_dim = X.shape
    out_dim = Y.shape[1]
    X_ext = torch.cat([X, torch.ones(n, 1, device=device)], dim=1)
    I = torch.eye(in_dim + 1, device=device)
    I[-1, -1] = 0.0
    XtX = X_ext.T @ X_ext + l2 * I
    XtY = X_ext.T @ Y
    Wb = torch.linalg.solve(XtX, XtY)
    W = Wb[:-1].T.contiguous()
    b = Wb[-1].T.contiguous()
    pred = X @ W.T + b
    mse = float(torch.mean((pred - Y) ** 2).item())
    return W, b, mse


@torch.no_grad()
def fit_linear_in_region(
    mask_bits: str,
    *,
    spec: FitSpec,
    selected_in_dims: Sequence[int],
    selected_out_dims: Sequence[int],
    dataset: str | Path,
    max_texts: int = 0,
    device: str | torch.device | None = "auto",
    batch_size: int = 1,
    l2: float = 1e-3,
    max_samples: int = 20000,
) -> RegionFitResult:
    dev = _resolve_device(device)
    model = get_model(device=dev)
    tok = get_tokenizer(device=dev)
    module, col_slice, neuron_idxs = _resolve_module_and_slices(model, spec)

    target_mask = torch.tensor([1 if ch == "1" else 0 for ch in mask_bits], dtype=torch.uint8)
    texts = _load_texts(dataset, max_texts=max_texts)
    if not texts:
        raise ValueError("No texts provided for PSPS fit")

    X_list: List[Tensor] = []
    Y_list: List[Tensor] = []
    collected = 0

    def _hook_capture(_m, inp, out):
        nonlocal collected
        y = out[0] if isinstance(out, tuple) else out
        xin = inp[0] if isinstance(inp, (tuple, list)) and inp else None
        if y is None or xin is None:
            return
        if y.dim() == 3:
            bt, sl, od = y.shape
            Y = y.reshape(bt * sl, od)
        else:
            Y = y
        if xin.dim() == 3:
            bt, sl, hid = xin.shape
            X = xin.reshape(bt * sl, hid)
        else:
            X = xin
        if col_slice is not None:
            Y_sel = Y[:, col_slice]
        elif neuron_idxs is not None:
            Y_sel = Y[:, neuron_idxs]
        else:
            Y_sel = Y
        Y_mask_src = Y[:, selected_out_dims]
        mask = (Y_mask_src > 0).to(torch.uint8)
        match = (mask == target_mask.to(mask.device).unsqueeze(0)).all(dim=1)
        if match.any():
            X_sel = X[:, selected_in_dims][match]
            Y_sel = Y_sel[match]
            X_list.append(X_sel.detach().to("cpu"))
            Y_list.append(Y_sel.detach().to("cpu"))
            collected += int(X_sel.shape[0])

    handle = module.register_forward_hook(lambda m, i, o: _hook_capture(m, i, o))
    try:
        for i in range(0, len(texts), batch_size):
            if collected >= max_samples:
                break
            batch = texts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=False)
            input_ids = enc["input_ids"].to(dev)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(dev)
            model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
    finally:
        handle.remove()

    if not X_list:
        in_dim = len(selected_in_dims)
        out_dim = target_mask.numel()
        W = torch.zeros((out_dim, in_dim))
        b = torch.zeros(out_dim)
        return RegionFitResult(mask_bits, 0, in_dim, out_dim, W, b, float("nan"), float("nan"), float("nan"))

    X_cat = torch.cat(X_list, dim=0)
    Y_cat = torch.cat(Y_list, dim=0)
    X_cat = X_cat[: max_samples]
    Y_cat = Y_cat[: max_samples]

    W, b, mse = _ridge_fit(X_cat, Y_cat, l2=float(l2))
    var = torch.var(Y_cat, dim=0, unbiased=False).mean().item() + 1e-8
    nmse = float(mse / var)
    r2 = float(max(0.0, 1.0 - nmse))
    return RegionFitResult(mask_bits, int(X_cat.shape[0]), int(X_cat.shape[1]), int(Y_cat.shape[1]), W, b, mse, nmse, r2)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _linear_phase_sum(
    *,
    W: Tensor,
    b: Tensor,
    selected_in_dims: Sequence[int],
    group: PhaseGroup,
    min_coeff_abs: float,
    max_terms: int,
) -> PhaseSum:
    if W.ndim != 2:
        raise ValueError("W must be [out_dim, in_dim]")
    in_dim = W.shape[1]
    if in_dim != len(selected_in_dims):
        raise ValueError("selected_in_dims length must match W columns")
    coeffs: Dict[str, float] = {}
    w_mean = W.mean(dim=0)
    for j, idx in enumerate(selected_in_dims):
        coeff = float(w_mean[j].item())
        if abs(coeff) < float(min_coeff_abs):
            continue
        ray = group.get_ray(int(idx) % group.n)
        coeffs[ray.name] = coeffs.get(ray.name, 0.0) + coeff
    ps = PhaseSum(group, coeffs, bias=float(b.mean().item()) if b.numel() else 0.0)
    return ps.prune(min_abs=min_coeff_abs, max_terms=max_terms)


def export_region_to_psps(
    *,
    region_mask: str,
    coverage: float,
    W: Tensor,
    b: Tensor,
    selected_in_dims: Sequence[int],
    group: PhaseGroup | None = None,
    min_coeff_abs: float = 0.01,
    max_terms: int = 5,
) -> PspsRegionExport:
    group = group or PhaseGroup(7)
    phase_sum = _linear_phase_sum(
        W=W,
        b=b,
        selected_in_dims=selected_in_dims,
        group=group,
        min_coeff_abs=min_coeff_abs,
        max_terms=max_terms,
    )
    terms = phase_sum.to_dict()
    return PspsRegionExport(
        region_mask=region_mask,
        coverage=float(coverage),
        mse=float("nan"),
        formula_str=str(phase_sum),
        terms=terms,
        bias=float(phase_sum.bias),
    )


__all__ = [
    "RegionInfo",
    "RegionizerResult",
    "FitSpec",
    "RegionFitResult",
    "PspsRegionExport",
    "build_piecewise_regions",
    "fit_linear_in_region",
    "export_region_to_psps",
]
