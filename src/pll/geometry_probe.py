from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# Reuse the project's cached loader to avoid duplicate HF loads
from pll.runtime import get_model


@dataclass
class PolygonFit:
    n: int
    phi: float
    coverage: float
    mean_abs_delta_deg: float
    median_abs_delta_deg: float


@dataclass
class DpnIndicators:
    n: int
    s1_sum_norm_over_sqrt_n: float
    s2_antipair_cos: float
    passed: bool


@dataclass
class StructMetrics:
    ray_cov: float
    mass_cov: float


@dataclass
class TplMetrics:
    phi: float
    ray_cov: float
    mass_cov: float


@dataclass
class BaselineStats:
    trials: int
    cn_cov12_mean: float
    cn_cov12_p_value: float
    s1_mean: float
    s1_p_value: float
    s2_mean: float
    s2_p_value: float
    ray_cov_mean: float = float('nan')
    ray_cov_p_value: float = float('nan')
    mass_cov_mean: float = float('nan')
    mass_cov_p_value: float = float('nan')


@dataclass
class PartReport:
    part: str
    rows: int
    hidden: int
    best_cn: Optional[PolygonFit]
    dpn: Optional[DpnIndicators]
    struct: Optional[StructMetrics] = None
    tpl: Optional[TplMetrics] = None
    baseline: Optional[BaselineStats] = None
    pca2_var_explained: Optional[float] = None
    pca3_var_explained: Optional[float] = None
    multi3_extra: Optional[float] = None
    error: Optional[str] = None


@dataclass
class LayerReport:
    layer: int
    parts: List[PartReport]


@dataclass
class FullReport:
    model_dir: str
    device: str
    layers: List[LayerReport]


def _resolve_device(spec: str | None) -> str:
    if spec is None or spec == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return spec


def _get_attn_submodule(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    return model.get_submodule(f"model.layers.{layer_idx}.self_attn")


def _get_attn_q_weight(model: torch.nn.Module, layer_idx: int) -> torch.Tensor:
    attn = _get_attn_submodule(model, layer_idx)
    if hasattr(attn, "q_proj") and hasattr(attn.q_proj, "weight"):
        return attn.q_proj.weight.detach().to("cpu")
    # Fallback for fused in_proj
    d_model = attn.embed_dim  # type: ignore[attr-defined]
    w = attn.in_proj_weight.detach().to("cpu")
    return w[:d_model, :]


def _get_attn_k_weight(model: torch.nn.Module, layer_idx: int) -> torch.Tensor:
    attn = _get_attn_submodule(model, layer_idx)
    if hasattr(attn, "k_proj") and hasattr(attn.k_proj, "weight"):
        return attn.k_proj.weight.detach().to("cpu")
    d_model = attn.embed_dim  # type: ignore[attr-defined]
    w = attn.in_proj_weight.detach().to("cpu")
    return w[d_model: 2 * d_model, :]


def _get_attn_v_weight(model: torch.nn.Module, layer_idx: int) -> torch.Tensor:
    attn = _get_attn_submodule(model, layer_idx)
    if hasattr(attn, "v_proj") and hasattr(attn.v_proj, "weight"):
        return attn.v_proj.weight.detach().to("cpu")
    d_model = attn.embed_dim  # type: ignore[attr-defined]
    w = attn.in_proj_weight.detach().to("cpu")
    return w[2 * d_model: 3 * d_model, :]


def _get_attn_o_weight(model: torch.nn.Module, layer_idx: int) -> torch.Tensor:
    attn = _get_attn_submodule(model, layer_idx)
    # Prefer o_proj, fallback to out_proj
    if hasattr(attn, "o_proj") and hasattr(attn.o_proj, "weight"):
        return attn.o_proj.weight.detach().to("cpu")
    if hasattr(attn, "out_proj") and hasattr(attn.out_proj, "weight"):
        return attn.out_proj.weight.detach().to("cpu")
    raise AttributeError("Neither o_proj nor out_proj found in self_attn")


def _row_normalize(w: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
    return w / norms


def _pca_2d(x: np.ndarray) -> np.ndarray:
    # x: [rows, dim]
    x = x - x.mean(axis=0, keepdims=True)
    # Compute top-2 right singular vectors via np.linalg.svd on covariance proxy
    # For stability, do thin SVD on [rows, dim]
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:2].T  # [dim, 2]
    return x @ comps  # [rows, 2]


def _pca_var_explained(x: np.ndarray) -> Tuple[float, float]:
    """Возвращает (var2, var3): доли объяснённой дисперсии для топ-2 и топ-3 ПК."""
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    power = (s ** 2)
    denom = float(power.sum()) + 1e-12
    var2 = float(power[: min(2, power.shape[0])].sum()) / denom
    var3 = float(power[: min(3, power.shape[0])].sum()) / denom
    return var2, var3


def _struct_cov_metrics(coords2d: np.ndarray, n: int, phi: float, *, beta: float = 0.35, ray_occ_frac: float = 0.5) -> Tuple[float, float]:
    """Структурные метрики покрытия: (ray_cov, mass_cov).
    tol = beta * pi / n, учитываем phi путём сдвига углов точек."""
    if coords2d.shape[0] == 0 or n <= 0:
        return 0.0, 0.0
    rows = coords2d.shape[0]
    tol = beta * math.pi / max(1, n)
    angles = _angles_2d(coords2d)
    angles = (angles - phi) % (2 * math.pi)
    rays = (2 * math.pi / n) * np.arange(n)
    diffs = np.abs(_angular_diff(angles[:, None], rays[None, :]))  # [rows, n]
    nearest_idx = np.argmin(diffs, axis=1)
    nearest_abs = diffs[np.arange(rows), nearest_idx]
    mass_cov = float((nearest_abs <= tol).mean())
    occ = np.zeros(n, dtype=int)
    for k in range(n):
        occ[k] = int(((nearest_idx == k) & (nearest_abs <= tol)).sum())
    occ_thresh = max(1.0, ray_occ_frac * (rows / max(1, n)))
    ray_cov = float((occ >= occ_thresh).mean())
    return ray_cov, mass_cov


def _random_baseline(
    rows: int,
    n: int,
    *,
    trials: int,
    cov12_obs: float,
    s1_obs: Optional[float],
    s2_obs: Optional[float],
    angle_tol_deg: float = 12.0,
    struct_beta: Optional[float] = None,
    ray_occ_frac: Optional[float] = None,
    struct_obs: Optional[Tuple[float, float]] = None,
) -> BaselineStats:
    """Монте-Карло baseline для равномерных углов на окружности.
    Возвращает средние по метрикам и односторонние p-value:
      - cov12: P(cov12 >= cov12_obs)
      - s1:    P(s1 <= s1_obs)
      - s2:    P(s2 >= s2_obs)
    Если s1_obs/s2_obs отсутствуют, p-value = nan.
    """
    rng = np.random.default_rng(seed=42)
    cov12_vals: List[float] = []
    s1_vals: List[float] = []
    s2_vals: List[float] = []
    ray_vals: List[float] = []
    mass_vals: List[float] = []

    # helper to get coverage and phi for random angles
    def _fit_cov12(angles: np.ndarray) -> Tuple[float, float]:
        best = _fit_cn(angles, n, angle_tol_deg=angle_tol_deg)
        return best.coverage, best.phi

    for _ in range(max(1, trials)):
        rand_angles = rng.uniform(0.0, 2 * math.pi, size=rows)
        cov12, phi = _fit_cov12(rand_angles)
        cov12_vals.append(cov12)
        # Build unit coords and compute DPN indicators as in main flow
        coords = np.stack([np.cos(rand_angles), np.sin(rand_angles)], axis=1)
        means = _cluster_means_on_vertices(coords, n, phi)
        s1, s2 = _dpn_indicators(means)
        s1_vals.append(s1)
        s2_vals.append(s2)
        if struct_beta is not None and ray_occ_frac is not None:
            ray_cov, mass_cov = _struct_cov_metrics(coords, n, phi, beta=struct_beta, ray_occ_frac=ray_occ_frac)
            ray_vals.append(ray_cov)
            mass_vals.append(mass_cov)

    cov12_arr = np.asarray(cov12_vals)
    s1_arr = np.asarray(s1_vals)
    s2_arr = np.asarray(s2_vals)
    # p-values
    p_cov12 = float((cov12_arr >= cov12_obs).mean())
    if s1_obs is None:
        p_s1 = float('nan')
    else:
        p_s1 = float((s1_arr <= s1_obs).mean())
    if s2_obs is None:
        p_s2 = float('nan')
    else:
        p_s2 = float((s2_arr >= s2_obs).mean())

    ray_mean = float('nan')
    p_ray = float('nan')
    mass_mean = float('nan')
    p_mass = float('nan')
    if struct_beta is not None and ray_occ_frac is not None and struct_obs is not None and len(ray_vals) > 0:
        ray_arr = np.asarray(ray_vals)
        mass_arr = np.asarray(mass_vals)
        ray_mean = float(ray_arr.mean())
        mass_mean = float(mass_arr.mean())
        ray_obs, mass_obs = struct_obs
        p_ray = float((ray_arr >= ray_obs).mean())
        p_mass = float((mass_arr >= mass_obs).mean())

    return BaselineStats(
        trials=trials,
        cn_cov12_mean=float(cov12_arr.mean()),
        cn_cov12_p_value=p_cov12,
        s1_mean=float(s1_arr.mean()),
        s1_p_value=p_s1,
        s2_mean=float(s2_arr.mean()),
        s2_p_value=p_s2,
        ray_cov_mean=ray_mean,
        ray_cov_p_value=p_ray,
        mass_cov_mean=mass_mean,
        mass_cov_p_value=p_mass,
    )


def _angles_2d(coords: np.ndarray) -> np.ndarray:
    return np.arctan2(coords[:, 1], coords[:, 0]) % (2 * math.pi)


def _angular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # smallest signed difference in [-pi, pi]
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


def _fit_cn(angles: np.ndarray, n: int, angle_tol_deg: float = 12.0) -> PolygonFit:
    # Grid-search phi to minimise mean |Δθ|
    bins = 360  # 1-degree granularity
    phis = np.linspace(0.0, 2 * math.pi, bins, endpoint=False)
    best_mean = 1e9
    best_phi = 0.0
    best_med = 1e9
    best_cov = 0.0
    grid = (2 * math.pi / n) * np.arange(n)
    for phi in phis:
        # nearest vertex per angle
        nearest = ((angles[:, None] - (grid[None, :] + phi) + math.pi) % (2 * math.pi)) - math.pi
        deltas = np.min(np.abs(nearest), axis=1)
        mean_abs = float(deltas.mean())
        if mean_abs < best_mean:
            best_mean = mean_abs
            best_phi = float(phi)
            best_med = float(np.median(np.abs(deltas)))
            cov = float((np.abs(deltas) <= (angle_tol_deg * math.pi / 180.0)).mean())
            best_cov = cov
    return PolygonFit(
        n=n,
        phi=best_phi,
        coverage=best_cov,
        mean_abs_delta_deg=best_mean * 180.0 / math.pi,
        median_abs_delta_deg=best_med * 180.0 / math.pi,
    )


def _cluster_means_on_vertices(coords: np.ndarray, n: int, phi: float) -> List[np.ndarray]:
    # Assign each point to nearest polygon vertex, return unit mean per vertex
    angles = _angles_2d(coords)
    grid = (2 * math.pi / n) * np.arange(n) + phi
    # indices per row
    nearest = np.argmin(np.abs(_angular_diff(angles[:, None], grid[None, :])), axis=1)
    means: List[np.ndarray] = []
    for k in range(n):
        sel = coords[nearest == k]
        if sel.shape[0] == 0:
            means.append(np.array([0.0, 0.0], dtype=float))
            continue
        m = sel.mean(axis=0)
        nrm = np.linalg.norm(m) + 1e-8
        means.append(m / nrm)
    return means


def _dpn_indicators(means: List[np.ndarray]) -> Tuple[float, float]:
    # S1: ||sum(means)|| / sqrt(n)
    n = len(means)
    S1 = float(np.linalg.norm(np.sum(means, axis=0))) / math.sqrt(max(1, n))
    # S2: average max_j cos(u_i, -u_j)
    arr = np.stack(means, axis=0)  # [n, 2]
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    U = arr / norms
    # cos(u_i, -u_j) = - u_i · u_j
    G = U @ U.T  # [n, n]
    anti = -G
    # ignore self (i==j)
    np.fill_diagonal(anti, -np.inf)
    best_per_i = anti.max(axis=1)
    S2 = float(np.mean(best_per_i))
    return S1, S2


def _pca_var_explained(x: np.ndarray, top_k: int = 3) -> Tuple[float, float]:
    """Возвращает (var2, var3): доли объяснённой дисперсии топ-2 и топ-3 ПК.
    Если размерность <3, var3 считает по доступным компонентам.
    """
    x = x - x.mean(axis=0, keepdims=True)
    # SVD на центрированных данных
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    power = (s ** 2)
    total = float(power.sum()) + 1e-12
    var2 = float(power[: min(2, power.shape[0])].sum()) / total
    var3 = float(power[: min(3, power.shape[0])].sum()) / total
    return var2, var3


def _struct_cov_metrics(coords2d: np.ndarray, n: int, phi: float, *, beta: float = 0.35, ray_occ_frac: float = 0.5) -> Tuple[float, float]:
    """Структурные метрики покрытия: (ray_cov, mass_cov).
    - mass_cov: доля точек в динамическом допуске tol=beta*pi/n от ближайшего луча (с учётом phi)
    - ray_cov: доля лучей с занятостью >= ray_occ_frac * (rows/n)
    """
    if coords2d.shape[0] == 0:
        return 0.0, 0.0
    rows = coords2d.shape[0]
    tol = beta * math.pi / max(1, n)
    angles = _angles_2d(coords2d)
    angles = (angles - phi) % (2 * math.pi)
    rays = (2 * math.pi / n) * np.arange(n)
    diffs = np.abs(_angular_diff(angles[:, None], rays[None, :]))  # [rows, n]
    nearest_idx = np.argmin(diffs, axis=1)
    nearest_abs = diffs[np.arange(rows), nearest_idx]
    mass_cov = float((nearest_abs <= tol).mean())
    # per-ray occupancy under tol
    occ = np.zeros(n, dtype=int)
    for k in range(n):
        occ[k] = int(((nearest_idx == k) & (nearest_abs <= tol)).sum())
    occ_thresh = max(1.0, ray_occ_frac * (rows / max(1, n)))
    ray_cov = float((occ >= occ_thresh).mean())
    return ray_cov, mass_cov


# (удалена старая версия _random_baseline без p-value)


def analyze_part(rows2d: np.ndarray, *, max_n: int = 12) -> Tuple[Optional[PolygonFit], Optional[DpnIndicators]]:
    if rows2d.shape[0] < 8:
        return None, None
    angles = _angles_2d(rows2d)
    fits: List[PolygonFit] = []
    for n in range(3, max_n + 1):
        fits.append(_fit_cn(angles, n))
    # choose by (coverage desc, mean_abs_delta asc)
    fits.sort(key=lambda f: (-f.coverage, f.mean_abs_delta_deg))
    best = fits[0] if fits else None
    dpn: Optional[DpnIndicators] = None
    if best and best.coverage >= 0.25:
        means = _cluster_means_on_vertices(rows2d, best.n, best.phi)
        s1, s2 = _dpn_indicators(means)
        passed = (s1 <= 0.30) and (s2 >= 0.70)
        dpn = DpnIndicators(n=best.n, s1_sum_norm_over_sqrt_n=s1, s2_antipair_cos=s2, passed=passed)
    return best, dpn


def extract_q_rows2d(model: torch.nn.Module, layer_idx: int) -> Tuple[np.ndarray, Tuple[int, int], float, float]:
    w = _get_attn_q_weight(model, layer_idx)  # [rows, hidden]
    rows, hidden = w.shape
    x = w.float().cpu().numpy()
    x = _row_normalize(x)
    coords2d = _pca_2d(x)
    var2, var3 = _pca_var_explained(x)
    return coords2d, (rows, hidden), var2, var3


def _extract_rows2d_from_weight(w: torch.Tensor) -> Tuple[np.ndarray, Tuple[int, int], float, float]:
    rows, hidden = w.shape
    x = w.float().cpu().numpy()
    x = _row_normalize(x)
    coords2d = _pca_2d(x)
    var2, var3 = _pca_var_explained(x)
    return coords2d, (rows, hidden), var2, var3


def extract_k_rows2d(model: torch.nn.Module, layer_idx: int) -> Tuple[np.ndarray, Tuple[int, int], float, float]:
    w = _get_attn_k_weight(model, layer_idx)
    return _extract_rows2d_from_weight(w)


def extract_v_rows2d(model: torch.nn.Module, layer_idx: int) -> Tuple[np.ndarray, Tuple[int, int], float, float]:
    w = _get_attn_v_weight(model, layer_idx)
    return _extract_rows2d_from_weight(w)


def extract_o_rows2d(model: torch.nn.Module, layer_idx: int) -> Tuple[np.ndarray, Tuple[int, int], float, float]:
    w = _get_attn_o_weight(model, layer_idx)
    return _extract_rows2d_from_weight(w)


def _get_mlp_module(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    return model.get_submodule(f"model.layers.{layer_idx}.mlp")


def _get_mlp_weights(model: torch.nn.Module, layer_idx: int) -> Dict[str, torch.Tensor]:
    mlp = _get_mlp_module(model, layer_idx)
    out: Dict[str, torch.Tensor] = {}
    # Qwen/LLaMA style
    if hasattr(mlp, "up_proj") and hasattr(mlp.up_proj, "weight"):
        out["mlp_up"] = mlp.up_proj.weight.detach().to("cpu")
    if hasattr(mlp, "gate_proj") and hasattr(mlp.gate_proj, "weight"):
        out["mlp_gate"] = mlp.gate_proj.weight.detach().to("cpu")
    if hasattr(mlp, "down_proj") and hasattr(mlp.down_proj, "weight"):
        out["mlp_down"] = mlp.down_proj.weight.detach().to("cpu")
    # Common alt names
    if not out:
        if hasattr(mlp, "fc1") and hasattr(mlp.fc1, "weight"):
            out["mlp_fc1"] = mlp.fc1.weight.detach().to("cpu")
        if hasattr(mlp, "fc2") and hasattr(mlp.fc2, "weight"):
            out["mlp_fc2"] = mlp.fc2.weight.detach().to("cpu")
    return out


def _get_heads_info(model: torch.nn.Module) -> Tuple[int, int, int]:
    cfg = getattr(model, 'config', None)
    head_dim = int(getattr(cfg, 'head_dim', 0) or 0)
    n_heads = int(getattr(cfg, 'num_attention_heads', 0) or 0)
    n_kv = int(getattr(cfg, 'num_key_value_heads', n_heads) or n_heads)
    if head_dim <= 0 or n_heads <= 0:
        # Fallback attempt via hidden size
        hidden = int(getattr(cfg, 'hidden_size', 0) or 0)
        if hidden and n_heads:
            head_dim = hidden // n_heads
    return head_dim, n_heads, n_kv


def run_probe(model_dir: Path, *, device: str, layers: List[int], max_n: int, baseline_trials: int = 0, struct_beta: float = 0.35, ray_occ_frac: float = 0.5, per_head: bool = False) -> FullReport:
    model = get_model(device=device)
    head_dim, n_heads, n_kv = _get_heads_info(model)
    reports: List[LayerReport] = []
    for layer in layers:
        parts: List[PartReport] = []
        # Attention projections: q,k,v,o
        for name, extractor in [
            ("q_proj", extract_q_rows2d),
            ("k_proj", extract_k_rows2d),
            ("v_proj", extract_v_rows2d),
            ("o_proj", extract_o_rows2d),
        ]:
            try:
                coords2d, shape, var2, var3 = extractor(model, layer)
                best, dpn = analyze_part(coords2d, max_n=max_n)
                struct = None
                tpl = None
                baseline = None
                if best is not None:
                    ray_cov, mass_cov = _struct_cov_metrics(coords2d, best.n, best.phi, beta=struct_beta, ray_occ_frac=ray_occ_frac)
                    struct = StructMetrics(ray_cov=ray_cov, mass_cov=mass_cov)
                    # TPL (n=3) quick fit and structural metrics
                    angles = _angles_2d(coords2d)
                    tpl_fit = _fit_cn(angles, 3)
                    tpl_ray, tpl_mass = _struct_cov_metrics(coords2d, 3, tpl_fit.phi, beta=struct_beta, ray_occ_frac=ray_occ_frac)
                    tpl = TplMetrics(phi=tpl_fit.phi, ray_cov=tpl_ray, mass_cov=tpl_mass)
                    # random baseline
                    if baseline_trials > 0:
                        cov12_obs = best.coverage
                        s1_obs = None
                        s2_obs = None
                        if dpn is not None:
                            s1_obs = dpn.s1_sum_norm_over_sqrt_n
                            s2_obs = dpn.s2_antipair_cos
                        baseline = _random_baseline(
                            shape[0], best.n,
                            trials=baseline_trials, cov12_obs=cov12_obs,
                            s1_obs=s1_obs, s2_obs=s2_obs,
                            struct_beta=struct_beta, ray_occ_frac=ray_occ_frac,
                            struct_obs=(ray_cov, mass_cov),
                        )
                parts.append(PartReport(part=name, rows=shape[0], hidden=shape[1], best_cn=best, dpn=dpn, struct=struct, tpl=tpl, baseline=baseline, pca2_var_explained=var2, pca3_var_explained=var3, multi3_extra=(var3 - var2) if (var3 is not None and var2 is not None) else None))
            except Exception as exc:
                parts.append(PartReport(part=name, rows=0, hidden=0, best_cn=None, dpn=None, error=str(exc)))

            # Per-head analysis for attention parts
            if per_head and name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
                try:
                    attn = _get_attn_submodule(model, layer)
                    w_tensor: Optional[torch.Tensor] = None
                    if name == "q_proj" and hasattr(attn, "q_proj"):
                        w_tensor = attn.q_proj.weight.detach().to("cpu")
                        h_count = n_heads
                        # rows grouped by head
                        for h in range(max(1, h_count)):
                            r0 = h * max(1, head_dim)
                            r1 = r0 + max(1, head_dim)
                            sub = w_tensor[r0:r1, :]
                            coords2d, shape, var2, var3 = _extract_rows2d_from_weight(sub)
                            best, dpn = analyze_part(coords2d, max_n=max_n)
                            struct = baseline = None
                            if best is not None:
                                ray_cov, mass_cov = _struct_cov_metrics(coords2d, best.n, best.phi, beta=struct_beta, ray_occ_frac=ray_occ_frac)
                                struct = StructMetrics(ray_cov=ray_cov, mass_cov=mass_cov)
                                if baseline_trials > 0:
                                    cov12_obs = best.coverage
                                    s1_obs = dpn.s1_sum_norm_over_sqrt_n if dpn else None
                                    s2_obs = dpn.s2_antipair_cos if dpn else None
                                    baseline = _random_baseline(shape[0], best.n, trials=baseline_trials, cov12_obs=cov12_obs, s1_obs=s1_obs, s2_obs=s2_obs, struct_beta=struct_beta, ray_occ_frac=ray_occ_frac, struct_obs=(ray_cov, mass_cov))
                            parts.append(PartReport(part=f"q_proj.h{h}", rows=shape[0], hidden=shape[1], best_cn=best, dpn=dpn, struct=struct, baseline=baseline, pca2_var_explained=var2, pca3_var_explained=var3))

                    if name == "k_proj" and hasattr(attn, "k_proj"):
                        w_tensor = attn.k_proj.weight.detach().to("cpu")
                        h_count = n_kv if n_kv else n_heads
                        for h in range(max(1, h_count)):
                            r0 = h * max(1, head_dim)
                            r1 = r0 + max(1, head_dim)
                            sub = w_tensor[r0:r1, :]
                            coords2d, shape, var2, var3 = _extract_rows2d_from_weight(sub)
                            best, dpn = analyze_part(coords2d, max_n=max_n)
                            struct = baseline = None
                            if best is not None:
                                ray_cov, mass_cov = _struct_cov_metrics(coords2d, best.n, best.phi, beta=struct_beta, ray_occ_frac=ray_occ_frac)
                                struct = StructMetrics(ray_cov=ray_cov, mass_cov=mass_cov)
                                if baseline_trials > 0:
                                    cov12_obs = best.coverage
                                    s1_obs = dpn.s1_sum_norm_over_sqrt_n if dpn else None
                                    s2_obs = dpn.s2_antipair_cos if dpn else None
                                    baseline = _random_baseline(shape[0], best.n, trials=baseline_trials, cov12_obs=cov12_obs, s1_obs=s1_obs, s2_obs=s2_obs, struct_beta=struct_beta, ray_occ_frac=ray_occ_frac, struct_obs=(ray_cov, mass_cov))
                            parts.append(PartReport(part=f"k_proj.h{h}", rows=shape[0], hidden=shape[1], best_cn=best, dpn=dpn, struct=struct, baseline=baseline, pca2_var_explained=var2, pca3_var_explained=var3))

                    if name == "v_proj" and hasattr(attn, "v_proj"):
                        w_tensor = attn.v_proj.weight.detach().to("cpu")
                        h_count = n_kv if n_kv else n_heads
                        for h in range(max(1, h_count)):
                            r0 = h * max(1, head_dim)
                            r1 = r0 + max(1, head_dim)
                            sub = w_tensor[r0:r1, :]
                            coords2d, shape, var2, var3 = _extract_rows2d_from_weight(sub)
                            best, dpn = analyze_part(coords2d, max_n=max_n)
                            struct = baseline = None
                            if best is not None:
                                ray_cov, mass_cov = _struct_cov_metrics(coords2d, best.n, best.phi, beta=struct_beta, ray_occ_frac=ray_occ_frac)
                                struct = StructMetrics(ray_cov=ray_cov, mass_cov=mass_cov)
                                if baseline_trials > 0:
                                    cov12_obs = best.coverage
                                    s1_obs = dpn.s1_sum_norm_over_sqrt_n if dpn else None
                                    s2_obs = dpn.s2_antipair_cos if dpn else None
                                    baseline = _random_baseline(shape[0], best.n, trials=baseline_trials, cov12_obs=cov12_obs, s1_obs=s1_obs, s2_obs=s2_obs, struct_beta=struct_beta, ray_occ_frac=ray_occ_frac, struct_obs=(ray_cov, mass_cov))
                            parts.append(PartReport(part=f"v_proj.h{h}", rows=shape[0], hidden=shape[1], best_cn=best, dpn=dpn, struct=struct, baseline=baseline, pca2_var_explained=var2, pca3_var_explained=var3))

                    if name == "o_proj":
                        # per-head split by columns
                        w_tensor = None
                        if hasattr(attn, "o_proj"):
                            w_tensor = attn.o_proj.weight.detach().to("cpu")
                        elif hasattr(attn, "out_proj"):
                            w_tensor = attn.out_proj.weight.detach().to("cpu")
                        if w_tensor is not None and n_heads and head_dim:
                            for h in range(n_heads):
                                c0 = h * head_dim
                                c1 = c0 + head_dim
                                sub = w_tensor[:, c0:c1]
                                coords2d, shape, var2, var3 = _extract_rows2d_from_weight(sub)
                                best, dpn = analyze_part(coords2d, max_n=max_n)
                                struct = baseline = None
                                if best is not None:
                                    ray_cov, mass_cov = _struct_cov_metrics(coords2d, best.n, best.phi, beta=struct_beta, ray_occ_frac=ray_occ_frac)
                                    struct = StructMetrics(ray_cov=ray_cov, mass_cov=mass_cov)
                                    if baseline_trials > 0:
                                        cov12_obs = best.coverage
                                        s1_obs = dpn.s1_sum_norm_over_sqrt_n if dpn else None
                                        s2_obs = dpn.s2_antipair_cos if dpn else None
                                        baseline = _random_baseline(shape[0], best.n, trials=baseline_trials, cov12_obs=cov12_obs, s1_obs=s1_obs, s2_obs=s2_obs, struct_beta=struct_beta, ray_occ_frac=ray_occ_frac, struct_obs=(ray_cov, mass_cov))
                                parts.append(PartReport(part=f"o_proj.h{h}", rows=shape[0], hidden=shape[1], best_cn=best, dpn=dpn, struct=struct, baseline=baseline, pca2_var_explained=var2, pca3_var_explained=var3))
                except Exception:
                    pass

        # MLP weights (up/gate/down or fc1/fc2)
        try:
            mlp_ws = _get_mlp_weights(model, layer)
            for mlp_name, w in mlp_ws.items():
                try:
                    coords2d, shape, var2, var3 = _extract_rows2d_from_weight(w)
                    best, dpn = analyze_part(coords2d, max_n=max_n)
                    struct = None
                    baseline = None
                    if best is not None:
                        ray_cov, mass_cov = _struct_cov_metrics(coords2d, best.n, best.phi, beta=struct_beta, ray_occ_frac=ray_occ_frac)
                        struct = StructMetrics(ray_cov=ray_cov, mass_cov=mass_cov)
                        # TPL (n=3)
                        angles = _angles_2d(coords2d)
                        tpl_fit = _fit_cn(angles, 3)
                        tpl_ray, tpl_mass = _struct_cov_metrics(coords2d, 3, tpl_fit.phi, beta=struct_beta, ray_occ_frac=ray_occ_frac)
                        tpl = TplMetrics(phi=tpl_fit.phi, ray_cov=tpl_ray, mass_cov=tpl_mass)
                        if baseline_trials > 0:
                            cov12_obs = best.coverage
                            s1_obs = None
                            s2_obs = None
                            if dpn is not None:
                                s1_obs = dpn.s1_sum_norm_over_sqrt_n
                                s2_obs = dpn.s2_antipair_cos
                            baseline = _random_baseline(
                                shape[0], best.n,
                                trials=baseline_trials, cov12_obs=cov12_obs,
                                s1_obs=s1_obs, s2_obs=s2_obs,
                                struct_beta=struct_beta, ray_occ_frac=ray_occ_frac,
                                struct_obs=(ray_cov, mass_cov),
                            )
                    parts.append(PartReport(part=mlp_name, rows=shape[0], hidden=shape[1], best_cn=best, dpn=dpn, struct=struct, tpl=tpl, baseline=baseline, pca2_var_explained=var2, pca3_var_explained=var3, multi3_extra=(var3 - var2) if (var3 is not None and var2 is not None) else None))
                except Exception as exc:
                    parts.append(PartReport(part=mlp_name, rows=0, hidden=0, best_cn=None, dpn=None, error=str(exc)))
        except Exception:
            pass
        reports.append(LayerReport(layer=layer, parts=parts))

    return FullReport(model_dir=str(model_dir), device=device, layers=reports)


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM weight geometry probe (Cn vs DPN)")
    ap.add_argument("--model-dir", type=Path, default=Path("models/Qwen3-0.6B"))
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"]) 
    ap.add_argument("--layers", type=str, default="0,1", help="comma-separated layer indices")
    ap.add_argument("--max-n", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("runs/geometry_probe_report.json"))
    ap.add_argument("--baseline-trials", type=int, default=0, help="число Монте-Карло прогонов для нулевой гипотезы")
    ap.add_argument("--struct-beta", type=float, default=0.35, help="бета для динамического допуска tol=beta*pi/n")
    ap.add_argument("--ray-occ-frac", type=float, default=0.5, help="минимальная доля занятости луча")
    ap.add_argument("--per-head", action="store_true", help="включить per-head анализ для Q/K/V/O")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]

    report = run_probe(
        args.model_dir,
        device=device,
        layers=layers,
        max_n=args.max_n,
        baseline_trials=args.baseline_trials,
        struct_beta=args.struct_beta,
        ray_occ_frac=args.ray_occ_frac,
        per_head=bool(args.per_head),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    # Human-readable stdout summary
    print("== Geometry Probe Summary ==")
    print(f"Model: {report.model_dir} | Device: {report.device}")
    for layer_report in report.layers:
        print(f"Layer {layer_report.layer}:")
        for part in layer_report.parts:
            if part.best_cn is None:
                print(f"  {part.part}: no data")
                continue
            cn = part.best_cn
            line = (
                f"  {part.part}: Cn_best=n={cn.n}, cov={cn.coverage:.2f}, "
                f"meanΔ={cn.mean_abs_delta_deg:.1f}°, medΔ={cn.median_abs_delta_deg:.1f}°"
            )
            print(line)
            if part.struct is not None:
                print(f"    struct: ray={part.struct.ray_cov:.2f}, mass={part.struct.mass_cov:.2f}")
            if part.tpl is not None:
                print(f"    TPL(n=3): ray={part.tpl.ray_cov:.2f}, mass={part.tpl.mass_cov:.2f}")
            if part.pca2_var_explained is not None:
                print(f"    PCA: var2={part.pca2_var_explained:.3f}, var3={part.pca3_var_explained:.3f}")
                if part.multi3_extra is not None:
                    print(f"    Multi3 proxy: extra={part.multi3_extra:.3f}")
            if part.dpn is not None:
                dp = part.dpn
                print(
                    f"    DPN(n={dp.n}): S1={dp.s1_sum_norm_over_sqrt_n:.2f}, "
                    f"S2={dp.s2_antipair_cos:.2f}, passed={'YES' if dp.passed else 'no'}"
                )
            if part.baseline is not None:
                b = part.baseline
                print(
                    f"    Baseline(trials={b.trials}): cov12_mean={b.cn_cov12_mean:.2f}, p={b.cn_cov12_p_value:.3f}; "
                    f"S1_mean={b.s1_mean:.2f}, p={b.s1_p_value:.3f}; S2_mean={b.s2_mean:.2f}, p={b.s2_p_value:.3f}"
                )


if __name__ == "__main__":
    main()
