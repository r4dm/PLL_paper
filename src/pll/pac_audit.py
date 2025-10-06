#!/usr/bin/env python
"""pac_audit.py
Phase-Alignment Consistency (PAC) audit for cyclic phase lattices Cn.

This artifact targets decoder-based phases: given a directory containing
`decoder_config.json` and `decoder.pt`, PAC is measured for the requested
layers/parts and compared with random-φ baselines.

Usage example
-------------
python -m pll.pac_audit \
  --texts-file data/texts.jsonl \
  --decoder models/decoder_v2 \
  --weight-layers 0,6,11 --n 7 --beta 0.35 --baseline-trials 50 --per-head
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
# Mapping mode omitted in the minimal artifact; decoder path is the supported one.
compute_token_polarities = None  # placeholder for compatibility

from pll.geometry_helpers import (
    _get_attn_submodule,
    _get_heads_info,
)


def _angular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # smallest signed difference in [-pi, pi]
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


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


def _fit_phi_for_n(angles: np.ndarray, n: int) -> float:
    """Grid-search phi that minimises mean |Δθ| for fixed n (like _fit_cn but fixed n)."""
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
    return best_phi


def _get_part_weight(model: torch.nn.Module, layer_idx: int, part: str) -> torch.Tensor:
    attn = _get_attn_submodule(model, layer_idx)
    if part == "q_proj":
        if hasattr(attn, "q_proj") and hasattr(attn.q_proj, "weight"):
            return attn.q_proj.weight.detach().to("cpu")
        # fused in_proj fallback
        d_model = attn.embed_dim  # type: ignore[attr-defined]
        return attn.in_proj_weight.detach().to("cpu")[:d_model, :]
    if part == "k_proj":
        if hasattr(attn, "k_proj") and hasattr(attn.k_proj, "weight"):
            return attn.k_proj.weight.detach().to("cpu")
        d_model = attn.embed_dim  # type: ignore[attr-defined]
        return attn.in_proj_weight.detach().to("cpu")[d_model: 2 * d_model, :]
    if part == "v_proj":
        if hasattr(attn, "v_proj") and hasattr(attn.v_proj, "weight"):
            return attn.v_proj.weight.detach().to("cpu")
        d_model = attn.embed_dim  # type: ignore[attr-defined]
        return attn.in_proj_weight.detach().to("cpu")[2 * d_model: 3 * d_model, :]
    if part in {"o_proj", "out_proj"}:
        if hasattr(attn, "o_proj") and hasattr(attn.o_proj, "weight"):
            return attn.o_proj.weight.detach().to("cpu")
        if hasattr(attn, "out_proj") and hasattr(attn.out_proj, "weight"):
            return attn.out_proj.weight.detach().to("cpu")
    raise ValueError(f"Unsupported part or not found: {part}")


def _phi_for_layer_part(model: torch.nn.Module, layer: int, part: str, n: int) -> Tuple[float, float, float]:
    """Return (phi, var2, var3) where phi is best phase for fixed n using PCA(2) on row-normalised weights."""
    w = _get_part_weight(model, layer, part)
    x = w.float().cpu().numpy()
    x = _row_normalize(x)
    coords2d = _pca_2d(x)
    angles = _angles_2d(coords2d)
    phi = _fit_phi_for_n(angles, n)
    # var explained for rough sanity
    x_center = x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x_center, full_matrices=False)
    power = (s ** 2)
    denom = float(power.sum()) + 1e-12
    var2 = float(power[: min(2, power.shape[0])].sum()) / denom
    var3 = float(power[: min(3, power.shape[0])].sum()) / denom
    return phi, var2, var3


def _phi_for_weight_sub(w_sub: torch.Tensor, n: int) -> Tuple[float, float, float]:
    x = w_sub.float().cpu().numpy()
    x = _row_normalize(x)
    coords2d = _pca_2d(x)
    angles = _angles_2d(coords2d)
    phi = _fit_phi_for_n(angles, n)
    x_center = x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x_center, full_matrices=False)
    power = (s ** 2)
    denom = float(power.sum()) + 1e-12
    var2 = float(power[: min(2, power.shape[0])].sum()) / denom
    var3 = float(power[: min(3, power.shape[0])].sum()) / denom
    return phi, var2, var3


def _phase_from_probs(probs: np.ndarray, n: int) -> Optional[float]:
    """Return circular mean phase in [0, 2π) from per-polarity probs over Cn."""
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


def _nearest_vertex_delta(theta: float, phi: float, n: int) -> float:
    """Smallest absolute angular distance from theta to any vertex of grid phi + 2πk/n."""
    rays = (2 * math.pi / n) * np.arange(n) + phi
    diffs = np.abs(_angular_diff(np.array([theta]), rays))
    return float(diffs.min())


def _compute_token_phases_with_mapping(texts: Sequence[str], mapping_path: Path, *, device: str, layers: Optional[Sequence[int]], n: int) -> List[List[Optional[float]]]:
    if compute_token_polarities is None:
        raise SystemExit("mapping-based PAC is not shipped in the minimal PLL artifact")
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    token_pols, energies_list = compute_token_polarities(  # type: ignore[assignment]
        list(texts), mapping, device=device, layers=layers, return_energy=True
    )
    phases: List[List[Optional[float]]] = []
    for energies in energies_list:
        # energies: [seq, rank]
        seq_phases: List[Optional[float]] = []
        for j in range(energies.shape[0]):
            e = energies[j]
            # convert to probs
            s = float(e.sum())
            if s <= 1e-12:
                seq_phases.append(None)
                continue
            probs = e / s
            ph = _phase_from_probs(probs, n=n)
            seq_phases.append(ph)
        phases.append(seq_phases)
    return phases


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


def pac_audit(
    texts: Sequence[str],
    *,
    device: str,
    mapping: Optional[Path],
    decoder: Optional[Path],
    weight_layers: Sequence[int],
    phase_layers: Optional[Sequence[int]],
    n: int,
    part: str,
    beta: float,
    baseline_trials: int = 0,
    per_head: bool = False,
) -> list[dict]:
    dev = _resolve_device(device)
    model = get_model(device=dev)

    # Token phases from mapping energies or decoder probs
    if mapping is not None:
        token_phases = _compute_token_phases_with_mapping(texts, mapping, device=dev, layers=phase_layers, n=n)
    elif decoder is not None:
        token_phases = _compute_token_phases_with_decoder(texts, decoder, device=dev, layers=phase_layers, n=n)
    else:
        raise SystemExit("either --mapping or --decoder must be provided")

    # Compute phi for each layer (fixed n)
    tol = beta * math.pi / max(1, n)
    all_apc = []
    print("== PAC Audit ==")
    print(f"Device: {dev} | Part: {part} | n={n} | tol={tol:.4f} rad")
    head_dim, n_heads, n_kv = _get_heads_info(model)
    json_out: list[dict] = []
    for layer in weight_layers:
        if not per_head:
            try:
                phi, var2, var3 = _phi_for_layer_part(model, layer, part, n)
            except Exception as exc:
                print(f"Layer {layer}: error extracting phi for {part}: {exc}")
                continue
            total = 0
            ok = 0
            for seq_ph in token_phases:
                for ph in seq_ph:
                    if ph is None:
                        continue
                    d = _nearest_vertex_delta(ph, phi, n)
                    total += 1
                    if d <= tol:
                        ok += 1
            apc = (ok / total) if total > 0 else float('nan')
            all_apc.append(apc)

            # Random-phi baseline (uniform φ in [0, 2π))
            b_mean = float('nan')
            b_p = float('nan')
            if baseline_trials and total > 0:
                rng = np.random.default_rng(seed=42)
                vals = []
                for _ in range(max(1, int(baseline_trials))):
                    phi_r = float(rng.uniform(0.0, 2 * math.pi))
                    ok_r = 0
                    for seq_ph in token_phases:
                        for ph in seq_ph:
                            if ph is None:
                                continue
                            d = _nearest_vertex_delta(ph, phi_r, n)
                            if d <= tol:
                                ok_r += 1
                    vals.append(ok_r / total)
                arr = np.asarray(vals, dtype=float)
                b_mean = float(arr.mean())
                # one-sided p-value: P(APC_baseline >= APC_obs)
                b_p = float((arr >= apc).mean())

            if baseline_trials and total > 0:
                print(
                    f"Layer {layer}: phi={phi:.3f} | var2={var2:.3f} var3={var3:.3f} | "
                    f"APC={apc:.3f} (ok={ok}/tot={total}) | baseline_mean={b_mean:.3f} p={b_p:.3f}"
                )
            else:
                print(
                    f"Layer {layer}: phi={phi:.3f} | var2={var2:.3f} var3={var3:.3f} | "
                    f"APC={apc:.3f} (ok={ok}/tot={total})"
                )
            json_out.append({
                "layer": int(layer),
                "part": str(part),
                "apc": float(apc),
                "ok": int(ok),
                "total": int(total),
                "phi": float(phi),
                "var2": float(var2),
                "var3": float(var3),
                "baseline_mean": float(b_mean) if baseline_trials and total>0 else None,
                "p": float(b_p) if baseline_trials and total>0 else None,
            })
        else:
            # per-head phi and APC
            try:
                attn = _get_attn_submodule(model, layer)
            except Exception as exc:
                print(f"Layer {layer}: error getting attn submodule: {exc}")
                continue
            head_results: List[Tuple[int, float, float, float, float, float]] = []
            # prepare weight tensor and head ranges
            if part == "o_proj" or part == "out_proj":
                w_tensor = None
                if hasattr(attn, "o_proj"):
                    w_tensor = attn.o_proj.weight.detach().to("cpu")
                elif hasattr(attn, "out_proj"):
                    w_tensor = attn.out_proj.weight.detach().to("cpu")
                if w_tensor is None or not n_heads or not head_dim:
                    print(f"Layer {layer}: cannot split per-head for {part}")
                    continue
                for h in range(int(n_heads)):
                    c0 = h * int(head_dim)
                    c1 = c0 + int(head_dim)
                    sub = w_tensor[:, c0:c1]
                    try:
                        phi_h, var2_h, var3_h = _phi_for_weight_sub(sub, n)
                    except Exception:
                        continue
                    total = 0
                    ok = 0
                    for seq_ph in token_phases:
                        for ph in seq_ph:
                            if ph is None:
                                continue
                            d = _nearest_vertex_delta(ph, phi_h, n)
                            total += 1
                            if d <= tol:
                                ok += 1
                    apc_h = (ok / total) if total > 0 else float('nan')
                    b_mean = float('nan')
                    b_p = float('nan')
                    if baseline_trials and total > 0:
                        rng = np.random.default_rng(seed=42)
                        vals = []
                        for _ in range(max(1, int(baseline_trials))):
                            phi_r = float(rng.uniform(0.0, 2 * math.pi))
                            ok_r = 0
                            for seq_ph in token_phases:
                                for ph in seq_ph:
                                    if ph is None:
                                        continue
                                    d = _nearest_vertex_delta(ph, phi_r, n)
                                    if d <= tol:
                                        ok_r += 1
                            vals.append(ok_r / total)
                        arr = np.asarray(vals, dtype=float)
                        b_mean = float(arr.mean())
                        b_p = float((arr >= apc_h).mean())
                    head_results.append((h, apc_h, b_mean, b_p, var2_h, var3_h))
                # print per-head results sorted by APC desc
                head_results.sort(key=lambda t: (-(t[1] if not (t[1] != t[1]) else -1.0)))
                for h, apc_h, b_mean, b_p, var2_h, var3_h in head_results:
                    print(
                        f"Layer {layer}.{part}.h{h}: APC={apc_h:.3f} | baseline_mean={b_mean:.3f} p={b_p:.3f} | var2={var2_h:.3f} var3={var3_h:.3f}"
                    )
                    json_out.append({
                        "layer": int(layer),
                        "part": str(part),
                        "head": int(h),
                        "apc": float(apc_h),
                        "phi": None,
                        "var2": float(var2_h),
                        "var3": float(var3_h),
                        "baseline_mean": float(b_mean) if baseline_trials and not (b_mean != b_mean) else None,
                        "p": float(b_p) if baseline_trials and not (b_p != b_p) else None,
                    })
            else:
                # q/k/v → split by rows
                w_tensor = None
                if part == "q_proj" and hasattr(attn, "q_proj"):
                    w_tensor = attn.q_proj.weight.detach().to("cpu")
                    h_count = int(n_heads or 0)
                elif part == "k_proj" and hasattr(attn, "k_proj"):
                    w_tensor = attn.k_proj.weight.detach().to("cpu")
                    h_count = int(n_kv or n_heads or 0)
                elif part == "v_proj" and hasattr(attn, "v_proj"):
                    w_tensor = attn.v_proj.weight.detach().to("cpu")
                    h_count = int(n_kv or n_heads or 0)
                else:
                    print(f"Layer {layer}: cannot split per-head for {part}")
                    continue
                if w_tensor is None or not head_dim or h_count <= 0:
                    print(f"Layer {layer}: invalid head split params for {part}")
                    continue
                head_results = []
                for h in range(h_count):
                    r0 = h * int(head_dim)
                    r1 = r0 + int(head_dim)
                    sub = w_tensor[r0:r1, :]
                    try:
                        phi_h, var2_h, var3_h = _phi_for_weight_sub(sub, n)
                    except Exception:
                        continue
                    total = 0
                    ok = 0
                    for seq_ph in token_phases:
                        for ph in seq_ph:
                            if ph is None:
                                continue
                            d = _nearest_vertex_delta(ph, phi_h, n)
                            total += 1
                            if d <= tol:
                                ok += 1
                    apc_h = (ok / total) if total > 0 else float('nan')
                    b_mean = float('nan')
                    b_p = float('nan')
                    if baseline_trials and total > 0:
                        rng = np.random.default_rng(seed=42)
                        vals = []
                        for _ in range(max(1, int(baseline_trials))):
                            phi_r = float(rng.uniform(0.0, 2 * math.pi))
                            ok_r = 0
                            for seq_ph in token_phases:
                                for ph in seq_ph:
                                    if ph is None:
                                        continue
                                    d = _nearest_vertex_delta(ph, phi_r, n)
                                    if d <= tol:
                                        ok_r += 1
                            vals.append(ok_r / total)
                        arr = np.asarray(vals, dtype=float)
                        b_mean = float(arr.mean())
                        b_p = float((arr >= apc_h).mean())
                    head_results.append((h, apc_h, b_mean, b_p, var2_h, var3_h))
                head_results.sort(key=lambda t: (-(t[1] if not (t[1] != t[1]) else -1.0)))
                for h, apc_h, b_mean, b_p, var2_h, var3_h in head_results:
                    print(
                        f"Layer {layer}.{part}.h{h}: APC={apc_h:.3f} | baseline_mean={b_mean:.3f} p={b_p:.3f} | var2={var2_h:.3f} var3={var3_h:.3f}"
                    )
                    json_out.append({
                        "layer": int(layer),
                        "part": str(part),
                        "head": int(h),
                        "apc": float(apc_h),
                        "phi": None,
                        "var2": float(var2_h),
                        "var3": float(var3_h),
                        "baseline_mean": float(b_mean) if baseline_trials and not (b_mean != b_mean) else None,
                        "p": float(b_p) if baseline_trials and not (b_p != b_p) else None,
                    })

    if all_apc:
        valid = [a for a in all_apc if not (a != a)]  # drop NaN
        if valid:
            print(f"APC mean={np.mean(valid):.3f} median={np.median(valid):.3f}")

    return json_out


def _load_texts(args: argparse.Namespace) -> List[str]:
    texts: List[str] = []
    if getattr(args, "texts_file", None):
        path = Path(args.texts_file)
        if not path.exists():
            raise FileNotFoundError(f"texts_file not found: {path}")
        # Support JSONL with {"text": ...} or plain TXT (one per line)
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
    else:
        texts = list(getattr(args, "texts", ["Blue * Yellow = Green"]))
    max_n = int(getattr(args, "max_texts", 0) or 0)
    if max_n > 0:
        texts = texts[:max_n]
    return texts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PAC-audit: angle–polarity consistency vs Cn lattice")
    p.add_argument("--texts", type=str, nargs="*", default=["Blue * Yellow = Green"], help="Input texts to analyse")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Torch device")
    p.add_argument("--texts-file", type=str, help="Path to TXT or JSONL with texts (JSONL expects {'text': ...})")
    p.add_argument("--max-texts", type=int, default=0, help="Use at most N texts from --texts-file (0 = all)")
    p.add_argument("--mapping", type=Path, help="Path to neuron→polarity mapping JSON")
    p.add_argument("--decoder", type=Path, help="Path to trained decoder directory (overrides mapping)")
    p.add_argument("--weight-layers", type=str, default="0,6,11", help="Comma-separated layer indices for weight phi")
    p.add_argument("--phase-layers", type=int, nargs="+", help="Transformer layer indices for token phases (e.g. -1)")
    p.add_argument("--n", type=int, default=7, help="Cn order for audit (e.g. 7)")
    p.add_argument("--part", type=str, default="o_proj", choices=["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"], help="Attention sub-part for phi")
    p.add_argument("--beta", type=float, default=0.35, help="Tolerance beta: tol = beta*pi/n")
    p.add_argument("--baseline-trials", type=int, default=0, help="Random-phi baseline trials (0 disables baseline)")
    p.add_argument("--per-head", action="store_true", help="Compute per-head APC (splitting q/k/v by rows, o by columns)")
    p.add_argument("--out", type=Path, help="Optional JSON output path with per-item summaries")
    return p


def main() -> None:
    args = build_parser().parse_args()
    weight_layers = [int(x) for x in args.weight_layers.split(",") if x.strip() != ""]
    texts = _load_texts(args)
    results = pac_audit(
        texts,
        device=args.device,
        mapping=args.mapping if args.mapping else None,
        decoder=args.decoder if args.decoder else None,
        weight_layers=weight_layers,
        phase_layers=args.phase_layers,
        n=int(args.n),
        part=str(args.part),
        beta=float(args.beta),
        baseline_trials=int(args.baseline_trials),
        per_head=bool(args.per_head),
    )
    if args.out:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open('w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Wrote JSON to {args.out}")
        except Exception as exc:
            print(f"Failed to write JSON to {args.out}: {exc}")


if __name__ == "__main__":
    main()
