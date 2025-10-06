#!/usr/bin/env python3
"""Extract Piecewise-SPS regions for a selected layer/part.

Pipeline:
1. Build top-K activation regions (binary masks over positive activations).
2. Fit lightweight ridge models inside each region.
3. Export linear functionals as symbolic phase sums (SPS/PSPS-style strings).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pll.geometry_helpers import _get_heads_info
from pll.piecewise_sps import (
    FitSpec,
    PspsRegionExport,
    build_piecewise_regions,
    export_region_to_psps,
    fit_linear_in_region,
)
from pll.phase_group import PhaseGroup
from pll.runtime import get_model, _resolve_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract Piecewise-SPS for selected layer part")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--part", type=str, default="mlp_gate", choices=["mlp_gate", "mlp_up", "mlp_down", "o_proj", "out_proj"])
    p.add_argument("--topk-neurons", type=int, default=128)
    p.add_argument("--regions", type=int, default=12)
    p.add_argument("--min-coverage", type=float, default=0.8)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--max-texts", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--head", type=int, default=-1, help="For o_proj/out_proj: head index for per-head PSPS")
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--n", type=int, default=7, help="Phase group size Cn for SPS export")
    p.add_argument("--min-coeff", type=float, default=0.01, help="Coefficient threshold for SPS terms")
    p.add_argument("--max-terms", type=int, default=5)
    return p


def main() -> None:
    args = build_parser().parse_args()

    dev = _resolve_device(str(args.device))
    head_slice: Optional[Tuple[int, int]] = None
    if str(args.part) in {"o_proj", "out_proj"} and int(args.head) >= 0:
        model = get_model(device=dev)
        head_dim, n_heads, _ = _get_heads_info(model)
        h = int(args.head)
        if h < 0 or h >= int(n_heads):
            raise SystemExit(f"--head must be in [0, {n_heads-1}] for part={args.part}")
        c0 = h * int(head_dim)
        c1 = c0 + int(head_dim)
        head_slice = (c0, c1)
    else:
        # ensure model cached for regionizer later
        get_model(device=dev)

    region_res = build_piecewise_regions(
        layer=int(args.layer),
        part=str(args.part),
        topk_neurons=int(args.topk_neurons),
        regions=int(args.regions),
        dataset=str(args.dataset),
        max_texts=int(args.max_texts),
        device=dev,
        batch_size=int(args.batch_size),
        head_slice=head_slice,
    )

    selected_in_dims = list(map(int, region_res.selected_input_dims))
    selected_out_dims = list(map(int, region_res.selected_output_neurons))

    report: Dict[str, object] = {
        "layer": int(args.layer),
        "part": str(args.part),
        "topk": int(args.topk_neurons),
        "regions_requested": int(args.regions),
        "min_coverage": float(args.min_coverage),
        "coverage_kept": float(region_res.coverage_kept),
        "total_tokens": int(region_res.total_tokens),
        "selected_neurons": selected_out_dims,
        "selected_inputs": selected_in_dims,
        "regions": [],
    }

    kept_cov = 0.0
    group = PhaseGroup(int(args.n))
    exports: List[PspsRegionExport] = []
    for reg in region_res.regions:
        if kept_cov >= float(args.min_coverage):
            break
        fit = fit_linear_in_region(
            reg.mask_bits,
            spec=FitSpec(layer=int(args.layer), part=str(args.part), head_slice=head_slice, neuron_indices=None),
            selected_in_dims=selected_in_dims,
            selected_out_dims=selected_out_dims,
            dataset=str(args.dataset),
            max_texts=int(args.max_texts),
            device=dev,
            batch_size=int(args.batch_size),
            l2=float(args.l2),
            max_samples=int(args.max_samples),
        )
        export = export_region_to_psps(
            region_mask=reg.mask_bits,
            coverage=float(reg.coverage),
            W=fit.W,
            b=fit.b,
            selected_in_dims=selected_in_dims,
            group=group,
            min_coeff_abs=float(args.min_coeff),
            max_terms=int(args.max_terms),
        )
        exports.append(export)
        kept_cov += float(reg.coverage)
        report["regions"].append({
            "mask": reg.mask_bits,
            "count": reg.count,
            "coverage": reg.coverage,
            "samples": len(reg.sample_locs),
            "mse": fit.mse,
            "nmse": fit.nmse,
            "r2": fit.r2,
            "phase_sum": export.formula_str,
            "terms": export.terms,
            "bias": export.bias,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved Piecewise-SPS report → {out_path}")


if __name__ == "__main__":
    main()
