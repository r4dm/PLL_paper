#!/usr/bin/env python3
"""Summarize Piecewise‑SPS JSONs into CSV + manifest.

Extracts per‑file coverage_kept and region metrics (R² and formula compactness).
Also writes a per‑region CSV for top regions by R².

Usage:
  python scripts/make_piecewise_psps_figs.py \
      --in runs/piecewise_sps_l0_mlp_gate.json \
      --out figs/piecewise_psps_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _count_terms(phase_str: str) -> int:
    # crude: count occurrences of '+', treat zero-only expressions as 0
    if not isinstance(phase_str, str) or phase_str.strip().startswith('0'):
        return 0
    # split by '+' and count non-empty chunks
    parts = [p.strip() for p in phase_str.split('+')]
    return sum(1 for p in parts if p)


def _collect_regions(obj: Any) -> Tuple[float, List[Dict[str, Any]]]:
    coverage_kept = None
    regions: List[Dict[str, Any]] = []

    def _walk(node: Any):
        nonlocal coverage_kept
        if isinstance(node, dict):
            # coverage
            for k in node.keys():
                if str(k).lower() == 'coverage_kept':
                    try:
                        coverage_kept = float(node[k])
                    except Exception:
                        pass
            # region with r2 and PSPS
            keys = set(k.lower() for k in node.keys())
            if ('r2' in keys or 'r_squared' in keys) and ('phase_sum' in keys or 'psps' in keys):
                r = {
                    'r2': node.get('r2', node.get('R2', node.get('r_squared'))),
                    'phase_sum': node.get('phase_sum', node.get('psps')),
                    'terms': _count_terms(node.get('phase_sum', node.get('psps', ''))),
                }
                regions.append(r)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(obj)
    return (coverage_kept if coverage_kept is not None else float('nan'), regions)


def main() -> None:
    ap = argparse.ArgumentParser(description='Summarize Piecewise‑SPS JSONs into CSV')
    ap.add_argument('--in', dest='inputs', type=Path, nargs='+', required=True)
    ap.add_argument('--out', dest='out_csv', type=Path, required=True)
    ap.add_argument('--top', dest='top_k', type=int, default=10, help='Export top‑K regions by R² to a tiles CSV')
    args = ap.parse_args()

    out_csv: Path = args.out_csv
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = ['file', 'coverage_kept', 'regions', 'frac_r2_ge_0_9', 'median_terms']
    rows: List[Dict[str, Any]] = []
    manifest = {'inputs': [], 'rows': 0}
    tiles: List[Dict[str, Any]] = []

    for p in args.inputs:
        obj = _load(p)
        cov, regs = _collect_regions(obj)
        r2_vals = [float(r.get('r2') or 0.0) for r in regs]
        frac = 0.0
        if r2_vals:
            frac = sum(1 for x in r2_vals if x >= 0.9) / len(r2_vals)
        terms = [int(r.get('terms') or 0) for r in regs]
        med = ''
        if terms:
            st = sorted(terms)
            med = st[len(st)//2]
        rows.append({
            'file': p.name,
            'coverage_kept': cov,
            'regions': len(regs),
            'frac_r2_ge_0_9': f'{frac:.3f}' if r2_vals else '',
            'median_terms': med,
        })
        # tiles
        regs_sorted = sorted(regs, key=lambda r: float(r.get('r2') or 0.0), reverse=True)
        for r in regs_sorted[: max(0, int(args.top_k))]:
            tiles.append({'file': p.name, 'r2': r.get('r2'), 'terms': r.get('terms'), 'phase_sum': r.get('phase_sum')})
        manifest['inputs'].append({'path': str(p), 'sha256': _sha256(p), 'regions': len(regs)})

    # Write summary
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # Write tiles
    tiles_csv = out_dir / 'piecewise_psps_tiles.csv'
    with tiles_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'r2', 'terms', 'phase_sum'])
        w.writeheader()
        for t in tiles:
            w.writerow(t)
    manifest['rows'] = len(rows)
    with (out_dir / 'manifest.json').open('w', encoding='utf-8') as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    print(f'Wrote {len(rows)} summaries → {out_csv}')
    print(f'Wrote tiles → {tiles_csv}')


if __name__ == '__main__':
    main()
