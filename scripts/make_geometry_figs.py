#!/usr/bin/env python3
"""Summarize geometry probe JSON into a CSV and a small manifest.

Reads 1+ JSON files produced by `scripts/geometry_probe.py` and
extracts per (layer, part[, head]) coverage, var2/var3 and, if present, DPN.

Usage:
  python docs/bridge_paper/scripts/make_geometry_figs.py \
      --in runs/geometry_probe_full_0_6_11.json \
      --out docs/bridge_paper/figs/geometry_summary.csv

If keys differ across JSONs, the script will include what it finds and leave
missing fields blank. It also writes `manifest.json` with input hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _flatten_records(obj: Any) -> List[Dict[str, Any]]:
    """Attempt to extract a list of records for (layer, part[, head]).

    The geometry probe scripts in this repo write a list/dict structure with
    fields including some of: layer, part, head, coverage, var2, var3, DPN.
    We try to be robust to minor format changes: walk nested lists/dicts and
    collect dicts that look like per‑item summaries.
    """
    records: List[Dict[str, Any]] = []

    # Special-case known report schema with top-level {layers: [...]}
    if isinstance(obj, dict) and isinstance(obj.get('layers'), list):
        for layer_obj in obj['layers']:
            try:
                layer_id = layer_obj.get('layer')
                parts = layer_obj.get('parts') or []
                for p in parts:
                    rec: Dict[str, Any] = {'layer': layer_id, 'part': p.get('part')}
                    bc = p.get('best_cn') or {}
                    rec['coverage'] = bc.get('coverage')
                    rec['mean_deg'] = bc.get('mean_abs_delta_deg')
                    dpn = p.get('dpn') or {}
                    rec['dpn_s1'] = dpn.get('s1_sum_norm_over_sqrt_n')
                    rec['dpn_s2'] = dpn.get('s2_antipair_cos')
                    records.append(rec)
            except Exception:
                continue
        return records

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            # Heuristic: contains at least a layer and part or coverage/var2
            keys = set(node.keys())
            if {'layer', 'part'} <= keys or {'coverage', 'var2', 'var3'} & keys:
                # Capture only simple JSON‑serializable fields
                rec: Dict[str, Any] = {}
                for k in ['layer', 'part', 'head', 'coverage', 'mean_deg', 'var2', 'var3', 'dpn_s1', 'dpn_s2']:
                    if k in node:
                        rec[k] = node[k]
                # Sometimes nested metrics live under sub‑dicts
                if 'DPN' in node and isinstance(node['DPN'], dict):
                    rec.setdefault('dpn_s1', node['DPN'].get('S1'))
                    rec.setdefault('dpn_s2', node['DPN'].get('S2'))
                if 'best_cn' in node and isinstance(node['best_cn'], dict):
                    rec.setdefault('coverage', node['best_cn'].get('coverage'))
                    rec.setdefault('mean_deg', node['best_cn'].get('mean_abs_delta_deg'))
                if rec:
                    records.append(rec)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(obj)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description='Summarize geometry probe JSON into CSV')
    ap.add_argument('--in', dest='inputs', type=Path, nargs='+', required=True,
                    help='Input geometry JSON file(s)')
    ap.add_argument('--out', dest='out_csv', type=Path, required=True,
                    help='Output CSV path (will create parent directories)')
    args = ap.parse_args()

    out_csv: Path = args.out_csv
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {'inputs': [], 'rows': 0}
    for p in args.inputs:
        data = _load_json(p)
        recs = _flatten_records(data)
        all_records.extend(recs)
        manifest['inputs'].append({'path': str(p), 'sha256': _sha256(p), 'count': len(recs)})

    # Deduce CSV headers
    headers = ['layer', 'part', 'head', 'coverage', 'mean_deg', 'var2', 'var3', 'dpn_s1', 'dpn_s2']
    # Sort records for stability
    def _key(rec: Dict[str, Any]):
        return (
            rec.get('layer', 1e9),
            str(rec.get('part', '')),
            rec.get('head', 1e9),
        )
    all_records.sort(key=_key)

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for rec in all_records:
            w.writerow({k: rec.get(k, '') for k in headers})
    manifest['rows'] = len(all_records)

    with (out_dir / 'manifest.json').open('w', encoding='utf-8') as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    print(f'Wrote {len(all_records)} rows to {out_csv}')
    print(f'Manifest: {out_dir / "manifest.json"}')


if __name__ == '__main__':
    main()
