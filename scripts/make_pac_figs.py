#!/usr/bin/env python3
"""Summarize PAC JSON outputs into CSV + manifest.

Accepts one or more JSON files. Supported formats include:
- Output of `eval_pla_apc.py`: dict with keys {apc, phi, var2, var3, ok, total}.
- Custom PAC exports that include {layer, part, head, apc, baseline_mean, p} per record.

Usage:
  python docs/bridge_paper/scripts/make_pac_figs.py \
      --in results/pla_eval_o0_h8.json \
      --out docs/bridge_paper/figs/pac_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _as_records(obj: Any, source: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        # eval_pla_apc.py format
        if {'apc', 'phi', 'var2', 'var3'} <= set(obj.keys()):
            recs.append({
                'source': source,
                'layer': obj.get('layer', ''),
                'part': obj.get('part', ''),
                'head': obj.get('head', ''),
                'apc': obj.get('apc'),
                'baseline_mean': obj.get('baseline_mean'),
                'p_value': obj.get('p'),
                'ok': obj.get('ok'),
                'total': obj.get('total'),
                'phi': obj.get('phi'),
                'var2': obj.get('var2'),
                'var3': obj.get('var3'),
            })
        else:
            # try single-record generic
            if 'apc' in obj:
                d = {'source': source}
                for k in ['layer', 'part', 'head', 'apc', 'baseline_mean', 'p', 'p_value', 'ok', 'total', 'phi', 'var2', 'var3']:
                    if k in obj:
                        d[k if k != 'p' else 'p_value'] = obj[k]
                recs.append(d)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                recs.extend(_as_records(item, source))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description='Summarize PAC JSON into CSV')
    ap.add_argument('--in', dest='inputs', type=Path, nargs='+', required=True)
    ap.add_argument('--out', dest='out_csv', type=Path, required=True)
    args = ap.parse_args()

    out_csv: Path = args.out_csv
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = ['source', 'layer', 'part', 'head', 'apc', 'baseline_mean', 'p_value', 'ok', 'total', 'phi', 'var2', 'var3']
    rows: List[Dict[str, Any]] = []
    manifest = {'inputs': [], 'rows': 0}

    for p in args.inputs:
        try:
            obj = _load(p)
        except Exception:
            continue
        rs = _as_records(obj, p.name)
        rows.extend(rs)
        manifest['inputs'].append({'path': str(p), 'sha256': _sha256(p), 'count': len(rs)})

    # Stable sort
    rows.sort(key=lambda r: (str(r.get('layer', '')), str(r.get('part', '')), str(r.get('head', ''))))

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, '') for h in headers})
    manifest['rows'] = len(rows)
    with (out_dir / 'manifest.json').open('w', encoding='utf-8') as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    print(f'Wrote {len(rows)} rows → {out_csv}')


if __name__ == '__main__':
    main()

