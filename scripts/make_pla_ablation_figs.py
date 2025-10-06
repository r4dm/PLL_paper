#!/usr/bin/env python3
"""Aggregate PLA run directories into a CSV + manifest.

For each provided directory, the script looks for:
- train_summary.json (required): steps, lr, mu, lambda_init, final_lambda, mean_lm, time_s, etc.
- apc_before_after.json (optional): {before, after}
- geometry_before_after.json (optional): {before:{phi,var2,var3}, ...}

Usage:
  python docs/bridge_paper/scripts/make_pla_ablation_figs.py \
      --in runs/pla_o0_h8_600 runs/pla_v0_h3_600 \
      --out docs/bridge_paper/figs/pla_ablation.csv
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


def _first_existing(p: Path, names: List[str]) -> Path | None:
    for n in names:
        q = p / n
        if q.exists():
            return q
    return None


def _parse_run_id(run_dir: Path) -> Dict[str, Any]:
    """Heuristically parse layer/part/head from run directory name.
    Expected tokens like: pla_l{layer}_{part}_h{head}_...
    part is one letter: o|v (o_proj or v_proj).
    """
    name = run_dir.name
    tokens = name.split('_')
    out: Dict[str, Any] = {}
    for t in tokens:
        if t.startswith('l') and len(t) > 1 and t[1:].isdigit():
            out['layer'] = int(t[1:])
        if t in ('o', 'v'):
            out['part'] = 'o_proj' if t == 'o' else 'v_proj'
        if t.startswith('h') and len(t) > 1 and t[1:].isdigit():
            out['head'] = int(t[1:])
        if t.startswith('steps'):
            num = t.replace('steps','')
            if num.isdigit():
                out['steps_in_name'] = int(num)
        if t in ('oonly','all'):
            out['mode'] = t
    return out


def _load_pac_baseline(paths: List[Path]) -> Dict[tuple, float]:
    """Load PAC baseline JSON (list[dict]) and map (layer,part,head) -> apc."""
    baseline: Dict[tuple, float] = {}
    import json
    for p in paths:
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        if isinstance(obj, list):
            for rec in obj:
                try:
                    key = (int(rec.get('layer')), str(rec.get('part')), int(rec.get('head')))
                    apc = float(rec.get('apc'))
                    if key[0] == key[0] and apc == apc:
                        baseline[key] = apc
                except Exception:
                    pass
    return baseline


def main() -> None:
    ap = argparse.ArgumentParser(description='Aggregate PLA runs into CSV')
    ap.add_argument('--in', dest='inputs', type=Path, nargs='+', required=True)
    ap.add_argument('--out', dest='out_csv', type=Path, required=True)
    ap.add_argument('--pac-baseline', dest='pac_baseline', type=Path, nargs='*', default=[],
                    help='Optional PAC baseline JSON(s) from pac_audit (list of dicts)')
    args = ap.parse_args()

    out_csv: Path = args.out_csv
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = [
        'run_dir', 'layer', 'part', 'head', 'steps_in_name', 'mode',
        'steps', 'seq_len', 'batch_size', 'accum', 'lr', 'mu',
        'lambda_init', 'final_lambda_mean', 'final_lambda_list', 'mean_lm', 'mean_off', 'time_s',
        'apc_before', 'apc_after', 'delta_apc', 'phi_before', 'var2_before', 'var3_before'
    ]
    rows: List[Dict[str, Any]] = []
    manifest = {'inputs': [], 'rows': 0}

    baseline_map = _load_pac_baseline(list(args.pac_baseline)) if args.pac_baseline else {}

    for run in args.inputs:
        run = run.resolve()
        entry = {'run_dir': str(run)}
        # enrich with inferred coords
        entry.update(_parse_run_id(run))
        ts_path = _first_existing(run, ['train_summary.json'])
        if not ts_path:
            continue
        ts = _load(ts_path)
        entry.update({k: ts.get(k) for k in ['steps', 'seq_len', 'batch_size', 'accum', 'lr', 'mu', 'lambda_init', 'mean_lm', 'mean_off', 'time_s']})
        fl = ts.get('final_lambda', []) or []
        if isinstance(fl, list):
            try:
                entry['final_lambda_mean'] = sum(fl) / max(1, len(fl))
                entry['final_lambda_list'] = ','.join(f'{x:.4f}' for x in fl)
            except Exception:
                entry['final_lambda_mean'] = ''
                entry['final_lambda_list'] = ''
        # APC before/after
        aa_path = _first_existing(run, ['apc_before_after.json'])
        if aa_path:
            aa = _load(aa_path)
            # Nested dicts: {'before': {'apc': ...}, 'after': {'apc': ...}}
            before_apc = None
            after_apc = None
            if isinstance(aa.get('before'), dict):
                before_apc = aa['before'].get('apc')
            elif isinstance(aa.get('before'), (int, float)):
                before_apc = aa.get('before')
            if isinstance(aa.get('after'), dict):
                after_apc = aa['after'].get('apc')
            elif isinstance(aa.get('after'), (int, float)):
                after_apc = aa.get('after')
            if before_apc is not None:
                entry['apc_before'] = before_apc
            if after_apc is not None:
                entry['apc_after'] = after_apc
            if before_apc is not None and after_apc is not None:
                try:
                    entry['delta_apc'] = float(after_apc) - float(before_apc)
                except Exception:
                    entry['delta_apc'] = ''
        else:
            # Try apc_after.json + PAC baseline map
            aft = _first_existing(run, ['apc_after.json'])
            if aft:
                try:
                    obj = _load(aft)
                    after_apc = obj.get('apc')
                    entry['apc_after'] = after_apc
                except Exception:
                    after_apc = None
                key = (entry.get('layer'), entry.get('part'), entry.get('head'))
                if None not in key and key in baseline_map and after_apc is not None:
                    entry['apc_before'] = baseline_map[key]
                    try:
                        entry['delta_apc'] = float(after_apc) - float(baseline_map[key])
                    except Exception:
                        entry['delta_apc'] = ''
        # Geometry before
        gb_path = _first_existing(run, ['geometry_before_after.json'])
        if gb_path:
            gb = _load(gb_path)
            b = gb.get('before', {})
            entry['phi_before'] = b.get('phi')
            entry['var2_before'] = b.get('var2')
            entry['var3_before'] = b.get('var3')

        rows.append(entry)
        manifest['inputs'].append({'path': str(run), 'sha256': _sha256(ts_path), 'has_apc': bool(aa_path), 'has_geom': bool(gb_path)})

    # Stable sort by delta_apc desc then time
    def _val(x):
        try:
            return float(x)
        except Exception:
            return float('-inf')
    rows.sort(key=lambda r: (-_val(r.get('delta_apc')), r.get('time_s') or 0))

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
