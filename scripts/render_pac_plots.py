#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='Render PAC bar plots (APC vs baseline)')
    ap.add_argument('--csv', type=Path, required=True, help='pac_summary.csv')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--top', type=int, default=20, help='Top‑K by APC to render')
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        raise SystemExit('matplotlib/numpy required: pip install matplotlib numpy')

    rows = []
    with args.csv.open('r', encoding='utf-8') as f:
        for rec in csv.DictReader(f):
            rows.append(rec)

    # Build identifiers and values
    items = []
    for r in rows:
        ident = '.'.join(str(x) for x in [r.get('layer',''), r.get('part',''), r.get('head','') if r.get('head','')!='' else ''])
        try:
            apc = float(r.get('apc') or 'nan')
            base = float(r.get('baseline_mean') or 'nan')
        except Exception:
            continue
        if apc != apc:
            continue
        items.append((ident.strip('.'), apc, base))

    # Sort by APC desc and cut top‑K
    items.sort(key=lambda t: -t[1])
    items = items[: max(1, int(args.top))]

    labels = [it[0] for it in items]
    apc_vals = [it[1] for it in items]
    base_vals = [it[2] for it in items]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(min(14, 0.6*len(labels)+4), 5))
    # Only plot baseline if present (finite)
    base_finite = [b for b in base_vals if b==b]
    plt.bar(x - (width/2 if base_finite else 0), apc_vals, width, label='APC', color='#4C78A8')
    if base_finite:
        plt.bar(x + width/2, base_vals, width, label='baseline', color='#E45756')
    plt.ylabel('Fraction within tol')
    plt.title('PAC (APC vs baseline) — Top heads/parts')
    plt.xticks(x, labels, rotation=45, ha='right')
    # Dynamic y‑limit around observed range
    ymax = max([v for v in apc_vals+base_vals if v==v] + [0.2]) * 1.2
    plt.ylim(0.0, min(1.05, max(0.3, ymax)))
    # Annotate bars with values
    for xi, v in zip(x, apc_vals):
        plt.text(xi - (width/2 if base_finite else 0), max(0, v)+0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    if base_finite:
        for xi, v in zip(x, base_vals):
            if v==v:
                plt.text(xi + width/2, max(0, v)+0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, color='#E45756')
    plt.legend()
    plt.tight_layout()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'pac_apc_vs_baseline.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
