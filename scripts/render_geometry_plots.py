#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='Render geometry mean |Δθ| (deg) combined plot')
    ap.add_argument('--csv', type=Path, required=True, help='geometry_summary.csv')
    ap.add_argument('--out-dir', type=Path, required=True, help='output directory for PNGs')
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise SystemExit('matplotlib is required: pip install matplotlib')

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with args.csv.open('r', encoding='utf-8') as f:
        for rec in csv.DictReader(f):
            rows.append(rec)

    # Collect layers and parts
    layers = sorted({r.get('layer') for r in rows if r.get('layer') not in (None, '')}, key=lambda x: int(x) if str(x).isdigit() else x)
    parts = sorted({r.get('part') for r in rows if r.get('part') not in (None, '')})

    # Build mean |Δθ| map: layer -> part -> mean
    deg_map = defaultdict(lambda: defaultdict(list))
    for r in rows:
        layer = r.get('layer')
        part = r.get('part')
        if not layer or not part:
            continue
        try:
            md = float(r.get('mean_deg') or 'nan')
        except Exception:
            md = float('nan')
        if md == md:
            deg_map[layer][part].append(md)

    # Prepare figure: one plot, multiple layers as lines
    plt.figure(figsize=(min(16, 0.6 * len(parts) + 6), 5))
    x = list(range(len(parts)))
    for idx, layer in enumerate(layers):
        y = []
        for p in parts:
            vals = deg_map[layer].get(p, [])
            if vals:
                y.append(sum(vals) / len(vals))
            else:
                y.append(float('nan'))
        line, = plt.plot(x, y, marker='o', label=f'layer {layer}')
        # annotate each point with its value (2 decimals), skip NaN
        for xi, yi in zip(x, y):
            if yi == yi:  # not NaN
                plt.text(xi, yi + 0.02, f'{yi:.2f}', ha='center', va='bottom', fontsize=7, color=line.get_color())

    plt.xticks(x, parts, rotation=30, ha='right')
    plt.ylabel('mean |Δθ| (deg)')
    plt.title('Geometry mean |Δθ| by part (combined across layers)')
    # Focused y‑range around observed band ~[5,6]
    plt.ylim(5.3, 5.7)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = out_dir / 'geometry_mean_deg_all_layers.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
