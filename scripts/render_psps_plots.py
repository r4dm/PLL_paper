#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='Render Piecewise‑SPS summary plots')
    ap.add_argument('--summary', type=Path, required=True, help='piecewise_psps_summary.csv')
    ap.add_argument('--tiles', type=Path, required=True, help='piecewise_psps_tiles.csv')
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise SystemExit('matplotlib required: pip install matplotlib')

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary bar: frac R²≥0.9 by file
    rows = []
    with args.summary.open('r', encoding='utf-8') as f:
        for rec in csv.DictReader(f):
            rows.append(rec)
    import re
    def _clean(name: str) -> str:
        # piecewise_sps_l0_mlp_gate.json -> mlp_gate
        # piecewise_sps_l0_o_h8_k64_r16.json -> o_h8_k64_r16
        m = re.match(r'^piecewise_sps_(?:l\d+_)?(.+)\.json$', name)
        return m.group(1) if m else name
    labels = [_clean(r['file']) for r in rows]
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get('frac_r2_ge_0_9') or 0.0))
        except Exception:
            vals.append(0.0)
    plt.figure(figsize=(min(14, 0.6*len(labels)+4), 4))
    plt.bar(labels, vals, color='#4C78A8')
    plt.ylim(0.0, 1.05)
    plt.ylabel('Frac R²≥0.9')
    plt.title('Piecewise‑SPS — fraction of high‑R² regions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out1 = out_dir / 'piecewise_frac_r2.png'
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f'Wrote {out1}')

    # Histogram: terms in top tiles
    tiles = []
    with args.tiles.open('r', encoding='utf-8') as f:
        for rec in csv.DictReader(f):
            tiles.append(rec)
    terms = []
    for t in tiles:
        try:
            terms.append(int(t.get('terms') or 0))
        except Exception:
            pass
    if terms:
        plt.figure(figsize=(6, 4))
        plt.hist(terms, bins=range(0, max(terms)+2), color='#72B7B2', edgecolor='black')
        plt.xlabel('# terms')
        plt.ylabel('count')
        plt.title('Piecewise‑SPS — term counts in top regions')
        plt.tight_layout()
        out2 = out_dir / 'piecewise_terms_hist.png'
        plt.savefig(out2, dpi=150)
        plt.close()
        print(f'Wrote {out2}')


if __name__ == '__main__':
    main()
