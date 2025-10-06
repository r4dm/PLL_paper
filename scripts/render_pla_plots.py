#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='Render PLA ΔAPC bar plot')
    ap.add_argument('--csv', type=Path, required=True, help='pla_ablation.csv')
    ap.add_argument('--out', type=Path, required=True, help='output PNG path')
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise SystemExit('matplotlib required: pip install matplotlib')

    rows = []
    with args.csv.open('r', encoding='utf-8') as f:
        for rec in csv.DictReader(f):
            rows.append(rec)

    # Sort by |delta_apc| desc and filter invalid
    def _to_float(s):
        try:
            return float(s)
        except Exception:
            return float('nan')

    rows = [r for r in rows if r.get('delta_apc') not in ('', None) and _to_float(r.get('delta_apc'))==_to_float(r.get('delta_apc'))]
    rows.sort(key=lambda r: -abs(_to_float(r.get('delta_apc'))))
    def _label(r):
        layer = r.get('layer')
        part = r.get('part')
        head = r.get('head')
        steps = r.get('steps') or r.get('steps_in_name')
        mode = r.get('mode')
        if layer not in ('', None) and part not in ('', None) and head not in ('', None):
            core = f"l{layer} {part}.h{head}"
            extra = []
            if steps not in ('', None):
                extra.append(str(steps))
            if mode not in ('', None):
                extra.append(str(mode))
            if extra:
                return core + ' ' + ' '.join(extra)
            return core
        return Path(r.get('run_dir','')).name

    labels = [_label(r) for r in rows]
    vals = [_to_float(r.get('delta_apc')) for r in rows]

    if not rows:
        print('No valid delta_apc values found; nothing to plot.')
        # still write an empty figure to avoid pipeline break
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,2))
        plt.text(0.5, 0.5, 'No ΔAPC data', ha='center', va='center')
        plt.axis('off')
        out: Path = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        plt.close()
        return

    import matplotlib.pyplot as plt
    plt.figure(figsize=(min(14, 0.6*len(labels)+4), 5))
    colors = ['#4C78A8' if v >= 0 else '#E45756' for v in vals]
    bars = plt.bar(labels, vals, color=colors)
    plt.ylabel('ΔAPC')
    plt.title('PLA ablations — ΔAPC by run')
    plt.xticks(rotation=45, ha='right')
    # Dynamic y-limits with margins
    vmin = min(vals)
    vmax = max(vals)
    span = max(0.005, (vmax - vmin))
    plt.ylim(vmin - 0.1*span, vmax + 0.1*span)
    # Annotate bars
    for bar, v in zip(bars, vals):
        y = bar.get_height()
        xpos = bar.get_x() + bar.get_width()/2
        offset = 0.0005 if v >= 0 else -0.0005
        va = 'bottom' if v >= 0 else 'top'
        plt.text(xpos, y + offset, f'{v:+.4f}', ha='center', va=va, fontsize=8)
    plt.tight_layout()
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
