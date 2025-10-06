#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _succ_criteria(report: dict) -> Dict[str, float | int | bool]:
    regions = report.get("regions", [])
    coverage = float(report.get("coverage_kept", 0.0))
    good = sum(1 for r in regions if float(r.get("r2", 0.0) or 0.0) >= 0.9)
    short = sum(1 for r in regions if isinstance(r.get("terms"), dict) and len(r.get("terms")) <= 5)
    frac_good = (good / max(1, len(regions))) if regions else 0.0
    frac_short = (short / max(1, len(regions))) if regions else 0.0
    success = (coverage >= 0.8) and (frac_good >= 0.6) and (frac_short >= 0.6)
    return {
        "coverage": coverage,
        "frac_good_r2": frac_good,
        "frac_short_terms": frac_short,
        "success": success,
        "regions": len(regions),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Piecewise-SPS reports and emit markdown")
    ap.add_argument("--inputs", type=str, nargs="+", required=True, help="JSON report paths")
    ap.add_argument("--out", type=str, required=True, help="Output markdown path")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    reports = [(_p, _load(_p)) for _p in paths]

    lines: List[str] = []
    lines.append("## PLL: Piecewise-SPS summary")
    lines.append("")

    ok_all = True
    for p, rep in reports:
        crit = _succ_criteria(rep)
        lines.append(f"### {p.name}")
        lines.append("- **coverage_kept**: {:.3f}".format(crit["coverage"]))
        lines.append("- **fraction regions with R²≥0.9**: {:.3f}".format(crit["frac_good_r2"]))
        lines.append("- **fraction regions with ≤5 terms**: {:.3f}".format(crit["frac_short_terms"]))
        lines.append("- **#regions**: {}".format(int(crit["regions"])))
        lines.append("- **success**: {}".format("YES" if crit["success"] else "no"))
        lines.append("")
        ok_all = ok_all and bool(crit["success"])  # type: ignore[arg-type]

    lines.append("---")
    lines.append("Result: {}".format("experiment PASSES" if ok_all else "additional tuning required."))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved summary → {out}")


if __name__ == "__main__":
    main()
