from __future__ import annotations

import json
import os


def read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _extract_apc_payload(payload):
    if payload is None:
        return None, None, None
    if isinstance(payload, dict):
        return (
            payload.get("apc"),
            payload.get("baseline_mean"),
            payload.get("p"),
        )
    if isinstance(payload, list):
        apc_vals = [item.get("apc") for item in payload if isinstance(item, dict) and "apc" in item]
        baseline_vals = [item.get("baseline_mean") for item in payload if isinstance(item, dict) and item.get("baseline_mean") is not None]
        p_vals = [item.get("p") for item in payload if isinstance(item, dict) and item.get("p") is not None]
        apc = sum(apc_vals) / len(apc_vals) if apc_vals else None
        baseline = sum(baseline_vals) / len(baseline_vals) if baseline_vals else None
        p_val = sum(p_vals) / len(p_vals) if p_vals else None
        return apc, baseline, p_val
    return None, None, None


def main() -> None:
    before_payload = read_json("runs/apc_before.json")
    after_payload = read_json("runs/apc_after.json")

    apc_before, baseline_before, p_before = _extract_apc_payload(before_payload)
    apc_after, baseline_after, p_after = _extract_apc_payload(after_payload)
    baseline = baseline_after if baseline_before is None else baseline_before
    if baseline is None:
        baseline = baseline_after
    delta = None
    if isinstance(apc_before, (int, float)) and isinstance(apc_after, (int, float)):
        delta = apc_after - apc_before

    lines = [
        "# PLL Report",
        "",
        f"APC before: {apc_before}",
        f"APC after: {apc_after}",
        f"Baseline mean: {baseline}",
        f"Delta APC: {delta}",
    ]
    if p_before is not None:
        lines.append(f"Mean p (before): {p_before}")
    if p_after is not None:
        lines.append(f"Mean p (after): {p_after}")

    os.makedirs("runs", exist_ok=True)
    with open("runs/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
