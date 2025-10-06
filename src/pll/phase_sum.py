"""Lightweight symbolic phase sums used in the PLL artifact.

The original project exposed a rich algebra of symbolic multipolar values. For
this artifact we only need a compact representation that can:
- store coefficients per phase ray
- apply simple pruning and ordering rules
- render a human-readable string compatible with the paper
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

from .phase_group import PhaseGroup, PhaseRay


@dataclass
class PhaseTerm:
    ray: PhaseRay
    coefficient: float

    def as_tuple(self) -> Tuple[str, float]:
        return (self.ray.name, float(self.coefficient))


class PhaseSum:
    """Dense phase sum with explicit coefficients for each ray."""

    def __init__(self, group: PhaseGroup, coeffs: Dict[str, float] | None = None, bias: float = 0.0) -> None:
        self.group = group
        # store by ray name for stable ordering
        base = {name: 0.0 for name in group.names()}
        if coeffs:
            for name, val in coeffs.items():
                base[name] = float(val)
        self._coeffs = base
        self.bias = float(bias)

    def items(self) -> Iterator[Tuple[str, float]]:
        for name in self.group.names():
            yield name, self._coeffs[name]

    def prune(self, *, min_abs: float = 1e-3, max_terms: int = 5) -> "PhaseSum":
        """Return a new PhaseSum keeping at most *max_terms* largest coefficients."""
        filtered = [(name, val) for name, val in self.items() if abs(val) >= float(min_abs)]
        filtered.sort(key=lambda t: -abs(t[1]))
        if max_terms and len(filtered) > int(max_terms):
            filtered = filtered[: int(max_terms)]
        coeffs = {name: val for name, val in filtered}
        return PhaseSum(self.group, coeffs, bias=self.bias)

    def to_dict(self) -> Dict[str, float]:
        return {name: float(val) for name, val in self.items() if abs(val) > 0.0}

    def as_terms(self) -> List[PhaseTerm]:
        return [PhaseTerm(ray=self.group.get_ray_by_name(name), coefficient=val) for name, val in self.items() if abs(val) > 0.0]

    def __str__(self) -> str:
        terms = []
        for name, val in self.items():
            if abs(val) <= 0.0:
                continue
            terms.append(f"({val:+.6f})*{name}")
        if abs(self.bias) > 0.0:
            terms.append(f"({self.bias:+.6f})")
        return " + ".join(terms) if terms else "0"


class PiecewiseSymbolicPhase:
    """Wrapper around a PhaseSum plus optional metadata for PSPS export."""

    def __init__(self, *, mask: str, coverage: float, phase_sum: PhaseSum) -> None:
        self.mask = str(mask)
        self.coverage = float(coverage)
        self.phase_sum = phase_sum

    def to_report(self) -> Dict[str, float | str]:
        return {
            "mask": self.mask,
            "coverage": self.coverage,
            "psps": str(self.phase_sum),
            "terms": {name: coeff for name, coeff in self.phase_sum.items() if abs(coeff) > 0.0},
        }


__all__ = [
    "PhaseSum",
    "PhaseTerm",
    "PiecewiseSymbolicPhase",
]
