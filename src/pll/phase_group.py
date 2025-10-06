"""Minimal phase-group algebra for cyclic lattices Cn.

The artifact only needs a lightweight subset of the original multipolar
framework, enough to express phase rays and perform simple arithmetic when
exporting symbolic phase sums (SPS / PSPS).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Iterator, List, Sequence


@dataclass(frozen=True)
class PhaseRay:
    """Ray on the cyclic phase lattice Cn."""

    index: int
    angle: float  # radians in [0, 2π)
    name: str

    def conjugate(self) -> "PhaseRay":
        """Return the additive inverse ray (angle + π modulo 2π)."""
        n = getattr(self, "_n", None)
        if n is None:
            raise AttributeError("PhaseRay missing lattice size metadata")
        idx = (self.index + n // 2) % n if n % 2 == 0 else (n - self.index) % n
        return PhaseRay(index=idx, angle=(self.angle + math.pi) % (2 * math.pi), name=f"PR{idx}_C{n}")


class PhaseGroup:
    """Cyclic phase group Cn with evenly spaced rays."""

    def __init__(self, n: int, *, name: str | None = None) -> None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        self.n = int(n)
        self.name = name or f"PhaseGroupC{n}"
        self._rays: List[PhaseRay] = []
        for idx in range(self.n):
            angle = (2 * math.pi * idx) / self.n
            ray = PhaseRay(index=idx, angle=angle, name=f"PR{idx}_C{self.n}")
            object.__setattr__(ray, "_n", self.n)  # attach lattice size for helpers
            self._rays.append(ray)
        self._index: Dict[int, PhaseRay] = {r.index: r for r in self._rays}
        self._name_to_ray: Dict[str, PhaseRay] = {r.name: r for r in self._rays}

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"PhaseGroup(n={self.n}, name={self.name!r})"

    # ------------------------------------------------------------------
    # Ray access helpers
    # ------------------------------------------------------------------
    def get_ray(self, index: int) -> PhaseRay:
        return self._index[int(index) % self.n]

    def get_ray_by_name(self, name: str) -> PhaseRay:
        if name not in self._name_to_ray:
            raise KeyError(f"Ray {name!r} not found in {self.name}")
        return self._name_to_ray[name]

    def rays(self) -> Sequence[PhaseRay]:
        return tuple(self._rays)

    # ------------------------------------------------------------------
    # Group operations
    # ------------------------------------------------------------------
    def multiply(self, left: PhaseRay | int, right: PhaseRay | int) -> PhaseRay:
        li = left.index if isinstance(left, PhaseRay) else int(left)
        ri = right.index if isinstance(right, PhaseRay) else int(right)
        return self.get_ray((li + ri) % self.n)

    def add(self, left: PhaseRay | int, right: PhaseRay | int) -> PhaseRay:
        return self.multiply(left, right)

    def inverse(self, ray: PhaseRay | int) -> PhaseRay:
        idx = ray.index if isinstance(ray, PhaseRay) else int(ray)
        return self.get_ray((-idx) % self.n)

    def divide(self, numerator: PhaseRay | int, denominator: PhaseRay | int) -> PhaseRay:
        return self.multiply(numerator, self.inverse(denominator))

    # ------------------------------------------------------------------
    # Iteration utilities
    # ------------------------------------------------------------------
    def angles(self) -> List[float]:
        return [r.angle for r in self._rays]

    def names(self) -> List[str]:
        return [r.name for r in self._rays]

    def enumerate(self) -> Iterator[PhaseRay]:
        yield from self._rays


__all__ = ["PhaseGroup", "PhaseRay"]
