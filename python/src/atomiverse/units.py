"""Unit operations on simulation data; Frame remains a data container.

A driver interprets its parameters in the chosen run style and adapts annotated
input/output data automatically. These functions are for imports and explicit
exchange between runs, outside numerical kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atomiverse._binding import _cpp
from atomiverse._units import STYLES, set_units, units_style


@dataclass(frozen=True)
class ReducedReference:
    """Physical scales for one reduced LJ model, in SI per particle."""
    identity: str
    mass_kg: float
    length_m: float
    energy_j: float

    def _args(self):
        return (self.mass_kg, self.length_m, self.energy_j), self.identity


def _reference(frame: Any) -> ReducedReference | None:
    names = ("units_lj_identity", "units_lj_mass_kg", "units_lj_length_m", "units_lj_energy_j")
    present = [name in frame.meta for name in names]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("incomplete LJ reference metadata")
    return ReducedReference(*(frame.meta[name] for name in names))


def annotate(frame: Any, style: str, *, reference: ReducedReference | None = None) -> None:
    """Declare the meaning of existing values without rescaling them."""
    if style != "electronic" and style not in STYLES:
        raise ValueError(f"unknown unit style: {style!r}")
    if reference is not None:
        if style != "lj":
            raise ValueError("a reduced reference belongs to the LJ style")
        _cpp._validate_unit_reference(*reference._args())
    frame.meta["units_style"] = style
    for key in ("identity", "mass_kg", "length_m", "energy_j"):
        name = "units_lj_" + key
        if reference is not None:
            frame.meta[name] = getattr(reference, key)
        elif name in frame.meta:
            del frame.meta[name]


def convert(frame: Any, style: str, *, reference: ReducedReference | None = None) -> Any:
    """Return a converted copy, preserving the input and its unit metadata."""
    result = frame.copy()
    _cpp._convert_frame_units(result, style, *(reference._args() if reference else (None, "")))
    return result


def from_style(style: str):
    """Resolve a run style using the same prepared coefficients as C++."""
    return _cpp._units_from_style(style)
