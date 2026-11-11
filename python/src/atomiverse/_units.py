"""LAMMPS ``units`` for the Python entry points.

A run's unit style belongs to its driver (``Driver::set_units``): every
parameter, the timestep, temperatures, pressures and the frame are in that
style, and nothing is converted. A Python script mirrors a LAMMPS input
script: :func:`set_units` near the top is its ``units`` command, and every
``MD`` / ``Minimize`` / ``Evaluate`` constructed afterwards without an
explicit ``units=`` runs in that style. A driver keeps the style it was
constructed with, so changing the default later never alters a driver that
already exists.
"""

from __future__ import annotations

#: The LAMMPS ``units`` styles (``Update::set_units``).
STYLES: tuple[str, ...] = (
    "lj",
    "real",
    "metal",
    "si",
    "cgs",
    "electron",
    "micro",
    "nano",
)

_default = "lj"


def _check(style: str) -> str:
    if style not in STYLES:
        raise ValueError(
            f"units: unknown LAMMPS style {style!r} ({', '.join(STYLES)})"
        )
    return style


def set_units(style: str) -> None:
    """LAMMPS ``units <style>`` for drivers constructed from now on.

    ``MD``, ``Minimize`` and ``Evaluate`` created afterwards without
    ``units=`` run in ``style`` (default ``"lj"``, as in LAMMPS). Existing
    drivers keep theirs.
    """
    global _default
    _default = _check(style)


def units_style() -> str:
    """The style new drivers adopt when no ``units=`` is given."""
    return _default


def resolve(units: str | None) -> str:
    """``units`` when given, else the current default."""
    return _default if units is None else _check(units)
