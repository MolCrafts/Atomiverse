"""Plane-wave DFT method API for periodic systems.

The PW-enabled wheel exposes the same object lifecycle as C++:

``PW`` (members) → ``build(frame)`` → ``run(frame)`` → ``PWResult``.

There is no bag named ``PWConfig`` / ``PWSolver`` and no transient functional
solve entry point.

Example::

    import atomiverse as av

    pw = av.PW()
    pw.gth_name = "GTH-PBE-q4"   # analytic GTH for Si; no external file
    pw.ecut_ry = 30.0
    pw.k_mesh = 4
    pw.build(frame)
    result = pw.run(frame)
    print(result.energy)  # Ry

Pseudopotential — exactly one of two, and there is no default
------------------------------------------------------------

``gth_name``
    An in-tree analytic Goedecker-Teter-Hutter parameterization, e.g.
    ``"GTH-PBE-q4"`` (Si) or ``"GTH-PBE-q1"`` (H). Needs no file: the
    published parameters ship with the engine. In-tree coverage is H, O,
    Na, Cl, Ar, Si, Ni, Zn, and one potential serves the whole cell, so
    the frame must be single-species.

``upf_path``
    A **norm-conserving** UPF file (ONCVPSP and similar). Ultrasoft and
    PAW files are rejected on read: the plane-wave path applies no
    augmentation charge, so integrating one would be wrong by an
    unbounded amount.

Setting both, or neither, raises. There is no toy/bare-Coulomb fallback —
a bare valence-charge potential is not representable in a plane-wave basis
at any finite cutoff, so it never converged with ``ecut_ry``.

When Atomiverse is built without ``ATV_BUILD_PW=ON``, this module remains
importable but exports no PW types. The package surface therefore reflects the
compiled capability instead of pretending an unavailable solver exists.
"""

from __future__ import annotations

from atomiverse._binding import _cpp

if hasattr(_cpp, "PW"):
    PW = _cpp.PW
    PWResult = _cpp.PWResult
    __all__ = ["PW", "PWResult"]
else:
    __all__: list[str] = []
