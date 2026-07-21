"""Explicit plane-wave DFT solver API for periodic systems.

The PW-enabled wheel exposes the same object lifecycle as C++:
``PWConfig`` → ``PWSolver.build(frame)`` → ``PWSolver.run(frame)``.
There is no transient functional solve entry point.

Example::

    import atomiverse as av

    cfg = av.PWConfig()
    cfg.upf_path = "Si.upf"
    cfg.ecut_ry = 30.0
    cfg.k_mesh = 4

    solver = av.PWSolver(cfg)
    solver.build(frame)
    result = solver.run(frame)
    print(result.energy)  # Ry

When Atomiverse is built without ``ATV_BUILD_PW=ON``, this module remains
importable but exports no PW types. The package surface therefore reflects the
compiled capability instead of pretending an unavailable solver exists.
"""

from __future__ import annotations

from atomiverse._binding import _cpp

if hasattr(_cpp, "PWSolver"):
    PWConfig = _cpp.PWConfig
    PWSolver = _cpp.PWSolver
    PWResult = _cpp.PWResult
    __all__ = ["PWConfig", "PWSolver", "PWResult"]
else:
    __all__: list[str] = []
