"""NDDO/AM1 semiempirical electronic structure (mirrors ``cpu/semiempirical/``).

Single-point only: ``AM1Evaluator`` carries ``Energy | Charge |
DensityMatrix`` capabilities — no forces, so it plugs into
``atomiverse.driver.single_point``, never into MD (AM1 gradients are a
future spec). Canonical deep paths: ``atomiverse.semiempirical.AM1Config``
/ ``AM1Evaluator`` / ``AM1Result``; the top-level names are the same
classes.

Example::

    import atomiverse as atv

    ev = atv.AM1Evaluator(atv.AM1Config(charge=0))
    atv.single_point(frame, ev)      # energy [eV] + charges [e] on frame
    res = ev.last_result()           # Hartree/kcal-mol detail + Mulliken
"""

from atomiverse._binding import _cpp

#: SCF configuration — ``cpu::semiempirical::AM1Config`` (charge [e],
#: multiplicity, max_iter, density_tol, energy_tol [Hartree]).
AM1Config = _cpp.AM1Config

#: The AM1 single-point Evaluator component.
AM1Evaluator = _cpp.AM1Evaluator

#: Converged-solve results: energies, Mulliken charges, iteration count.
AM1Result = _cpp.AM1Result

__all__ = ["AM1Config", "AM1Evaluator", "AM1Result"]
