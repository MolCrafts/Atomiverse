"""CPU plane-wave DFT with immutable norm-conserving potentials.

Construct a physical provider once and share it between independent methods::

    from atomiverse.pw import PW, GthPseudopotential
    potential = GthPseudopotential(14, "GTH-PBE-q4")
    pw = PW(potentials=[potential], ecut_ry=15.0, k_mesh=2)
    result = pw.run(frame)
    forces = pw.forces(frame)

``UpfPseudopotential(path)`` reads a scalar NC UPF once and checks its element,
valence and physical channels. Both providers use the same CPU PW operators.
Supply one provider for each atomic number in the frame. Provider order is
independent of atom order.
USPP, PAW, NLCC and spin-orbit potentials are refused.

Frame positions and forces use Angstrom and eV/Angstrom; PWResult energies use
Ry (``energy``) and Ha (``energy_ha``). ``run`` preserves force columns;
``forces`` overwrites them. ``stress`` returns mechanical stress in eV/A^3.
SCF controls remain writable on the method. Mount the method with
``MD.set_evaluator(pw)`` to reuse the same electronic loop in dynamics.
The C++ public NC interface also supports independent third-party providers;
Python subclass callbacks are outside this implementation slice.
"""
from __future__ import annotations
from atomiverse._binding import _cpp, require_domain
require_domain(__name__, "pw")
PW = _cpp.PW
PWResult = _cpp.PWResult
NormConservingPotential = _cpp.NormConservingPotential
GthPseudopotential = _cpp.GthPseudopotential
UpfPseudopotential = _cpp.UpfPseudopotential
__all__ = ["PW", "PWResult", "NormConservingPotential", "GthPseudopotential", "UpfPseudopotential"]
