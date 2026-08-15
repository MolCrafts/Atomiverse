"""Pair (non-bonded) potentials — ``cpu/potentials/`` pair styles.

A *pair style* gives the energy of one pair of atoms as a function of their
separation ``r``; the total is the sum over all pairs. Every style here is
*truncated* at a cutoff ``r_c`` [Å] — pairs farther apart are skipped — which
is what keeps the sum affordable at scale and is defensible because these
interactions are short-ranged. ``r_c`` is an explicit constructor argument
everywhere except :data:`WCA`, which derives it from σ.

**"Shifted at the cutoff" means this and only this:** the style subtracts the
constant ``U(r_c)`` from the energy of every pair it does count, so the energy
reaches zero *continuously* at ``r_c`` instead of stepping there. A constant
has zero derivative, so the force is untouched — which means the *force* still
steps at ``r_c`` for most styles, and that discontinuity is a slow source of
energy drift in a long run. Two styles avoid even that, by construction rather
than by subtraction: :data:`WCA` cuts exactly at the Lennard-Jones minimum,
where the force is already zero, and :data:`Dpd`'s weight ``1 − r/r_c``
vanishes there on its own.
"""

from atomiverse._binding import _cpp

#: Lennard-Jones 12-6 — ``cpu::PairLJCut(epsilon [eV], sigma [Å], cutoff [Å])``.
LJCut = _cpp.PairLJCut

#: Truncated Coulomb — ``cpu::PairCoulCut(cutoff [Å])``. The bare
#: ``q_i q_j / r`` sum over the neighbor list, hard-cut at ``cutoff``; needs
#: no reciprocal-space partner, so it is the whole electrostatic model of
#: the run that installs it. Needs a ``NeighborList`` built with a cutoff at
#: least as large. Charges come from the frame's per-atom ``charge``
#: column [e]; a frame without that column is rejected by ``run()`` rather
#: than read as neutral. The Ewald-split styles — accurate periodic
#: electrostatics, at the cost of a matched pair of components — live in
#: :mod:`atomiverse.longrange`.
CoulCut = _cpp.PairCoulCut

#: Morse V = D(1 − e^{−α(r − r₀)})², shifted at the cutoff —
#: ``cpu::PairMorse(D [eV], alpha [Å⁻¹], r0 [Å], cutoff [Å])``. The tree's
#: "a new pair style costs two files" demonstration: it walks every pair
#: O(N²) with no neighbor list, so use it at the scale where that is honest
#: (a dimer, a small cluster, a teaching example), not on bulk.
Morse = _cpp.PairMorse

#: Buckingham U = A e^{−r/ρ} − C/r⁶, shifted at the cutoff (LAMMPS
#: ``pair_style buck``) — ``cpu::PairBuck(A [eV], rho [Å], C [eV·Å⁶],
#: cutoff [Å])``. Single-type. The Born-Mayer wall is finite while −C/r⁶ is
#: not, so this form collapses attractively as r → 0; :data:`Born` adds the
#: +D/r⁸ term that cures it.
Buck = _cpp.PairBuck

#: Born-Mayer-Huggins U = A e^{(σ−r)/ρ} − C/r⁶ + D/r⁸, shifted at the cutoff
#: (LAMMPS ``pair_style born``) — ``cpu::PairBorn``. The Fumi-Tosi
#: short-range half of a molten-salt model; the Coulomb term is a separate
#: evaluator.
#:
#: Constructed from parallel sequences, one entry per table row:
#: ``Born(type_i, type_j, A [eV], rho [Å], sigma [Å], C [eV·Å⁶], D [eV·Å⁸],
#: cutoff [Å])`` — the seven sequences are one table seen column-wise and must
#: share a length, checked when the evaluator is installed rather than here.
#: The type keys are ATOMIC NUMBERS (the frame's type column comes from molrs
#: ``atomic_number``), so an Na-Cl row is ``11, 17``. Every interacting pair
#: must be listed and unlisted pairs do not interact — Fumi-Tosi parameters
#: are not Lorentz-Berthelot, so there is no mixing rule to fall back on.
Born = _cpp.PairBorn

#: Screened Coulomb (Yukawa) U = A e^{−κr}/r, shifted at the cutoff —
#: ``cpu::PairYukawa(A [eV·Å], kappa [Å⁻¹], cutoff [Å])``. Reads no per-atom
#: ``charge`` column: A is the whole interaction strength, which makes this a
#: colloid / screened-plasma model rather than an electrostatic one. For real
#: charges use :data:`CoulCut` or :mod:`atomiverse.longrange`.
Yukawa = _cpp.PairYukawa

#: Weeks-Chandler-Andersen — LJ 12-6 cut and shifted at r_c = 2^{1/6}σ, so
#: U(r_c) = F(r_c) = 0 and the style is purely repulsive.
#: ``cpu::PairWCA(epsilon [eV], sigma [Å])`` takes NO cutoff: it is derived
#: from σ, not chosen, and beyond it the contribution is exactly zero.
WCA = _cpp.PairWCA

#: Groot-Warren dissipative particle dynamics (DPD) —
#: ``cpu::PairDpd(a [eV/Å], gamma [eV·fs/Å²], temperature [K], cutoff [Å],
#: dt [fs], seed)``. Coarse-grained: a particle is a blob of fluid, not an
#: atom, so the conservative force is soft — ``F^C = a·w(r)·r̂`` with
#: ``w = 1 − r/r_c``, reported energy ``U^C = ½·a·r_c·w²`` [eV] — and the
#: coarsened-away motion returns as a pairwise dissipative/random thermostat
#: with ``σ² = 2·gamma·k_B·T``. It is therefore a THERMOSTAT as well as a
#: potential: install it instead of, not beside, a thermostat (it does *not*
#: replace the integrator), and pass the integrator's own ``dt`` — the random
#: force scales as ``1/√dt``. The reported stress carries the conservative
#: virial only, and the random force is redrawn every step from a stream keyed
#: on atom indices, so a run repeats for a given seed, step budget and atom
#: ordering; ``gamma=0`` switches both thermostat terms off and is the only
#: reproducible *single point*. Only the single-``a`` constructor is bound;
#: the C++ multi-component ``a_matrix`` one is not.
Dpd = _cpp.PairDpd

__all__ = [
    "LJCut",
    "CoulCut",
    "Morse",
    "Buck",
    "Born",
    "Yukawa",
    "WCA",
    "Dpd",
]
