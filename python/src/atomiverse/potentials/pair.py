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

#: One Lennard-Jones type-pair row — ``ATV::LJTypeCoeff(type_i, type_j,
#: epsilon [eV], sigma [Å], cutoff=None [Å])``, LAMMPS ``pair_coeff i j eps
#: sigma [cut_lj]``. Types are the frame's per-atom type values. An unset
#: ``cutoff`` on a listed row is the style's global cutoff.
LJTypeCoeff = _cpp.LJTypeCoeff

#: One force-shifted r⁻⁴ attraction row — ``ATV::R4TypeCoeff(type_i, type_j,
#: strength [eV], sigma [Å], cutoff [Å])``:
#: U(r) = −S [ (σ/r)⁴ − (σ/r_c)⁴ + 4 (σ/r_c)⁴ (r/r_c − 1) ] for r < r_c,
#: else 0, so U and F both vanish at r_c. S > 0 attracts (e.g. a Li⁺–ether
#: oxygen solvation well in a coarse-grained polymer electrolyte). An
#: ATV-native term with no LAMMPS style; a pair without a row has none.
R4TypeCoeff = _cpp.R4TypeCoeff

#: Lennard-Jones 12-6, shifted to zero at each pair's cutoff by default — two
#: forms:
#:
#: * ``LJCut(epsilon [eV], sigma [Å], cutoff [Å], s12=1, s13=1, s14=1,
#:   energy_shift=True)`` — one ε/σ for every pair
#:   (``cpu::PairLJCut(epsilon, sigma, cutoff)``);
#: * ``LJCut(coeffs, cutoff [Å], s12=1, s13=1, s14=1, energy_shift=True)`` —
#:   a list of
#:   :data:`LJTypeCoeff` rows (``cpu::PairLJCut(coeffs, cutoff)`` /
#:   ``cuda::PairLJCut``). Completed with LAMMPS ``init_one`` semantics: an
#:   unlisted pair mixes Lorentz–Berthelot (geometric ε, arithmetic σ) and
#:   takes the arithmetic mean of the two diagonal cutoffs; a listed row
#:   overrides; a type with no diagonal row has no LJ site. ``cutoff`` is the
#:   default for rows that name none; the neighbor list must cover the
#:   largest per-pair cutoff.
#:
#: ``s12``, ``s13``, ``s14`` are the 1-2 / 1-3 / 1-4 weights (fractional
#: weights such as Amber 0/0/½ on both backends; on CUDA a fractional weight
#: needs a single GPU). ``energy_shift`` is LAMMPS ``pair_modify shift``
#: (``False`` is CPU-only). Chainable setters (each returns the object):
#: ``.set_r4_attraction([R4TypeCoeff, ...])`` (per-type form only) adds the
#: r⁻⁴ term in the same pair walk, scaled by the same weight;
#: ``.set_type_coeff(LJTypeCoeff)`` (per-type form only) adds or overrides
#: one row; ``.set_tail_correction(on)`` adds the LAMMPS ``pair_modify tail
#: yes`` energy and pressure (default off). The task codec stores every
#: field.
LJCut = _cpp.PairLJCut

#: Fused typed Lennard-Jones + Ewald real-space Coulomb — LAMMPS
#: ``pair_style lj/cut/coul/long`` — ``cpu::PairLJCutCoulLong(coeffs, alpha
#: [1/Å], cutoff [Å])`` / ``cuda::PairLJCutCoulLong`` with identical
#: semantics. ``coeffs`` are :data:`LJTypeCoeff` rows completed as for
#: :data:`LJCut`; Coulomb uses ``cutoff``, each LJ pair its own cutoff. The
#: real-space half of an Ewald split: install ``KSpacePME`` at the same
#: ``alpha``, and a per-atom ``charge`` column. Chainable setters (each
#: returns the object): ``set_special_bonds(lj12, lj13, lj14, c12, c13,
#: c14)``, ``set_special_bonds_amber()`` (the default: LJ 0/0/½, Coulomb
#: 0/0/⅚), ``set_energy_shift(on)`` (default on; OpenMM comparisons turn it
#: off), ``set_r4_attraction(rows)`` (scaled by the LJ special weight),
#: ``set_type_coeff(row)`` (add or override one :data:`LJTypeCoeff` row),
#: ``set_tail_correction(on)`` (LAMMPS ``pair_modify tail yes``, each pair at
#: its own LJ cutoff; default off). On CUDA, a pair whose nearest bond-graph
#: shell has weights ≠ (1, 1) is an exception evaluated in a sparse pass with
#: the LAMMPS cutoffs (LJ inside the pair's LJ cutoff, Coulomb correction
#: inside the Coulomb cutoff).
LJCutCoulLong = _cpp.PairLJCutCoulLong

#: Truncated Coulomb — ``cpu::PairCoulCut(cutoff [Å], s12=1, s13=1, s14=1,
#: coulomb=1)`` (the LAMMPS lj qqr2e), with 1-2 / 1-3 / 1-4 weights and the
#: Coulomb constant a force field states (folded with its dielectric). The bare
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
#: The type keys are FORCE-FIELD TYPES: the frame's ``atoms.type_id`` column
#: (u64), not ``atomic_number`` (the species). A frame without ``type_id``
#: is refused with a how-to. Every interacting pair
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
    "LJTypeCoeff",
    "R4TypeCoeff",
    "LJCut",
    "LJCutCoulLong",
    "CoulCut",
    "Morse",
    "Buck",
    "Born",
    "Yukawa",
    "WCA",
    "Dpd",
]
