"""Monte Carlo samplers (mirrors the ``cpu/montecarlo/`` backend dir).

What these samplers do
----------------------
Molecular dynamics moves atoms by integrating Newton's equations. Monte
Carlo does not: it produces configurations by a guided random walk, and
never computes a force. The walk is a *Markov chain* — each configuration
is proposed from the previous one and from nothing else — and one step of
it is called a **trial**:

1. a *mover* proposes a small random change (displace one atom, rescale
   the cell, insert or delete one atom),
2. the registered energy sources score the whole changed configuration,
3. the proposal is **accepted** (kept) or **rejected** (undone).

The acceptance rule is what decides *which distribution* the visited
configurations are drawn from. Metropolis acceptance,
``min(1, exp(-dE / (k_B T)))`` — with ``dE`` [eV] the trial energy minus
the energy of the last accepted configuration, ``k_B`` the Boltzmann
constant and ``T`` [K] the temperature — makes them Boltzmann-distributed,
so a plain average over the chain is a canonical ensemble average. Every
downhill trial (``dE <= 0``) is accepted; an uphill one is accepted with
probability ``exp(-dE/(k_B T))``, which is how the chain climbs out of a
local minimum instead of getting stuck in it.

An *ensemble* is the set of macroscopic quantities held fixed. Three
presets over one shared trial loop, plus the four movers they compose:

=========================  =================================================
``MC(T, seed)``            canonical NVT — fixed N, V, T
``NptMC(T, P, seed)``      isothermal-isobaric — fixed N, P, T; V sampled
``GCMC(T, mu, m, seed)``   grand-canonical muVT — fixed mu, V, T; N sampled
``TranslateMove(delta)``   displace one random atom in [-delta, delta]^3
``VolumeMove(dv_max)``     isotropic volume change (NPT)
``InsertDeleteMove(z, m)`` insert / delete one atom of species z (muVT)
``MoveSelector()``         weighted mix — fires exactly ONE child per trial
=========================  =================================================

**NVT** holds the particle number N, the cell volume V and the temperature
T fixed — the plain canonical ensemble. **NPT** holds N, the external
pressure P and T fixed and samples V instead, so the cell settles at the
size where the system's own pressure matches P (the ensemble to use for a
density at a given pressure). **muVT** holds the chemical potential mu, V
and T fixed and samples N, treating the cell as being in contact with an
infinite particle reservoir at that mu — the ensemble for adsorption and
vapour problems, where the average population is the answer.

Canonical deep paths are ``atomiverse.montecarlo.<Name>``; the top-level
``atomiverse.<Name>`` are the same objects. All seven resolve on an
extension built with ``ATV_BUILD_MC=ON``, which is the CMake default and
what the shipped wheel uses, so they stay in the top-level ``__all__``.
The exception is a staged CLI install, which selects modules explicitly:
include ``mc`` there (``python -m atomiverse config md mc ...``) or the
extension is built without these names. On such a build this module still
imports and exports nothing (the same capability probe the compile-optional
``KSpacePME`` / ``PW`` use), and ``atomiverse.MC`` raises ``AttributeError``
naming ``ATV_BUILD_MC`` instead of failing bare.

Monte Carlo is CPU-only: there is no ``device=`` argument on these
constructors.

Units
-----
Temperature [K]; pressure [eV/A^3] (LAMMPS sign, positive = compressive;
1 bar ~ 6.2415e-7 eV/A^3); chemical potential [eV], total, including the
ideal part (see *Chemical potential* below); mass [amu]; displacement [A];
volume [A^3]; energy [eV].

Shape
-----
Categorical setters as in MD — ``set_evaluator`` (plural: energy sources
are summed), ``set_neighbor`` and the MC-only ``set_mover`` (singular) —
and the frame is still the only I/O currency. That is the whole handle
surface: there is no ``set_thermostat`` / ``set_thermo`` / ``set_record``
and no explicit ``build()``; the first ``run()`` compiles the plan::

    import atomiverse as av

    sim = av.MC(300.0, seed=42)
    sim.set_evaluator(av.PairLJCut(0.0104, 3.4, 8.5))
    sim.set_neighbor(av.NeighborList(8.5, 1.0))
    sim.set_mover(av.TranslateMove(0.3))
    sim.run(frame, 100_000)
    print(sim.acceptance_ratio(), sim.current_energy())

``set_neighbor`` takes the ordinary ``NeighborList(cutoff, skin)``, but
builds it in the MC-correct form (``every=1, delay=0, check=yes``): an
accepted trial move can cross the skin at any trial, so the rebuild check
is never deferred the way MD's ``delay=10`` default defers it.

Two sizing rules are the caller's, not the engine's: the neighbor cutoff
must be at least the pair potential's cutoff, and ``cutoff + skin`` must
stay below half the shortest box edge, because pair distances are taken
under the minimum-image convention (each pair is seen only through its
nearest periodic copy). Under ``NptMC`` that has to hold at the smallest
volume the chain visits, not only at the starting cell. Neither is
checked.

Acceptance rules
----------------
One symbol set for all three: ``dE`` [eV] is the trial energy minus the
last accepted energy, ``beta = 1/(k_B T)`` [1/eV], ``N`` the particle
count, ``V`` the cell volume [A^3] and ``V'`` [A^3] the trial volume.

* NVT displacement — ``min(1, exp(-beta dE))``.
* NPT volume move —
  ``min(1, exp(-beta [dE + P (V' - V) - N k_B T ln(V'/V)]))``. The middle
  term is the work done against the external pressure P [eV/A^3]; the last
  one credits the free volume the N atoms gain, and it is what pairs with
  drawing the trial volume uniformly in V (``VolumeMove``) rather than in
  ln V.
* muVT insertion, N -> N+1 —
  ``min(1, V / (Lambda^3 (N+1)) * exp(beta (mu - dE)))``.
* muVT deletion, N -> N-1 —
  ``min(1, Lambda^3 N / V * exp(-beta (mu + dE)))``.

These are Frenkel & Smit, *Understanding Molecular Simulation* (2nd ed.
2002) §5.4 and §5.6.1, and they are the rules the C++ policies implement
in ``include/atomiverse/cpu/montecarlo/npt_mc.h`` and ``gcmc.h``.

Chemical potential and the thermal wavelength
---------------------------------------------
``mu`` [eV] passed to ``GCMC`` is the **total** chemical potential — the
ideal-gas part included, not the excess part alone. It enters the two muVT
rules only through the *activity* z (a letter clash: the ``z`` argument of
``InsertDeleteMove`` is an atomic number, nothing to do with this one)::

    z = exp(beta mu) / Lambda**3        [1/A^3]
    Lambda = h / sqrt(2 pi m k_B T)     [A]

``Lambda`` is the thermal de Broglie wavelength, the quantum length scale
of a free particle of mass m [amu] at temperature T [K]; it is what
carries the "per unit volume" in z. For an ideal gas z *is* the number
density, so ``mu = k_B T ln(z Lambda**3)`` converts a target density into
the argument to pass. This is the convention of Frenkel & Smit eq. (5.6.9)
and of ``regressions/mc/gcmc_lj/REFERENCE.md``, which pins <N>(z) against
the Mayer activity expansion using exactly this relation.

Because ``GCMC(T, mu, m, seed)`` builds ``Lambda`` from its ``m``, the
``InsertDeleteMove(z, m)`` that actually stamps atoms into the cell must
carry the **same** mass. Two different masses mean the sampled chemical
potential is not the requested one — a silent error, since nothing
compares them.

Chain continuation
------------------
Consecutive ``run()`` calls on one sampler extend ONE Markov chain — the
standard ``run(equilibrate); run(sample)`` pattern — so the tallies
accumulate and the configuration carries over. The frame is materialized
once, on the first call: edits made to the ``molrs.Frame`` between calls
are ignored, and a fresh chain means a fresh sampler object.

The execution plan is likewise compiled once, on that first call, so a
``set_*`` call afterwards would have no effect on the running chain and
raises ``RuntimeError`` instead of silently doing nothing. Configure the
sampler fully before the first ``run()``.

``run(frame, n_trials)`` counts single trial moves, not sweeps: each count
is one proposal plus one accept/reject decision, so a system of N atoms
needs roughly N times more trials here than it would "one move per
particle" sweeps.

Statistics are handle accessors, not frame metadata (the C++ loop keeps no
acceptance keys): ``attempts()``, ``accepts()`` and ``acceptance_ratio()``
take an optional *verb* — the kind of move a trial turned out to be:
``"displace"``, ``"box"``, ``"insert"``, ``"delete"`` — and aggregate over
every verb when called with none. An unknown name raises ``ValueError``.
The aggregate also counts the trials that ended with no verb at all (a
deletion drawn on an empty cell, a volume draw that would take V <= 0;
both always rejected), so ``attempts()`` is exactly the number of trials
run while the four named verbs can sum to less.

What comes back on the frame
---------------------------
After every ``run()`` the same ``molrs.Frame`` holds the current
configuration: positions [A], the box, and the last accepted energy in
``frame.meta["energy"]`` [eV] (bit-identical to ``current_energy()`` — a
rejected trial's energy never survives). Because the sampler scores
energies and never assembles a gradient, ``fx``/``fy``/``fz`` come back
**zeroed**; they are not physical forces. Velocities are left untouched
(an inserted atom gets zero velocity), since nothing here moves in time.

Variable particle number
------------------------
muVT changes N, and ``to_molrs`` cannot write a changed row count into an
existing block. After a ``GCMC.run()`` that inserted or deleted atoms, the
frame's ``atoms`` block is therefore *rebuilt* at the new N — every column
comes back, including ``mass`` and ``atomic_number``, which the fixed-N
path treats as immutable inputs and does not write. Accumulate ``n_atoms()``
along the chain for <N>.

Single rank
-----------
Monte Carlo is single-rank by design: no MC task exists in the MPI launcher
protocol, so ``run()`` raises ``RuntimeError`` under
``atomiverse-launcher`` rather than silently sampling one independent chain
per rank.

The quantitative ensemble physics is pinned in ``regressions/mc/`` —
``npt_lj`` for <V> and its exact NPT width, ``gcmc_lj`` for <N>(z) against
the Mayer activity expansion.
"""

from __future__ import annotations

from atomiverse._binding import _cpp

# The whole surface is gated on ATV_HAS_MC in bind_mc.cpp, so one probe on the
# preset the extension always carries when MC is on covers all seven names.
if hasattr(_cpp, "MC"):
    #: Canonical NVT Metropolis — ``cpu::mc::MC(temperature [K], seed)``.
    MC = _cpp.MC

    #: Isothermal-isobaric MC —
    #: ``cpu::mc::NptMc(temperature [K], pressure [eV/A^3], seed)``.
    NptMC = _cpp.NptMC

    #: Grand-canonical muVT MC —
    #: ``cpu::mc::Gcmc(temperature [K], mu [eV], mass [amu], seed)``.
    GCMC = _cpp.GCMC

    #: Single-particle displacement Mover —
    #: ``cpu::mc::TranslateMove(delta [A])``.
    TranslateMove = _cpp.TranslateMove

    #: Isotropic volume-change Mover — ``cpu::mc::VolumeMove(dv_max [A^3])``.
    VolumeMove = _cpp.VolumeMove

    #: Grand-canonical insertion/deletion Mover —
    #: ``cpu::mc::InsertDeleteMove(z = atomic number, mass [amu])``.
    InsertDeleteMove = _cpp.InsertDeleteMove

    #: Weighted mix of Movers, one child per trial —
    #: ``cpu::mc::MoveSelector()``.
    MoveSelector = _cpp.MoveSelector

    __all__ = [
        "MC",
        "NptMC",
        "GCMC",
        "TranslateMove",
        "VolumeMove",
        "InsertDeleteMove",
        "MoveSelector",
    ]
else:
    __all__: list[str] = []
