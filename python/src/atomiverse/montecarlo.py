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

For a symmetric NVT proposal, Metropolis acceptance is
``min(1, exp(-dE / (k_B T)))``, where ``dE`` is the trial energy minus the
last accepted energy, ``k_B`` is the Boltzmann constant and ``T`` is the
temperature. Asymmetric proposals also need the reverse to forward
proposal probability ratio. Equilibrium averages include the repeated
state after a rejected trial.

An *ensemble* is the set of macroscopic quantities held fixed. Configure
``MC(seed=...)`` with ``set_nvt(T)``, ``set_npt(T, P)`` or
``set_gcmc(T, mu, m)``. The following constructors configure the same driver:

===========================  =================================================
``MC(T, seed)``              canonical NVT — fixed N, V, T
``NptMC(T, P, seed)``        isothermal-isobaric — fixed N, P, T; V sampled
``GCMC(T, mu, m, seed)``     grand-canonical muVT — fixed mu, V, T; N sampled
``TranslateMove(delta)``     displace one random atom in [-delta, delta]^3
``VolumeMove(dv_max)``       isotropic volume change (NPT)
``InsertDeleteMove(t,z,m)``  insert / delete one atom: type t, species z
``MoveSelector()``           weighted mix — fires exactly ONE child per trial
===========================  =================================================

**NVT** holds the particle number N, the cell volume V and the temperature
T fixed — the plain canonical ensemble. **NPT** holds N, the external
pressure P and T fixed and samples V instead, so the cell settles at the
size where the system's own pressure matches P (the ensemble to use for a
density at a given pressure). **muVT** holds the chemical potential mu, V
and T fixed and samples N, treating the cell as being in contact with an
infinite particle reservoir at that mu — the ensemble for adsorption and
vapour problems, where the average population is the answer.

Canonical deep paths are ``atomiverse.montecarlo.<Name>``; the top-level
``atomiverse.<Name>`` are the same objects. They exist on an extension built
with the ``mc`` module (``ATV_BUILD_MC=ON``). On one built without it,
importing this module raises ``MissingDomainError`` (an ``ImportError``)
naming the flag, and ``atomiverse.MC`` raises ``AttributeError`` with the
same message.

Monte Carlo is CPU-only: there is no ``device=`` argument on these
constructors.

Units
-----
Equilibrium MC uses the selected LAMMPS ``units_style`` for temperature,
pressure, chemical potential, mass, displacement, volume and energy.
The ``units=`` argument selects the style, with the default taken from
``atomiverse.units_style()`` (initially reduced ``lj``). For example,
``metal`` uses K, bar, eV, g/mol and Å. Pressure is positive in compression.
Chemical potential
includes the ideal part. Kinetic rates and the KMC clock use ps-based
units regardless of the Frame unit style.

Shape
-----
Categorical setters as in MD — ``set_evaluator`` (plural: energy sources
are summed), ``set_neighbor`` and the MC-only ``set_mover`` (singular) —
and the frame is still the only I/O currency. That is the whole handle
surface: there is no ``set_thermostat`` / ``set_thermo`` / ``set_record``
and no explicit ``build()``; the first ``run()`` compiles the plan::

    import atomiverse as av

    sim = av.MC(300.0, seed=42)
    sim.set_units("metal")
    sim.set_evaluator(av.PairLJCut(0.0104, 3.4, 8.5))
    sim.set_neighbor(av.NeighborList(8.5, 1.0))
    sim.set_mover(av.TranslateMove(0.3))
    sim.run(frame, 100_000)
    print(sim.acceptance_ratio(), sim.current_energy())

``set_neighbor`` takes the ordinary ``NeighborList(cutoff, skin)``, but
builds it in the MC-correct form (``every=1, delay=0, check=yes``, the
LAMMPS defaults MD uses too): an accepted trial move can cross the skin at
any trial, so the rebuild check must never be deferred.

Two sizing rules are the caller's, not the engine's: the neighbor cutoff
must be at least the pair potential's cutoff, and ``cutoff + skin`` must
stay below half the shortest box edge, because pair distances are taken
under the minimum-image convention (each pair is seen only through its
nearest periodic copy). Under ``NptMC`` that has to hold at the smallest
volume the chain visits, not only at the starting cell. Neither is
checked.

Acceptance rules
----------------
One symbol set for all three: ``dE`` in run energy units is the trial energy minus the
last accepted energy, ``beta = 1/(k_B T)``, ``N`` the particle
count, ``V`` the cell volume and ``V'`` the trial volume.

* NVT displacement — ``min(1, exp(-beta dE))``.
* NPT volume move —
  ``min(1, exp(-beta [dE + P (V' - V) - N k_B T ln(V'/V)]))``. The middle
  term is the work done against the external pressure P converted to energy per volume; the last
  one credits the free volume the N atoms gain, and it is what pairs with
  drawing the trial volume uniformly in V (``VolumeMove``) rather than in
  ln V.
* muVT insertion, N -> N+1 —
  ``min(1, V / (Lambda^3 (N+1)) * exp(beta (mu - dE)))``.
* muVT deletion, N -> N-1 —
  ``min(1, Lambda^3 N / V * exp(-beta (mu + dE)))``.

These reference moves declare zero extra proposal correction. Custom C++
movers must set ``Transaction::log_proposal_ratio`` for every nonempty
trial. It adds the log reverse/forward proposal ratio and any change of
measure omitted from the reference acceptance factor.

The reference formulas are from Frenkel & Smit, *Understanding Molecular Simulation* (2nd ed.
2002) §5.4 and §5.6.1, and they are the rules the C++ policies implement
in ``include/atomiverse/cpu/montecarlo/npt_mc.h`` and ``gcmc.h``.

Chemical potential and the thermal wavelength
---------------------------------------------
``mu`` [run energy units] passed to ``GCMC`` is the **total** chemical potential — the
ideal-gas part included, not the excess part alone. It enters the two muVT
rules only through the *activity* z (a letter clash: the ``z`` argument of
``InsertDeleteMove`` is an atomic number, nothing to do with this one)::

    z = exp(beta mu) / Lambda**3        [inverse volume]
    Lambda = h / sqrt(2 pi m k_B T)     [length]

``Lambda`` is the thermal de Broglie wavelength, the quantum length scale
of a free particle of mass m at temperature T in the run unit system; it is what
carries the "per unit volume" in z. For an ideal gas z *is* the number
density, so ``mu = k_B T ln(z Lambda**3)`` converts a target density into
the argument to pass. This is the convention of Frenkel & Smit eq. (5.6.9)
and of ``regressions/mc/metropolis/REFERENCE.md``, which pins <N>(z) against
the Mayer activity expansion using exactly this relation.

Because ``GCMC(T, mu, m, seed)`` builds ``Lambda`` from its ``m``, the
``InsertDeleteMove(type, atomic_number, m)`` that actually stamps atoms into
the cell must carry the same mass. The policy checks this match.
``InsertDeleteMove`` checks that the Frame is unbonded and contains
only the configured neutral species.

Chain continuation
------------------
Consecutive ``run()`` calls on one sampler extend ONE Markov chain — the
standard ``run(equilibrate); run(sample)`` pattern — so the tallies
accumulate and the configuration carries over. The frame is materialized
once, on the first call: edits made to the ``molpy.Frame`` between calls
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
After every ``run()`` the same ``molpy.Frame`` holds the current
configuration: positions [A], the box, and the last accepted energy in
``frame.meta["energy"]`` in run energy units (bit-identical to ``current_energy()`` — a
rejected trial's energy never survives). Because the sampler scores
energies and never assembles a gradient, ``fx``/``fy``/``fz`` come back
**zeroed**; they are not physical forces. Velocities are left untouched
(an inserted atom gets zero velocity), since nothing here moves in time.

Variable particle number
------------------------
muVT changes N, and ``to_molpy`` cannot write a changed row count into an
existing block. After a ``GCMC.run()`` that inserted or deleted atoms, the
frame's ``atoms`` block is therefore *rebuilt* at the new N — every column
comes back, including ``mass`` and whichever identity columns the frame
carried (``atomic_number`` for the species, ``type_id`` for the force-field
type), which the fixed-N path treats as immutable inputs and does not
write. An identity column the input frame lacked is not invented. Accumulate ``n_atoms()``
along the chain for <N>.

Single rank
-----------
Monte Carlo is single-rank by design: no MC task exists in the MPI launcher
protocol, so ``run()`` raises ``RuntimeError`` under
``atomiverse-launcher`` rather than silently sampling one independent chain
per rank.

Kinetic Monte Carlo
-------------------
Configure ``KMC(seed=...)`` with ``set_event(StateTransitions(edges))`` and
``set_solver(LinearKineticSolver())``.
Each edge is ``(source, destination, rate)`` with a rate in inverse picoseconds.
On a ``molpy.Frame``, set the integer initial label in ``meta["kmc_state"]``.
The driver writes ``kmc_time_ps``, ``kmc_waiting_time_ps`` and
``kmc_event_id`` to Frame metadata after each event. ``StateTransitions``
uses the zero-based table row as its channel ID. A custom C++ Event can
report an unwrapped ``kmc_displacement`` in Frame length units when
``kmc_has_displacement`` is true. Collect output at each event to obtain
event-resolved histories. The waiting time belongs to the state before
the event, so use that state for residence-time averages.

``run(frame, n_events)`` returns the number executed and can stop at an
absorbing state. Supply isolated state labels via ``states``. KMC uses no
trial acceptance statistics. The rates are constant between events. Their
physical interpretation depends on the supplied model. The solver receives
a full, freshly enumerated rate vector on each step.

The quantitative ensemble physics is pinned in ``regressions/mc/`` —
``npt_lj`` for <V> and its exact NPT width, ``gcmc_lj`` for <N>(z) against
the Mayer activity expansion.
"""

from __future__ import annotations

from atomiverse._binding import _cpp, require_domain

require_domain(__name__, "mc")

MC = _cpp.MC
KMC = _cpp.KMC
StateTransitions = _cpp.StateTransitions
LinearKineticSolver = _cpp.LinearKineticSolver


def NptMC(
    temperature: float, pressure: float, seed: int, units: str | None = None
) -> MC:
    """Construct an MC driver configured for isothermal-isobaric sampling."""
    driver = MC(seed=seed, units=units)
    driver.set_npt(temperature, pressure)
    return driver


def GCMC(
    temperature: float, mu: float, mass: float, seed: int, units: str | None = None
) -> MC:
    """Construct an MC driver configured for grand-canonical sampling."""
    driver = MC(seed=seed, units=units)
    driver.set_gcmc(temperature, mu, mass)
    return driver


#: Single-particle displacement Mover —
#: ``cpu::mc::TranslateMove(delta [A])``.
TranslateMove = _cpp.TranslateMove

#: Isotropic volume-change Mover — ``cpu::mc::VolumeMove(dv_max [A^3])``.
VolumeMove = _cpp.VolumeMove

#: Grand-canonical insertion/deletion Mover —
#: ``cpu::mc::InsertDeleteMove(type, atomic_number, mass)``.
InsertDeleteMove = _cpp.InsertDeleteMove

#: Weighted mix of Movers, one child per trial —
#: ``cpu::mc::MoveSelector()``.
MoveSelector = _cpp.MoveSelector

ConRotMove = _cpp.ConRotMove
ParallelRotationMove = _cpp.ParallelRotationMove
TorsionMove = _cpp.TorsionMove
AtomIdentityExchangeMove = _cpp.AtomIdentityExchangeMove
EndMerRotationMove = _cpp.EndMerRotationMove
LibrationMove = _cpp.LibrationMove
DimerFlipMove = _cpp.DimerFlipMove
PivotMove = _cpp.PivotMove
ReptationMove = _cpp.ReptationMove
GeneralizedReptationMove = _cpp.GeneralizedReptationMove
CBMCMove = _cpp.CBMCMove
ExtendedCBMCMove = _cpp.ExtendedCBMCMove

__all__ = [
    "MC",
    "KMC",
    "NptMC",
    "GCMC",
    "TranslateMove",
    "VolumeMove",
    "InsertDeleteMove",
    "MoveSelector",
    "ConRotMove",
    "ParallelRotationMove",
    "TorsionMove",
    "AtomIdentityExchangeMove",
    "EndMerRotationMove",
    "LibrationMove",
    "DimerFlipMove",
    "PivotMove",
    "ReptationMove",
    "GeneralizedReptationMove",
    "CBMCMove",
    "ExtendedCBMCMove",
    "StateTransitions",
    "LinearKineticSolver",
]
