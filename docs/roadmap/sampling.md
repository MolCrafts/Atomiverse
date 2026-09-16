# Sampling and Free Energy

Legend and caveats: [Roadmap index](index.md).

Monte Carlo exists as three CPU **presets** over one shared trial loop
(`cpu::mc::McBase`): `MC` (NVT), `NptMc` (NPT), and `Gcmc` (μVT) — each is a
constructor that attaches its **outcome policy** (`Metropolis`,
`NptMetropolis`, `GcmcMetropolis`; accept/reject → `Commit`/`Rollback`). The
trial changeset is a **Frame patch** (`mc::Transaction`: the touched atoms'
original rows + box + pre-trial `nlocal`), so displacement, insertion,
deletion, and volume moves all roll back through one splice — bitwise, with
topology. Movers: `TranslateMove`, `VolumeMove`, `InsertDeleteMove` (single
species), composed one-trial-per-sweep by `MoveSelector`. Variable N reuses
the migration primitives (`Atom::copy_row`, `pack/unpack_exchange`,
`acquire_tag`). Single-rank; trial energy is an O(N) full recompute by design
(the only form valid for all potentials — `energy_of_rows()` is the follow-up
performance lever). NVT is validated against NIST LJ reference data
(`/* benchmarks removed; re-add later */ lj/validation/`); μVT and NPT against the analytic Mayer
activity expansion / finite-N ⟨V⟩ relation for the exact cut-shifted
potential (`regressions/mc/gcmc_lj`, `regressions/mc/npt_lj`).

**Enhanced sampling is a working textbook suite, not a PLUMED replacement.** A
CV protocol (`cpu::bias::CollectiveVariable`) plus `DistanceCV`, `AngleCV`,
`DihedralCV` and `HarmonicBias` support single-window umbrella sampling
(`reg_umbrella_harmonic`, `reg_umbrella_angle`, `reg_umbrella_dihedral`).
Temperature REMD (`cpu::path::Remd`) and Hamiltonian REMD on a λ ladder
(`cpu::path::HRemd`, `reg_hremd_soft_lambda`) ship as method presets over N
owned `MD` programs. Soft-core λ TI and **Widom** also exist.
**Well-tempered metadynamics:** 1D (`Metadynamics1D`, `reg_metad_double_well`)
and multi-CV (`Metadynamics`, `reg_metad_2d`). **Multi-window WHAM** is a thin
C++ façade over molrs (`cpu::Wham`, `reg_wham_double_well`). Still absent:
multi-walker metaD, OPES, ABF, MBAR production PMF protocols. No PLUMED
interface.

## 1. Monte Carlo Methods

| Method                                      | CPU | CUDA | Kokkos |
| ------------------------------------------- | :-: | :--: | :---: |
| Metropolis Monte Carlo                      |  ✓  |   ○  |   ○  |
| Canonical Monte Carlo, NVT                  |  ✓  |   ○  |   ○  |
| Isothermal–isobaric Monte Carlo, NPT        |  ✓  |   ○  |   ○  |
| Grand-canonical Monte Carlo                 |  ✓  |   ○  |   ○  |
| Gibbs-ensemble Monte Carlo                  |  ○  |   ○  |   ○  |
| Semi-grand-canonical Monte Carlo            |  ○  |   ○  |   ○  |
| Configurational-bias Monte Carlo            |  ○  |   ○  |   ○  |
| Continuous-fractional-component Monte Carlo |  ○  |   ○  |   ○  |
| Hybrid Monte Carlo                          |  ○  |   ○  |   ○  |
| Hamiltonian Monte Carlo                     |  ○  |   ○  |   ○  |
| Replica-exchange Monte Carlo                |  ○  |   ○  |   ○  |
| Simulated tempering                         |  ○  |   ○  |   ○  |
| Wang–Landau sampling                        |  ○  |   ○  |   ○  |
| Multicanonical sampling                     |  ○  |   ○  |   ○  |
| Nested sampling                             |  ○  |   ○  |   ○  |
| Kinetic Monte Carlo                         |  ○  |   ○  |   ○  |
| Rejection-free kinetic Monte Carlo          |  ○  |   ○  |   ○  |
| Reaction-ensemble Monte Carlo               |  ○  |   ○  |   ○  |

NPT-MC and GCMC confirmed the prediction this roadmap made before they were
written: each was a small `OutcomePolicy` plus its mover, with **nothing new
in the loop** — no driver class, no category. The remaining ensembles keep
that shape: a Gibbs-ensemble exchange and a replica-exchange swap are each one
policy object plus a mover (REMD additionally a nesting question —
`cpu/replica/` already supplies `ReplicaSet`/`ReplicaExpansion`). GCMC's known
ceilings: single rank, single insertable species, O(N) trial energy
(`energy_of_rows()` is the performance follow-up), no CBMC (molecular
insertion needs the topology loaders re-run per trial).

## 2. Enhanced-Sampling Methods

| Method                                  | CPU | CUDA | Kokkos |
| --------------------------------------- | :-: | :--: | :---: |
| Umbrella sampling                       |  ✓  |   ○  |   ○  |
| Adaptive umbrella sampling              |  ○  |   ○  |   ○  |
| Multidimensional umbrella sampling      |  ○  |   ○  |   ○  |
| Well-tempered metadynamics              |  ✓  |   ○  |   ○  |
| Multi-CV well-tempered metadynamics     |  ✓  |   ○  |   ○  |
| Multiple-walker metadynamics            |  ○  |   ○  |   ○  |
| Bias-exchange metadynamics              |  ○  |   ○  |   ○  |
| Parallel-bias metadynamics              |  ○  |   ○  |   ○  |
| Funnel metadynamics                     |  ○  |   ○  |   ○  |
| Adaptive biasing force                  |  ○  |   ○  |   ○  |
| Extended-system adaptive biasing force  |  ○  |   ○  |   ○  |
| Adaptive biasing potential              |  ○  |   ○  |   ○  |
| Accelerated weight histogram            |  ○  |   ○  |   ○  |
| Variationally enhanced sampling         |  ○  |   ○  |   ○  |
| OPES                                    |  ○  |   ○  |   ○  |
| OPES Metad                              |  ○  |   ○  |   ○  |
| OPES Expanded                           |  ○  |   ○  |   ○  |
| Gaussian accelerated molecular dynamics |  ○  |   ○  |   ○  |
| Temperature-accelerated MD              |  ○  |   ○  |   ○  |
| Hyperdynamics                           |  ○  |   ○  |   ○  |
| Integrated tempering sampling           |  ○  |   ○  |   ○  |
| Solute tempering                        |  ○  |   ○  |   ○  |
| REST2                                   |  ○  |   ○  |   ○  |
| Replica-exchange molecular dynamics     |  ✓  |   ○  |   ○  |
| Hamiltonian replica exchange            |  ✓  |   ○  |   ○  |
| Multidimensional replica exchange       |  ○  |   ○  |   ○  |
| Expanded-ensemble sampling              |  ○  |   ○  |   ○  |
| Weighted-ensemble sampling              |  ○  |   ○  |   ○  |
| Transition-path sampling                |  ○  |   ○  |   ○  |
| Transition-interface sampling           |  ○  |   ○  |   ○  |
| Forward-flux sampling                   |  ○  |   ○  |   ○  |
| Milestoning                             |  ○  |   ○  |   ○  |

## 3. Free-Energy Methods

| Method                                  | CPU | CUDA | Kokkos |
| --------------------------------------- | :-: | :--: | :---: |
| Free-energy perturbation                |  ✓  |   ○  |   ○  |
| Thermodynamic integration               |  ✓  |   ○  |   ○  |
| Multistate Bennett acceptance ratio     |  ○  |   ○  |   ○  |
| Umbrella integration                    |  ○  |   ○  |   ○  |
| Weighted histogram analysis method      |  ✓  |   ○  |   ○  |
| Potential-of-mean-force calculation     |  ○  |   ○  |   ○  |
| Adaptive integration of mean force      |  ○  |   ○  |   ○  |
| Jarzynski equality                      |  ○  |   ○  |   ○  |
| Crooks fluctuation theorem              |  ○  |   ○  |   ○  |
| Nonequilibrium switching                |  ○  |   ○  |   ○  |
| Bidirectional nonequilibrium estimation |  ○  |   ○  |   ○  |
| Widom insertion                         |  ✓  |   ○  |   ○  |
| Absolute solvation free energy          |  ○  |   ○  |   ○  |
| Relative binding free energy            |  ○  |   ○  |   ○  |
| Absolute binding free energy            |  ○  |   ○  |   ○  |
| Double-decoupling method                |  ○  |   ○  |   ○  |
| Alchemical lambda dynamics              |  ○  |   ○  |   ○  |
| Constant-pH free-energy calculation     |  ○  |   ○  |   ○  |

Umbrella *samples* are collected by `HarmonicBias` on MD. **Multi-window WHAM**
ships as `cpu::Wham` (thin C++ over molrs histogram math;
`reg_wham_double_well`) — free-energy / PMF reconstruction from pre-binned
windows, not a turnkey multi-window campaign driver. **Widom insertion** ships
as `cpu::Widom` (Observer / snapshot estimator): μ_ex = −kT ln⟨e^{−βΔU}⟩ via
ghost insertions (LJ cut-and-shift or hard sphere). Not GCMC; not absolute
solvation protocols. Soft-core LJ/Coulomb for alchemical work is
`cpu::PairLJSoft` / `cpu::PairCoulSoft`. **Kirkwood TI** on fixed soft LJ
(`dU_dlambda`, `reg_fep_soft_dimer`) is the free-energy *primitive* — not a
multi-window BAR/MBAR campaign (see
[Force Fields](force-fields.md#2-classical-nonbonded-potentials)).
