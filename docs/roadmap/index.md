# Scientific Methods Roadmap

This roadmap is the honest inventory of what Atomiverse can actually compute
today, method by method, on each execution backend. It is written against the
code on `dev`, not against intent: a row is only marked implemented if the
method is reachable from the public API and validated.

## Status

| Symbol | Meaning                                 |
| ------ | --------------------------------------- |
| ○      | Not implemented                         |
| ✓      | Implemented and numerically correct     |
| ⚡      | Numerically correct and fully optimized |

A CUDA implementation must execute its principal computation on CUDA. A CPU
fallback is not considered CUDA support.

### What this roadmap does not list

A method that a **strictly better algorithm dominates** — same purpose, same or lower
cost, no regime where the older one wins — is simply **not in these tables**. It is not
marked, not tracked, and not planned. Listing it, even as "will not implement", would
suggest the roadmap has an opinion worth reading about it; it does not.

So there is no BFGS row (L-BFGS matches its convergence at O(mN) memory instead of
O(N²)), no Fletcher–Reeves row (Polak–Ribière⁺ restarts strictly better at the same
cost), no Berendsen-thermostat row (CSVR samples the canonical ensemble; Berendsen does
not sample it at all).

This is only about *dominance*, never about our schedule. A method that survives because
it occupies a genuinely different cost/accuracy niche stays ○ and stays listed — steepest
descent is dominated by CG for minimization proper, but it is the robust first resort on a
badly broken start geometry, so it keeps its row. Brownian dynamics keeps its row too: it
is different physics from Langevin, not a worse integrator.

### How the symbols are assigned here

**✓** means the method runs from the public API and its numbers are pinned
against an external reference or an analytic result — a validation case under
`benchmarks/<domain>/<case>/validation/` (RHF, MP2, FCI, GPW, vibrations,
plane-wave Si, LJ MD, bonded MD vs LAMMPS, LJ MC vs NIST), or a unit test that
checks the physics rather than the plumbing.

**⚡** is deliberately scarce. It is reserved for kernels that are both correct
*and* carry a committed performance suite measured against a production engine.
Today that is exactly one method family: the Lennard–Jones 12-6 pair style,
benchmarked against LAMMPS/Kokkos on CPU (xsimd, `-march=native`) and on GPU
(`/* benchmarks removed; re-add later */ lj/performance/`). Everything else that is correct but has no
standing performance gate is a ✓, including code that is fast in practice.

**○** means not implemented. It does not distinguish "planned" from "not
planned"; the sections below are a method census, not a commitment schedule.

## Chapters

Each chapter is one **method family**. Periodicity, gradients, and excited-state
response are *axes*, not families — a periodic calculation is still DFT — so
those rows live inside the method that owns them: periodic DFT (plane-wave, GPW)
is a section of the DFT chapter, periodic HF a section of the Hartree–Fock one.

| Chapter | Covers |
| ------- | ------ |
| [Hartree–Fock](hartree-fock.md) | RHF/UHF/ROHF/GHF, HF gradients and Hessians, periodic HF, SCF infrastructure |
| [Density-Functional Theory](dft.md) | Kohn–Sham methods, functionals (LDA → double hybrid), dispersion, **periodic DFT: plane-wave, GPW, forces, stress, phonons** |
| [Post-Hartree–Fock](post-hf.md) | Møller–Plesset, configuration interaction, coupled cluster, multireference |
| [Excited States and Many-Body](excited-states.md) | CIS/TDHF/TDDFT, ADC, EOM-CC, GW/BSE/RPA |
| [Semiempirical and Tight Binding](semiempirical.md) | NDDO/AM1, DFTB, extended tight binding (xTB) |
| [Force Fields](force-fields.md) | Bonded, nonbonded, electrostatics, polarizable, many-body/reactive, coarse-grained, machine-learning potentials |
| [Molecular Dynamics](molecular-dynamics.md) | Integrators, thermostats, barostats, ensembles, constraints, nonequilibrium MD, AIMD, nuclear quantum, nonadiabatic, vibrations |
| [Geometry Optimization](optimization.md) | Minimizers, transition states, reaction paths |
| [Sampling and Free Energy](sampling.md) | Monte Carlo, enhanced sampling, free-energy methods |
| [Multiscale and Properties](multiscale.md) | Atomic charges, implicit solvent, QM/MM, embedding, relativistic, intermolecular interactions |

## Work in progress — known gaps and caveats

The tables tell you *whether* a method exists. This section tells you what to
watch out for in the ones that do.

### Gaussian SCF

- **Cartesian basis functions only.** There is no spherical-harmonic (pure)
  transform, so `d` and higher shells carry their contaminant `s`/`p`
  components. The native Rys ERI engine covers `l ≤ 3` (s/p/d/f).
- **No analytic Hessian anywhere.** Every second derivative in the engine —
  frequencies, thermochemistry — is a central finite difference of analytic
  forces (`cpu::vib::Hessian`). CPHF is not implemented.
- **Open shell defaults to unrestricted.** ROKS does not exist; ROHF exists as
  an energy-only method (`cpu::scf::ROHF`, PySCF canonical / Guest–Saunders
  convention) — no analytic gradient, no ROHF-MP2.
- **CUDA SCF is a restricted-HF energy engine, not a general one.** It runs the
  Fock build, density build, and diagonalization (cuSOLVER) on device, but it
  has no DIIS, no UHF, no Kohn–Sham path, and **no gradients** — so there is no
  GPU AIMD. Its device-native integral kernels are s-type only; anything beyond
  s shells must be fed a host-built integral pack.
- **MINAO guess covers H, C, N, O, F only**; other elements silently fall back
  to the Hcore guess.

### Post-Hartree–Fock

- **MP2 is closed-shell canonical only** — an RHF reference, a full O(N⁵)
  AO→MO transform, no frozen core, no density fitting/RI, no gradients.
- **FCI is a reference tool, not a production method.** It builds the dense
  determinant CI matrix and diagonalizes it, and throws past a hard cap
  (`n_mo > 14` or `N_det > 2000`). It exists to pin the exact answer for tiny
  systems (H₂/STO-3G), not to correlate real ones.

### Exchange-correlation

- The functional set is exactly `{LSDA, PBE, B3LYP, PBE0}`. B88 exchange and
  LYP correlation are implemented, but only as B3LYP components — there is no
  standalone BLYP or BP86 selector.
- **libxc is not a runtime dependency.** All functionals are hand-coded; libxc
  appears only as an optional *test oracle* (`ATV_BUILD_XC_ORACLE`) that the
  native implementations must match to 1e-12.
- **No XC on the GPU outside plane waves.** The only device functional is
  Slater + PW92 LDA on the plane-wave grid.

### Plane-wave DFT

- **Ionic forces: analytic local HF + Ewald + KB nonlocal**
  (`pw_forces_analytic`); FD (`pw_forces_fd`) remains the oracle.
  **Cell stress via strain-FD of SCF energy** (`pw_stress`, Voigt `kStress`);
  Nielsen–Martin analytic decomposition remains a follow-on.
- **Effectively LDA.** The XC builder calls the PBE routine with `sigma = 0`,
  which is the Slater + PW92 LDA limit. There is no true GGA and no spin
  polarization in the plane-wave path.
- Pseudopotentials are norm-conserving UPF (v1/v2) with KB projectors. No
  ultrasoft, no PAW. An empty UPF path selects a toy local potential that runs
  but is unphysical.
- On CUDA the Davidson H·ψ hot path, FFTs (cuFFT), Poisson, V_eff, and LDA all
  run on device; the Rayleigh–Ritz subspace bookkeeping and the SCF control
  loop stay on the host.

### GPW

- **Γ-point only, closed-shell only.** No k-points, no spin polarization, no
  stress tensor, no hybrids.
- Analytic forces (Hellmann–Feynman + Pulay) are available and validated, so
  GPW *can* drive AIMD. `GpwConfig.local_pp_on_grid` must be consistent between
  energy and gradient — the force is the derivative of whatever functional the
  SCF actually converged.
- **DFT-D3 is zero-damping only**, with PBE parameters. Becke–Johnson damping,
  D2, and D4 are not implemented.

### Semiempirical

- AM1 is the only parameterization (no MNDO/PM3/PM6/PM7/RM1). **Analytic
  nuclear forces** are available via `AM1Evaluator` (`with_gradient`, default
  on) on the shared `NDDOSolver` path, so AM1 can drive Evaluate / MD / Minimize
  force phases. Hessian remains finite-difference (`cpu::vib::Hessian`).
- AM1-BCC charges are implemented on top of it (AM1 Mulliken + molrs BCC typing).
- DFTB / xTB are not implemented; when they are, they get their own Solver +
  Evaluator (+ kind), not a new Driver and not `NDDOSolver`.

### Classical force fields

- **The pair-style catalogue is thin but GAFF-capable on CPU**: LJ 12-6 (multi-type
  + Lorentz–Berthelot; per-type path is scalar), cut Coulomb, real-space
  `coul/long` + PME, and **Amber `special_bonds` scales** (including 1–4 = 0.5)
  on LJ, CoulCut, and CoulLong. `load_lammps_data` reads `atom_style full` (charges).
  Still missing for full GAFF productization: multi-term Fourier dihedral component,
  complete `.ff` text parser, CUDA Amber parity, paper-length polymer protocols.
- **Improper harmonic exists** (periodic/CVFF still open). No reactive potentials
  (ReaxFF, AIREBO), no coarse-grained models.
- **Many-body: EAM and Stillinger–Weber now exist** (CPU, single-element, MPI-parallel
  via a ghost-of-ghost halo; see [Chapter 16](../tutorial/ch16-many-body-potentials.md)).
  Tersoff/MEAM remain absent, as does any CUDA mirror.
- **Transition states: CI-NEB runs** (`cpu::PathDriver` — N coupled replicas, stopping
  on the `AllReplicasConverged` criterion). It is CPU-only and single-replica-
  per-rank (the band is replicated; the spatial decomposition lives inside each image).
  No dimer, no string method, no IRC. The replica infrastructure (`ReplicaSet`,
  `ReplicaExpansion`) is deliberately general — REMD/PIMD/umbrella would reuse it — but
  none of those exist.
- **Constraints: RATTLE runs** (`cpu::md::FixRattle`, CPU, MPI-parallel). SETTLE and
  LINCS are deliberately not written — they are optimizations of a solver that already
  holds rigid geometry to 5 × 10⁻¹¹. No rigid bodies, no virtual sites.
- **The torch backend is not a machine-learning potential.** It is a
  differentiable classical force field: LJ and harmonic bond with autograd
  backward passes w.r.t. positions *and* parameters (ε, σ, K, r₀) — a
  parameter-fitting surface. It has no integrator, no driver, no thermostat.

### Molecular dynamics

- The standalone `NoseHoover` thermostat is a **single** chain (scalar ξ), a
  documented divergence from LAMMPS. The real M-chain NHC (default M = 3) exists
  only inside the MTK NPT integrator.
- The MTK barostat couples the **diagonal** of the cell only. There is no
  off-diagonal/shear coupling, hence no true Parrinello–Rahman and no
  constant-stress dynamics.
- **NPH is not reachable**: the MTK integrator fuses the thermostat chain with
  the barostat, and there is no thermostat-free barostat integrator.
- `Deform` is **uniaxial only** (one axis, engineering or true strain). No
  shear, no SLLOD, no Lees–Edwards.
- The reciprocal-space solvers (Ewald, PME) are **single-GPU only** on CUDA, and
  the CPU PME replicates the grid across ranks rather than distributing the FFT.

### Sampling

- Monte Carlo is CPU-only, **three ensembles**: NVT (`mc::MC`), NPT
  (`mc::NptMc`, isotropic volume moves), and μVT GCMC (`mc::Gcmc`,
  single-species insert/delete, single rank). Mixed-move ensembles compose via
  `MoveSelector`. Trial energy is a deliberate O(N) full recompute
  (`energy_of_rows()` is the follow-up performance lever). No CBMC, no Widom
  insertion, no Gibbs ensemble.
- There is **no enhanced sampling and no free-energy machinery** — no umbrella
  sampling, metadynamics, FEP, TI, or BAR.
- Transition states: **CI-NEB runs** (`cpu::PathDriver`; see Force Fields
  above). No dimer, no string method.

### Optimization

- CPU minimizers are LBFGS and Polak–Ribière⁺ CG (steepest descent exists only
  as the degenerate β = 0 case); CUDA additionally ships a true steepest-descent
  component. FIRE is not implemented.
- All minimizers move **atoms only** — there is no cell or variable-cell
  optimization.
- Optimization is Cartesian; there are no internal or redundant-internal
  coordinates.
