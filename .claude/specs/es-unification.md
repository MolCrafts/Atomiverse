# ES unification — front/back split (live)

Goal: every electronic-structure method family presents the MD shape — a
public front (concrete method, Frame in/out, ctor parameters) over
a private back (engines per family per device). Delete when every phase closes.
Census evidence: `.claude/notes/es-architecture-debt.md` (rewrite it as phases
land).

Extension contracts and loop/model ownership are specified in
[es-extensibility.md](es-extensibility.md). A physical model or single-pass
method is not required to inherit Driver; only a genuine execution loop owns
Driver state. That distinction governs the phases below.

## Target contract (front)

- **One class per method per backend**, same shape on `cpu::` and `cuda::`:
  `RHF`, `UKS`, `PW`, `GpwDft`, `AM1`, `DFTB2`, `GFN2`, …
- **Parameters are ctor arguments**, fixed after construction. No member
  assignment as the parameter story, no `*Params`/`*Config` copies (task specs
  and Python bind the ctor, not a field list).
- **Real SCF/SCC loop = `Driver<<family>::Phase>`**, owned or inherited,
  with typed stages; stages publish
  residuals into `ConvergenceMetrics`, a Criterion judges. No hand-written
  copies of iteration policy and no algorithm-side `converged = …`.
  Single-pass methods remain Components; terms/data/kernels are not Drivers.
- **Entries:** `run(Frame&)` → `kEnergy` (+`kEnergyElectronic`);
  `run(const Frame&)` → nothing; `forces(Frame&)` → `kAtom_f*`, energy;
  `stress(Frame&)` (periodic) → `kStress` = σΩ, returns σ. Overwrite.
  Hartree/Ry only on the returned result object.
- **One bind rule:** `outer.on(phase).add(method)` mounts a single-pass
  Component or the method's genuine nested loop **on the parent Runtime** and
  **accumulates** (`+=`)
  energy/forces/stress. What it computes follows the parent's
  `forces_requested` / `stress_requested`; there is no `*Evaluator` wrapper and
  no `with_gradient`.
- **Properties** (charges, dipoles, populations) are written to Frame columns.
- **Composition, not wrappers:** electronic propagators (CPMD, second-gen CP,
  XL-BOMD, ASPC) are one composed node over any method that exposes an
  electronic-state protocol; QM/MM `set_engine(method)` runs the method through
  the same nested mount on the QM sub-frame (no dummy Context).
- **Post-SCF methods** (post-HF, TDDFT, DFPT, bands/DOS, Berry, Wannier) take a
  converged method and read its state; they never run a private SCF.

## Target contract (back)

- Engines (integrals/grids/FFT, Hamiltonian build, eigensolver, mixer,
  density) are `internal`, owned by the method, rebuilt from Frame. No public
  header includes `internal/`; nothing `internal` appears in a public signature
  or result type.
- Public mathematical/data extension contracts are separate from private
  engines; external implementations must not supply a Context/Runtime or
  access private workspace. See `es-extensibility.md` for the narrow seams.
- Algorithm pieces with ≥ 2 users live once under a shared ES internal module
  (Pulay/DIIS, Anderson/Broyden charge mixing, dense eigensolve, convergence
  metrics); families do not keep private copies.
- Additive energy terms that do not need the density (D3, D3BJ, D4, TS with
  free volumes) are `Potential` components with Frame forces/stress, usable
  next to any method; density-dependent terms stay inside the method.
- CUDA replaces engines, never the front; host setup may reuse CPU internals.

## Progress

- Phase 1, Gaussian SCF: nested `MeanFieldMethod` runs its stage plan on the
  parent Runtime (per-solve build/release).
- Phase 1, NDDO: `AM1`/`RM1` mount directly (`NddoMethod::build_on` /
  `run_nested_erased`, molecular unwrap, accumulate, force request from the
  phase); `AM1Evaluator`/`RM1Evaluator` deleted; task wire `AM1Params`;
  Python `AM1` is the solve object (read-only parameters).
- Phase 1, PW: `PW` mounts directly (one SCF per step; forces on request;
  virial only when the plan consumes stress — `Plan::stress_consumed`, set
  from `requires_system_stress` on components and criteria; cell relaxers and
  stress/pressure criteria now declare it); `PWEvaluator` deleted; task wire
  `PWParams`; Python `set_evaluator(PW)`.
- Phase 1, GPW: `GpwDft` mounts directly for Born–Oppenheimer work (DIIS
  stages on the parent Runtime, warm start from its previous converged
  density, forces on request, `gpw_stress_at_state_impl` virial on demand).
  `gpw-bomd-water` and `gpw-variable-cell-ice` mount it.
  **Transitional debt (owner: phase 5):** `GPWEvaluator` still hosts the
  electronic propagators (CPMD, second-gen CP, XL-BOMD) and remaining plain BO
  regression sections plus `test_phonon_gpw_wiring` still use it, so the plain BO
  write-back exists twice (`GpwDft::run_nested_erased`,
  `GPWEvaluator::launch`). Phase 5 moves the propagators to the composed
  node, migrates those users and deletes `GPWEvaluator`.
- DFTB0 mounts as a single-pass Component. DFTB2 now owns a persistent SCC
  Driver and mounts directly; DFTB3/GFN1/GFN2/IPEA remain Components composing
  the shared SCC helper. Direct mounting is not contingent on inheritance.
  Their lifecycle/request/provider gaps are tracked in the SE extension review.
- Phase 2: identity parameters are ctor arguments and `const` members on
  Gaussian SCF (basis/charge/occupations/xc), `PW` (CPU+CUDA, pseudopotential
  tag `UpfPseudopotential`/`GthPseudopotential`, cutoff, k-mesh, xc), `GpwDft`
  (CPU+CUDA, kinds/cutoff/xc), AM1/RM1/DFTB2/DFTB3/GFN0/GFN1/GFN2/IPEA
  (charge, multiplicity, SK set); no default ctor where no identity default
  exists; move-assignment deleted. `ContinuumSolvent` deleted (`WithSolvent`
  takes model + ε). `Counterpoise` requires a neutral method instead of
  rewriting its charge. The duplicate `ScfElectronicThermostat` /
  `ElectronicThermostat` bags go with the propagator node (phase 5).
- Phase 3, SCC: DFTB2 owns persistent stages and uses Driver nested execution;
  DFTB3/GFN1/GFN2/IPEA compose `SccDriver`. They share SCC stage/criterion
  primitives, but iteration/lifecycle orchestration still has two paths.
  Consolidate the matching loop implementation while retaining method-owned
  workspace; no blanket Component-to-Driver conversion. NDDO convergence is
  `NddoConverged`: it judges
  the density and energy residuals `scf_check` publishes (`<=` either
  tolerance). The solver no longer assigns `result_.converged`; `drive_`
  copies the Driver verdict onto the result after the loop. Still open in
  this phase: GPW OT as stages, and PW post-processors on method state.

## Phases (each keeps every regression green)

1. **Mount** — nested ES methods run on the parent Runtime and accumulate;
   driven by the parent's force/stress request. Removes `PWEvaluator`,
   `GPWEvaluator`, `AM1Evaluator`, `RM1Evaluator`, the task `*Params` copies,
   and the private-Runtime path of `MeanFieldMethod`.
2. **Parameters** — ctor arguments for RKS/UKS/ROHF/ROKS/GHF (basis), PW, GPW,
   NDDO, DFTB/xTB, post-processors; drop `NDDOConfig`, `ContinuumSolvent` bag.
3. **Loops** — real SCC loops use Driver; DFTB2's persistent and the helper
   orchestration still need consolidation. CPU NDDO shares a Driver and
   Criterion; CUDA NDDO's four execution copies need consolidation. Single-pass
   methods and physical terms remain non-Drivers. Still open: GPW OT as stages
   (no inline DIIS copy); PW post-processors on the method's state.
4. **Surface** — Frame `forces`/`stress` on every method that has them
   (DFTB/xTB, Gaussian periodic stress where implemented); dispersion
   components; remove internal leaks from public headers/results; GPW
   regressions off `internal/`.
5. **Composition** — one propagator node for CPMD/XL-BOMD/ASPC across Gaussian,
   GPW and DFTB; QM/MM through the nested mount; post-HF write semantics
   (`kEnergy` = correlated energy).
6. **CUDA fronts** — `cuda::pw::PW` and `cuda::gpw::GpwDft` own or inherit
   Drivers for their genuine electronic loops, with the
   CPU shape (no `GpwAtom`/`cell` public, `stress(Frame&)`). CUDA RHF already
   has `run(const Frame&)`; UHF still needs that overload. Verify accumulation
   on nested mount; no public device pointers.

DFT status refresh: current architecture closures and surviving numerical
recheck targets are in `.claude/notes/es-architecture-debt.md`; the
method/backend/quantity evidence matrix is only in `docs/roadmap/dft.md`.
Source/registration review does not close this spec's numerical gates.

Order inside each phase: Gaussian SCF → NDDO/DFTB/xTB → PW → GPW (smallest to
largest blast radius). CPU before CUDA.
