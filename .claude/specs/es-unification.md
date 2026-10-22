# ES unification — front/back split (live)

Goal: every electronic-structure method family presents the MD shape — a
public front (concrete method = loop node, Frame in/out, ctor parameters) over
a private back (engines per family per device). Delete when every phase closes.
Census evidence: `.claude/notes/es-architecture-debt.md` (rewrite it as phases
land).

## Target contract (front)

- **One class per method per backend**, same shape on `cpu::` and `cuda::`:
  `RHF`, `UKS`, `PW`, `GpwDft`, `AM1`, `DFTB2`, `GFN2`, …
- **Parameters are ctor arguments**, fixed after construction. No member
  assignment as the parameter story, no `*Params`/`*Config` copies (task specs
  and Python bind the ctor, not a field list).
- **Loop = `Driver<<family>::Phase>`** with typed stages; stages publish
  residuals into `ConvergenceMetrics`, a Criterion judges. No hand-written
  `for (iter)` SCF loop and no algorithm-side `converged = …`.
- **Entries:** `run(Frame&)` → `kEnergy` (+`kEnergyElectronic`);
  `run(const Frame&)` → nothing; `forces(Frame&)` → `kAtom_f*`, energy;
  `stress(Frame&)` (periodic) → `kStress` = σΩ, returns σ. Overwrite.
  Hartree/Ry only on the returned result object.
- **One bind rule:** `outer.on(phase).add(method)` mounts the method as a
  nested node that runs **on the parent Runtime** and **accumulates** (`+=`)
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
- Next in phase 1: PW, then GPW (nesting currently rejected), DFTB/xTB.

## Phases (each keeps every regression green)

1. **Mount** — nested ES methods run on the parent Runtime and accumulate;
   driven by the parent's force/stress request. Removes `PWEvaluator`,
   `GPWEvaluator`, `AM1Evaluator`, `RM1Evaluator`, the task `*Params` copies,
   and the private-Runtime path of `MeanFieldMethod`.
2. **Parameters** — ctor arguments for RKS/UKS/ROHF/ROKS/GHF (basis), PW, GPW,
   NDDO, DFTB/xTB, post-processors; drop `NDDOConfig`, `ContinuumSolvent` bag.
3. **Loops** — DFTB/xTB SCC as `Driver<scc::Phase>` (one loop for the 9 copies);
   GPW OT as stages (no inline DIIS copy); PW post-processors on the method's
   state; NDDO/DFTB convergence through Criteria.
4. **Surface** — Frame `forces`/`stress` on every method that has them
   (DFTB/xTB, Gaussian periodic stress where implemented); dispersion
   components; remove internal leaks from public headers/results; GPW
   regressions off `internal/`.
5. **Composition** — one propagator node for CPMD/XL-BOMD/ASPC across Gaussian,
   GPW and DFTB; QM/MM through the nested mount; post-HF write semantics
   (`kEnergy` = correlated energy).
6. **CUDA fronts** — `cuda::pw::PW` and `cuda::gpw::GpwDft` as Drivers with the
   CPU shape (no `GpwAtom`/`cell` public, `stress(Frame&)`); CUDA RHF/UHF gain
   `run(const Frame&)`, accumulate on nested mount; no public device pointers.

Order inside each phase: Gaussian SCF → NDDO/DFTB/xTB → PW → GPW (smallest to
largest blast radius). CPU before CUDA.
