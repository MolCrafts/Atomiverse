# Notes — current understanding

Conclusions only. Binding rules live in docs/developer/project-contract.md; history is git's.

## Topic notes

| Area | Note |
| --- | --- |
| CPU MD execution vs LAMMPS | [cpu-lammps-alignment.md](cpu-lammps-alignment.md) |
| CPU MD communication | [md-comm.md](md-comm.md) |
| CUDA MD: NVIDIA libraries, OpenMM method | [cuda-md-nvidia-libraries.md](cuda-md-nvidia-libraries.md), [cuda-openmm-md-perf.md](cuda-openmm-md-perf.md) |
| GAFF benchmark (CPU vs LAMMPS, CUDA vs OpenMM) | [gaff-md-perf.md](gaff-md-perf.md) |
| Machines, accounts, workbenches | [arrhenius-workbench.md](arrhenius-workbench.md) |
| BLAS threading traps | [openblas-thread-binding.md](openblas-thread-binding.md) |
| ES public-shape debt | [es-architecture-debt.md](es-architecture-debt.md) |
| SE loop/model extension review | [se-extensibility-audit.md](se-extensibility-audit.md); design [es-extensibility.md](../specs/es-extensibility.md) |
| ES optimization state | [es-optimization-order.md](es-optimization-order.md) |
| CUDA ES (RHF, PW) | [cuda-rhf-device-only.md](cuda-rhf-device-only.md), [cuda-only-no-dh.md](cuda-only-no-dh.md) |
| CUDA GPW | [cuda-gpw-public-shape.md](cuda-gpw-public-shape.md) |
| GPW perf vs CP2K | [gpw-perf-architecture.md](gpw-perf-architecture.md), [gpw-perf-baseline.md](gpw-perf-baseline.md) |
| GPW numerics | [gpw-dftu-projector-divergence.md](gpw-dftu-projector-divergence.md), [gpw-gamma-vs-kpoint-density-divergence.md](gpw-gamma-vs-kpoint-density-divergence.md) |
| Oracle pins | [oracle-pin-provenance.md](oracle-pin-provenance.md) |
| TB loop | [dftb-vs-dftbplus-cleanup.md](dftb-vs-dftbplus-cleanup.md), [tb-blocked-capabilities.md](tb-blocked-capabilities.md), [dftb-plus-u.md](dftb-plus-u.md) |
| MetalWalls loop | [metalwalls-capability-map.md](metalwalls-capability-map.md) |

Loop queues live under `.claude/campaigns/` (gitignored): `cp2k-method-queue.md`,
`gpw-perf-queue.md`, `es-cpu-cuda-queue.md`, `arch-campaign-queue.md`,
`claim-completion-queue.md`. Always wipe builds after a gpw-perf wave.

## Open engineering leads

- **MD unification (CPU/CUDA).** CPU and CUDA run different execution
  machinery (docs/developer/project-contract.md › Known debt). Open plan:
  `.claude/specs/md-unification.md`; CPU contract work:
  `.claude/specs/cpu-orthogonality.md`.
- **ES public-shape debt.** Do not extend dialects;
  [es-architecture-debt.md](es-architecture-debt.md).
- **GPW MG MAIN wall** is the worst CP2K ratio (2.33×), but it was measured
  with OpenMP binding on, i.e. ATV BLAS single-threaded — re-measure unbound
  before treating it as a gap ([openblas-thread-binding.md](openblas-thread-binding.md)).
- **Open-shell cluster GPW mixer.** Si₂-triplet GTH in a cluster box is
  SCF-ill-conditioned for ATV (linear/DIIS/OT) where CP2K converges with
  Broyden; a better mixer is the gap and blocks the `gpw-fd-phonon-si` e2e.
- **CPU MD vs LAMMPS.** Same-input LJ/EAM/SW+LJ harness:
  `benchmarks/compare/lj-lammps` (results not in git). Open: Comm category slower than LAMMPS (includes
  ATV reductions), SW+LJ neighbor build ~1.5–2×, no x86/cluster numbers.
- **Component clone ceiling** — revisit when a real N-replica inner-loop user
  appears.

## Open debt found, not yet fixed (verify before acting)

- `src/cpu/potentials/pair_morse.cpp` (≈43–50) and `pair_yukawa.cpp` negate
  dx/dy/dz twice on the listed path — dead code left by the image rewrite.
- Stale comments: `core/plan.h` ("shared by CPU and CUDA"), `core/stage.h`
  md::Phase docs (pre_exchange/pre_neighbor/post_force described as doing the
  comm Runtime now does), `cpu/component/context.h` (`Backend::init()` names),
  `docs/developer/architecture.md` (shared initial-force schedule / ledger),
  `src/cuda/dynamics/brownian.cu` ("step is an int").
- `regressions/mc/metropolis` is red (⟨N⟩ below the activity expansion on the
  rewritten trial loop).
- Minimize/Optimizer vocabulary is non-orthogonal (`cpu::minimizer::`,
  `Minimize` preset, `Optimizer` kind, `CG`/`LBFGS`/`FIRE`) — needs a
  /mol:discuss decision, overlaps namespace alignment.
- RKS/PBE forces: ~3e-3 eV/Å XC-grid noise on Ar₂ at the default grid, and
  slow (28 s per def2-SVP force vs 0.25 s RHF).
- GPW historical runtime/convergence/repeatability reports need a current
  run. Surviving method cases and sections are listed in
  [es-architecture-debt.md](es-architecture-debt.md#numerical-recheck-targets);
  deleted functional-swap and self-pinned correlated cases are not current
  failing tests. Keep the ≤60 s per-section policy when selecting smaller cases.
- DFTB: no open-shell Δq ≠ 0 force gate; 3ob `with_stress` still uses
  multi-SCC strain FD; MioSketch O/H rows give backwards water charges.
- PW collinear branches: convergence measured on the post-mix density change
  (β × residual) and plain linear mixing; most PW absolute-energy pins are
  self-freezes (replace with invariants or external pins).
- Sweep owed: any regression that can silently degrade its own physics on a
  missing input (the old `pw-smearing-al` toy-Hamiltonian fallback pattern).

## Build-shape conclusions

- **molpy is derived** (`ATV_BUILD_ANALYSIS` / `IO` / `PYTHON` / `GPW` / `HF` /
  `SEMIEMPIRICAL` force `ATV_BUILD_MOLPY=ON`). Every bridge surface Atomiverse
  uses is required of the molpy checkout; there are no capability probes.
- **molpy source:** `ATV_MOLPY_PATH` selects the local MolCrafts/molpy
  repository (default `../molpy`). Corrosion imports `molcxx/Cargo.toml`;
  Python builds `molpy/`. Both consume the same checkout; no source revision
  is pinned.
- **Unit-suite determinism:** bit-for-bit comparisons pin `OMP_NUM_THREADS=1`
  via test ENVIRONMENT; FD-vs-analytic tests choose `h` so round-off does not
  drown the tolerance.
- Never put a SIMD reduction behind a loop-invariant branch: GCC 14.3 at
  `-O3 -march=native` reassociated the `PairLJCut` virial chains (37 % short)
  while energy/forces stayed exact. Guard: `tests/cpu/potentials/test_pair_virial.cpp`.
