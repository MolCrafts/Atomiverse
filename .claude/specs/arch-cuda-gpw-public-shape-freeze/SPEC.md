# arch-cuda-gpw-public-shape-freeze

## Intent

Document + enforce: the full CUDA GPW method product **equals** the CPU
`GpwDft` public shape after multi-path DIIS Driver (`arch-gpw-driver-paths`
merged). Stage footholds under `cuda/gpw/internal/` compose the product; they
are not a second public API. CPU OT free-loop residual is documented and does
**not** block this freeze.

## Lever

**Architecture (ES order step 1 → CUDA product unlock):** freeze the public
contract before `cuda-gpw-full-product` is staffed. Pattern research:
`MeanFieldMethod` + `cpu/scf/stages/` (Driver stages, Frame `run`/`forces`) and
landed `GpwDft : cpu::Driver<gpw::Phase>` multi-path DIIS. CUDA twin for
classical shape is `cuda::scf::RHF` / `cuda::pw::PW` — members +
`build`/`run`/`last_result` + forces free functions; no `*Config`, no
`make_*` / `build_*` factories.

## Shape change (named)

| Surface | Before freeze | After freeze (binding) |
| --- | --- | --- |
| CUDA GPW public method | **none** (footholds only) | **must** mirror CPU `GpwDft` when product lands |
| Stage units (`KsChainDevice`, MG, …) | internal, green | stay `internal/`; compose only — never public method |
| CPU OT free loops | residual | residual; does **not** block freeze or DIIS product path |
| `cuda-gpw-full-product` | blocked “needs freeze” | staffable **after this tip merges** under freeze contract |

## Public surface (frozen target for CUDA product)

CPU oracle of record (`include/atomiverse/cpu/gpw/gpw_solver.h` + forces/stress
helpers + `GPWEvaluator` for Frame workflows):

- Concrete type `GpwDft` (not a freestanding `SCFSolver` bag)
- Parameters as **public members** + default/move ctor (no `*Config`)
- `build(atoms)` fail-fast validation
- `run(atoms, …)` → `GpwResult&`; `last_result()`
- Forces / stress as free functions (`gpw_forces_analytic` / `gpw_stress` family)
  or via `GPWEvaluator` Frame I/O — not a second method class
- Post-SCF (`mp2` / `direct_rpa` / `g0w0`) optional product depth; not required
  for first CUDA method slice
- Forbidden on the public product: `*Config`, `make_*`/`build_*` factories,
  public Context, hand-driven Component lifecycle, promoting foothold types to
  user API

CUDA product first slice may implement a **subset of capability** (e.g. Γ RKS
DIIS only) but **must not invent a different public shape**.

## Acceptance

- [x] Freeze note: `.claude/notes/cuda-gpw-public-shape.md`
- [x] Spec-lite at `.claude/specs/arch-cuda-gpw-public-shape-freeze/`
- [x] Harness reg `reg_arch-cuda-gpw-public-shape-freeze` (shape-only, no SCF)
- [x] Queue unlock wording for `cuda-gpw-full-product` (freeze green ⇒ staffable)
- [x] Notes / harness rewrite current understanding only
- [x] Scoped build+test green; wipe builds
- [x] `acceptance_met=true`

## Out

Implementing `cuda::gpw::GpwDft`; OT stage migration; MG MAIN structural;
hybrid/GAPW/PAW/meta-GGA; pin softening; classical MD CUDA.
