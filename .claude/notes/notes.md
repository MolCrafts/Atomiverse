# Notes — current understanding

Conclusions only. Binding rules live in CLAUDE.md; module inventory in
[architecture.md](architecture.md); loop state in the sibling topic notes.
History is git's job — superseded designs are deleted, not archived here.

## Open engineering leads

- **ES optimization order (binding in CLAUDE.md).** Wave state:
  [es-optimization-order.md](es-optimization-order.md). Structure + OT +
  windows + collint + STVnl polish landed; per-wave walls there are
  as-claimed-at-the-time and superseded —
  [gpw-perf-baseline.md](gpw-perf-baseline.md) is the single wall authority
  (MG MAIN re-measured **1.6 s / 2.33×** CP2K quiet 0.688 s). CUDA ES stays
  CPU-first (only the sanctioned next waves in
  [cuda-rhf-device-only.md](cuda-rhf-device-only.md)); classical MD CUDA
  independent.
- **e2e-in-tests migration (wave 1 + wave 2 done on `dev`).** Wave-1 deleted
  **78** heavy full-method sources; wave-2 removed residual multi-iter and
  emptied `ATV_TEST_HEAVY_REGEX` to `^$`. Ledger:
  [e2e-in-tests-campaign.md](../../docs/developer/e2e-in-tests-campaign.md).
  Optional wave-3 only if unique pin gaps appear after inventory.
- **Open-shell cluster GPW mixer.** GPW analytic forces are validated
  (analytic-vs-FD ≤ 3.6e-6 Ha/Bohr across Si₂ RKS/UKS setups), but Si₂-triplet
  GTH in a cluster box is SCF-ill-conditioned for both ATV and CP2K in
  complementary settings windows. ATV has linear + DIIS + OT; CP2K wins with
  Broyden. A better mixer for open-shell cluster GPW is the actionable gap,
  and it also blocks the `gpw-fd-phonon-si` e2e (cost/SCF exceeds the 60 s
  tier everywhere CP2K converges).
- **GPW MG MAIN residual.** OT convergence is closed: `p0-ot-cp2k-match` +
  `p0-ot-match-residual` match CP2K at **10 iterations** (OT wall ~2.99 s vs
  CP2K 2.50 s ≈ **1.20×**, SCF body ≤1.0× —
  [gpw-perf-queue.md](../../docs/developer/gpw-perf-queue.md)). The biggest
  remaining CPU-GPW lever vs CP2K is now MG MAIN **2.33×** (coll/int
  ~2.0×/2.8×); kernel-level SIMD, structure, and sphere/contract cuts are
  exhausted — further cut needs non-pin-safe windows or a task-list rewrite
  beyond the current loop.
- **Component clone ceiling.** Components cannot be cloned, so one inner
  Driver body over N replicas (N L-BFGS histories) is not expressible yet.
  Undecided; revisit when a real N-replica inner-loop user appears.

## Loop state pointers

| Loop | State notes |
| --- | --- |
| cp2k-method-loop | queue `docs/developer/cp2k-method-queue.md`; pin law [oracle-pin-provenance.md](oracle-pin-provenance.md) (**strict**: real CP2K run + matched algorithm + perf; no GTH↔NC-UPF windows) |
| gpw-perf-loop | queue [gpw-perf-queue.md](../../docs/developer/gpw-perf-queue.md); arch [gpw-perf-architecture.md](gpw-perf-architecture.md); **always wipe builds**; first structural cut `p1-gpw-workspace` → `p1-ks-grid-build` ([es-optimization-order.md](es-optimization-order.md)) |
| tb-method-loop | [tb-blocked-capabilities.md](tb-blocked-capabilities.md), [dftb-plus-u.md](dftb-plus-u.md), [dftb-vs-dftbplus-cleanup.md](dftb-vs-dftbplus-cleanup.md), [gfn2-xtb-chon.md](gfn2-xtb-chon.md) |
| metalwalls-method-loop | [metalwalls-capability-map.md](metalwalls-capability-map.md), [oracle-pin-provenance.md](oracle-pin-provenance.md) |
| cuda-rhf-reshape | [cuda-rhf-device-only.md](cuda-rhf-device-only.md) (waves 0–10 done; free-loop `SCFSolver` **folded** via **arch-campaign** `arch-cuda-scf-solver-fold` → private `RhfScfCycle`; nested `cuda::Driver<scf::Phase>` residual); CUDA GPW public shape **frozen** to CPU `GpwDft` on `dev` — [cuda-gpw-public-shape.md](cuda-gpw-public-shape.md); full product staffable under that contract (not auto-landed) |
| **arch-campaign** | queue [arch-campaign-queue.md](../../docs/developer/arch-campaign-queue.md); loop [arch-campaign-loop.md](../../docs/developer/arch-campaign-loop.md); spine **GpwDft/PW Driver nodes → residual paths → CUDA SCF fold → shape freeze → MG structural**; landed: `arch-gpw-driver-node`, `arch-gpw-driver-paths` (multi-path DIIS; OT free-loop residual), `arch-pw-driver-node`, `arch-cuda-scf-solver-fold`, `arch-e2e-tests-wave{1,2}`, `arch-cuda-gpw-public-shape-freeze` (merge `0147fe02`); harness wave-8 **pending=0** unheld / **in_progress=2** (`arch-pw-driver-paths` @ `c20bb04f`, `arch-mg-main-structural` @ `9ee9f49e`; rebase onto `0147fe02` before gate) / **gaps_remaining=3** / **new_slugs=0**; unmerged tip ≠ done |
| CUDA MD perf | [cuda-openmm-md-perf.md](cuda-openmm-md-perf.md) (A/B methodology + precision mode map vs LAMMPS Kokkos); **not** blocked by ES CPU priority |
| **claim-completion** | queue [claim-completion-queue.md](../../docs/developer/claim-completion-queue.md) — CUDA/docs/Python completion for claimed domains (operator 2026-08-14); torch backend **removed**; molrs submodule at latest (`95dde84`, 0.13.1); python layer migrated to molrs 0.13.1 schema (`atomic_number` uint + plain-value meta), suite 176 green |

## Build-shape conclusions

- **molrs is a derived dependency.** Products control molrs
  (`ATV_BUILD_ANALYSIS` / `IO` / `PYTHON` / `GPW` / `HF` / `SEMIEMPIRICAL`
  force `ATV_BUILD_MOLRS=ON` in `cmake/AtvProductModules.cmake`); never a peer
  option, never `FATAL_ERROR "X requires MOLRS"`. Feature slices that need a
  specific molrs pin (e.g. WHAM ⇒ `ATV_MOLRS_HAS_WHAM`) gate their sources
  **and** their regressions on the same condition.
- **Toolchain gotchas** (login node): `OPENBLAS_ROOT="$OPENBLASDIR"`,
  `RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu`, mirror `LIBRARY_PATH`
  into `LD_LIBRARY_PATH` before running anything. GPU node: default aarch64
  Rust is correct; CUDA only builds there.
- **Unit-suite determinism:** bit-for-bit comparisons pin
  `OMP_NUM_THREADS=1` via test ENVIRONMENT (`test_cpu_qm_engine`); FD-vs-
  analytic derivative tests pick `h` large enough that round-off does not
  drown the tolerance (`test_dftb3_scc` uses `h=1e-4`).
