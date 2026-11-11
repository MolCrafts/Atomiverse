# Test and regression consolidation (live)

Goal: one executable for unit tests, one for regressions, one regression per
method. Delete when every phase closes.

## Why

- ~900 executables (525 regression + 334 test targets), each TU ~5 s
  (278k preprocessed lines; xtensor parse + expression instantiation), so a
  header change costs ~40 min at -j2. Links are cheap (0.3 s).
- 414 regression cases because roadmap rows split one method by quantity
  (energy / forces / stress), variant (spin / k-points / metallic) and backend
  (34 `cuda-*` siblings), plus implementation probes (`cpu-simd-*`).
- xtensor stays the user-facing array type; the cost is re-parsing it per tiny
  TU, which unity build + PCH remove without touching the API.

## Target

- **`atv_tests`** — one gtest executable (CPU sources; CUDA `.cu` sources
  when `ATV_BUILD_CUDA`), sources gated by module flags, tests discovered
  per gtest case. MPI tests keep their own `main` in `atv_tests_mpi`, run at
  each rank count.
- **`atv_regressions`** — one executable; each case registers with
  `ATV_REGRESSION("domain/method")` and returns non-zero printing
  `got / want / tol`. ctest still lists one test per case
  (`atv_regressions --case domain/method [--device=…]`, `TIMEOUT 60`,
  per-case `OMP_NUM_THREADS`, `mpiexec -n R` for ranked cases), so failure
  location and the time budget stay per case. Case dir comes from `__FILE__`;
  `ATV_REG_DATA_DIR` is global. Each case body lives in its own namespace so
  unity batches cannot collide.
- **One case per method**: a case checks every quantity the method claims
  (energy, forces, stress, variants) on every backend it claims; a roadmap row
  is a method, quantities and backends are its columns/notes.
- Build: unity build + precompiled headers on both executables; Ninja +
  ccache recommended in the build docs.

## Phases

1. **Mechanics** — done (414 cases → 90 method cases, 377 CPU sections;
   tutorial method pages 273 → 65): `atv_regressions` registry + CMake (unity batches of
   16), every case's `main` converted to a registered `run`; unit tests in
   `atv_tests` / `atv_tests_cuda` / `atv_tests_mpi` (PCH), with dedicated
   executables only for tests needing their own link line; Ninja + ccache.
2. **One per method**, per domain (scf and gpw first). Each merged case:
   - keeps every external pin it absorbs and shares one SCF across the
     quantities of one system (energy, forces, stress);
   - folds `cuda-*` siblings in as `DEVICES both` (`cpu-twin` gates inside);
   - is named after the method, not history (`*-gate`, `*-product-*`,
     `*-foothold`, `arch-*`, quantity suffixes go);
   - drops implementation probes (`cpu-simd-*`, `arch-*`);
   - stays ≤ 60 s; census over-budget cases (21, mostly GPW) and the three
     red ones (`gpw-fd-phonon-si`, `const-pot-il-edl-box`, `gpw-uks-stress`)
     are fixed or shrunk here; `gpw-dftu-kpoint-forces` NiO UKS converges to
     thread-dependent states (|ΔE| ~1e-4 Ha between runs) and needs a
     deterministic setup;
   - rewrites its `REFERENCE.md` into the template (oracle named) and its
     roadmap row (row names the case, case names the row).
   `internal/` includes in 27 GPW cases close with the ES public-shape phase.
   Known section defects: `dynamics/xl-bomd/xl-bomd-rhf-water.cpp:99`
   `scf_tol` is never passed (runs at 1e-9, record says 1e-6);
   ATV-vs-ATV gates in `gpw/rks/gpw-kpoint-stress.cpp:131-139` and
   `md/constant-potential/dipole-electrode-joint`; `gfn2-xtb-forces` and
   `gfn2-aes-fock` run identical pins; several `REFERENCE.md` blocks cite
   deleted unit tests (dftb2/3, td-dftb, car-parrinello, xl-bomd, gpw
   d3/d4/mbd/xdm/uks). Wall per case today (CPU, -j10 shared): dfpt 29 min,
   constant-potential 17 min, pw/bands-dos 14 min — the SCF-sharing targets.
3. **Harness** — rewrite docs/developer/project-contract.md iron law 1a and the build section.
