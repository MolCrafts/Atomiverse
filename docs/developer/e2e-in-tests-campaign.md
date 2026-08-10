# e2e-in-tests campaign (iron law)

`tests/` holds **unit tests only**. Full method solves, SCF convergence loops,
BOMD/CPMD campaigns, and oracle pins live in `regressions/` (or benchmarks),
never under ctest’s default gate.

## Wave 1 (2026-08-10) — delete duplicates

**Action:** remove **78** heavy-gated sources that were full-method / multi-step
campaigns already covered by `regressions/**` (or illegal under the iron law
even when short).

| Domain | Examples deleted | Coverage now |
| --- | --- | --- |
| GPW | solver/method, forces/stress/uks/ot, BOMD/CPMD/XL, dispersion, post-SCF | `regressions/gpw/*` |
| PW | method/si/evaluator, forces/stress/spin/BOMD, DFPT, bands, wannier | `regressions/pw/*` |
| SCF / electronic | solver family, post-HF e2e, CPMD/XL, embedding/PCM/CPHF | `regressions/scf/*`, `regressions/*cpmd*` |
| SE | AM1 scaffold/thermo, DFTB0/2/3 SCC campaigns | `regressions/semiempirical/*` |
| MD glue | `md_no_alloc`, cpmd conserved e2e | dynamics/regs |

**Also:** slimmed `ATV_TEST_HEAVY_REGEX` — **dropped kernel-unit tokens**
(`gpw_poisson`, `gpw_hartree`, `gpw_grid_fft`, `gpw_xc_grid_gga`,
`gpw_mt_poisson`, `gpw_thread`) so true kernel unit binaries rejoin the default
(pre-push) gate.

## Wave 2 (2026-08-10) — residual multi-iter + empty HEAVY

**Action:** audit + delete residual multi-iter SCF / MPI / multiscale under
`tests/` that wave-1 left (not heavy-gated, or false-heavy unit names). Slim
mixed files to bind/build/throw/math only. Empty `ATV_TEST_HEAVY_REGEX` to `^$`.

| Class | Wave-2 action | Coverage |
| --- | --- | --- |
| TDA / NAC / ALDA e2e units | **deleted** | `regressions/scf/h2_{tda_force,analytic_tda_force,tda_nac,nac_vectors,full_grid_alda,alda_vs_cis,rks_alda_tda}` |
| Analytic Hessian multi-SCF | **deleted** | `regressions/scf/{h2o_rhf_freq,uhf-hessian-o2,…}` |
| delta-SCF / Koopmans / lindep / level_shift smokes | **deleted** | `regressions/scf/{n2_ionization,delta-scf-n2,h2o_rhf_sto3g,…}` |
| MPI PW/GPW kpoint multi-iter | **deleted** | `regressions/pw/mpi-pw-scaling-4rank`, `regressions/gpw/mpi-gpw-scaling-4rank` |
| GPW vol-forces product SCF (max_iter=60) | **slimmed** (geometry-only FD duals) | `regressions/gpw/{gpw-hirshfeld-forces,gpw-hirshfeld-kpoint-forces,…}` |
| `test_qm_engine` multi-iter evaluate | **slimmed** (protocol/bind only) | `regressions/multiscale/*`, `architecture/lj_rhf_coupled` |
| multiscale RHF.launch multi-iter | **slimmed** (build/throw/math + link-count) | `regressions/multiscale/*` |
| `test_soscf` micro RHF.run pin | **slimmed** (stepper units only) | `regressions/scf/soscf_hard_case` |
| `test_method_driver_wire` nested run/MD | **slimmed** (bind/build only) | `regressions/architecture/lj_rhf_coupled` |
| `test_source_cobinding` RHF co-bind | **slimmed** (stubs only) | `regressions/architecture/lj_rhf_coupled` |
| `test_spin_resolution` UHF converge | **slimmed** (math + throw) | `regressions/scf/*uhf*` |
| `ATV_TEST_HEAVY_REGEX` false hits | **emptied** (`^$`) | units rejoin default gate |
| Harness twin | `reg_arch-e2e-tests-wave2` | architecture |

## Residual (not wave-2)

| Class | Policy |
| --- | --- |
| Unique pin only discovered after green inventory | Migrate to `regressions/<domain>/<slug>/` with REFERENCE.md |
| New unit that collides with a future heavy token | Prefer rename over reintroducing a fat HEAVY_REGEX |
| Classical MD multi-step skin/neighbor unit loops | Out of ES iron-law wave; do not re-open fat HEAVY_REGEX for them |

## Do not reintroduce

- Full `GpwDft::run` / `PW::run` / `RHF::run` SCF loops under `tests/`
- Multi-step `MD` / `Minimize` campaigns driven by electronic methods under `tests/`
- Oracle/live third-party calls under ctest
- Fat unanchored `ATV_TEST_HEAVY_REGEX` tokens that false-hit unit names

## Next waves

- Wave 3: optional migrate of *unique* coverage discovered only after green
  regression inventory (if any gap remains after wave-2 inventory)
