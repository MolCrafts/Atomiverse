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
`gpw_mt_poisson`, `gpw_thread`) so true unit binaries rejoin the default
(pre-push) gate.

## Residual (not wave-1)

| Class | Policy |
| --- | --- |
| Remaining heavy name hits | Safety-net regex only; should be empty/near-empty after wave 1 |
| Unique e2e still only in deleted files | **none intentional** — if a pin was only in a deleted test, re-add as `regressions/<domain>/<slug>/` with REFERENCE.md |
| False-heavy renames | Prefer rename over regex gymnastics when a new unit collides with residual tokens |

## Do not reintroduce

- Full `GpwDft::run` / `PW::run` SCF loops under `tests/`
- Multi-step `MD` / `Minimize` campaigns under `tests/`
- Oracle/live third-party calls under ctest

## Next waves

- Wave 2 (`arch-e2e-tests-wave2`, pending): residual multi-iter under `tests/`
  after wave-1 — GapCensus wave-2 concrete hits include
  `test_cpu_gpw_vol_forces` (max_iter=60), `test_soscf` / `test_full_grid_alda`,
  multiscale + `test_qm_engine` (max_iter 40–100), MPI `test_gpw_kpoint` /
  `test_pw_kpoint` (max_iter 20–30), plus fat residual `ATV_TEST_HEAVY_REGEX`
- Wave 3: optional migrate of *unique* coverage discovered only after green
  regression inventory
