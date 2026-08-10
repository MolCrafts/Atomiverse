# Acceptance — arch-e2e-tests-wave2

| # | Criterion | Met |
| --- | --- | --- |
| 1 | Pure multi-iter residual sources listed in wave-2 audit are **gone** from `tests/` | yes |
| 2 | Slimmed units (soscf, spin_resolution, method_driver_wire, source_cobinding, ecp gate, vol_forces, qm_engine, multiscale) perform **no multi-iter SCF converge pin** | yes |
| 3 | MPI PW/GPW kpoint unit sources removed; coverage = `reg_mpi-*-scaling-4rank` | yes |
| 4 | `ATV_TEST_HEAVY_REGEX` empty (`^$`) — no false-heavy skip of true units | yes |
| 5 | Scoped: `ctest` for touched units green; `run_reg_arch-e2e-tests-wave2` green | yes |
| 6 | Public `run`/`forces` API untouched | yes |

acceptance_met=true
