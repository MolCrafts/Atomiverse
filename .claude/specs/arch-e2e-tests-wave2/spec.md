# arch-e2e-tests-wave2

## Intent

Residual **e2e-in-tests** iron-law cleanup after wave-1: multi-iter SCF solves,
MPI PW/GPW campaigns, GPW product SCF under unit names, multiscale/qm_engine
launch campaigns, and false-heavy `ATV_TEST_HEAVY_REGEX` hits still living
under `tests/` after the 78-file wave-1 delete.

## Shape change (lever)

Same pattern as SCF unit discipline already used by `test_scf_stages` /
`test_orbital_hessian` / `test_tdhf_pack`: **no full method `run` / multi-iter
`evaluate`/`launch` loop under `tests/`**. Delete pure multi-iter sources with
regression twins; slim mixed files to bind/build/throw/math only; empty HEAVY
name regex so true units rejoin the default gate. `MeanFieldMethod` /
`cpu/scf/stages/` remain the production Driver pattern — this slug does not
reshape solvers.

## In scope

- Delete residual multi-iter unit sources (TDA/NAC/ALDA/hessian/delta-SCF smoke,
  MPI kpoint, converge-only level_shift/lindep/koopmans, orbital_info minimize).
- Slim `test_soscf`, `test_spin_resolution`, `test_method_driver_wire`,
  `test_source_cobinding`, ECP x2c gate.
- Slim `test_cpu_gpw_vol_forces` (geometry FD only), `test_qm_engine`,
  multiscale link/electrostatic/subsystem/polarizable (no multi-iter launch).
- Empty `ATV_TEST_HEAVY_REGEX` (remove false hits on unit names).
- Harness regression `reg_arch-e2e-tests-wave2`.

## Out of scope

- New electronic methods; pin-softening; classical MD CUDA; GPW/PW Driver paths.
- Re-seeding every deleted smoke as a new regression (twins already exist).

## Public API

Unchanged: `RHF`/`RKS`/`GpwDft`/`PW` `run`/`forces` stay as on `dev`.
