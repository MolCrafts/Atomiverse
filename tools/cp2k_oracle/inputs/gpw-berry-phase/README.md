# Native validation: exact GPW Berry Fourier overlaps

This directory archives native correctness/runtime evidence, not new CP2K
oracle pins. The build12 (2026-10-25) CTest case output is
`native-validation.txt`; `native-validation.json` records SHA-256, command,
thread width, complete-section timers and per-axis profiles. The literature
symmetry/ASR pins remain in `regressions/gpw/berry-phase/REFERENCE.md`.
Existing external inputs are `../si_periodic_berry_gpw.inp` and
`../si_periodic_berry_gpw_uks.inp`; no fresh CP2K result or timing is claimed.

The canonical batched operator retains the exact Cartesian Gaussian Fourier
integral, groups only exactly equal short wavevectors, and shares image-local
integrals before each ket's Bloch phase is applied. Matrix image/primitive
sum order is preserved. Group/row OpenMP uses thread-local Hermite scratch.
UKS borrows common AO operators for its two occupied manifolds. Each direction
independently uses the ordinary GPW SCF path with unchanged settings/gates.

Reproduce with a CPU HF/PW/GPW-enabled build of `atv_regressions`:

```sh
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 ATV_GPW_PROF=1 /tmp/atv-xc-build/regressions/atv_regressions --case gpw/berry-phase
```

The actual invocation ran from `/tmp/atv-xc-build/regressions`. On Apple M4
(10 physical/logical cores), RKS and UKS ran sequentially in one process.
Whole sections took **10.070322 / 15.201491 s**, including construction and
assertions; both pass the enforced 60 s bound. Ideal total polarization is
zero at printed precision. Displaced Δp_z is +0.000069 e in both sections,
with printed Born charges +0.0034/+0.0035 e. These observations do not become
expected values.

`native-math-validation.txt` and its JSON metadata preserve actual build13
result lines: all 17 GpwOneElectron tests (seven primitive Fourier tests) and
all three GpwFourierLinks tests pass. The latter cover absolute analytic image
sums/equal and distinct wavevectors, split/batch identity, mixed-angular
translation/conjugation covariance and worker-exception propagation. All
**1771 registered units pass in 46.66 s**. Build13 adds the missing test
registration; native mathematical code is unchanged from the build12
regression measurements. `native-math-build12.txt/.json` retain the preceding
actual unit record without retroactively claiming those batch tests ran.

Reproduce the focused registered tests after building `atv_tests`:

```sh
ctest --test-dir /tmp/atv-xc-build -R '^(GpwOneElectron|GpwFourierLinks)\.' --output-on-failure
```

`native-build10.txt/.json` and `native-math-build10.txt` retain the preceding
actual measurements. Historical CP2K ratios in the regression reference were
measured with an earlier kernel and another host. The current archive proves
native numerical and whole-section budget gates, not a refreshed fair
ATV/CP2K performance ratio.
