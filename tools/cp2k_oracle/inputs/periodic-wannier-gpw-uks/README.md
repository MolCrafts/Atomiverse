# GPW Wannier evidence

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

`si_uks_wannier_bct.inp` is the existing CP2K input supporting the external
energy pin in `regressions/gpw/wannier/REFERENCE.md`. Native outputs here are
validation evidence and do not generate oracle pins.

The current build12 (2026-10-25) case output is `native-validation.txt`, with
SHA-256, command, settings and boundaries in `native-validation.json`.
At OpenMP/OpenBLAS 8/8 on Apple M4, RKS and UKS whole sections take
**7.889850 / 9.886750 s** and pass all existing numerical and 60 s gates.
Shared AO-link construction takes 4.574 / 3.971 s for all 48 links; channel
localization takes 27 ms (RKS) and 25/21 ms (UKS).

The common batched AO operator retains the exact Fourier integral and all
links. Exactly equal short wavevectors reuse image-local primitive integrals
before the distinct ket Bloch phases are accumulated. Per-matrix image and
primitive sum order is preserved, with independent group/row parallelism.
Both spin contractions borrow the same compute-local matrices by const
reference; there is no persistent global cache.

RKS maximum midpoint distance is 0.02245 Å and spread ratio 0.1068. UKS has
0.01726 Å maximum distance per spin, spread ratio 0.1041, and printed energy
−15.76993360 Ha (error 2.306724e-5 Ha against the unchanged CP2K pin).
The MV convergence flag remains false and produces the existing warning.
The acceptance scope is SCF convergence, center/spread/energy checks and
whole-section runtime; these results do not establish a stationary MV optimum.

Reproduce with a CPU HF/PW/GPW build:

```sh
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 ATV_GPW_PROF=1 /tmp/atv-xc-build/regressions/atv_regressions --case gpw/wannier
```

The actual invocation ran from `/tmp/atv-xc-build/regressions`, RKS then UKS
in one process. Whole-section boundaries include construction and assertions.

Historical diagnostics are preserved unchanged in `native-build10.txt/.json`
and `native-build11.txt/.json`: UKS whole-section 103.975982 and 84.018104 s
failed the retained 60 s bound before batched Fourier reuse. No numerical
or physical settings were relaxed. Primitive Fourier math is covered by the
actual build13 unit archive at `../gpw-berry-phase/native-math-validation.txt`
and its JSON metadata: 17 GpwOneElectron and three GpwFourierLinks fixtures
pass, as do all 1771 registered units (46.66 s). Build13 only adds test
registration; native mathematical code is unchanged from the build12
regression measurements. This archive
does not supply a new external timing comparison or replace any oracle pin.
