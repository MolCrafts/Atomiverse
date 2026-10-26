# Scientific benchmark

Commit: 54d0b9c9d91bffc680d14f4f687578cf83cf70af; dirty=True; campaign=28302a8025644ed3865c8691052df55f.

Times are warm end-to-end medians [min, max], seconds. Speedup = reference / ATV. Missing results are explicit. Local/unqualified evidence does not promote roadmap marks.

| Method / model / task | Systems | atv.cpu.direct | atv.cpu.stored | atv.cuda | pyscf.cpu | gpu4pyscf.cuda | Notes: hardware / environment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RHF STO-3G energy (hcore) | 16 parallel waters at 4 Angstrom intervals, 112 STO-3G AOs | water_chain16: pass; 7.75752 s [6.88613, 8.10655]; energy_hartree Δmax=1.31e-08 (limit 1e-06) Ha; ratio ineligible: native binary fingerprint missing, invalid or changed between samples | water_chain16: pass; 21.8271 s [17.3918, 26.6995]; energy_hartree Δmax=2.98e-11 (limit 1e-06) Ha; ratio ineligible: native binary fingerprint missing, invalid or changed between samples | water_chain16: binding_missing; BindingMissing: Public atomiverse.scf.CudaRHF is absent from the loaded binary | water_chain16: pass; 1.94159 s [1.49639, 7.03998]; energy_hartree Δmax=0 (limit 1e-06) Ha; ratio ineligible: native binary fingerprint missing, invalid or changed between samples | water_chain16: dependency_missing; No module named 'gpu4pyscf' | H1; FP64; reference=pyscf.cpu |

H1: CPU=Apple M4; physical cores=10; logical CPUs=10; requested OMP/BLAS threads=1; GPU=unknown (GPU query unavailable); host RAM=25769803776 bytes; OS=macOS-27.0.1-arm64-arm-64bit; Python=3.12.12; 5 warm samples, 1 warmup(s)/sample; 2026-10-26T12:38:47.029957+00:00. Actual engine/compiler/threadpool versions and memory budgets are in the observations.

Summary: `{"complete": true, "failed": 0, "historical_status": "unbaselined", "incomplete": 0, "optional_unavailable": 2, "passed": 2, "performance_regressions": 0, "required": 2}`.
