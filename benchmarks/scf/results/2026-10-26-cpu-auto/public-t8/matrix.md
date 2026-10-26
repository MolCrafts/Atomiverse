# Scientific benchmark

Commit: c9862a288828e6a1f43f90a2fa1656347c802aed; dirty=True; campaign=46e7f59da9404eb882ab906f941e9df8.

Times are warm end-to-end medians [min, max], seconds. Speedup = reference / ATV. Missing results are explicit. Local/unqualified evidence does not promote roadmap marks.

| Method / model / task | Systems | atv.cpu | atv.cuda | pyscf.cpu | gpu4pyscf.cuda | Notes: hardware / environment |
| --- | --- | --- | --- | --- | --- | --- |
| RHF STO-3G energy (hcore) | 16 parallel waters at 4 Angstrom intervals, 112 STO-3G AOs | water_chain16: pass; 0.57688 s [0.537226, 0.908381]; energy_hartree Δmax=1.59e-12 (limit 1e-06) Ha; ratio ineligible: actual engine thread resources differ from the declared profile | water_chain16: binding_missing; BindingMissing: Public atomiverse.scf.CudaRHF is absent from the loaded binary | water_chain16: pass; 1.91791 s [1.05976, 2.51366]; energy_hartree Δmax=1.82e-12 (limit 1e-06) Ha; ratio ineligible: actual engine thread resources differ from the declared profile | water_chain16: dependency_missing; No module named 'gpu4pyscf' | H1; FP64; reference=pyscf.cpu |

H1: CPU=AMD EPYC 9755 128-Core Processor; physical cores=256; logical CPUs=256; requested OMP/BLAS threads=8; GPU=unknown (GPU query unavailable); host RAM=811083767808 bytes; OS=Linux-5.14.0-687.59.1.el9_8.x86_64-x86_64-with-glibc2.34; Python=3.13.13; 5 warm samples, 1 warmup(s)/sample; 2026-10-26T17:47:04.091401+00:00. Actual engine/compiler/threadpool versions and memory budgets are in the observations.

Summary: `{"complete": true, "failed": 0, "historical_status": "unbaselined", "incomplete": 0, "optional_unavailable": 2, "passed": 2, "performance_regressions": 0, "required": 2}`.
