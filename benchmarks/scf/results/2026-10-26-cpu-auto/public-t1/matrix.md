# Scientific benchmark

Commit: c9862a288828e6a1f43f90a2fa1656347c802aed; dirty=True; campaign=70a70c8f61484e808a2645cbe693b7f0.

Times are warm end-to-end medians [min, max], seconds. Speedup = reference / ATV. Missing results are explicit. Local/unqualified evidence does not promote roadmap marks.

| Method / model / task | Systems | atv.cpu | atv.cuda | pyscf.cpu | gpu4pyscf.cuda | Notes: hardware / environment |
| --- | --- | --- | --- | --- | --- | --- |
| RHF STO-3G energy (hcore) | 16 parallel waters at 4 Angstrom intervals, 112 STO-3G AOs | water_chain16: pass; 0.987649 s [0.914192, 1.00091]; energy_hartree Δmax=1.59e-12 (limit 1e-06) Ha; 1.2× vs pyscf.cpu | water_chain16: binding_missing; BindingMissing: Public atomiverse.scf.CudaRHF is absent from the loaded binary | water_chain16: pass; 1.18167 s [1.0904, 1.27477]; energy_hartree Δmax=0 (limit 1e-06) Ha; 1× vs pyscf.cpu | water_chain16: dependency_missing; No module named 'gpu4pyscf' | H1; FP64; reference=pyscf.cpu |

H1: CPU=AMD EPYC 9755 128-Core Processor; physical cores=256; logical CPUs=256; requested OMP/BLAS threads=1; GPU=unknown (GPU query unavailable); host RAM=811083767808 bytes; OS=Linux-5.14.0-687.59.1.el9_8.x86_64-x86_64-with-glibc2.34; Python=3.13.13; 5 warm samples, 1 warmup(s)/sample; 2026-10-26T17:46:28.787172+00:00. Actual engine/compiler/threadpool versions and memory budgets are in the observations.

Summary: `{"complete": true, "failed": 0, "historical_status": "unbaselined", "incomplete": 0, "optional_unavailable": 2, "passed": 2, "performance_regressions": 0, "required": 2}`.
