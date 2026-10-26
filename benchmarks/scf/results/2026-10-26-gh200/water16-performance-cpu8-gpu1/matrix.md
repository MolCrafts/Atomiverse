# Scientific benchmark

Commit: 54d0b9c9d91bffc680d14f4f687578cf83cf70af; dirty=True; campaign=df912087ac26491584bac28145d38ec3.

Times are warm end-to-end medians [min, max], seconds. Speedup = reference / ATV. Missing results are explicit. Local/unqualified evidence does not promote roadmap marks.

| Method / model / task | Systems | atv.cpu.direct | atv.cpu.stored | atv.cuda | pyscf.cpu | gpu4pyscf.cuda | Notes: hardware / environment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RHF STO-3G energy (hcore) | 16 parallel waters at 4 Angstrom intervals, 112 STO-3G AOs | water_chain16: pass; 1.46419 s [1.39271, 1.46876]; energy_hartree Δmax=1.31e-08 (limit 1e-06) Ha; ratio ineligible: actual engine thread resources differ from the declared profile | water_chain16: pass; 3.29714 s [3.24003, 3.30414]; energy_hartree Δmax=3.21e-11 (limit 1e-06) Ha; ratio ineligible: actual engine thread resources differ from the declared profile | water_chain16: pass; 0.570016 s [0.561882, 0.5705]; energy_hartree Δmax=4.57e-11 (limit 1e-06) Ha; ratio ineligible: actual engine thread resources differ from the declared profile | water_chain16: pass; 1.85496 s [1.70588, 1.92007]; energy_hartree Δmax=1.14e-12 (limit 1e-06) Ha; ratio ineligible: actual engine thread resources differ from the declared profile | water_chain16: dependency_missing; No module named 'gpu4pyscf' | H1; FP64; reference=pyscf.cpu |

H1: CPU=aarch64; physical cores=288; logical CPUs=288; requested OMP/BLAS threads=8; GPU=['NVIDIA GH200 120GB, GPU-6d2ec6f2-f9df-c924-6ce3-cc43d7f7db1c, 97871, 580.178.04']; host RAM=921772163072 bytes; OS=Linux-5.14.0-687.59.1.el9_8.aarch64+64k-aarch64-with-glibc2.34; Python=3.13.17; 5 warm samples, 1 warmup(s)/sample; 2026-10-26T13:01:15.927079+00:00. Actual engine/compiler/threadpool versions and memory budgets are in the observations.

Summary: `{"complete": true, "failed": 0, "historical_status": "unbaselined", "incomplete": 0, "optional_unavailable": 1, "passed": 3, "performance_regressions": 0, "required": 3}`.
