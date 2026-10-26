# Scientific benchmark

Commit: 54d0b9c9d91bffc680d14f4f687578cf83cf70af; dirty=True; campaign=d6bc653c63334780be4c929f6eaccd9a.

Times are warm end-to-end medians [min, max], seconds. Speedup = reference / ATV. Missing results are explicit. Local/unqualified evidence does not promote roadmap marks.

| Method / model / task | Systems | atv.cpu.direct | atv.cpu.stored | atv.cuda | pyscf.cpu | gpu4pyscf.cuda | Notes: hardware / environment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RHF STO-3G energy (hcore) | 16 parallel waters at 4 Angstrom intervals, 112 STO-3G AOs | water_chain16: pass; 5.70386 s [5.70062, 5.70768]; energy_hartree Δmax=1.31e-08 (limit 1e-06) Ha; 0.22× vs pyscf.cpu | water_chain16: pass; 14.2432 s [14.2022, 14.2634]; energy_hartree Δmax=3.14e-11 (limit 1e-06) Ha; 0.0881× vs pyscf.cpu | water_chain16: pass; 0.5709 s [0.570004, 0.572794]; energy_hartree Δmax=4.64e-11 (limit 1e-06) Ha; 2.2× vs pyscf.cpu | water_chain16: pass; 1.25492 s [1.25292, 1.25842]; energy_hartree Δmax=0 (limit 1e-06) Ha; 1× vs pyscf.cpu | water_chain16: dependency_missing; No module named 'gpu4pyscf' | H1; FP64; reference=pyscf.cpu |

H1: CPU=aarch64; physical cores=288; logical CPUs=288; requested OMP/BLAS threads=1; GPU=['NVIDIA GH200 120GB, GPU-6d2ec6f2-f9df-c924-6ce3-cc43d7f7db1c, 97871, 580.178.04']; host RAM=921772163072 bytes; OS=Linux-5.14.0-687.59.1.el9_8.aarch64+64k-aarch64-with-glibc2.34; Python=3.13.17; 5 warm samples, 1 warmup(s)/sample; 2026-10-26T12:56:55.316309+00:00. Actual engine/compiler/threadpool versions and memory budgets are in the observations.

Summary: `{"complete": true, "failed": 0, "historical_status": "unbaselined", "incomplete": 0, "optional_unavailable": 1, "passed": 3, "performance_regressions": 0, "required": 3}`.
