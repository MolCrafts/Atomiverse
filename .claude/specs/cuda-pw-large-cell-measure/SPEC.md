# Spec — `cuda-pw-large-cell-measure`

| Field | Value |
| --- | --- |
| slug | `cuda-pw-large-cell-measure` |
| method | cuda |
| branch | `wf/escuda-cuda-pw-large-cell-measure` |
| oracle | same-settings CPU twin (`cpu::pw::PW`); correctness via `pw-cuda-lda-si` |
| status | measurement (no kernel change) |

## Problem

ES order step 4: residency polish on `cuda::pw::PW` (kpoint hoist, Zheevd,
devptr-nrm2) cannot be shown to pay on the 2-atom Si correctness cell —
process/context init dominates ~2 s. Need the first **large-cell** CPU↔CUDA
wall with GH200 log evidence.

## Lever

| | |
| --- | --- |
| Class | Measurement harness + note fill |
| Files | `tools/bench_cuda_pw_cell.cu`, `tools/gpu_job_cuda_pw_large_cell.sh`, `.claude/notes/cuda-pw-large-cell.md` |
| Metric | `run(frame)` wall_s for Si n=1/2/3 supercells (8/64/216 atoms), LSDA Γ ecut 15 Ry |
| Non-goal | Physics change, CMake product targets, forces/stress/PBE |

## Acceptance

1. GH200 job log contains `BENCH_RESULT` lines for CUDA and CPU (or CPU
   timeout lower-bound with `WALL_LOWER_BOUND` / `LEG_TIMEOUT`).
2. `reg_pw-cuda-lda-si --device=both` PASS on the same job (slice sanity).
3. Same-settings CPU wall **> 60 s** at some ladder rung (gate).
4. Note `.claude/notes/cuda-pw-large-cell.md` filled with methodology + table
   + job id (not PENDING).
5. No global “CUDA wins” claim on n_rep=1; large-cell ratio reported at the
   gate-met size.
6. Build dirs wiped after the job / worker exit.

## Dual gate

| Gate | How |
| --- | --- |
| Correctness | `pw-cuda-lda-si` both-device; \|E_cuda−E_cpu\| < 1e-3 Ry |
| Wall | CPU twin at identical settings; T_cuda / T_cpu reported; within_10x or documented lower-bound |

## Evidence (authoritative)

Job **1035278** (see note): n=2 CUDA **9.664 s**, CPU **>5400 s** → gate met;
slice sanity \|ΔE\|=1.243e-14 Ry.
