# ES CPU+CUDA optimization — wave 7 acceptance report

**Date:** 2026-08-10  
**Parent:** `dev` @ `5de10b67` (wave-7 harness close)  
**Stop reason:** Staffable queue empty after landing both CUDA pending rows.  
**cpu_pending:** 0 (no pin-safe fair-wall lever; MG MAIN **2.33×** blocked `mg-main-structural`)  
**cuda_pending:** 0 after wave  
**No push** (per loop law).

## Waves run

| Wave | Tracks | Outcome |
| --- | --- | --- |
| 7 | CUDA only (cpu empty) | **2 CUDA merges** |

## Merged this wave (`docs/developer/es-cpu-cuda-queue.md` done)

| Slug | Merge | GH200 evidence |
| --- | --- | --- |
| `cuda-pw-metallic-smear` | `d2ccc627` / feat `2891e824` | job **1051119**: ctest 9/9; Al dual-gate E both **−7.5496489573 Ry**, n_occ=3, \|ΔE\|=**1.24e-14** Ry (tol 5e-3) |
| `cuda-rhf-nondefault-stream` | `018dca0d` / feat `a8e23747` | job **1051227**: `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS** (2.98 + 0.86 s) |

## CPU track

- **cpu_merged = 0** — honest staffable set empty (queue + baseline).
- Worst OT/MG wall ratio **unchanged: 2.33×** (`gpw-multigrid-kpoints` MAIN 1.6 s / CP2K 0.688 s quiet).
- Water OT ~1.20× / UKS DIIS 1.077× closed; no promote from gpw-perf-queue.

## CUDA track

- **cuda_merged = 2**
- Restricted metallic smear on `cuda::pw::PW`: FD/Gaussian/MP, `extra_bands`, multi-k device Ritz cache, Mermin free energy.
- Spin+smear, smear forces, smear stress still reject (as designed).
- RHF residual/V_nuc D2H bound to Evaluate non-default stream (not default stream).
- GPW full product still **blocked** (stage units only).

## CUDA parity status

| Twin | Result |
| --- | --- |
| Al FD free energy CPU↔CUDA | \|ΔE\|=1.24e-14 Ry ≪ 5e-3 |
| RHF H₂/H₂O unit suite | 100% PASS (tols unchanged) |

## Remaining

- **pending_remaining = 0** staffable
- **gaps_remaining ≈ 5** residual/blocked: `mg-main-structural`, `cpu-collint-int-residual-2` no-land, `p1-products-default-k`, `p2-mpi-product-grid`, `cuda-gpw-full-product`
- Out of loop: hybrid/GAPW/PAW/meta-GGA; classical MD CUDA

## Re-run failed tips

```bash
# Metallic smear dual-gate (GH200)
cd <tree-at-2891e824-or-later>
sbatch tools/gpu_job_cuda_pw_metallic_smear.sh
# RHF stream (GH200)
sbatch tools/gpu_job_cuda_rhf_nondefault_stream.sh
```

Logs: `.claude/cuda-pw-metallic-smear-1051119.out`, `.claude/cuda-rhf-nondefault-stream-1051227.out`

## Disk law

`builds_wiped=true` — post-job wipe in scripts + clearup; `du -sh build build-*` empty on exit.

## CPU fair-wall authority

Unchanged: [`.claude/notes/gpw-perf-baseline.md`](../.claude/notes/gpw-perf-baseline.md).
