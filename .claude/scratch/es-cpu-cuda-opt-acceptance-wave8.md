# ES CPU + CUDA opt — acceptance report (wave 8 attempt / post wave-7)

**Date:** 2026-08-06  
**Parent:** `dev` @ `a3f5f58f` (wave-7 harness mark; code land `8b76cbe7` / `c6659278`)  
**Stop reason:** Optional CPU residual (`cpu-collint-int-residual-2`) remeasured on login host — **no landable win** vs wave-7 baseline; CUDA track **empty** (foothold chain closed wave 7). No push.

## Waves run

| Wave | Landed | Notes |
| ---: | --- | --- |
| 1–7 | yes | See `docs/developer/es-cpu-cuda-queue.md` done |
| 8 | **no merge** | `cpu-collint-int-residual-2` attempted on `wf/escuda-cpu-collint-int-residual-2`; branch deleted after clearup |

## CPU track (this session)

Staffed: `cpu-collint-int-residual-2` off current `dev`.

Host baseline remeasure (arrhenius login, OMP=4 bind, Release, same binary class as wave-7):

| Metric | Wave-7 note | Host remeasure (w7 tree) | Attempt A (z-range fused) | Attempt B (per-chunk hsum) |
| --- | ---: | ---: | ---: | ---: |
| int ms/iter (MAIN `ngrid=37800`) | ~45 | **~47–49** | ~48–50 (noise) | **~68–72 (regress)** |
| coll ms/iter | ~29 | ~30–33 | ~33 | ~31 |
| MAIN wall | ~1.6 s | **~1.7 s** | ~1.6–1.7 | **~1.8 s** |
| pin \|ΔE\| | 2.029e-05 Ha | 2.029e-05 | same | same |
| FD `gpw-forces-water` | max\|dF\|=3.243e-04 | — | PASS 3.243e-04 | PASS 3.243e-04 |

**Decision:** do **not** merge. Per-chunk CP2K-style hsum into `cx` clearly worse on AVX-512 Zen5; fused zero/reduce across sphere sections is noise-flat vs wave-7 segment accumulate. Kernel polish on `product_collint` integrate is **exhausted** at this host’s ~47–49 ms class (~2.4–2.5× CP2K int ~19 ms). Further int cut needs a non-kernel lever (task/MPI product layout, or products-default-k after closer parity — both gated elsewhere).

**Pending CPU (unchanged):**

- `cpu-collint-int-residual-2` (P2, optional) — still open; no free kernel win found  
- `cpu-simd-2center-setup` (P3) — not staffed (slab-class only; invisible on four fair twins)

**Worst OT ratio:** `ot-uks-o2` **~1.4×** (≤3× aim already met; not primary).  
**MG MAIN ratio:** **~1.25–1.33×** vs CP2K quiet 1.28 s (host MAIN ~1.6–1.7 s).

## CUDA track

No open pending after wave-7 GapCensus. Foothold chain closed:

- RHF s/p energy + forces + level_shift + path-select + device CDIIS  
- Γ GPW Poisson/FFT, Padé LDA XC, product ρ collocate, product V·shape integrate  

**Blocked:** `cuda-gpw-full-product` (architecture freeze / full method out of scope).  
**No CUDA staff this session.** GH200 not required (no CUDA change).

## Merged slugs (loop total, all on `dev`)

### CPU
`cpu-xc-spectral-fft`, `cpu-ot-diis-outer`, `cpu-ot-residual`, `cpu-stvnl-residual`, `cpu-collint-residual`, `cpu-products-mg-setup`, `cpu-collint-shell-tasks`, `cpu-collint-integrate-residual`

### CUDA
`cuda-rhf-pshell-forces`, `cuda-rhf-device-only-polish`, `cuda-rhf-level-shift`, `cuda-rhf-path-select-host`, `cuda-rhf-cdiis-device`, `cuda-gpw-foothold`, `cuda-gpw-xc-lda-foothold`, `cuda-gpw-rho-collocate-foothold`, `cuda-gpw-product-integrate-foothold`

**cpu_done this loop:** 8  
**cuda_done this loop:** 9  
**Wave-8 new merges:** 0

## Pending remaining

- 2 CPU optional (`cpu-collint-int-residual-2`, `cpu-simd-2center-setup`)  
- 0 CUDA open  
- blocked: `global-xtensor-xsimd`, `cuda-gpw-full-product`

## Clearup

- Worktree `Atomiverse-wf-escuda-cpu-collint-int-residual-2` removed  
- Branch `wf/escuda-cpu-collint-int-residual-2` deleted (no merge)  
- All `build` / `build-*` wiped on touched trees; repo size ~129 M (sources only)

## How to re-run failed tips

```bash
# CPU fair twin (login)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:$LD_LIBRARY_PATH"
  export OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON \
    -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
    -DATV_BUILD_GPW=ON -DATV_BUILD_CUDA=OFF -DATV_BUILD_SEMIEMPIRICAL=OFF \
    -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target reg_gpw-multigrid-kpoints reg_gpw-forces-water \
    test_cpu_gpw_collocate test_cpu_gpw_multigrid
  ATV_GPW_PROF=1 ./build/regressions/reg_gpw-multigrid-kpoints
  ./build/regressions/reg_gpw-forces-water
  rm -rf build build-* CMakeCache.txt CMakeFiles Testing .cache
'

# CUDA (GH200 batch only)
# sbatch -p gpu -A naiss2026-4-715-gpu --gpus 1 -t 01:00:00 ...
# job: module load GPU/buildenv-gcccuda/2026.03-cu13.0
# cmake -DCMAKE_CUDA_ARCHITECTURES=90 -DATV_BUILD_CUDA=ON -DATV_BUILD_SEMIEMPIRICAL=OFF ...
# ctest -R 'test_cuda_scf_rhf|test_cuda_gpw_'
```

## Operator return summary

Loop is **stable post wave-7**. Wave 8 GapCensus found only optional CPU residual; attempted integrate residual-2 on real hardware and **stopped without merge** (no wall/int win; per-chunk path regressed). CUDA parity footholds remain green from prior GH200 lands; full GPW CUDA product still blocked. Next free wall lever is not a thin collint kernel tweak — re-open only with a new oracle-backed thin unit (CUDA) or a non-kernel CPU architecture step after explicit gate review.
