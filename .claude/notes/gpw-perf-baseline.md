<!-- mol:note:topic:gpw-perf-baseline -->
## GPW fair-wall baseline

Standing **ATV_GPW_PROF** table for the four performance twins. Compare on the
same host class with OMP=4 + bind. **CP2K walls must be re-run on this host** —
do not trust older REFERENCE.md wall numbers when the node or module differs.

| | |
| --- | --- |
| Date | 2026-08-06 (wave-4: collint residual hybrid SIMD + cab expand) |
| Host | `arrhenius1` — AMD EPYC 9255 ×2, OMP=4, `OMP_PROC_BIND=close` |
| Build | Release, HF+PW+GPW, CUDA OFF, OpenBLAS |
| Env | `ATV_GPW_PROF=1` |

### How to reproduce

```bash
export OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores
export ATV_GPW_PROF=1
/usr/bin/time -f 'WALL_SEC %e' ./build/regressions/reg_<twin>
# CP2K twin (MG kpoints):
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
# run tools/cp2k_oracle/inputs/gpw-multigrid-kpoints/si_kpoints_mg.inp
```

### Fair walls (MAIN / single-twin SCF)

| Twin | ATV [s] | CP2K [s] | ratio | Notes |
| --- | ---: | ---: | ---: | --- |
| `gpw-multigrid-kpoints` MAIN | **~2.4** remeasure | **~1.28** quiet | **~1.9** | post collint residual; coll ~63 / int ~76 ms |
| `gpw-uks-o2-gamma` | ~19–24 | ~21 (older pin) | ~1 | DIIS path; not product MG |
| `gpw-kpoints-si` MAIN | ~few s KS after setup | ~3.3 (older) | ~1 | dense χ^k, not MG products |
| `ot-minimizer-water` | **25** OT it (was 32 / 93) | OT class ~2.7 (older) | improved | OUTER_SCF polish |
| `ot-uks-o2` | **27** OT it / OT **~11.8 s** | **8.68** OT-only | **~1.4** | was ~1.5× / ~13 s wave-3 |

**Supersedes** pre-collint MAIN **~42 s / ~47×**, mid-collint **~27 s / ~30×**,
wave-1 water ~16.2 s / ~6×, wave-2 UKS ~4.4×, wave-3 MAIN ~2.3 s / ~1.8×.
Keep energy pins; refresh walls every perf wave.

### Per-step KS (MG MAIN class, ATV quiet builds)

| stage | ms / rebuild (order of) |
| --- | ---: |
| collocate | **~63** (was ~80–90; residual hybrid SIMD + cab expand) |
| integrate | **~76** (was ~85–90; \|V(pc)\| once per geometry) |
| Poisson + XC | ~5–10 (MG LDA-class); GGA OT water XC ~**95–100** ms after spectral FFT |
| setup STVnl_k (once) | **~435** (Hermite geometry-key reuse + phase flatten; was ~520 / ~1500) |

CP2K same twin: `grid_collocate` ~0.018 s/iter, `grid_integrate` ~0.019 s/iter.

### Wave-4 collint residual (landed)

Hybrid xsimd FMA + 4-wide residual (AVX-512 hosts), lp_z≤9, cab-style alpha for
large geometry groups, TLS multi-imr sphere bounds, gamma group_idx nthread
private grids. Re-pin 2026-08-06 OMP=4 bind (merge `5fa8b0c6` / feat `6da97a85`):

| Twin | iters / setup | wall | pin / FD |
| --- | ---: | ---: | --- |
| `gpw-multigrid-kpoints` | coll **~63** / int **~76** ms | MAIN **~2.4 s** | pin \|ΔE\|=5.7e-6 Ha; **~1.9×** vs 1.28 s |
| `ot-uks-o2` | **27** | OT **~11.8 s** | pin PASS; ratio **~1.4×** vs 8.68 s |
| `ot-minimizer-water` | **25** | — | pin PASS |
| `gpw-forces-water` | — | — | FD PASS max\|dF\|=3.2e-4 |
| ratio worst OT | — | **~1.4×** | was ~1.5× (wave-3) |

### Wave-3 OT residual + STVnl residual (landed)

OUTER_SCF polish (fixed-H eigen + density mix when residual moderate; skip
under DFT+U) + STVnl Hermite geometry-key reuse. Superseded MAIN numbers by
wave-4 remeasure above.

### Wave-2 OT outer (landed `cpu-ot-diis-outer`)

g·e B-matrix (CP2K `ot_diis_step` default) + SAFE_DIIS/Armijo + PR+ CG fallback
on stall. Superseded for worst-ratio by wave-3 polish (water 93→32; UKS
180→~90 / ~4.4× at land).

### Wave-1 spectral FFT (landed `cpu-xc-spectral-fft`)

Fused multi-out howmany=3 FFT + FFTW_MEASURE plans for GGA spectral grad/div.
XC residual ~108→**95** ms/iter. Non-bit-exact MEASURE plans — stated tol.

### Active lever

**MG residual** toward ~1×: products_mg setup (~0.3 s) + remaining collint gap
vs CP2K ~20 ms class (shell-level tasks open); MAIN **~1.9×**. OT worst
**~1.4×** (≤3× aim met). See
[gpw-perf-architecture.md](gpw-perf-architecture.md) and
[es-cpu-cuda-queue.md](../../docs/developer/es-cpu-cuda-queue.md).
