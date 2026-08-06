<!-- mol:note:topic:gpw-perf-baseline -->
## GPW fair-wall baseline

Standing **ATV_GPW_PROF** table for the four performance twins. Compare on the
same host class with OMP=4 + bind. **CP2K walls must be re-run on this host** —
do not trust older REFERENCE.md wall numbers when the node or module differs.

| | |
| --- | --- |
| Date | 2026-08-06 (wave-3: OT OUTER_SCF polish + STVnl Hermite key reuse) |
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
| `gpw-multigrid-kpoints` MAIN | **~2.3** quiet best | **~1.28** quiet | **~1.8** | post STVnl residual; STVnl ~435 ms |
| `gpw-uks-o2-gamma` | ~19–24 | ~21 (older pin) | ~1 | DIIS path; not product MG |
| `gpw-kpoints-si` MAIN | ~few s KS after setup | ~3.3 (older) | ~1 | dense χ^k, not MG products |
| `ot-minimizer-water` | **25** OT it (was 32 / 93) | OT class ~2.7 (older) | improved | OUTER_SCF polish |
| `ot-uks-o2` | **27** OT it / OT **~13 s** | **8.68** OT-only | **~1.5** | was ~4.4× / ~90 it / ~38.5 s |

**Supersedes** pre-collint MAIN **~42 s / ~47×**, mid-collint **~27 s / ~30×**,
wave-1 water ~16.2 s / ~6×, wave-2 UKS ~4.4×. Keep energy pins; refresh walls
every perf wave.

### Per-step KS (MG MAIN class, ATV quiet builds)

| stage | ms / rebuild (order of) |
| --- | ---: |
| collocate | **~80–90** |
| integrate | **~85–90** (group-idx + segment SIMD accumulate) |
| Poisson + XC | ~5–10 (MG LDA-class); GGA OT water XC ~**95** ms after spectral FFT (was ~108) |
| setup STVnl_k (once) | **~435** (Hermite geometry-key reuse + phase flatten; was ~520 / ~1500) |

CP2K same twin: `grid_collocate` ~0.018 s/iter, `grid_integrate` ~0.019 s/iter.

### Wave-3 OT residual + STVnl residual (landed)

OUTER_SCF polish (fixed-H eigen + density mix when residual moderate; skip
under DFT+U) + STVnl Hermite geometry-key reuse. Re-pin 2026-08-06 OMP=4 bind
(worker evidence on merge `4249fd3f` / `dfe68fdd`):

| Twin | iters / setup | wall | pin / FD |
| --- | ---: | ---: | --- |
| `ot-uks-o2` | **27** (was 87–94) | OT **~13 s** | pin PASS; ratio **~1.53×** vs 8.68 s |
| `ot-minimizer-water` | **25** (was 32) | — | pin PASS; FD forces-water PASS |
| `gpw-multigrid-kpoints` | STVnl **~435 ms** | MAIN **~2.3 s** | pin bit-stable; **~1.8×** vs 1.28 s |
| ratio worst OT | — | **~1.5×** | was ~4.4× (wave-2) / ~7.1× |

### Wave-2 OT outer (landed `cpu-ot-diis-outer`)

g·e B-matrix (CP2K `ot_diis_step` default) + SAFE_DIIS/Armijo + PR+ CG fallback
on stall. Superseded for worst-ratio by wave-3 polish (water 93→32; UKS
180→~90 / ~4.4× at land).

### Wave-1 spectral FFT (landed `cpu-xc-spectral-fft`)

Fused multi-out howmany=3 FFT + FFTW_MEASURE plans for GGA spectral grad/div.
XC residual ~108→**95** ms/iter. Non-bit-exact MEASURE plans — stated tol.

### Active lever

**MG residual** toward ~1× (STVnl ~435 ms + products_mg/kbg + collint ~4× CP2K
per-iter; MAIN **~1.8×**). OT worst **~1.5×** (≤3× aim met). See
[gpw-perf-architecture.md](gpw-perf-architecture.md) and
[es-cpu-cuda-queue.md](../../docs/developer/es-cpu-cuda-queue.md).
