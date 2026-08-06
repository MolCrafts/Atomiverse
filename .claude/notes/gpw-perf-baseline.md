<!-- mol:note:topic:gpw-perf-baseline -->
## GPW fair-wall baseline

Standing **ATV_GPW_PROF** table for the four performance twins. Compare on the
same host class with OMP=4 + bind. **CP2K walls must be re-run on this host** —
do not trust older REFERENCE.md wall numbers when the node or module differs.

| | |
| --- | --- |
| Date | 2026-08-06 (wave-2: OT g·e DIIS outer; MG STVnl numbers still 2026-08-04) |
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
| `gpw-multigrid-kpoints` MAIN | **~2.1** quiet best-of-4 | **~1.28** quiet | **~1.6** | pin OK; STVnl ~0.52 s; coll/int ~80–90 ms |
| `gpw-uks-o2-gamma` | ~19–24 | ~21 (older pin) | ~1 | DIIS path; not product MG |
| `gpw-kpoints-si` MAIN | ~few s KS after setup | ~3.3 (older) | ~1 | dense χ^k, not MG products |
| `ot-minimizer-water` | **32** OT it; reg wall **~11.6–14 s** (OT+DIIS) | OT class ~2.7 (older) | **~3–4×** class | was 93 it / ~16.2 s / ~6× |
| `ot-uks-o2` | **87–94** OT it / OT **~38.5 s** | **8.68** OT-only | **~4.4** | was 180 it / ~62 s / ~7.1×; land claim 3.74× |

**Supersedes** pre-collint MAIN **~42 s / ~47×**, mid-collint **~27 s / ~30×**,
pre-OT-outer water ~16.2 s / ~6× and UKS ~7.1×. Keep energy pins; refresh walls
every perf wave.

### Per-step KS (MG MAIN class, ATV quiet builds)

| stage | ms / rebuild (order of) |
| --- | ---: |
| collocate | **~80–90** |
| integrate | **~85–90** (group-idx + segment SIMD accumulate) |
| Poisson + XC | ~5–10 (MG LDA-class); GGA OT water XC ~**95** ms after spectral FFT (was ~108) |
| setup STVnl_k (once) | **~520** (gen-stamp Hermite + s–s fast path; was ~1500) |

CP2K same twin: `grid_collocate` ~0.018 s/iter, `grid_integrate` ~0.019 s/iter.

### Wave-2 OT outer (landed `cpu-ot-diis-outer`)

g·e B-matrix (CP2K `ot_diis_step` default) + SAFE_DIIS/Armijo + PR+ CG fallback
on stall. Re-pin 2026-08-06 OMP=4 bind:

| Twin | iters | wall | pin / FD |
| --- | ---: | ---: | --- |
| `ot-minimizer-water` | **32** (was 93) | reg ~11.6–14 s | pin PASS; \|E_OT−E_DIIS\|~4e-11 |
| `ot-uks-o2` | **87–94** (was 180) | OT ~38.5 s | pin PASS; FD forces-water PASS |
| ratio worst | — | **~4.4×** vs CP2K 8.68 s | was ~7.1× |

### Wave-1 spectral FFT (landed `cpu-xc-spectral-fft`)

Fused multi-out howmany=3 FFT + FFTW_MEASURE plans for GGA spectral grad/div.
XC residual ~108→**95** ms/iter. Non-bit-exact MEASURE plans — stated tol.

### Active lever

**MG residual** toward ~1× (STVnl + products_mg + kbg + collint ~4× CP2K
per-iter). OT outer no longer dominates (~4.4× worst). See
[gpw-perf-architecture.md](gpw-perf-architecture.md) and
[es-cpu-cuda-queue.md](../../docs/developer/es-cpu-cuda-queue.md).
