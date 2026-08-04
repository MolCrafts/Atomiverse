<!-- mol:note:topic:gpw-perf-baseline -->
## GPW fair-wall baseline

Standing **ATV_GPW_PROF** table for the four performance twins. Compare on the
same host class with OMP=4 + bind. **CP2K walls must be re-run on this host** —
do not trust older REFERENCE.md wall numbers when the node or module differs.

| | |
| --- | --- |
| Date | 2026-08-04 (STVnl gen-stamp + integrate group-idx) |
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
| `ot-minimizer-water` | ~9–11 reg wall | OT class ~2.7 (older) | ~3–4 | includes DIIS gate in reg |

**Supersedes** pre-collint MAIN **~42 s / ~47×**, mid-collint **~27 s / ~30×**,
and the stale wave2 CP2K wall 23.6 s. Keep energy pins; refresh walls every
perf wave.

### Per-step KS (MG MAIN class, ATV quiet builds)

| stage | ms / rebuild (order of) |
| --- | ---: |
| collocate | **~80–90** |
| integrate | **~85–90** (group-idx + segment SIMD accumulate) |
| Poisson + XC | ~5–10 |
| setup STVnl_k (once) | **~520** (gen-stamp Hermite + s–s fast path; was ~1500) |

CP2K same twin: `grid_collocate` ~0.018 s/iter, `grid_integrate` ~0.019 s/iter.

### Active lever

Residual ~0.8 s vs CP2K: setup (STVnl + products_mg + kbg) and
collint still ~4× CP2K per-iter. Next only if chasing ~1× — see
[gpw-perf-architecture.md](gpw-perf-architecture.md).
