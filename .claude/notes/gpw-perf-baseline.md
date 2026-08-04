<!-- mol:note:topic:gpw-perf-baseline -->
## GPW fair-wall baseline

Standing **ATV_GPW_PROF** table for the four performance twins. Compare on the
same host class with OMP=4 + bind. **CP2K walls must be re-run on this host** —
do not trust older REFERENCE.md wall numbers when the node or module differs.

| | |
| --- | --- |
| Date | 2026-08-04 (post product_collint geometry-batch) |
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
| `gpw-multigrid-kpoints` MAIN | **~27** quiet (load can push 45+) | **0.885** (re-run 2026-08-04) | **~30** | pin −15.8024164 Ha; post collint batch |
| `gpw-uks-o2-gamma` | ~19–24 | ~21 (older pin) | ~1 | DIIS path; not product MG |
| `gpw-kpoints-si` MAIN | ~few s KS after setup | ~3.3 (older) | ~1 | dense χ^k, not MG products |
| `ot-minimizer-water` | ~9–11 reg wall | OT class ~2.7 (older) | ~3–4 | includes DIIS gate in reg |

**Supersedes** pre-collint MAIN **~42 s / ~47×** (1D SIMD only) and the
stale wave2 CP2K wall 23.6 s. Keep energy pins; refresh walls every perf wave.

### Per-step KS (MG MAIN class, ATV quiet builds)

| stage | ms / rebuild (order of) |
| --- | ---: |
| collocate | ~550–760 |
| integrate | **~740–800** (was ~4200–4500) |
| Poisson + XC | ~10 |

CP2K same twin: `grid_collocate` ~0.018 s/iter, `grid_integrate` ~0.019 s/iter.

### Active lever

Still collint-class: reflection pairs, tighter sphere cache, task-list
locality beyond geometry sort — see [gpw-perf-architecture.md](gpw-perf-architecture.md).
