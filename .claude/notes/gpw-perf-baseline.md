<!-- mol:note:topic:gpw-perf-baseline -->
## GPW fair-wall baseline

Standing **ATV_GPW_PROF** + wall table for the four performance twins. Compare
on the **same host** with OMP=4 + bind. **CP2K walls must be re-run on this
host** — do not trust older REFERENCE.md wall numbers when the node, module, or
binary differs.

| | |
| --- | --- |
| Date | **2026-08-06** (cp2k-perf Baseline re-pin + harness wave-2 MG + wave-5 UKS ATV refresh, parent `dev`) |
| Host | `arrhenius1` — AMD EPYC 9255 ×2, OMP=4, `OMP_PROC_BIND=close` `OMP_PLACES=cores` |
| Build (ATV) | Release, HF+PW+GPW, CUDA OFF, OpenBLAS (`-DBLA_VENDOR=OpenBLAS`) |
| Env | `ATV_GPW_PROF=1` for stage ms; fair walls without prof where noted |
| Metric | \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) on **matched twin single SCF job**, 1 MPI × OMP=4 |

### CP2K binary provenance (binding)

| Field | Value |
| --- | --- |
| Binary | `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/bin/cp2k.psmp` |
| Activate | `source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh` |
| Version | **CP2K 2026.1** `git:5e54ba2` |
| Flags | `omp libint fftw3 libxc elpa parallel scalapack mpi_f08 spglib` |
| Compiler | GCC 14.3.0 |
| Data | `CP2K_DATA_DIR=/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/src/data` |
| Launch | `OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores mpirun -np 1 cp2k.psmp -i <inp> -o <out>` |

**Fair wall convention:** twin `.inp` physics unchanged; `PRINT_LEVEL` forced to
**LOW** for the wall pin (MEDIUM I/O inflates the CP2K timer ~2–3× on the same
job). \(T_{\mathrm{cp2k}}\) = TIMING table **`CP2K` TOTAL TIME MAXIMUM** (not
`/usr/bin/time` wall, which includes mpirun startup). Energy lines below are
from the same quiet runs (digits match the offline energy pins).

### How to reproduce

```bash
# --- CP2K (each twin) ---
source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
export CP2K_DATA_DIR=/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/src/data
export OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
# copy twin .inp; ensure PRINT_LEVEL LOW; then:
mpirun -np 1 cp2k.psmp -i <twin>.inp -o <twin>.out
# pin lines: ENERGY| Total FORCE_EVAL …  and  TIMING  CP2K  … MAXIMUM

# --- ATV ---
module load buildenv-gcc/2026.03-mpich
export OPENBLAS_ROOT="$OPENBLASDIR" RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores
cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
cmake --build build -j$(nproc) --target atomiverse \
  reg_gpw-uks-o2-gamma reg_gpw-kpoints-si reg_gpw-multigrid-kpoints reg_ot-minimizer-water
/usr/bin/time -f 'WALL_SEC %e' ./build/regressions/reg_<twin>
```

### Fair walls (MAIN / single-twin SCF) — 2026-08-06 re-pin

| Twin | ATV [s] | CP2K [s] | ratio | Notes |
| --- | ---: | ---: | ---: | --- |
| `gpw-multigrid-kpoints` MAIN | **1.6** | **0.688** | **2.33** | post-merge `p0-stvnl-setup-residual` harness re-measure (quiet ×3, MAIN print; E bit-identical); CP2K quiet timer MAX unchanged |
| `gpw-uks-o2-gamma` | **14.75** | **13.691** | **1.077** | post-merge `p0-uks-o2-scf-wall` (`3f991946`); dual-spin cbg fuse + packed χ + lazy multi-FFT; CP2K quiet timer MAX unchanged |
| `gpw-kpoints-si` MAIN | **2.88** | **2.45** | **1.18** | ATV MAIN = STVnl+kbg+6 KS from `ATV_GPW_PROF` (ngrid=37800 nk=4); full reg ~13.6 s (7 SCF solves, not fair); not re-measured this wave |
| `ot-minimizer-water` OT-only | **7.55** | **2.50** | **3.02** | ATV OT segment via dual runtime-init timestamps (25 OT it); full OT+DIIS reg ~9.7 s (path untouched) |
| **worst_ratio** | — | — | **3.02** | OT water |

CP2K walls from 2026-08-06 Baseline re-pin (provenance below). ATV MG MAIN
refreshed post-merge `e1eaf084` / harness wave-2; UKS ATV refreshed post-merge
`3f991946` / harness wave-5. Energy pins unchanged.

### CP2K pin detail (energy + wall + output line)

#### `gpw-multigrid-kpoints`

| Field | Value |
| --- | --- |
| Input | `tools/cp2k_oracle/inputs/gpw-multigrid-kpoints/si_kpoints_mg.inp` (PRINT_LEVEL LOW for wall) |
| Energy line | `ENERGY\| Total FORCE_EVAL ( QS ) energy [hartree]            -15.802416412503845` |
| Timer line | `CP2K … TOTAL TIME MAXIMUM **0.688**` (warm pass2; pass1 0.689) |
| `/usr/bin/time` | WALL_SEC 1.00 (includes mpirun; not the pin) |
| grid_collocate_task_list | 0.203 s / 14 calls ≈ **14.5 ms**/call |
| grid_integrate_task_list | 0.222 s / 13 calls ≈ **17.1 ms**/call |
| SCF | 13 steps |
| ATV MAIN | **1.6 s** (quiet ×3 post `p0-stvnl-setup-residual`; loadavg high but stable); E = −15.8023961242 Ha; \|ΔE\|pin = 2.029e-05 Ha |
| ATV setup (prof MAIN) | ST_k **~340–346** ms + Vnl_k **~57–62** ms; products_mg **~19–21** ms |
| steady coll/int (MAIN ngrid=37800) | collocate **~29–30** ms; integrate **~47–49** ms |
| ratio | **1.6 / 0.688 = 2.33** |

#### `gpw-kpoints-si`

| Field | Value |
| --- | --- |
| Input | `tools/cp2k_oracle/inputs/gpw-kpoints-si.inp` (PRINT_LEVEL LOW for wall) |
| Energy line | `ENERGY\| Total FORCE_EVAL ( QS ) energy [hartree]            -15.802416412655166` |
| Timer line | `CP2K … MAXIMUM **2.451**` (warm; pass1 2.443) |
| WALL_SEC | 2.77 |
| ATV MAIN | **2.875 s** prof-sum (STVnl 433 + kbg 1393 + 6×KS coll/xc/int); E = −15.8023941333 Ha; \|ΔE\| = 2.228e-05 |
| Full reg wall | ~13.6 s (not the fair twin) |
| ratio MAIN | **2.875 / 2.451 = 1.17** |

#### `gpw-uks-o2-gamma`

| Field | Value |
| --- | --- |
| Input | `tools/cp2k_oracle/inputs/gpw-uks-o2-gamma/o2_uks_pade.inp` (PRINT_LEVEL LOW for wall) |
| Energy line | `ENERGY\| Total FORCE_EVAL ( QS ) energy [hartree]            -31.769803499314229` (warm) |
| Timer line | `CP2K … MAXIMUM **13.691**` (warm; pass1 14.039) |
| WALL_SEC | 14.03 warm |
| ATV | **14.75 s** post `p0-uks-o2-scf-wall` (`3f991946`; dual-spin cbg fuse + packed χ + lazy multi-FFT); prior pin **25.08 s**; E pin holds (\|ΔE\| ~1.26e-5); not bit-exact (triangle) |
| ratio | **14.75 / 13.691 = 1.077** |

#### `ot-minimizer-water`

| Field | Value |
| --- | --- |
| Input | `tools/cp2k_oracle/inputs/ot-minimizer-water-ot.inp` (already PRINT_LEVEL LOW) |
| Energy line | `ENERGY\| Total FORCE_EVAL ( QS ) energy [hartree]            -17.212848291913609` (warm) |
| Timer line | `CP2K … MAXIMUM **2.496**` (warm; pass1 2.466); 19 OT outer steps |
| WALL_SEC | 2.83 |
| ATV OT-only | **7.55 s** (runtime init → second runtime init before DIIS path); 25 OT it |
| ATV OT+DIIS reg | **9.7 s** full binary (not fair vs CP2K OT alone) |
| E_OT | −17.2118582735 Ha; \|ΔE\|CP2K = 9.900e-04 |
| ratio OT | **7.55 / 2.496 = 3.02** |

### Per-step KS (MG MAIN class, ATV)

| stage | ms / rebuild (order of) |
| --- | ---: |
| collocate (steady) | **~29–30** |
| integrate (steady) | **~47–49** |
| Poisson + XC | ~4–5 (MG PBE-class; first build higher) |
| setup products_mg (once) | **~19–21** |
| setup ST_k / Vnl_k (once) | **~340–346 / ~57–62** (post `p0-stvnl-setup-residual`) |

CP2K same twin quiet: collocate ~14.5 ms/call, integrate ~17.1 ms/call
(TOTAL time / CALLS on the TIMING table).

### Active lever

**worst_ratio = 3.02** (`ot-minimizer-water`). MG MAIN **2.33×** vs this
host’s quiet CP2K 0.688 s (integrate still ~2.8× CP2K per-call class; STVnl
setup residual landed). UKS O₂ **1.077×** (post `p0-uks-o2-scf-wall`; ≤1.2×
gate closed). Dense-χ^k MAIN **~1.2×**. See
[gpw-perf-architecture.md](gpw-perf-architecture.md) and
[cp2k-perf-loop.md](../../docs/developer/cp2k-perf-loop.md).

### Prior wave notes (history, not the pin)

Wave-7 integrate residual claimed MAIN ~1.6 s vs ~1.28 s CP2K (~1.25×) on an
older quiet definition; OT UKS O₂ ~1.4× was a different twin (`ot-uks-o2`).
This Baseline row **replaces** those wall numbers with re-runnable provenance
on `arrhenius1` + source-install CP2K 2026.1.
