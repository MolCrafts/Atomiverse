<!-- mol:note:topic:gpw-perf-baseline -->
## GPW fair-wall baseline

Single authority for GPW walls vs CP2K. Compare on the **same host**; re-run
CP2K there — never trust older `REFERENCE.md` walls when node, module or binary
differ.

> **Suspect:** every ATV and CP2K wall below was taken with
> `OMP_PROC_BIND=close OMP_PLACES=cores`, which collapses an OpenMP-flavour
> OpenBLAS to one thread ([openblas-thread-binding.md](openblas-thread-binding.md)).
> Energy pins are unaffected. Re-measure unbound before quoting a ratio as an
> ATV gap.

| | |
| --- | --- |
| Host | `arrhenius1` — AMD EPYC 9255 ×2, OMP = 4 |
| ATV build | Release, HF+PW+GPW, CUDA OFF, `-DBLA_VENDOR=OpenBLAS` |
| Profiling | `ATV_GPW_PROF=1` for stage ms; fair walls without it |
| Metric | \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) on the matched single-SCF twin, 1 MPI × OMP 4 |

### CP2K binary provenance

| Field | Value |
| --- | --- |
| Binary | `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/bin/cp2k.psmp` |
| Activate | `source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh` |
| Version | CP2K 2026.1 `git:5e54ba2` (`omp libint fftw3 libxc elpa parallel scalapack mpi_f08 spglib`, GCC 14.3.0) |
| Data | `CP2K_DATA_DIR=/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/src/data` |

**Wall convention:** twin `.inp` physics unchanged; `PRINT_LEVEL LOW` (MEDIUM
I/O inflates the timer 2–3×). \(T_{\mathrm{cp2k}}\) = TIMING table `CP2K`
TOTAL TIME **MAXIMUM** (not `/usr/bin/time`, which includes mpirun startup).

### Reproduce

```bash
# CP2K (each twin)
source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
export CP2K_DATA_DIR=/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/src/data
export OMP_NUM_THREADS=4; unset OMP_PROC_BIND OMP_PLACES
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
mpirun -np 1 cp2k.psmp -i <twin>.inp -o <twin>.out
# pin lines: "ENERGY| Total FORCE_EVAL …" and "CP2K … MAXIMUM"

# ATV (docs/developer/project-contract.md build recipe, plus)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
cmake --build build -j$(nproc) --target atomiverse \
  reg_gpw-uks-o2-gamma reg_gpw-kpoints-si reg_gpw-multigrid-kpoints reg_ot-minimizer-water
OMP_NUM_THREADS=4 /usr/bin/time -f 'WALL_SEC %e' ./build/regressions/reg_<twin>
```

### Walls (bound — see warning)

| Twin | Input (`tools/cp2k_oracle/inputs/`) | CP2K energy [Ha] | ATV [s] | CP2K [s] | ratio |
| --- | --- | --- | ---: | ---: | ---: |
| `gpw-multigrid-kpoints` MAIN | `gpw-multigrid-kpoints/si_kpoints_mg.inp` | −15.802416412503845 | 1.6 | 0.688 | **2.33** |
| `gpw-uks-o2-gamma` | `gpw-uks-o2-gamma/o2_uks_pade.inp` | −31.769803499314229 | 14.75 | 13.691 | 1.08 |
| `gpw-kpoints-si` MAIN | `gpw-kpoints-si.inp` | −15.802416412655166 | 2.88 | 2.451 | 1.17 |
| `ot-minimizer-water` OT | `ot-minimizer-water-ot.inp` | −17.212848291913609 | 2.99 (SCF body ~1.57) | 2.496 | 1.20 (body ≤ 1.0) |

- ATV MAIN of `gpw-kpoints-si` = STVnl + kbg + 6 KS from `ATV_GPW_PROF`; the
  full regression (~13.6 s, 7 SCF solves) is not the fair twin.
- MG MAIN per step (ATV): collocate ~29–30 ms, integrate ~47–49 ms, Poisson+XC
  ~4–5 ms; setup ST_k ~340 ms, Vnl_k ~60 ms, products_mg ~20 ms. CP2K same twin:
  collocate ~14.5 ms/call, integrate ~17.1 ms/call (13 SCF steps).
- ATV energies vs pins: MG |ΔE| 2.0e-5 Ha, k-points 2.2e-5, UKS ~1.3e-5,
  OT water 9.9e-4.

Architecture and levers: [gpw-perf-architecture.md](gpw-perf-architecture.md).
