# CP2K pin host notes (Arrhenius login)

## Source build (2026-08-01)

| Item | Value |
| --- | --- |
| Source | `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/src` (release tarball **CP2K 2026.1**, `REVISION` = `git:5e54ba2`) |
| Build script | `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/build_cp2k.sh` |
| Prefix | `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install` |
| Binary | `install/bin/cp2k.psmp` (loads `install/lib64/libcp2k.so.2026.1`) |
| Env | `source …/install/env.sh` |
| Fair twin runner | `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/fair_compare.sh` |
| Compiler | GCC 14.3.0 + MPICH 4.3.2 (EasyBuild) |
| Linked deps | OpenBLAS 0.3.30, FFTW 3.3.10, ScaLAPACK 2.2.2-fb, libint-cp2k 2.6.0 lmax-6, libxc 7.0.0, DBCSR 2.10.0, ELPA 2026.02.002, spglib 2.7.0 |
| Flags | `omp libint fftw3 libxc elpa parallel scalapack mpi_f08 spglib` |

**Not** the EasyBuild module binary — CP2K itself was compiled from the tarball; deps are site modules.

## Runtime recipe

```bash
source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
export OMP_NUM_THREADS=4
cd /tmp/work && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
mpirun -np 1 cp2k.psmp -i twin.inp -o twin.out
```

Site module alternative (same revision `git:5e54ba2`, more flags e.g. plumed/hdf5):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
```

## Fair re-pin snapshot (2026-08-01, source binary, 1 MPI × OMP=4)

| Twin | E_cp2k [Ha] | REFERENCE pin [Ha] | \|Δ\| | T_cp2k wall [s] |
| --- | ---: | ---: | ---: | ---: |
| `gpw-uks-o2-gamma` | −31.769803499314229 | −31.76980349931425 | 2e-14 | 11.97 |
| `ot-minimizer-water-ot` | −17.212848291913637 | −17.212848291913666 | 3e-14 | 2.69 |
| `gpw-kpoints-si` | −15.802416412655170 | −15.802416412655166 | 4e-15 | 2.73 |
| `gpw-multigrid-kpoints` | −15.802416412503838 | −15.802416412503845 | 7e-15 | 0.97 |
| D3 isolate (E_D3−E_plain) | −8.139921320e-4 | −8.1399213205e-4 | ~1e-13 | 8.6 |

Logs: `/tmp/atv-cp2k-fair/`.
