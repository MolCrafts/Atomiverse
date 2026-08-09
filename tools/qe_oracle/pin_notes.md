# QE pin host notes (Arrhenius x86_64 login)

## Source build (2026-08-09)

| Item | Value |
| --- | --- |
| Source | `/nobackup/proj/disk/teoroo/personal/jicli594/work/qe/src` (GitLab archive of tag **`qe-7.6`**, `https://gitlab.com/QEF/q-e/-/archive/qe-7.6/q-e-qe-7.6.tar.gz`) |
| Version string | `Program PWSCF v.7.6` (printed by `pw.x`) |
| Prefix | `/nobackup/proj/disk/teoroo/personal/jicli594/work/qe/install` |
| Binaries | `install/bin/pw.x` (+106 others incl. `bands.x`, `dos.x`, `pp.x`) |
| Env | `source …/qe/install/env.sh` (loads `buildenv-gcc/2026.03-mpich`, mirrors `LIBRARY_PATH` into `LD_LIBRARY_PATH`, prepends `install/bin`) |
| Compiler | GCC 14.3.0 + MPICH 4.3.2 via `mpicc` / `mpif90` (module `buildenv-gcc/2026.03-mpich`) |
| BLAS/LAPACK | OpenBLAS 0.3.32 (openmp, lp64): `/software/sse2/el9_epyc9005/manual/OpenBLAS/0.3.32/g14/hpc1/openmp/lp64/lib/libopenblas.so` |
| FFT | FFTW 3.3.11 external: `libfftw3.so` + `libfftw3_omp.so` under `/software/sse2/el9_epyc9005/manual/FFTW/3.3.11/g14/mp43/hpc1/lib` |
| ScaLAPACK | OFF |
| Vendored externals | auto-cloned by QE CMake at QE-pinned hashes: mbd `89a3cc19`, devxlib `a6b89ef7`, wannier90 `1d6b1873` (`external/submodule_commit_hash_records`) |
| Logs | `…/work/qe/configure.log`, `build.log`, `install.log` |

**Build line** (login shell, `bash -lc`, from `…/work/qe`):

```bash
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
export CMAKE_PREFIX_PATH="$OPENBLASDIR:$FFTW3_ROOT:$CMAKE_PREFIX_PATH"
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/nobackup/proj/disk/teoroo/personal/jicli594/work/qe/install \
      -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpif90 \
      -DQE_ENABLE_MPI=ON -DQE_ENABLE_OPENMP=ON -DQE_ENABLE_SCALAPACK=OFF \
      -DBLA_VENDOR=OpenBLAS
cmake --build build -j 24
cmake --install build
```

Gotcha: without `-DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpif90`,
CMake picks the system `/usr/bin/cc`+`/usr/bin/f95` (GCC 11.5) which cannot
read MPICH's gfortran-14 `mpi.mod`.

## Runtime recipe

```bash
source /nobackup/proj/disk/teoroo/personal/jicli594/work/qe/install/env.sh
export OMP_NUM_THREADS=4        # pin and record for fair walls
cd /tmp/work && mpirun -np 1 pw.x -in twin.in > twin.out
```

## Install validation (2026-08-09, not a regression pin)

Si diamond, ATV in-tree ONCVPSP NC UPF `tests/data/pw/Si_lda_nc.upf`,
`ecutwfc = 15 Ry`, 2×2×2 MP, `conv_thr = 1e-8`; input committed at
`inputs/smoke-si/si.in` (set `pseudo_dir` to the ATV checkout's
`tests/data/pw`). Key output lines (1 MPI × OMP=4):

```text
     Program PWSCF v.7.6 starts on  9Aug2026 at 21:52:23
     Exchange-correlation= PZ
!    total energy              =     -15.67520726 Ry
     convergence has been achieved in   7 iterations
     PWSCF        :      1.06s CPU      0.52s WALL
```

This validates that `pw.x` reads the exact UPF files the ATV PW regressions
use — the twin precondition for `pw-perf-fair-walls-batch1+`. Case pins and
fair walls land in each case's `REFERENCE.md`, not here.
