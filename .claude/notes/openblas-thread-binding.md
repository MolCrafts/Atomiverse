# OpenBLAS threading traps

Three silent failures, each an order of magnitude or more. Read before
recording any wall clock or shipping a Python wheel.

## 1. OpenMP binding single-threads an OpenMP-flavour OpenBLAS

- With OpenBLAS 0.3.32 (`USE_OPENMP`, the build `-DBLA_VENDOR=OpenBLAS`
  selects), **any** of `OMP_PROC_BIND=close|spread|true` or
  `OMP_PLACES=cores|threads` makes `openblas_get_num_threads()` report 1.
- At 16 threads: `dgemm` n=2000 807 → 106 GFLOP/s; `dsygvd` n=2432 0.52 s →
  9.7 s (worse than unbound single-thread, 2.09 s).
- Thread count is not monotonic even unbound (`dsygvd` best at 16, slower at
  32) — sweep, do not assume.
- `DeviceRuntime::configure_threads` reports the failure (several OpenMP threads
  while OpenBLAS says 1) instead of recommending binding.

**Rules**

- Never set `OMP_PROC_BIND` / `OMP_PLACES` in a harness that times a BLAS-bound
  path: SCF, GPW, plane-wave, DFTB SCC.
- Classical-MD harnesses (memory-bound OpenMP loops) may bind.
- A recorded fair wall states its environment in `REFERENCE.md`; "bound" must be
  justified.

**Perf numbers measured bound (physics pins unaffected; re-measure before
quoting):** `benchmarks/scf/validation/ri_speed/README.md`;
[gpw-perf-baseline.md](gpw-perf-baseline.md) (whole CP2K ratio table);
`regressions/scf/{uks-hessian-o2-sto3g,spin-flip-tda-o2,b2plyp-h2o,sf-tddft-o2,roks-grad-o2-sto3g}`;
`regressions/gpw/{gpw-optb88-forces,gpw-vdw-uks-multigrid,gpw-stress-water,gpw-forces-water,gpw-variable-cell-molopt}`.

## 2. OpenMP-flavour OpenBLAS deadlocks when `dlopen`ed into CPython

- The Python extension **hangs** in threaded GEMM if it resolves to the
  `USE_OPENMP` OpenBLAS (workers spin in `inner_thread`). Reproducer:
  `ctypes.CDLL(".../openmp/lp64/lib/libopenblas.so.0", RTLD_GLOBAL)` then one
  1088×1088×256 dgemm. Static executables are unaffected.
- Not the cause: numpy's bundled BLAS, import order, `OPENBLAS_NUM_THREADS`. The
  pthread/sequential flavours of the same install work.
- All flavours share the soname `libopenblas.so.0`; the runtime pick is the
  first `DT_RPATH` directory, which CMake builds from the implicit link path,
  i.e. **`LIBRARY_PATH`** (the module points it at the OpenMP flavour).
  `-DBLAS_LIBRARIES`, `LD_LIBRARY_PATH` and `BUILD_RPATH` cannot override it.
- Fix — select the flavour with `LIBRARY_PATH` when configuring the wheel:

```bash
BL=/software/.../OpenBLAS/0.3.32/g14/hpc1/pthread/lp64/lib
export LIBRARY_PATH="$BL:$LIBRARY_PATH"
cmake -B build-py -DATV_BUILD_PYTHON=ON -DBLAS_LIBRARIES=$BL/libopenblas.so -DLAPACK_LIBRARIES=$BL/libopenblas.so ...
ldd build-py/_atomiverse*.so | grep openblas   # must NOT say /openmp/
```

- `CMakeLists.txt` warns when `ATV_BUILD_PYTHON` links a libgomp/libomp BLAS
  and pins the linked BLAS directory into the rpath — a tripwire, not a
  guarantee. **Verify with `ldd`.** The pthread wheel matches C++ numbers
  (1088-atom PEG vs DFTB+) at ~1.5× the C++ wall.

## 3. Login-node CPU affinity moves under you

`sched_getaffinity` on the Arrhenius login node can drop from 48 to 4 cores
mid-session (8.3 s → 434 s for the same DFTB single point). Print
`len(os.sched_getaffinity(0))` next to any login-node number; prefer batch for
anything entering a `REFERENCE.md` or benchmark README.
