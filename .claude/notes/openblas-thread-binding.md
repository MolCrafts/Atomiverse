# OpenBLAS threading traps

Two independent OpenBLAS failures, both silent, both costing an order of
magnitude or more. Read both before recording any wall-clock number or
shipping a Python wheel.

---

# 1. OpenMP thread binding silently single-threads OpenBLAS

**Any** OpenMP affinity request collapses an OpenMP-flavour OpenBLAS to one
thread. This is not a tuning preference — it is a 10× cliff on every
BLAS-bound path, and every fair-wall number in the tree that was measured with
binding on is an ATV-handicapped number.

## The measurement

Host: `arrhenius` EPYC 9005 login node, 48 cores visible.
OpenBLAS 0.3.32 (`USE_OPENMP`, ZEN, lp64) — the build every documented recipe
selects with `-DBLA_VENDOR=OpenBLAS`.

`openblas_get_num_threads()` with `OMP_NUM_THREADS=16`:

| environment | threads reported |
| --- | --- |
| (nothing else set) | **16** |
| `OMP_PROC_BIND=close` | 1 |
| `OMP_PROC_BIND=spread` | 1 |
| `OMP_PROC_BIND=true` | 1 |
| `OMP_PLACES=cores` | 1 |
| `OMP_PLACES=threads` | 1 |
| `OMP_PROC_BIND=close OMP_PLACES=cores` | 1 |

Consequences at `OMP_NUM_THREADS=16`:

| kernel | unbound | bound |
| --- | --- | --- |
| `dgemm` n=2000 | **807 GFLOP/s** | 106 GFLOP/s |
| `dsygvd` n=2432 (`diagonalize_fock_generalized`) | **0.52 s** | 9.7 s |

Note the eigensolver is *worse than single-threaded* when bound (1 thread
unbound is 2.09 s): the affinity settings both disable OpenBLAS's parallelism
and add OpenMP overhead inside LAPACK's divide-and-conquer recursion.

Thread count is not monotonic even unbound — `dsygvd` n=2432 is 2.09 s at 1,
0.62 s at 8, **0.52 s at 16**, and 2.0–4.1 s at 32. Sweep, do not assume.

## Why the tree had it backwards

`cpu/device/DeviceRuntime::configure_threads` used to *print a warning
recommending* `OMP_PROC_BIND=close OMP_PLACES=cores`. That advice is right for
ATV's own OpenMP loops (pair styles, neighbour build, the SK and γ walks, which
are memory-bound and want pinned threads) and wrong for anything whose cost is
BLAS3 or LAPACK. It now reports the one unambiguous failure instead: several
OpenMP threads while `openblas_get_num_threads()` says 1.

## Standing rule

- **Never set `OMP_PROC_BIND` or `OMP_PLACES` in a benchmark harness that times
  a BLAS-bound path** — SCF, GPW, plane-wave, and the DFTB SCC (one dense
  generalized eigenproblem per iteration).
- Classical-MD harnesses may still bind; that is where the advice came from.
- When a fair wall is recorded, the environment goes in the `REFERENCE.md`
  alongside the number, and "bound" now has to be justified rather than
  assumed.

## Fixed

| Path | What |
| ---- | ---- |
| `src/cpu/device/device_runtime.cpp` | advice replaced by a BLAS-thread-count diagnostic |
| `benchmarks/scf/validation/ri_speed/run.sh` | `unset OMP_PROC_BIND OMP_PLACES` |
| `benchmarks/semiempirical/dftb_peg_liquid/run.sh` | unbound by construction |

## Suspect, not yet re-measured

Every wall-clock ratio below was taken with `OMP_PROC_BIND=close`, i.e. with
ATV's BLAS on one thread while the oracle ran however it ran. The **physics
pins in these files are unaffected** — only the perf columns are.

- `benchmarks/scf/validation/ri_speed/README.md` — `ours-RI-JK 1.617 s` vs
  `PySCF-DF 0.104 s`
- `.claude/notes/gpw-perf-baseline.md` — the whole CP2K ratio table, including
  the `gpw-multigrid-kpoints` 2.33× that is currently blocked as
  `mg-main-structural`
- `regressions/scf/{uks-hessian-o2-sto3g,spin-flip-tda-o2,b2plyp-h2o,
  sf-tddft-o2,roks-grad-o2-sto3g}/REFERENCE.md`
- `regressions/gpw/{gpw-optb88-forces,gpw-vdw-uks-multigrid,gpw-stress-water,
  gpw-forces-water,gpw-variable-cell-molopt}/REFERENCE.md`

Re-measuring these is its own slug. Do not quote the existing ratios as
evidence of an ATV performance gap until it is done.

---

# 2. An OpenMP-flavour OpenBLAS deadlocks when `dlopen`ed into CPython

The Python extension **hangs forever** in threaded GEMM if it resolves to an
OpenBLAS built `USE_OPENMP`. Not slow — hung: every worker spins in
`inner_thread` and never returns.

## Minimal reproducer

No Atomiverse, no numpy:

```python
import ctypes
ob = ctypes.CDLL("<prefix>/openmp/lp64/lib/libopenblas.so.0", mode=ctypes.RTLD_GLOBAL)
# one dgemm at 1088x1088x256, "T","N"  ->  never returns
```

The same library, same call, in a statically linked executable runs at
415 GFLOP/s. That is why every C++ regression and benchmark is unaffected and
only the wheel sees it.

Ruled out along the way, so nobody re-checks them: numpy's own bundled BLAS
(its threads sit idle in `pthread_cond_wait` throughout), import order, and
`OPENBLAS_NUM_THREADS`. The pthread and sequential flavours of the *same*
OpenBLAS 0.3.32 install run the same GEMM in 0.05 s / 0.03 s.

## Why `-DBLAS_LIBRARIES` does not fix it

All three flavours carry the **same soname**, `libopenblas.so.0`. What wins at
run time is the first directory in the extension's `DT_RPATH` that contains
one — and CMake builds that RPATH from the compiler's implicit link path,
i.e. from **`LIBRARY_PATH`**, which the `buildenv-gcc` module points at the
OpenMP flavour. So `-DBLAS_LIBRARIES` selects the link-time library while the
runtime one stays whatever the module chose, and `LD_LIBRARY_PATH` cannot
override it because `DT_RPATH` is searched first.

`BUILD_RPATH` does not help either — CMake appends it *after* the implicit
directories, which is the wrong end.

## The fix: select the flavour with `LIBRARY_PATH`

On this toolchain `LIBRARY_PATH` *is* the flavour selector. Put the
dlopen-safe flavour first when configuring the Python build:

```bash
BL=/software/.../OpenBLAS/0.3.32/g14/hpc1/pthread/lp64/lib
export LIBRARY_PATH="$BL:$LIBRARY_PATH"
cmake -B build-py -DATV_BUILD_PYTHON=ON       -DBLAS_LIBRARIES=$BL/libopenblas.so -DLAPACK_LIBRARIES=$BL/libopenblas.so ...
ldd build-py/_atomiverse*.so | grep openblas   # must NOT say /openmp/
```

`CMakeLists.txt` warns at configure time when `ATV_BUILD_PYTHON` is on and the
selected BLAS links libgomp/libomp, and pins the linked BLAS's directory into
the extension's rpath. Neither can outrank `LIBRARY_PATH`, so the check is a
tripwire, not a guarantee — **verify with `ldd`.**

## Verified

1088-atom PEG liquid through the Python surface, pthread flavour: total energy
2.25e-8 eV/atom and forces 8.1e-6 eV/Å MAE against DFTB+ 25.1 — the same
numbers the C++ path gives. Wall is ~1.5x the C++ path (pthread BLAS is slower
than the OpenMP one, plus the molpy frame conversion), which is a cost, not a
pathology.

---

# 3. Login-node CPU affinity moves under you

`sched_getaffinity` on the `arrhenius` login node returned **48** cores early
in one session and **4** later the same day, with no action from the user.
A 1088-atom DFTB single point that took 8.3 s on 16 threads took 434 s on the
4-core allocation — same binary, same input.

**Never record a wall-clock number from a login node without printing
`len(os.sched_getaffinity(0))` next to it**, and prefer a batch allocation for
anything that goes into a `REFERENCE.md` or a benchmark README.

Related: [[es-optimization-order]], [[gpw-perf-baseline]].
