# The `atomiverse` driver

The compiled `atomiverse` binary is a **LAMMPS-shaped CLI**. Native input
needs no Python:

```bash
atomiverse -in examples/in.lj
atomiverse -in examples/in.lj -log log.atv
```

Python scripts (after a native wheel / `python -m atomiverse install`):

```bash
python -m atomiverse run atv.py        # serial, no MPI
python -m atomiverse atv.py
mpirun -np 128 atomiverse atv.py       # MPI host (ATV_BUILD_LAUNCHER)
srun  -n  128 atomiverse atv.py
```

Under the launcher every rank starts the native `atomiverse` executable.
Rank 0 alone embeds a CPython interpreter and runs your script; ranks
1..N-1 never touch Python — they block in a native worker loop. Each
MD/Minimize driver `run()` (and each AM1 `single_point`) becomes ONE task-granular
in-memory broadcast: the task description plus the Frame go out, every
rank computes natively (MPI domain decomposition for MD and Minimize; full-replica
solves for whole-molecule methods like AM1), and the gathered result is
written back into the same `molpy.Frame` your script holds. There is no
intermediate file and no visible IR.

Semantics to know:

- **The script is unchanged** between hostings; results agree to
  domain-decomposition summation order (the shipped parity regression
  pins LJ MD at ~1e-15 relative and AM1 bitwise).
- **Each `run()` is one stateless task**: a fresh driver is materialized
  per dispatch, so seed-based components (`InitVelocity`) re-fire per
  `run()` — do one `run(frame, n)` per equilibration/production stage,
  not a Python-side per-step loop (that would pay one broadcast+gather
  per iteration anyway).
- **`ev.last_result()` is serial-path only** for AM1: under the launcher
  the solve happens inside per-rank materializers; energy (eV) and
  Mulliken charges (e) come back through the Frame.
- **Minimize returns control metadata separately from state**:
  `iterations`, `converged`, and `stop_reason` are the returned
  `MinimizeResult`; coordinates, forces, energy, box, and topology remain in
  the caller's same `molpy.Frame`.
- Subprocesses spawned by your script (multiprocessing workers, shelled
  `python`) import atomiverse in plain serial mode automatically.

## Which Python does the launcher embed?

Resolved at startup, from the launcher's own environment — never from
the build machine:

1. `ATV_PYTHON=/path/to/python` if set (the explicit override);
2. otherwise the first `python3` on `PATH` (an activated venv or conda
   env, or an Lmod-loaded Python, puts its interpreter first);
3. otherwise `python` on `PATH`;
4. otherwise a fatal error naming this search order.

The resolved interpreter's `MAJOR.MINOR` must match the libpython the
launcher was built against (checked at startup; mismatch is a fatal
error naming both versions). `sys.prefix`/site-packages follow the
resolved interpreter's own semantics — a venv's `pyvenv.cfg` is
honoured, so `import atomiverse` finds the package installed in that
environment.

**libpython visibility:** the binary carries `RPATH $ORIGIN/../lib`
(covers venv/conda layouts where `bin/atomiverse` sits beside
`lib/libpython3.X.so`). With a module-loaded Python (Lmod), the module's
`LD_LIBRARY_PATH` supplies libpython — load the module before running.

## Installing

Two routes; both need MPI at build time:

```bash
# 1. In the wheel (opt-in — an MPI + libpython-linked binary is bound to
#    the environment it was built in; it is NOT manylinux-distributable):
ATV_BUILD_LAUNCHER=ON pip install .        # from python/
# -> the real ELF lands in <env>/bin/atomiverse (no console-script shim)

# 2. Plain CMake:
cmake -B build -DATV_BUILD_LAUNCHER=ON -DATV_BUILD_PYTHON=ON ...
cmake --build build && cmake --install build --prefix <prefix>
# -> <prefix>/bin/atomiverse, plus libatomiverse + headers
```

`ATV_BUILD_LAUNCHER=ON` requires `ATV_USE_MPI=ON` and
`ATV_BUILD_PYTHON=ON`; an incoherent configuration fails at configure
time with a message naming the missing switch.
