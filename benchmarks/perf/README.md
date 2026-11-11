# Native performance harnesses

These are standalone C++/CUDA executables that call the public Atomiverse API
and print wall times or a JSON report. They aren't registered with CTest and
don't pass or fail on timing. A performance claim is the number a harness
printed on a named node.

All targets are declared in [`CMakeLists.txt`](CMakeLists.txt). They are built
only with `-DATV_BUILD_BENCHMARKS=ON`, and every binary goes to `<build>/bin`:

```sh
cmake -S . -B "$BUILD" -DATV_BUILD_BENCHMARKS=ON ...   # plus the gates below
cmake --build "$BUILD" --target <binary>
"$BUILD/bin/<binary>" ...
```

The *Gate* column lists the CMake options that must be `ON` in addition to
`ATV_BUILD_BENCHMARKS`. Scripts under [`../compare/`](../compare) use several
of these binaries to compare Atomiverse with external engines.

## md

Every md target also needs `ATV_BUILD_MD`, `ATV_BUILD_POTENTIALS` and
`ATV_BUILD_NEIGHBOR`.

| Binary | Source | Gate | Measures | Run |
| --- | --- | --- | --- | --- |
| `bench_lj_cpu` | `md/bench_lj_cpu.cpp` | (md only) | CPU MD timing and initial/one-step force checks. Models: LJ fcc, EAM Cu (fcc) and SW+LJ Si (diamond); orthorhombic or triclinic cell. Runs under MPI when Atomiverse is built with `ATV_USE_MPI`. `eam` and `sw-lj` need `ATV_BUILD_MOLPY`. Driven by `compare/lj-lammps`. | `bench_lj_cpu NSIDE WARM STEPS THREADS OUTPUT_PREFIX [nested] [lj\|eam\|sw-lj] [--tilt XY XZ YZ]` writes `OUTPUT_PREFIX.*` LAMMPS inputs and per-stage dumps. |
| `run_atv_gaff` | `md/run_atv_gaff.cpp` | `ATV_USE_MPI`, `ATV_BUILD_KSPACE` | MPI GAFF MD: `lj/cut/coul/long` + PPPM, Nosé–Hoover NVT/NPT and optional SHAKE on H bonds. Also a single-point force dump. Driven by `compare/gaff-liquid` and `compare/gaff-peo-lammps`. | `mpirun -np N run_atv_gaff --data D [--ff F] [--steps N --warmup N] [--npt] [--shake] [--mesh K]`, or `--single-point --alpha A --mesh K --forces-out FILE`. The full flag list is in the source header. |
| `fd_check` | `md/fd_check.cpp` | `ATV_USE_MPI`, `ATV_BUILD_KSPACE` | Compares the analytic force with a central finite difference for the fused `lj/cut/coul/long` + PPPM pair path, using two opposite charges in a cubic box. | `mpirun -np 1 fd_check` |
| `bench_nonbonded_style` | `md/bench_nonbonded_style.cu` | `ATV_BUILD_CUDA` | Single-GPU LJ NVE on the OpenMM cluster/tile engine. Prints ms/step, M atom·steps/s and energy. Driven by `compare/lj-lammps` and `compare/gaff-peo-openmm`. | `bench_nonbonded_style [N_or_nside=10000] [steps=50] [warmup=10]`. A first argument ≥ 1000 is an atom count; a smaller one is the lattice side. |
| `bench_multi_gpu_force` | `md/bench_multi_gpu_force.cu` | `ATV_BUILD_CUDA` | Two-GPU forward halo plus combined LJ and bonded force evaluation on a bonded grid. Needs ≥ 2 visible GPUs. | `bench_multi_gpu_force [iterations=100] [warmup=10]` |
| `bench_gaff` | `md/bench_gaff.cu` | `ATV_BUILD_CUDA`, `ATV_BUILD_KSPACE`, `ATV_BUILD_EVALUATE` | CUDA GAFF PEO: initial forces, then timed NVE with PME. Optional per-term isolation and an energy-drift series. Set `ATV_BENCH_TIMING=1` for per-stage timing. Driven by `compare/gaff-peo-openmm`. | `bench_gaff INPUT OUTPUT [steps=100] [warmup=10] [all\|nonbonded\|bond\|angle\|dihedral] [samples=3] [drift_every=0]` writes `OUTPUT`, `OUTPUT.final` and `OUTPUT.drift` (the last only when `drift_every` > 0). |
| `md-residency` | `md/residency/run.cu` | `ATV_BUILD_CUDA`, `ATV_BUILD_KSPACE` | Steady-state CUDA MD (LJ NVE, LJ + Nosé–Hoover, LJ NPT, charged LJ + PME) from a melting simple-cubic liquid. Prints `us_per_step` over the steady interval and writes the final state as hexadecimal floats. `--profile` brackets the steady steps with `cudaProfilerStart/Stop` for an Nsight capture. Driven by `md/mps_pack.sh`. | `md-residency --case lj-nve\|lj-nh\|lj-npt\|ion-pme --out FILE [--steps N=400] [--warm N=200] [--side N=16] [--profile]` |

`md/mps_pack.sh OUT BUILD_DIR...` measures MPS packing throughput: K = 1, 2,
4 and 8 concurrent `md-residency` processes on one GPU, for every listed
build tree.

`md/residency/trace.py CAPTURE.sqlite OUT.json --steps N` audits an Nsight
capture of `md-residency --profile` (`nsys profile
--capture-range=cudaProfilerApi`, exported to SQLite). It fails if the stepping
thread synchronizes, allocates or copies device data to pageable memory
before its last launch. `md/residency/compare.py A B` compares two final-state
files bitwise, for example from a build before and after a change.

## minimize

Every minimize target also needs `ATV_BUILD_MINIMIZER`, `ATV_BUILD_POTENTIALS`
and `ATV_BUILD_NEIGHBOR`.

| Binary | Source | Gate | Measures | Run |
| --- | --- | --- | --- | --- |
| `bench_minimize_cpu` | `minimize/bench_minimize_cpu.cpp` | (minimize only) | CPU `Minimize` wall time per iteration for L-BFGS, CG and FIRE, and force-phase launch counts (initial, line-search trial, committed). `rosen` is a chained Rosenbrock with an O(N) force, so its time is optimizer and driver overhead. `lj` is a perturbed fcc Lennard-Jones crystal with a neighbor list. The criterion never converges, so every run uses the full budget unless the optimizer stops itself. Prints one `BENCH` line with the best of `REPEATS` and the final energy. | `bench_minimize_cpu rosen\|lj N_or_NSIDE lbfgs\|cg\|fire ITERS THREADS REPEATS`. For `lj` the size is the fcc side (4·side³ atoms); `THREADS=0` keeps the runtime default. |

## scf

Every scf target also needs `ATV_BUILD_HF`.

| Binary | Source | Gate | Measures | Run |
| --- | --- | --- | --- | --- |
| `bench_hf_cpu` | `scf/bench_hf_cpu.cpp` | (hf only) | CPU RHF/UHF/ROHF/GHF complete calls. Each sample uses a fresh Frame and method and prints one JSON line with energy, iterations, optional forces and seconds. Set `ATV_BENCH_MPI=1` for collective J/K, and `ATV_LOG_LEVEL=debug` for J/K phase logs. | `bench_hf_cpu METHOD BASIS SYSTEM WATERS THREADS REPEATS CACHE_MB FORCES`. `BASIS` is `sto3g` or `def2svp`; `SYSTEM` is `water`, `h2`, `o2` or `h3`; `THREADS=0` keeps the runtime default. |
| `bench_mp2_cpu_vs_cuda` | `scf/bench_mp2_cpu_vs_cuda.cu` | `ATV_BUILD_CUDA` | Complete RHF + canonical MP2 calls on STO-3G water chains, CPU against CUDA, for a first call and repeated calls. | `bench_mp2_cpu_vs_cuda MAX_WATERS[1..32] WARM_REPEATS[1..100] cpu\|cuda\|both [scf_tol=1e-10] [workspace_mib=4096]` |

`scf/profile_hf_cpu.py --binary "$BUILD/bin/bench_hf_cpu" --out DIR [--threads N] [--repeats N] [--validate]`
runs `bench_hf_cpu` beside live PySCF. It writes per-case logs and
`DIR/results.json`, which include the exact-J/K build, J/K and SCF-loop phase
times. The default runs a 16-water STO-3G chain and 4- and 8-water def2-SVP
chains with 4000 MB and 0 MB budgets for both engines. It verifies the
actual cached and streamed paths. The def2-SVP chains also run with
analytic forces under `--validate`. PySCF checkpoint
writes are disabled for this in-memory comparison. It needs NumPy, PySCF and
threadpoolctl. `DIR` must not exist yet;
keep it under `build/` or outside the tree.

## pw

| Binary | Source | Gate | Measures | Run |
| --- | --- | --- | --- | --- |
| `bench_pw_large_cell` | `pw/bench_pw_large_cell.cu` | `ATV_BUILD_PW`, `ATV_BUILD_CUDA` | CPU against CUDA plane-wave SCF wall time on Si supercells: the 8-atom diamond cube replicated `n_rep` times per axis. The timer covers `run(frame)` only. | `bench_pw_large_cell UPF N_REP cpu\|cuda\|both [ecut_ry=15] [k_mesh=1] [tol=1e-6] [max_iter=100]` |

## semiempirical

Every semiempirical target also needs `ATV_BUILD_SEMIEMPIRICAL`. Each one
requires `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. The CUDA targets also
require FP64 storage, work and CUDA precision:

- `ATV_CUDA_PREC_MODE=double`
- `ATV_REAL_STORAGE=double`
- `ATV_REAL_WORK=double`

| Binary | Source | Gate | Measures | Run |
| --- | --- | --- | --- | --- |
| `bench_nddo_allocations` | `semiempirical/nddo_forces/count_allocations.cpp` | (semiempirical only) | C++ `new` counts and requested bytes for NDDO derivative primitives and complete public CPU force calls. | `bench_nddo_allocations OUTPUT.json [AM1\|RM1\|MNDO\|PM3]` |
| `bench_nddo_forces` | `semiempirical/nddo_forces/run.cu` | `ATV_BUILD_CUDA`, `ATV_BUILD_EVALUATE` | AM1/RM1/MNDO/PM3 complete force calls, CPU against CUDA, including a resident Evaluate loop. | `bench_nddo_forces OUTPUT.json [am1\|rm1\|mndo\|pm3\|all] [waters] [warm] [profile] [fixed\|moving\|all]` |
| `bench_nddo_energy` | `semiempirical/nddo_energy/run.cu` | `ATV_BUILD_CUDA`, `ATV_BUILD_EVALUATE` | AM1/RM1 energy and charge calls, CPU against CUDA, including a resident Evaluate loop. | `bench_nddo_energy OUTPUT.json [am1\|rm1\|both] [waters] [warm] [profile]` |

See [`semiempirical/nddo_forces/README.md`](semiempirical/nddo_forces/README.md)
and [`semiempirical/nddo_energy/README.md`](semiempirical/nddo_energy/README.md)
for the timing boundaries, the checks, Nsight capture and the `analyze.py`
trace summaries. `semiempirical/nddo_forces/run.py` times the same CPU force
calls through the Python package.
