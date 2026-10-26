# Automatic CPU HF: 2026-10-26

CPU molecular exact HF now exposes one automatic J/K engine. It caches s8
integrals in the working AO basis when memory permits; otherwise it contracts
screened density differences. The public `integral_mode` and `compute_eri`
switches and separate direct/stored benchmark participants are removed.
RHF/UHF/ROHF/GHF share this engine, including nonsymmetric GHF spin blocks.
Post-HF consumers materialize their required dense integrals after SCF.

The reference strategy is PySCF 2.14.0's
[`RHF.get_jk`, `_is_mem_enough`, and `get_veff`](https://pyscf.org/_modules/pyscf/scf/hf.html),
with canonical AO-pair row scheduling following
[`nr_incore.c`](https://github.com/pyscf/pyscf/blob/v2.14.0/pyscf/lib/vhf/nr_incore.c).
ATV retains its native integral primitives and SCF driver; these measurements
do not establish identical solver stopping criteria or performance on other systems.

## Public Python performance

Same node/allocation: **n229, AMD EPYC 9755**, Slurm **4466409**, eight allocated
logical CPUs. Python 3.13.13, PySCF 2.14.0, GCC 14.3.0. ATV links OpenBLAS
0.3.32 pthread; the bundles record every engine's actual BLAS/OpenMP libraries,
thread counts, binary hashes, source hashes, affinity and resolved controls.
`OMP_WAIT_POLICY=PASSIVE`; no `OMP_PROC_BIND` or `OMP_PLACES` setting.

48 atoms, 112 STO-3G AOs; identical water-chain coordinates, Hcore guess,
1e-10 energy tolerance, 1e-8 gradient tolerance, DIIS space 8/start cycle 1.
One cold observation and five warm observations per engine; each warm observation
has one fresh warmup solve. Every timed call creates fresh scientific state.

| Threads | ATV median | PySCF median | ATV warm range | PySCF warm range |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.98765 s | 1.18167 s | 0.914–1.001 s | 1.090–1.275 s |
| 8 | 0.57688 s | 1.91791 s | 0.537–0.908 s | 1.060–2.514 s |

ATV takes 12 iterations and PySCF 10. Maximum cross-engine energy difference
is below 2e-12 Ha. ATV's s8 cache is **160,199,648 bytes**, compared with
1,258,815,488 bytes for the previous full 112^4 double tensor.
The eight-thread comparison has a thread-resource mismatch: PySCF reports one
single-threaded OpenBLAS alongside its eight-thread pools. the timing
values, substantial variability and different BLAS builds are visible in the
raw samples but do not establish a qualified parallel speedup. These numbers must not be compared directly with the earlier
GH200/M4 wall times or generalized to all HF workloads.

- [Single-thread bundle](public-t1/results.json), [table](public-t1/matrix.html).
- [Eight-thread bundle](public-t8/results.json), [table](public-t8/matrix.html).
- [Machine-readable summary](summary.json).

Both selected CPU campaigns pass the numerical checks. The framework flags
the eight-thread observations for a thread-resource mismatch, so their timings
are diagnostic and it does not qualify an eight-thread performance ratio.
CUDA and gpu4pyscf are optional and unavailable in these CPU campaigns; CUDA
was checked separately below. Source files were stable during each run.
The binaries were built locally, but these bundles have **no formal build
attestation** (`build_verified=false`). Coverage is explicitly limited to the
selected chain row, not the entire HF roadmap.

The installed molci exporter lacks `Manifest.observation_id`; snapshot export
failed after measurement. Local tables were regenerated with the framework's
`report(..., snapshot=False)`. No source, timing, numerical gate or measured
result was changed to accommodate that exporter mismatch.

## Correctness and combinations

- **62 C++ unit tests passed**, including independent dense-reference J/K,
  nonsymmetric densities, spherical d shells, four-thread contraction,
  incremental spin-channel histories and geometry resets. [Log](checks/pthread-unit.log).
- **27 Python tests passed**, four CUDA-specific tests skipped in the CPU
  extension. [Log](checks/pthread-python-tests.log). CLI contracts: 46 passed;
  benchmark contracts: 34 passed.
- **16 live PySCF comparisons passed**: RHF water, UHF/ROHF O2, STO-3G and
  spherical def2-SVP, plus GHF H2/noncollinear H3; each with cached and zero-budget
  execution. Maximum energy error 5.7e-13 Ha; maximum analytic force component
  error 1.7e-9 Ha/Bohr. GHF energy only, since PySCF lacks its gradient oracle.
  [Results and phase logs](numerical/results.json).
- MP2's delayed dense-integral materialization was checked with the existing
  water example: correlation energy -0.03551367905 Ha agrees with live PySCF.
  [Native](checks/mp2-smoke.log), [oracle](checks/mp2-oracle.log).
- Minimal CPU **HF+MPI**, with other product modules disabled, built and ran
  RHF/UHF/ROHF/GHF using both memory budgets. Eight comparisons between local
  execution and two ranks × two OpenMP threads agree within 8.9e-16 Ha.
  The `checks/mpi-local-*` and `checks/mpi-two-*` logs preserve each run.
- Minimal **HF+CUDA**, both with MPI OFF and ON, built and ran on GH200.
  CPU/CUDA water energies differ by at most 1.1e-12 Ha:
  [MPI OFF](checks/gpu-minimal-OFF-smoke.log), [MPI ON](checks/gpu-minimal-ON-smoke.log).
  Two-rank CPU GHF also ran in the CUDA+MPI build:
  [H3](checks/gpu-minimal-mpi-ghf.log).
  This verifies feature composition; CUDA HF itself remains a local GPU solver.

MPI is opt-in at build time (`ATV_USE_MPI=ON`) and solve time (`method.mpi=true`).
All ranks must supply the same complete frame and controls after initializing
MPI. The s8 cache is replicated; J/K contraction is distributed and reduced.
OpenMP is required for HF and uses its runtime thread count by default.

## Phase diagnostics and reproduction

The separate C++ diagnostic records integral-cache construction, each J/K
contraction and the SCF loop: [one thread](phases-t1/results.json),
[eight threads](phases-t8/results.json). These are different entry boundaries
from the public Python table and are not substituted for its timings.
The one-thread integral build remains the largest phase; native primitive
evaluation and the remaining SCF iteration gap are still optimization targets.

```sh
python -m atomiverse config --hf                 # CPU, OpenMP; MPI/CUDA off
python -m atomiverse config --hf --mpi
python -m atomiverse config --hf --cuda
python -m atomiverse config --hf --cuda --mpi

# After compiling/installing, on a compute allocation with the documented BLAS:
OMP_WAIT_POLICY=PASSIVE python -m benchmarks run --method rhf \
  --tier performance --system water_chain16 --threads 8 --repeats 5 --warmup 1 \
  --out /path/to/fresh/results
```

The standalone diagnostic is `benchmarks/scf/profile_cpu.py`; build
`bench_hf_cpu` with `ATV_BUILD_BENCHMARKS=ON`. Its `--validate` option exercises
the 16 numerical comparisons without adding SCF-loop workloads to CTest.
