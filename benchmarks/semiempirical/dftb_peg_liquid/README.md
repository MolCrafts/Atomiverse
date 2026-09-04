# dftb_peg_liquid — SCC-DFTB2 single point, ATV vs DFTB+

Fair wall-clock of **one SCC-DFTB2 single point (energy + forces)** on the same
1088-atom PEG liquid cell, same parameter set, same thread count, same SCC
tolerance. Not part of ctest.

## System

HO-(CH₂CH₂O)₂-H, 64 molecules, 1088 atoms (C₂₅₆H₆₄₀O₁₉₂), cubic cell
V = 10463 Å³ (L ≈ 21.8 Å), Γ point, mio-1-1, charge 0, closed shell.
n_ao = 2432 (H: s, C and O: s+p).

Geometry: `.../peo-polar-length/experiments/n-series/dftb_bench/n2/omp16_sp/geo.gen`
(override with `GEO=`). It is the production cell of the PEG/MACE-POLAR
labelling experiment, which is why it is the benchmark: it is the size and the
chemistry the engine actually has to serve.

## Oracle

DFTB+ 25.1 development (commit `a23bfb2`), `nompi` OpenMP build, same
`mio-main/skfiles` tree that ATV's static tables were transcribed from
(byte-identical to `mio-1-1` for every H/C/O pair — verified by `diff`).

DFTB+ numbers come from the experiment's own sweep,
`dftb_bench/n2/bench.json` + `omp16_sp/dftb.stdout`. `SCCTolerance` is DFTB+'s
default **1e-5**; `Analysis { CalculateForces = Yes }`.

## Methodology

- **Matched tolerance.** ATV's `scc_tol` is `max|Δq_out − Δq_in|`; run it at
  1e-5 to compare against DFTB+'s default. Timing an ATV run at 1e-10 against a
  DFTB+ run at 1e-5 is not a comparison — it is 27 iterations against 11.
- **Matched threads.** `OMP_NUM_THREADS` only.
- **No thread binding.** `OMP_PROC_BIND` / `OMP_PLACES` are explicitly unset:
  any OpenMP affinity request drops this OpenBLAS to one thread and costs the
  eigensolver an order of magnitude. See
  [`.claude/notes/openblas-thread-binding.md`](../../../.claude/notes/openblas-thread-binding.md).
- Median of 3, forces on for both sides.
- Default mixer (`mix_beta = 0.3`). This cell converges in 8 iterations at
  `mix_beta = 0.5`, but 0.3 is the default because it is the value that
  converges every in-tree case; benchmarking a per-system tuning would not be
  the shipped configuration.

## Result

Median of 3, `scc_tol = 1e-5`, forces on, no thread binding. ATV needs 10 SCC
iterations at every thread count; DFTB+ needs 11. DFTB+ column from the
experiment's own sweep (`dftb_bench/n2/bench.json`, `gamma_sp_s`).

| threads | ATV [s] | DFTB+ [s] | **T_atv / T_dftb+** |
| --- | --- | --- | --- |
| 4 | 10.78 | 14.83 | **0.73** |
| 16 | 6.79 | 10.34 | **0.66** |
| 32 | 6.78 | 10.89 | **0.62** |
| 64 | 130.70 | 53.73 | 2.43 |

The gate is `T_atv / T_dftb+ ≤ 1.0` at matched threads: met at 4, 16 and 32.

**One asymmetry to keep honest:** the ATV column is a median of 3 runs in one
process, so runs 2–3 do not re-pay OpenMP/OpenBLAS team start-up, while the
DFTB+ column is a single cold run. A cold single ATV run at 16 threads is
**8.34 s**, i.e. ratio **0.81** rather than 0.66. Quote 0.81 when the
comparison has to be conservative; 0.66 is what a production run that evaluates
many frames in one process actually gets.

**64 threads is oversubscription, not scaling** — the node has 48 cores. Both
engines degrade there and ATV degrades harder (its eigensolver is already
non-monotonic past 16: `dsygvd` n=2432 is 0.62 s at 8 threads, 0.52 s at 16,
2.0–4.1 s at 32). Do not read the 64-thread row as a parallel-efficiency
statement about either code; read it as "do not run either code oversubscribed".

Neither engine gains anything from 16 → 32 threads. 16 is the operating point.

Same total energy at every thread count, −35805.6778561470 eV, matching DFTB+
to 2.4×10⁻⁵ eV over 1088 atoms (2.3×10⁻⁸ eV/atom); forces to 2.5×10⁻⁶ eV/Å
MAE. The correctness gates are `regressions/semiempirical/dftb2-ewald-gamma/`.

Per-iteration the gap is wider than the totals suggest: ATV is ~0.51 s per
generalized eigenproblem at n_ao = 2432 against DFTB+'s ~0.9 s (its
`detailed.out` timing table: SCC 9.90 s of 10.30 s over 11 iterations).

## What the cost is made of

Phase breakdown of one 1088-atom single point at `scc_tol = 1e-10`
(27 iterations, 16 threads), from the temporary profiler used during the
optimisation pass:

| phase | s | note |
| --- | --- | --- |
| SCC diagonalisation | 13.74 | 27 × 0.51 s — 82% |
| forces + stress | 0.77 | SK walk + Ewald γ gradient |
| apply shift | 1.02 | |
| density (GEMM) | 0.54 | |
| build γ (Ewald) | 0.05 | built once per geometry |
| build H⁰/S | 0.08 | |
| Mulliken | 0.13 | |

A DFTB single point **is** a stack of dense generalized eigenproblems; nothing
else is close. The corollary for MD is that the lever is the iteration count,
which is what `DFTB2::predictor()` is for.

## Reproduce

```bash
bash -lc 'module load buildenv-gcc/2026.03-mpich
          export OPENBLAS_ROOT="$OPENBLASDIR"
          export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
          BUILD_DIR=$PWD/build-dftb \
            benchmarks/semiempirical/dftb_peg_liquid/run.sh'
```

Needs a build with `-DATV_BUILD_SEMIEMPIRICAL=ON -DATV_BUILD_REGRESSIONS=ON`
(the script reuses `reg_dftb2-ewald-gamma`'s compile/link recipe rather than
hardcoding a toolchain).

DFTB+ side, if it has to be re-measured:

```bash
ATV_DFTBPLUS_HOST=25.1 source tools/tb_oracle/setup_dftbplus_env.sh
cd <a scratch dir with geo.gen and dftb_in.hsd>   # Gamma, CalculateForces
OMP_NUM_THREADS=16 dftb+          # read the running-times table in stdout
```
