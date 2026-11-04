# nddo_sqm_pbix — AM1 single point, ATV vs AmberTools SQM

Wall clock of **one AM1 single point** on closed-shell PBIX-Br molecules,
Atomiverse CPU against AmberTools SQM on the same deck. Not part of ctest; the
script asserts no timing.

## Systems

| Deck | Atoms | Orbitals | Geometry |
| --- | --- | --- | --- |
| `trimer205.sqm.in` | 205 (C/H/O/F/Br) | 538 | MMFF94s-optimised trimer |
| `chain674.sqm.in` | 674 (C/H/O/F/Br) | 1784 | MMFF94s-relaxed chain (`chain_relaxed`) |

Both are neutral singlets run with `maxcyc=0`, `scfconv=1e-8`. The deck is also
the geometry source for ATV.

## Method

```bash
python benchmarks/semiempirical/nddo_sqm_pbix/run.py --sqm "$AMBERHOME/bin/sqm" \
    --threads 1 2 4 10 --repeats 5 --sqm-repeats 3
python benchmarks/semiempirical/nddo_sqm_pbix/run.py --sqm "$AMBERHOME/bin/sqm" \
    --deck benchmarks/semiempirical/nddo_sqm_pbix/chain674.sqm.in --repeats 3 --sqm-repeats 1
```

- ATV: `atomiverse.single_point(frame, AM1(charge=0))` in a fresh interpreter
  per sample (cold: OpenMP/BLAS start-up and the first build included). This
  entry computes SCF, Mulliken charges and analytic forces.
- SQM: the deck as is (one SCF and its gradient). Its SCF is serial; both
  sides get the same `OMP_NUM_THREADS`. No thread binding.
- Median of the samples. Every row checks ΔHf against SQM's printed value and
  max |Δq| against its 1e-3-rounded charges.
- Tolerances differ: ATV requires max|ΔP| ≤ 1e-8 and |ΔE| ≤ 1e-10 eV; SQM's
  `scfconv=1e-8` is an energy criterion in kcal/mol with a looser density
  criterion. ATV is timed at its tighter defaults. SQM needs 19 (trimer) and
  24 (chain) cycles (`verbosity=2`).

## Results (Apple M4, 4P+6E cores, Accelerate, 2026-11-04)

Final code, quiet machine (load average 2.0–2.4):

| threads | trimer ATV [s] | trimer SQM [s] | ratio | chain ATV [s] | chain SQM [s] | ratio | chain ATV it |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.30 | 1.32 | 0.23 | 5.68 | 31.2 | 0.18 | 25 |
| 2 | 0.23 | 1.31 | 0.18 | 4.34 | 33.3 | 0.13 | 21 |
| 4 | 0.20 | 1.32 | 0.15 | 5.23 | 32.7 | 0.16 | 23 |
| 10 | 0.20 | 1.30 | 0.15 | 5.46 | 33.7 | 0.16 | 23 |

Trimer ATV takes 21 iterations at every thread count. The chain's count depends
on the thread count (different summation order), and is reproducible for a
given count.

History on the trimer (1 / 10 threads; earlier rows ran on a shared, loaded
machine, load average 8–19):

| Stage | ATV [s] | iterations |
| --- | --- | --- |
| Plain Roothaan, scalar density loop | 2.39 / 2.08 | 38 |
| + dsyrk density, DIIS, OpenMP Fock, compensated energy | 0.48 / 0.60 | 21 |
| + two-center cache, BLAS-3 pseudo-diagonalization, threaded dense loops | 0.30 / 0.20 | 21 |

Chain before the last stage: 10.6 s (1 thread) / 13.4 s (10 threads), load ~8.

Observables at every row and stage: ΔHf(ATV − SQM) = −0.188 kcal/mol (trimer),
−0.604 kcal/mol (chain); max |Δq| = 0.0006 e. Against the previous ATV code,
energy changes ≤ 4e-14 relative, charges ≤ 2e-8 e, forces ≤ 6e-7 eV/Å. The
ΔHf offset to SQM is its STO-6G overlap and legacy constants
(`tools/sqm_oracle/README.md`); ATV matches OpenMOPAC.

## What the time goes into, and what changed

1. **Density** P = 2 C_occ C_occᵀ: one BLAS `dsyrk` (it was a scalar triple
   loop, 43 % of the original solve).
2. **Pulay DIIS** on the Fock matrix (residual F P − P F, six vectors):
   38 → 21 iterations. Convergence is judged on the unextrapolated changes.
3. **Compensated energy summation.** Naive summation left ~5e-10 eV of
   round-off at 200 atoms, above the 1e-10 eV tolerance; the iteration count
   then depended on the thread count.
4. **Two-center integrals cached per geometry**, built once per solve with
   OpenMP over atom pairs (each block to its own slot): 5.8 MB for the
   trimer, 64 MB for the chain, 2 GiB budget (beyond it, recomputed per
   pair). The Fock pair loop gathers each pair's 10/10/16 density entries,
   forms the local Fock blocks and scatters them once. Analytic forces still
   differentiate the integrals in their own pass.
5. **Diagonalization.** Measured with Accelerate at n = 1688: `dsyevd` 0.21 s,
   `dsyevr` (all) 0.40 s, `dsyevr` (occupied only) 0.51 s; the new-LAPACK
   symbols and `VECLIB_MAXIMUM_THREADS` change nothing, and Homebrew
   OpenBLAS `dsyevd` is 0.30 s at 1–10 threads. `dsyevd` therefore stays for
   the first iterations. Once max|FP − PF| ≤ 0.05 eV (from iteration 3) the
   SCF uses a **BLAS-3 pseudo-diagonalization**: first-order occupied–virtual
   rotation κ_ia = F_ia/(ε_i − ε_a) of the previous orbitals as matrix
   products, then Cholesky re-orthonormalization (≈ 3 dgemm-equivalents; dgemm
   runs at ~450 GFLOP/s). A gap below 0.5 eV or a failed Cholesky falls back
   to `dsyevd`. Typically five full diagonalizations remain per solve.
6. OpenMP over the remaining n² loops (DIIS unpack/commutator, orbital
   transposes), all with fixed, order-independent reductions.

Peak dense workspace for the chain: about 8 n² doubles (≈ 200 MB) plus six
DIIS vectors (≈ 150 MB) and the 64 MB integral cache.

Still open: past one or two threads the chain is limited by the remaining
`dsyevd` calls and memory-bound n² work, so scaling to 10 threads is flat.
