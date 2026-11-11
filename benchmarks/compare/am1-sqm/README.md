# AM1 single point: Atomiverse vs AmberTools SQM

`run.py` times **one AM1 single point** on closed-shell PBIX-Br molecules in
Atomiverse's CPU Python API and in AmberTools `sqm`, using the same
input deck. It also prints the observable differences. It is not part of
ctest and asserts nothing about timing.

## Systems

| Deck | Atoms | Orbitals | Geometry |
| --- | --- | --- | --- |
| `trimer205.sqm.in` | 205 (C/H/O/F/Br) | 538 | MMFF94s-optimized trimer |
| `chain674.sqm.in` | 674 (C/H/O/F/Br) | 1784 | MMFF94s-relaxed chain |

Both decks are neutral singlets with `maxcyc=0` and `scfconv=1e-8`. Each deck
is also the geometry source for Atomiverse.

## Prerequisites

- AmberTools `sqm`: pass `--sqm PATH`, set `SQM`, or put `sqm` on `PATH`.
  Without one, the script exits with an error.
- The `atomiverse` Python package with the semiempirical module, plus
  `molpy` and `numpy`, importable by the running interpreter (see
  `docs/user-guide/python.md`).

## Run

```sh
python benchmarks/compare/am1-sqm/run.py --sqm "$AMBERHOME/bin/sqm" \
    --threads 1 2 4 10 --repeats 5 --sqm-repeats 3
python benchmarks/compare/am1-sqm/run.py --sqm "$AMBERHOME/bin/sqm" \
    --deck benchmarks/compare/am1-sqm/chain674.sqm.in --repeats 3 --sqm-repeats 1
```

Each output row gives one thread count: median wall times, the ATV/SQM
ratio, the SCF iteration counts, ΔHf (ATV − SQM) and max |Δq|. The script also
prints the load average before and after. Run on a quiet machine and record
the CPU, the BLAS library and the load average next to any number you quote.

## Method

- **Atomiverse**: `atomiverse.single_point(frame, AM1(charge=0))` runs in a
  fresh interpreter for each sample. That makes the timing cold: OpenMP/BLAS
  start-up and the first build are included. The call computes the SCF,
  Mulliken charges and analytic forces.
- **SQM**: runs the deck as is, one SCF plus its gradient. Its SCF is serial.
- Both sides get the same `OMP_NUM_THREADS`, with no thread binding. Each
  reported time is the median of its samples.
- The convergence criteria differ, and ATV is the tighter one. ATV requires
  max|ΔP| ≤ 1e-8 and |ΔE| ≤ 1e-10 eV. SQM's `scfconv=1e-8` is an energy
  criterion in kcal/mol with a looser density criterion. SQM prints its cycle
  count only at `verbosity=2`; otherwise the column shows `-`.

## Checks

There is no pass/fail threshold here. The observables are reported, not
gated:
- **Charges**: max |Δq| is measured against SQM's charges, which it prints
  rounded to 1e-3 e. So agreement can only be shown at that resolution.
- **Heat of formation**: a small constant ΔHf offset of a fraction of a
  kcal/mol is expected. It comes from SQM's STO-6G overlap and legacy
  constants; ATV matches OpenMOPAC. See `tools/sqm_oracle/README.md`.
- A timing is meaningful only if ΔHf and max |Δq| stay at those levels for
  every thread count.
- ATV's iteration count may vary with the thread count, because the
  summation order changes, but it is reproducible for a given count.

The gated SQM charge parity lives in the Python test suite, not here:
`python/tests/test_semiempirical.py::TestSqmMonomerParity`.
