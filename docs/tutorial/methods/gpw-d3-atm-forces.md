# GPW DFT-D3 three-body ATM (C9) analytic ionic forces

*(Technical-report page; CP2K capability loop.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d3-atm-forces` |
| Regression | `regressions/gpw/gpw-d3-atm-forces/` |
| Public API | `cpu::gpw::GPWEvaluator{d3 = true, d3_atm = true, compute_forces = true}` |
| Roadmap / queue | `docs/roadmap/dft.md` §8 Dispersion; queue wave 14 |

## Summary

The **Axilrod–Teller–Muto three-body C9** term of DFT-D3 (CP2K
`CALCULATE_C9_TERM .TRUE.`) contributes its **analytic gradient** to the GPW
ionic forces: the explicit-geometry legs of every ABC triple *and* the
coordination-number chain \(\partial C_6/\partial \mathrm{CN}\) folded into
`dEdCN`. The energy landed with `gpw-d3-atm`; this capability pins the
**force**, in both boundary conditions:

- **isolated** water dimer (12 Å, MT) — home-cell triples;
- **periodic 6 Å cell** — the lattice-image triple classes of CP2K's
  `qs_dispersion_d3` `doabc` (AB home + C off-home; B and C off-home with
  `fac0 = 1/3`), which only a compressed periodic cell makes measurable.

## Why the pin is a difference of two CP2K runs

D3 never enters the Kohn–Sham potential, so per boundary condition two CP2K
`ENERGY_FORCE` runs differing only in `CALCULATE_C9_TERM` share the same
converged density and

\[
F_A(\mathrm{C9}) - F_A(\mathrm{two\text{-}body}) = -\nabla_A E_{\mathrm{ATM}} .
\]

The electronic force cancels to ~2e−15 Ha/Bohr (the symmetry-zero
y-components of the printed tables differ only in the 15th digit), leaving the
pure ATM gradient at 4.6e−7 (iso) / 6.9e−7 (per) Ha/Bohr — ~500× above the
`FORCES|` print resolution.

## Measured (2026-07-31)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| Iso / per \(E_{\mathrm{disp}}\)(ATM) vs CP2K | 1e−9 Ha | 6.4e−11 / 2.8e−10 |
| Iso ATM force vs CP2K diff | 1e−8 Ha/Bohr | **5.0e−10** (signal 4.6e−7) |
| Per ATM force vs CP2K diff | 1e−8 Ha/Bohr | **9.8e−10** (signal 6.9e−7) |
| Product wiring | 1e−7 Ha/Bohr | 2.2e−17 |
| Unit FD iso / per | 1e−7 Ha/Bohr | 4.1e−14 / 2.2e−13 |
| Wall | ≤ 60 s | 34.5 s |

## Performance

Twin walls from `REFERENCE.md` (CP2K TIMING `CP2K` TOTAL TIME, `mpirun -np 2`,
`OMP_NUM_THREADS=2`); ATV wall is the full regression (iso + per SCF + forces).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| water dimer iso 12 Å + per 6 Å, PBE-D3(ATM) forces | 10.5 | 34.5 | 3.3 | ✓ | ATV single run covers both cells; CP2K wall is the dominant iso member |

## Two defects this pin exposed

1. **CASE 3 double-count** — the triple loop's AB pairs must replicate CP2K's
   *half* neighbor list (`iatom ≤ jatom`, one orientation per off-home image),
   as the two-body loop already did; both orientations give every lattice-image
   triple class 6 anchored representatives against `fac0 = 1/3`, inflating the
   periodic ATM energy 8× (+6.25e−6 vs +7.85e−7 Ha) on this twin.
2. **pbc anchoring** — CP2K's `doabc` runs on `pbc()`-wrapped coordinates, and
   the hard `r < r_cut` leg gates over a finite image window make the per-atom
   force split depend on the wrap while the energy is invariant (4e−19): with
   raw coordinates the two molecules' ATM forces came out exactly *swapped*
   relative to CP2K (1.8e−7 err, each value matching the other molecule to
   ~1e−10). `accumulate_atm_c9` now wraps first (`ANINT` parity).

Dual gate: `GpwDftD3Atm.AnalyticForcesMatchFiniteDifference` (isolated) and
`GpwDftD3Atm.PeriodicAtmForcesMatchFiniteDifference` (periodic; FD of the
ATM-only energy difference, so the two-body gradient cancels exactly) in
`tests/cpu/gpw/test_cpu_gpw_dftd3.cpp`.

Not covered here: the ATM **virial** is still zero (`d3_atm` under
`compute_stress` misses the three-body term) — a separate queue row.

## Run it

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_REGRESSIONS=ON \
      -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON -DATV_BUILD_CUDA=OFF
cmake --build build --target reg_gpw-d3-atm-forces -j$(nproc)
./build/regressions/reg_gpw-d3-atm-forces
```

Offline oracle (never run from ctest): CP2K 2026.1 inputs under
`tools/cp2k_oracle/inputs/gpw-d3-atm-forces/`.
