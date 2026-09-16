# Reference — isolated H₂O / PBE / DZVP-MOLOPT-GTH (Martyna–Tuckerman)

## Method

Γ-point closed-shell GPW Kohn–Sham DFT (CP2K Quickstep family): contracted
Gaussian basis for the wavefunction, uniform real-space grid + FFT for Hartree
and XC, GTH pseudopotentials. Isolated-molecule electrostatics via the
Martyna–Tuckerman (MT) Poisson solver (`PERIODIC NONE` / `PSOLVER MT` in CP2K;
`nonperiodic = true` on `cpu::gpw::GPWEvaluator`).

Public API: `cpu::gpw::GPWEvaluator` + `cpu::single_point` (Frame in/out).
Kinds use lookup constructors `GaussianBasis(Element, name)` /
`GthPotential(Element, name)` — no public `*Config` bag, no factories.

## System

Isolated water, CP2K getting-started geometry (Å), cubic box \(L = 10\) Å:

| Atom | x | y | z |
| ---- | --: | --: | --: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H | 5.7586 | 5.0000 | 5.5043 |
| H | 4.2414 | 5.0000 | 5.5043 |

- Basis: **DZVP-MOLOPT-GTH** (O, H)
- Pseudopotentials: **GTH-PBE-q6** (O), **GTH-PBE-q1** (H)
- XC: **PBE**
- Density cutoff: **400 Ry**
- Valence electrons: 8 → \(n_\mathrm{occ} = 4\)
- SCF: mix 0.3, max_iter 200, tol \(10^{-7}\)

## Pin (offline CP2K)

**\(E_\mathrm{tot} = -17.21233714458516\) Hartree**

Source: CP2K Quickstep getting-started tutorial input `h2o.inp` (same geometry,
basis/PP family, CUTOFF 400, PERIODIC NONE / PSOLVER MT, PBE). Published energy
decomposition (Hartree units) from that twin:

| Term | CP2K |
| ---- | ---: |
| self energy | −44.34715500 |
| core Hamiltonian | +13.07138998 |
| Hartree | +18.34339895 |
| XC | −4.27997143 |
| **total** | **−17.21233714458516** |

This is an **external-engine oracle**, not a golden snapshot of Atomiverse.
Cross-recorded in-tree by `examples/09_gpw.cpp`, `examples/results/09_gpw.json`,
and `docs/tutorial/ch09-gpw.md`. Historical engine milestone
(`6d7f0955`, 2026-06-18) reproduced this pin to ~0.43 mHa.

Optional twin input (for offline re-pin only; **never** launched by ctest):
`tools/cp2k_oracle/inputs/gpw-water-pbe-mt.inp`.

## Tolerance and rationale

**\(1 \times 10^{-3}\) Ha** absolute on \(E_\mathrm{tot}\).

The known Atomiverse residual on this minimal isolated-water stack is
~\(4.2 \times 10^{-4}\) Ha (ch09 live table: \(E_\mathrm{ATV} \approx
-17.212758\) Ha). A 1 mHa e2e band covers that residual with margin while still
failing gross method/geometry/cutoff mistakes. Bit-parity vs CP2K is a
follow-on (erf-screened MT tuning), not this regression's bar.

## What is checked

1. SCF converges (`eval.last_result().converged`).
2. \(E_\mathrm{tot}\) from `eval.last_energy_hartree()` within 1 mHa of the pin.
3. Finite energy on Frame (`frame.meta.get(kEnergy)` [eV]).

## Atomiverse result (verified 2026-07-23)

| Quantity | Value |
| -------- | ----- |
| \(E_\mathrm{tot}\) | −17.2127580657 Ha |
| \(\lvert\Delta E\rvert\) vs pin | \(4.209 \times 10^{-4}\) Ha |
| SCF iters | 9 |
| Converged | yes |
| Wall (\(T_\mathrm{atv}\), OMP=1, Release) | ~17.7 s |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No CP2K binary / module on the build host at pin time (`cp2k`, `cp2k.psmp`, `cp2k.sopt` not found; `module avail cp2k` empty). Fair wall twin deferred until offline CP2K is available. |
| \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\) | n/a (exempt) |
| `within_10x` | n/a (exempt) |

When a CP2K binary is available, re-run the twin in
`tools/cp2k_oracle/inputs/gpw-water-pbe-mt.inp`, time both single-rank runs on
the same host/core count, and replace this section with measured \(r\).
