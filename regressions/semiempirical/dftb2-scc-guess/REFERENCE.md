# dftb2-scc-guess — ASPC extrapolation of the SCC charges

```text
oracle: literature
oracle_version: Kolafa ASPC (formula identity); DFTB+ not applicable
family_oracle: dftb+
```

This case has **no absolute oracle pin and needs none**. It gates an
optimisation, and an optimisation has exactly two things to prove: that it
speeds something up, and that it changes nothing. Both are internal
comparisons against the same code with the feature off. DFTB+ is not a twin
here because its own charge restart (`ReadInitialCharges` / `charges.bin`) is a
different scheme with different iteration counts; comparing the two counts
would measure the mixers, not this feature.

## Method

Kolafa's always-stable predictor–corrector (Kolafa, *J. Comput. Chem.* **25**,
335 (2004), DOI [10.1002/jcc.10385](https://doi.org/10.1002/jcc.10385)),
order K = 3, applied to the Mulliken \(\Delta q\) of SCC-DFTB2 (Elstner et al.,
*Phys. Rev. B* **58**, 7260 (1998)). Implementation is the shared
`cpu::electronic::ASPCPredictor` — the same type Gaussian SCF and GPW use on
the density matrix — driven through `DFTB2::predictor(order)`.

The extrapolated \(\Delta q\) seeds the SCC. The residual gate is untouched, so
the converged point cannot move; only the number of iterations to reach it can.

## System

The `dftb2-ewald-gamma` cell: 8 water molecules, cubic 7.0 Å, Γ, mio-1-1,
`boundary = lattice`, `scc_tol = 1e-8`. Polar by construction — extrapolating
\(\Delta q\) is worth nothing on a cell where \(\Delta q \equiv 0\).

16 launches along a smooth scripted displacement,
\(\delta_{i\alpha}(s) = 0.01\,\mathrm{Å} \cdot s \cdot \sin(0.37 i + 1.13
\alpha)\). Scripted rather than driven by `md::MD` so the case builds in the
semiempirical-only configuration (`ATV_BUILD_MD=OFF`); the predictor's contract
is per-launch, which is what this exercises.

## Gates

| Gate | Tolerance | Observed |
| ---- | --------- | -------- |
| **Converged energy unchanged** | 1×10⁻⁶ eV | **2.3×10⁻¹³ eV** |
| SCC iteration speedup | ≥ 1.25× | **1.90×** (188 → 99) |

The energy gate is the load-bearing one: a predictor that moved the answer
would be a bug wearing a speedup's clothes.

The speedup gate sits well under the observed value on purpose. Two things cap
what a guess can buy:

- The first K+1 = 4 launches have no history and run cold.
- Past that the floor is the **mixer's** convergence rate, not the guess. A
  better starting point lowers the initial residual; getting from there to
  `scc_tol` still costs Anderson/Pulay its own iterations. Raising the ceiling
  further means changing the mixer (DFTB+ defaults to Broyden), not the guess.

## Trajectory dependence — read this before quoting 1.90×

The speedup depends on how smooth the charge history is:

| trajectory | iterations off → on | speedup |
| --- | --- | --- |
| smooth scripted displacement (this case) | 188 → 99 | 1.90× |
| `md::MD` + `VelocityVerlet`, 0.2 fs, Maxwell-Boltzmann 300 K, 16 steps | 176 → 119 | 1.48× |
| same, 40 steps | 440 → 306 | 1.44× |

Thermal velocities make \(\Delta q(t)\) less polynomial than a scripted ramp,
so a production NVT run should be expected around **1.4–1.5×**, not 1.9×.

## Re-run

```bash
cmake --build <build> --target run_reg_dftb2-scc-guess
```
