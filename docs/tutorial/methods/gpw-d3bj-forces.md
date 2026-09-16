# GPW PBE-D3(BJ) analytic ionic forces — periodic water dimer

*(Technical-report page; CP2K capability loop wave 14.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d3bj-forces` |
| Regression | `regressions/gpw/gpw-d3bj-forces/` |
| Public API | `cpu::gpw::GPWEvaluator{d3bj = true, compute_forces = true}` |
| Roadmap / queue | `docs/roadmap/dft.md` §8 Dispersion; queue done 2026-07-26 |

## Summary

Sibling of [gpw-d3-forces](gpw-d3-forces.md): the **rational (Becke–Johnson)**
damping function on a **periodic** cell, so the dispersion lattice sum and its
gradient are both exercised.

\[
E_{\mathrm{disp}} = -\sum_{A<B}\Bigl[\frac{s_6 C_6^{AB}}{R^6+f^6}
  + \frac{s_8 C_8^{AB}}{R^8+f^8}\Bigr],\qquad
f = a_1\sqrt{C_8^{AB}/C_6^{AB}} + a_2
\]

PBE parameters \(s_6=1\), \(s_8=0.7875\), \(a_1=0.4289\), \(a_2=4.4407\) Bohr,
CP2K coordination-number form, pair cutoff `kD3Cp2kPairCutoffBohr`
(= 2 × `R_CUTOFF 20` Å = 75.589 Bohr).

Together the two pages cover both damping functions × both boundary conditions
for the dispersion force.

## Measured (2026-07-26)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs CP2K | 1e−8 Ha | **8.9e−11** |
| \(F_{\mathrm{disp}}\) vs CP2K difference | 5e−6 Ha/Bohr | **1.21e−06** |
| Total \(F\) vs CP2K | 1e−2 Ha/Bohr | **4.30e−03** |
| \(\lvert\sum_A F_A\rvert\) | 3e−3 Ha/Bohr | **1.34e−03** |
| Wiring | 1e−7 Ha/Bohr | **5.9e−12** |
| Wall | ≤ 60 s | **32.2 s** (CP2K 11.9 s → ratio 2.7) |

Unit dual-gate: `GpwForces.AnalyticMatchesFdWithD3Bj` — **2.4e−6 Ha/Bohr** on
cheap MT H₂.

## Reading the tolerances honestly

The total-force comparison is bounded **from below by the oracle**: CP2K's own
`FORCES| Sum` on this twin is 5.43e−3 Ha/Bohr, i.e. its force field is not
translation invariant at `CUTOFF 300`. Atomiverse's own net force is 1.34e−3 —
4× tighter — and both shrink with cutoff. A 1e−3 component gate would be pinning
CP2K's grid noise, so the tight, physically meaningful gates are the dispersion
pins (1e−8 Ha / 5e−6 Ha/Bohr) and the net-force check.

See `regressions/gpw/gpw-d3bj-forces/REFERENCE.md` for the pin tables.
