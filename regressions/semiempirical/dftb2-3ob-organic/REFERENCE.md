# Reference — CH4 SCC-DFTB2 with static 3ob-3-1 SK

## Oracle

```text
oracle: dftb+
oracle_version: 24.1
```

Offline pin only — ctest never launches DFTB+.

## Method

- **Hamiltonian:** SCC-DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260
  (1998); DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)).
- **SK set:** DFTB.org **3ob-3-1** organic H/C/N/O (Gaus, Goez, Elstner,
  *J. Chem. Theory Comput.* **9**, 338 (2013); DOI
  [10.1021/ct300849w](https://doi.org/10.1021/ct300849w)). Static tables in
  `basis_data/dftb_3ob_tables.h` (CC-BY-SA 4.0 transcription; no runtime `.skf`).
- **Pin Hamiltonian twin:** DFTB+ `SCC = Yes`, **no** `ThirdOrder` (pure DFTB2
  with 3ob SK files). 3ob was trained for DFTB3; this case pins **SK fidelity**
  under DFTB2.

## Geometry

Methane, tetrahedral, \(r(\mathrm{C{-}H}) = 1.09\) Å:

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C | 0 | 0 | 0 |
| H | \(s\) | \(s\) | \(s\) |
| H | \(s\) | \(-s\) | \(-s\) |
| H | \(-s\) | \(s\) | \(-s\) |
| H | \(-s\) | \(-s\) | \(s\) |

with \(s = 1.09 / \sqrt{3}\). Isolated (cubic box 20 Å, molecular path).

## Pin

| Quantity | Value | Units |
|----------|-------|-------|
| Total energy (DFTB+ 24.1) | **−3.2291103259** | Ha |
| Total energy (public ATV) | **−87.86856831** | eV |

Conversion: \(E[\mathrm{eV}] = E[\mathrm{Ha}] \times 27.211386245988\).

DFTB+ detailed components (documentation only):

| Term | Ha | eV |
|------|-----|-----|
| Electronic | −3.1685644194 | −86.2210 |
| SCC | 0.0011039499 | 0.0300 |
| Repulsive | −0.0605459065 | −1.6475 |
| **Total** | **−3.2291103259** | **−87.8686** |

## Gate

- Public API: `cpu::Evaluate` + `cpu::semiempirical::DFTB2` with
  `sk_set = DftbSkSet::ThreeOb`, `with_forces = false`.
- Assert `|E_ATV − (−87.86856831)| ≤ 0.05` eV (SK tables + Elstner
  γ vs DFTB+ poly-interp / γ implementation residual).

## Unmoved by the 2026-07-28 SCC repair — this is the control

`E = −87.84300546 eV` vs the DFTB+ 24.1 pin `−87.86856831 eV`
(|ΔE| = 2.6e-02 eV) is byte-unchanged across the charge-mixing and
double-counting fixes. That is the point: the 3ob path **already** assembled
`E = Tr(ρH⁰) + E² + E_rep` (the correct route) and its SCC **already**
converged, so the only in-tree DFTB2 number with a real DFTB+ oracle behind it
is the one number that must not move — and did not. Only the MioSketch path,
which was on the `E_band − E²` route, moved. The 3ob analytic *gradient* did
change (wrong-sign γ term; see
`regressions/semiempirical/dftb2-analytic-forces-threeob/REFERENCE.md`), but
this case runs with `with_forces = false`.

## Float → double re-emission (reproducibility, not accuracy)

`tools/gen_dftb_3ob_tables.py` now transcribes 3ob-3-1 `.skf` into
`basis_data/dftb_3ob_tables.h` with `Dftb3obSkPoint` stored as eight
`double`s at 17 significant digits (was `float`, 9 digits). This is a
**reproducibility** fix — a generator and a full-precision table — not an
accuracy fix. The float32 storage bound on this Hamiltonian is
≤ 2.03e-06 Ha (5.53e-05 eV) and cannot explain a 9.4e-04 Ha
(~2.6e-02 eV) residual (~470×).

Measured CH₄ energy on the same public `DFTB2` / `DftbSkSet::ThreeOb` /
`DftbBoundary::cluster` launch used by this regression:

| Table | \(E\) [eV] | vs DFTB+ pin −87.86856831 eV |
|-------|------------|------------------------------|
| committed `float` 9-digit | −87.868568739305 | −4.3e-07 eV |
| re-emitted `double` 17-digit | −87.868568232331 | +7.8e-08 eV |

**Movement from this re-emission: +5.07e-07 eV** (float table → double
table), well under the expected 1e-04 eV. The regression golden
(DFTB+ pin, 0.05 eV gate) is unchanged.

The historical ATV snapshot \(E = -87.84300546\) eV in the section above
(|ΔE| = 2.6e-02 eV vs DFTB+) is already stale on current code: even the
committed float tables sit on the DFTB+ pin to ~4e-07 eV. That 2.6e-02 eV
gap is not present on this Hamiltonian today and was not closed by
widening storage.

## Perf (optional, not ctest)

DFTB+ wall for this molecule: ~0.12 s (single core, release). ATV expected
well under 10× (small dense eigenproblem).
