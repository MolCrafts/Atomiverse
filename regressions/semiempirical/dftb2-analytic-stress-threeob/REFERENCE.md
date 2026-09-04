# Reference — DFTB2 **ThreeOb** closed-form cell stress (CH₄ / 3ob-3-1)

oracle: literature
oracle_version: Hourahine et al., J. Chem. Phys. 152, 124101 (2020); Gaus
  et al., J. Chem. Theory Comput. 9, 338 (2013) 3ob-3-1 — pair-virial
  ThreeOb cell stress
family_oracle: dftb+

No DFTB+ run produced an absolute energy or stress for this asymmetric CH₄
geometry. The gate is analytic \(\sigma\) vs strain-FD of the same ATV 3ob
energy, plus the isotropic-trace identity. Do not read `oracle: dftb+` into
those identities.

## Method

SCC-DFTB2 (Elstner et al., Phys. Rev. B **58**, 7260 (1998)) on the static
DFTB.org **3ob-3-1** organic SK tables (`DftbSkSet::ThreeOb`; Gaus, Cui,
Elstner, J. Chem. Theory Comput. **9**, 338 (2013)):

\[
E = \mathrm{Tr}(P H^0)
  + \tfrac12\sum_{AB}\gamma_{AB}\,\Delta q_A\Delta q_B
  + E_{\mathrm{rep}}
\]

Intensive cell stress in the mechanical convention used by DFTB+ and by ATV's
PW/GPW `kStress` (stress theorem: Nielsen & Martin, Phys. Rev. B **32**, 3780
(1985); DFTB+ stress module: Hourahine et al., J. Chem. Phys. **152**, 124101
(2020)):

\[
\sigma_{\alpha\beta} = -\frac{1}{\Omega}\frac{\partial E}{\partial\varepsilon_{\alpha\beta}}
\]

evaluated **closed form** at the SCC density as the **pair virial** of the same
analytic force terms already assembled by `analytic_forces_3ob`:

\[
\frac{\partial E}{\partial\varepsilon_{\alpha\beta}}
  = \sum_{\text{pairs}} \left(\frac{\partial E}{\partial R_\alpha}\right) R_\beta ,
\qquad
\frac{\partial E}{\partial \mathbf R_{AB}}
  = \underbrace{2\,P\!\cdot\!\partial_R H^0}_{\text{Hellmann–Feynman}}
  + \underbrace{2\,(P\,\varepsilon^{\text{shift}}_{AB} - W)\!\cdot\!\partial_R S}_{\text{Pulay}}
  + \underbrace{\Delta q_A\Delta q_B\,\partial_R\gamma_{AB}}_{\text{SCC}}
  + \partial_R E^{\text{rep}}_{AB}
\]

with \(W\) the energy-weighted density matrix. Sign and Voigt order
`(xx, yy, zz, xy, xz, yz)` are **taken from the restricted-MioSketch path
already in tree** (`analytic_forces_restricted`), not assumed; off-diagonals
carry the factor 2 because the affine strain is symmetric
(\(\varepsilon_{\alpha\beta}=\varepsilon_{\beta\alpha}=h\)).

`scc_3ob` is a raw-distance **molecular cluster** sum (no lattice images, no
min-image), so \(E\) depends on geometry *only* through the pair bond vectors
and the identity above is **exact**, not a truncation — which is why the
measured analytic-vs-FD residual is 1e-13 rather than the 1e-10 of the
lattice-summed MioSketch path.

**This replaces a product path.** Before this slug, `with_stress` on ThreeOb ran
**12 full re-SCC** energy evaluations (6 Voigt × ±h) of an affine strain. That
strain-FD path is **deleted** from `src/cpu/semiempirical/dftb2.cpp`; the only
surviving finite difference is the FD oracle inside this regression and inside
`tests/cpu/semiempirical/test_dftb2_3ob.cpp`.

## Hamiltonian / tables

- ATV class: `cpu::semiempirical::DFTB2`, `sk_set = DftbSkSet::ThreeOb`
- SK / Hubbard / repulsive: static `basis_data/dftb_3ob_tables.h` (DFTB.org
  3ob-3-1 H/C/N/O). Same tables as `dftb2-3ob-organic` (DFTB+ 24.1 energy pin)
  and `dftb2-analytic-forces-threeob`.
- Radial SK grid: 0.02 Bohr, **linearly interpolated**.

## Geometry (Å) — asymmetric CH₄

C at the origin; H along the four tetrahedral directions
\(\hat u = (\pm1,\pm1,\pm1)/\sqrt3\) (even parity) with **four distinct** bond
lengths:

| atom | Z | r [Å] | x | y | z |
| ---- | - | ----- | - | - | - |
| C  | 6 | —       |  0.00000000 |  0.00000000 |  0.00000000 |
| H1 | 1 | 1.09540 |  0.63242948 |  0.63242948 |  0.63242948 |
| H2 | 1 | 1.12714 |  0.65075458 | −0.65075458 | −0.65075458 |
| H3 | 1 | 1.06364 | −0.61409284 |  0.61409284 | −0.61409284 |
| H4 | 1 | 1.10598 | −0.63853785 | −0.63853785 |  0.63853785 |

(the regression constructs these as `r_ch[i] * dir[i]`, exactly as tabulated).

Cell: cubic **20 Å** (Ω = 8000 Å³), isolated molecule.
`scc_tol = 1e-10`, `max_scc = 80`, `with_forces = false`, `with_stress = true`.

**Why asymmetric.** Symmetric tetrahedral CH₄ has *identically zero* shear
stress (the four \(R_\alpha R_\beta\) products cancel pairwise), so a symmetric
geometry makes σ_xy/σ_xz/σ_yz a vacuous gate. Four distinct radii break that
cancellation — all six Voigt components are non-zero here.

**Why these radii.** The 3ob SK tables are **linearly interpolated** on a
0.02 Bohr grid, so a finite-difference interval that straddles a grid knot
compares two different interpolant slopes and is not a valid oracle. Each
\(r_{\mathrm{CH}}\) is placed mid-way between knots (knot index
103.500 / 106.499 / 100.499 / 104.500) and the six H–H distances land at index
171.473 / 166.574 / 169.833 / 169.049 / 172.284 / 167.397. The closest approach
to a knot is H2–H3 at 0.049 grid intervals ≈ 1.0e-3 Bohr; with \(h = 10^{-5}\)
the strained displacement is ≤ 3.5e-5 Bohr ≈ 1.7e-3 grid intervals, i.e. a
≈30× margin on every pair.

## Pins (hard-coded gates)

1. **Energy** finite and bound: \(|E| > 10^{-3}\) eV.
2. **Stress** all six Voigt components finite and \(|\sigma|^2 > 10^{-14}\)
   (i.e. the shears are genuinely exercised).
3. **Strain-FD identity** — analytic σ vs central affine strain-FD of the same
   converged 3ob energy (\(h = 10^{-5}\)), **all six** components, within
   **1×10⁻⁸ eV/Å³**.
4. **Isotropic trace identity** — for \(r\to(1+\varepsilon)r\),
   \(\Omega(\varepsilon)=(1+\varepsilon)^3\Omega_0\) so
   \(\mathrm d\Omega/\mathrm d\varepsilon|_0 = 3\Omega_0\) and
   \(\mathrm dE/\mathrm d\varepsilon = \sum_\alpha \partial E/\partial\varepsilon_{\alpha\alpha}
   = -\Omega_0\,\mathrm{tr}\,\sigma\). Hence
   \[
     \tfrac13\mathrm{tr}\,\sigma \;=\; -\frac{\mathrm dE}{\mathrm d\Omega}
       \;=\; -\frac{1}{3\Omega_0}\left.\frac{\mathrm dE}{\mathrm d\varepsilon}\right|_0
   \]
   gated to **1×10⁻⁸ eV/Å³**.

Offline pin note:
`tools/tb_oracle/inputs/dftbplus/dftb2_analytic_stress_threeob_note.md`.

Measured 2026-07-29 (`wf/dftb2-analytic-stress-threeob`):

```text
E = -87.80014496 eV
sigma = (-8.7312820639e-05, -8.7404698573e-05, -8.7216929384e-05,
         -2.8645997634e-05,  2.1439935241e-04, -1.5258617867e-04) eV/A^3

sigma_v0 analytic=-8.7312820639e-05  fd=-8.7312820707e-05  |d|=6.813e-14
sigma_v1 analytic=-8.7404698573e-05  fd=-8.7404698412e-05  |d|=1.609e-13
sigma_v2 analytic=-8.7216929384e-05  fd=-8.7216928968e-05  |d|=4.161e-13
sigma_v3 analytic=-2.8645997634e-05  fd=-2.8645997840e-05  |d|=2.061e-13
sigma_v4 analytic= 2.1439935241e-04  fd= 2.1439935285e-04  |d|=4.401e-13
sigma_v5 analytic=-1.5258617867e-04  fd=-1.5258617916e-04  |d|=4.846e-13

tr(sigma)/3 = -8.7311482865e-05   -dE/dOmega = -8.7311482548e-05  |d|=3.177e-13
```

The gate is set at 1e-8 eV/Å³ — five orders above the measured 1e-13 — so it
catches a formula/sign defect without tracking FD noise. (For scale: the
γ-gradient sign bug closed on 2026-07-28 moved the analogous MioSketch stress
by ~1.3e-4 eV/Å³, i.e. **four orders above this gate**.)

## Tolerance rationale

A converged SCC-DFTB2 gradient is stationary in Δq, so the fixed-Δq analytic
virial is FD-exact up to (a) central-difference truncation \(O(h^2)\) and
(b) SCC-exit noise \(O(\text{scc\_tol}^2)\) — both ≲ 1e-13 eV/Å³ at
\(h=10^{-5}\), `scc_tol = 1e-10`, Ω = 8000 Å³. There is no "charge-response
residual" to absorb; a loose gate here would mask a real defect.

## Gaps (still open)

1. **No absolute DFTB+ σ.** This slug pins an *identity* (analytic virial ≡
   strain-FD of the same functional, plus the isotropic trace identity), not
   a DFTB+ 24.1 number. `dftb_in.hsd` for the twin CH₄/3ob *energy* pin lives
   at `tools/tb_oracle/inputs/dftbplus/ch4_3ob_dftb2/`; adding
   `Analysis { PrintForces = Yes }` there would convert this into an
   absolute σ pin, and that run has not been taken.
2. **No periodic ThreeOb.** `scc_3ob` is a molecular cluster sum — no lattice
   images. The stress is therefore the isolated-molecule virial and the cell
   enters only through Ω. A Γ-point lattice-summed ThreeOb (like the MioSketch
   path) is a separate capability; until it exists, ThreeOb σ is only physical
   for molecules in a defined box.
3. **D3 dispersion is absent from the stress** (inherited gap, shared with
   `dftb2-analytic-stress`): `DFTB2::d3` contributes to E and F but not to
   `kStress` on any path. A D3 virial is still owed.
4. **Open-shell / multi-k stress still missing.** Collinear open shell
   (MioSketch) is still multi-SCC strain-FD (`dftb2-spin-analytic-stress`);
   multi-k throws (`dftb2-multi-k-forces`).

## Perf

`perf_exempt: true` — no DFTB+ absolute-stress twin was run for this
geometry (strain-FD identity only). The analytic path replaces **12**
multi-SCC energy evaluations (6 Voigt × ±h) with **one** post-SCC pair
contraction that is shared with the nuclear-force assembly.
