# H₂O RHF/6-31G* C-PCM analytic free-energy forces

## System

- Geometry (Å): O (0, 0, 0.1173), H (0, ±0.7572, −0.4692) — same as
  `h2o_rhf_631gs_cpcm` and other H₂O regressions.
- Method: RHF / 6-31G* (`Pople631Gs`, Cartesian d).
- Solvent: water-like **ε = 78.4**, cavity = **Bondi × 1.2**,
  dielectric scaling **x = 0** (C-PCM), Lebedev **302** points with
  **hard bury-delete** of buried tesserae.
- Quantity: nuclear gradient of the electrostatic free energy
  \(G = E_\mathrm{HF}[P] + \Delta G_\mathrm{el}[P,R]\) [Ha/Bohr].
  Non-electrostatic terms are **not** included.

## External pin (PySCF 2.14.0)

Offline only — never imported by CMake/ctest:

```bash
# tools/setup_pyscf_env.sh first if needed
python tools/pyscf_oracle/pin_scf.py rhf \
  --geom water --basis '6-31g*' --cart --grad --pcm \
  --pcm-eps 78.4 --pcm-scale 1.2 --pcm-lebedev-order 29 \
  --conv-tol 1e-10
```

| Quantity | Value |
|---|---|
| PySCF version | **2.14.0** |
| `cart` | **True** (Atomiverse 6-31G* default) |
| E_C-PCM total | **−76.02209801044438 Ha** |
| `mf.with_solvent.e` (ΔG_el) | **−0.01274509533852 Ha** |
| ∂G/∂R_O [Ha/Bohr] | (0, 0, **+9.3696733492e-03**) |
| ∂G/∂R_H1 [Ha/Bohr] | (0, **+7.0215556208e-03**, **−4.6848366746e-03**) |
| ∂G/∂R_H2 [Ha/Bohr] | (0, **−7.0215556208e-03**, **−4.6848366746e-03**) |

### Model differences (honest)

| Piece | Atomiverse | PySCF PCM |
|---|---|---|
| Surface | hard bury-delete | SWIG (Lange–Herbert) |
| AO potential at tessera | point-charge `nuclear_block` | Gaussian-charge `int3c2e` |
| S matrix | Lange–Herbert erf(ξr)/r (same XI table) | same + SWIG switch |

Energy residual is already documented in `h2o_rhf_631gs_cpcm` (~0.06 kcal/mol).
**Forces therefore do not match PySCF at 1e-5 Ha/Bohr** — the literature pin
is retained for context only.

## Acceptance (this regression)

Primary oracle = **finite difference of the Atomiverse C-PCM free energy**
(same free-energy functional the analytic path differentiates):

\[
\frac{\partial G}{\partial R_{A,d}}
  \approx \frac{G(R+h\hat e_d)-G(R-h\hat e_d)}{2h},\quad h=10^{-3}\ \mathrm{Bohr}.
\]

- Element-wise \(|g_\mathrm{analytic}-g_\mathrm{FD}| \le 10^{-5}\) Ha/Bohr
- SCF converges; total wall ≤ 60 s on one CPU

## Physics / implementation

Free energy \(G = E_\mathrm{HF}^\mathrm{vac}[P] + \Delta G_\mathrm{el}\) with
\(\Delta G_\mathrm{el}=\tfrac12 q\cdot V\), \(S q = -f(\varepsilon) V\).
At SCF the orbital response cancels under the solvent Fock Brillouin condition;
`ScfGradient` therefore adds:

1. vacuum HF nuclear gradient on the solvent-polarized \(P,W\);
2. \(\partial\Delta G_\mathrm{el}/\partial R|_P\) by central FD of a rebuilt
   cavity + unit-charge AO potentials at fixed \(P\) (h = 10⁻⁴ Bohr).

Public API: `RHF` members `solvent_eps`, `solvent_radii_scale`, `solvent_x`
then `rhf.forces(frame)` — no `*Config` bag.

Refs: Barone & Cossi, *J. Phys. Chem. A* **102**, 1995 (1998),
DOI 10.1021/jp9716997; Lange & Herbert, *J. Chem. Phys.* **133**, 244111 (2010).
