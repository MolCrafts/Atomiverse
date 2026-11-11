# scf/solvation

One case; every section below is run by `atv_regressions --case scf/solvation` and keeps its own oracle record.


## H₂O RHF/6-31G* C-PCM electrostatic free energy

### System

- Geometry (Å): O (0, 0, 0.1173), H (0, ±0.7572, −0.4692) — same as other
  H₂O regressions in this tree.
- Method: RHF / 6-31G* (`Pople631Gs`).
- Solvent: water-like dielectric **ε = 78.4**, cavity = **Bondi × 1.2**,
  dielectric scaling **x = 0** (C-PCM, Barone–Cossi), Lebedev **302** points
  per sphere with hard bury-delete of buried tesserae.
- Quantity: electrostatic free energy **ΔG_el = ½ Σ_k q_k V_k** (Hartree →
  kcal/mol with 627.5094740631). Non-electrostatic terms (cavitation,
  dispersion, repulsion) are **not** included.

### External pin (PySCF 2.14.0)

```python
from pyscf import gto, scf
mol = gto.M(
    atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
    basis="6-31g*", unit="Angstrom", verbose=0)
mf = scf.RHF(mol).PCM()
mf.with_solvent.method = "C-PCM"
mf.with_solvent.eps = 78.4
mf.with_solvent.vdw_scale = 1.2
mf.with_solvent.lebedev_order = 29   # 302 grids
mf.conv_tol = 1e-10
mf.kernel()
print(mf.with_solvent.e)             # Hartree
```

| Quantity | Value |
|---|---|
| PySCF version | **2.14.0** |
| `mf.with_solvent.e` (ΔG_el) | **−0.01268465021284 Ha** |
| ΔG_el [kcal/mol] | **−7.95973818** |
| E_gas (RHF/6-31G*) | −76.00910803237647 Ha |
| E_C-PCM total | −76.02066073847446 Ha |

Notes on model differences (honest):

- Both use **modified Bondi H = 1.1 Å**, scale 1.2, ε = 78.4, x = 0 (C-PCM),
  Lebedev 302, and the Lange–Herbert Gaussian-smeared S matrix (erf(ξr)/r).
- PySCF PCM additionally applies **SWIG** surface switching
  (Lange & Herbert, JCP 133, 244111 (2010)); Atomiverse uses **hard
  bury-delete** (point inside any other sphere is discarded). The cavity
  surface is therefore discontinuous in nuclear coordinates — intentional.
- AO potentials in Atomiverse use the point-charge nuclear-attraction
  primitive; PySCF uses Gaussian-charge `int3c2e`. For H₂O the residual
  is well inside the 0.1 kcal/mol gate.

Measured Atomiverse ΔG_el on this geometry: **−8.018 kcal/mol**
(|Δ − pin| ≈ 0.059 kcal/mol).

### Acceptance

- `|ΔG_el(Atomiverse) − pin| ≤ 0.1 kcal/mol`
- Pin lies in **[−8, −6] kcal/mol**
- Total C-PCM energy strictly below gas phase; SCF converges ≤ 60 s


## H₂O RHF/6-31G* IEF-PCM electrostatic free energy

### System

- Geometry (Å): O (0, 0, 0.1173), H (0, ±0.7572, −0.4692) — same as other
  H₂O regressions in this tree.
- Method: RHF / 6-31G* (`Pople631Gs`, Cartesian d shells).
- Solvent: water-like dielectric **ε = 78.4**, cavity = **Bondi × 1.2**,
  model = **IEF-PCM** (Cancès–Mennucci–Tomasi; PySCF `method='IEF-PCM'`),
  Lebedev **302** points per sphere. Atomiverse uses hard bury-delete of
  buried tesserae (PySCF default is SWIG surface switching).
- Quantity: electrostatic free energy **ΔG_el = ½ Σ_k q_k V_k** (Hartree).
  Non-electrostatic terms (cavitation, dispersion, repulsion) are **not**
  included. Observable `e_solv` ≡ `e_solvent()` ≡ ΔG_el.

### External pin (PySCF 2.14.0)

Offline oracle (never imported by ctest):

```bash
source tools/setup_pyscf_env.sh
python tools/pyscf_oracle/pin_scf.py rhf --geom water --basis '6-31g*' \
  --cart --pcm --pcm-method 'IEF-PCM' --pcm-eps 78.4 --pcm-scale 1.2 \
  --pcm-lebedev-order 29 --conv-tol 1e-10
```

Equivalent script:

```python
from pyscf import gto, scf
mol = gto.M(
    atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
    basis="6-31g*", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol).PCM()
mf.with_solvent.method = "IEF-PCM"
mf.with_solvent.eps = 78.4
mf.with_solvent.vdw_scale = 1.2
mf.with_solvent.lebedev_order = 29   # 302 grids
mf.conv_tol = 1e-10
mf.kernel()
print(mf.with_solvent.e)             # Hartree ΔG_el
```

| Quantity | Value |
|---|---|
| PySCF version | **2.14.0** |
| `mf.with_solvent.e` (ΔG_el / e_solv) | **−0.01265803490853 Ha** |
| ΔG_el [kcal/mol] | **−7.94303683** |
| E_gas (RHF/6-31G* cart) | −76.01050498825606 Ha |
| E_IEF-PCM total | −76.02202516714988 Ha |

Notes on model differences (honest):

- Both use **modified Bondi H = 1.1 Å**, scale 1.2, ε = 78.4,
  f = (ε−1)/(ε+1), Lebedev 302, and the Lange–Herbert Gaussian-smeared S/D
  matrices (erf(ξr)/r).
- PySCF PCM additionally applies **SWIG** surface switching
  (Lange & Herbert, JCP 133, 244111 (2010)); Atomiverse uses **hard
  bury-delete** (point inside any other sphere is discarded). The cavity
  surface is therefore discontinuous in nuclear coordinates — intentional.
- AO potentials in Atomiverse use the point-charge nuclear-attraction
  primitive; PySCF uses Gaussian-charge `int3c2e`. Residual vs pin is gated
  at **1e-4 Ha** (~0.06 kcal/mol), consistent with the C-PCM twin
  (`h2o_rhf_631gs_cpcm`).

### Acceptance

- `|e_solv(Atomiverse) − pin| ≤ 1e-4 Ha` (and pin is the literature oracle)
- `e_solv < 0` (stabilizing electrostatic free energy)
- Total IEF-PCM energy strictly below gas phase; SCF converges ≤ 60 s

Public API:

```cpp
cpu::scf::RHF mf{cpu::scf::BasisSet::Pople631Gs};
cpu::scf::WithSolvent<cpu::scf::RHF> sol{
    std::move(mf), cpu::scf::SolventModel::IEFPCM, /*eps=*/78.4};
sol.radii_scale = 1.2;
sol.run(frame);
double e_solv = sol.e_solvent(); // [Hartree]
```


## H₂O RHF/6-31G* C-PCM analytic free-energy forces

### System

- Geometry (Å): O (0, 0, 0.1173), H (0, ±0.7572, −0.4692) — same as
  `h2o_rhf_631gs_cpcm` and other H₂O regressions.
- Method: RHF / 6-31G* (`Pople631Gs`, Cartesian d).
- Solvent: water-like **ε = 78.4**, cavity = **Bondi × 1.2**,
  dielectric scaling **x = 0** (C-PCM), Lebedev **302** points with
  **hard bury-delete** of buried tesserae.
- Quantity: nuclear gradient of the electrostatic free energy
  \(G = E_\mathrm{HF}[P] + \Delta G_\mathrm{el}[P,R]\) [Ha/Bohr].
  Non-electrostatic terms are **not** included.

### External pin (PySCF 2.14.0)

Offline only — never imported by CMake/ctest:

```bash
## tools/setup_pyscf_env.sh first if needed
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

#### Model differences (honest)

| Piece | Atomiverse | PySCF PCM |
|---|---|---|
| Surface | hard bury-delete | SWIG (Lange–Herbert) |
| AO potential at tessera | point-charge `nuclear_block` | Gaussian-charge `int3c2e` |
| S matrix | Lange–Herbert erf(ξr)/r (same XI table) | same + SWIG switch |

Energy residual is already documented in `h2o_rhf_631gs_cpcm` (~0.06 kcal/mol).
**Forces therefore do not match PySCF at 1e-5 Ha/Bohr** — the literature pin
is retained for context only.

### Acceptance (this regression)

Primary oracle = **finite difference of the Atomiverse C-PCM free energy**
(same free-energy functional the analytic path differentiates):

\[
\frac{\partial G}{\partial R_{A,d}}
  \approx \frac{G(R+h\hat e_d)-G(R-h\hat e_d)}{2h},\quad h=10^{-3}\ \mathrm{Bohr}.
\]

- Element-wise \(|g_\mathrm{analytic}-g_\mathrm{FD}| \le 10^{-5}\) Ha/Bohr
- SCF converges; total wall ≤ 60 s on one CPU

### Physics / implementation

Free energy \(G = E_\mathrm{HF}^\mathrm{vac}[P] + \Delta G_\mathrm{el}\) with
\(\Delta G_\mathrm{el}=\tfrac12 q\cdot V\), \(S q = -f(\varepsilon) V\).
At SCF the orbital response cancels under the solvent Fock Brillouin condition;
`ScfGradient` therefore adds:

1. vacuum HF nuclear gradient on the solvent-polarized \(P,W\);
2. \(\partial\Delta G_\mathrm{el}/\partial R|_P\) by central FD of a rebuilt
   cavity + unit-charge AO potentials at fixed \(P\) (h = 10⁻⁴ Bohr).

Public API: `WithSolvent<RHF>{std::move(rhf), SolventModel::CPCM, eps}`
(knobs `radii_scale`, `x`) then `sol.forces(frame)` — no `*Config` bag.

Refs: Barone & Cossi, *J. Phys. Chem. A* **102**, 1995 (1998),
DOI 10.1021/jp9716997; Lange & Herbert, *J. Chem. Phys.* **133**, 244111 (2010).
