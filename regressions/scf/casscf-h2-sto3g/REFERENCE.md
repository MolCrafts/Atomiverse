# H2 CASSCF(2,2)/STO-3G — total energy pin

Closed-shell **CASSCF** with a full valence active space on H₂ / STO-3G.
Product API: `RHF::casscf(frame, /*ncas=*/2, /*nelecas=*/2)` (drives a fresh
RHF, freezes no core, dense FCI in the 2-orbital active space; orbital
optimization has no non-redundant pairs when `ncas = n_mo`).

Literature / theory:

- B. O. Roos, *The Complete Active Space Self-Consistent Field Method and its
  Applications in Electronic Structure Calculations*, Adv. Chem. Phys. **69**,
  399 (1987). DOI [10.1002/9780470142943.ch7](https://doi.org/10.1002/9780470142943.ch7)
- On this two-electron / two-orbital foothold, CASSCF(2e,2o) is **identical** to
  CASCI / full CI in the basis (active–active rotations are redundant; no core
  or virtual outside the active window).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂, charge 0, singlet |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — `pin_scf` `h2` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` (s only; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| CASSCF | `pyscf.mcscf.CASSCF(mf, ncas=2, nelecas=2)`, `conv_tol=1e-12` |
| AO / MO | 2 AO; 1 occupied, 1 virtual spatial MO; ncore=0 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py casscf \
    --geom h2 --basis sto-3g --ncas 2 --nelecas 2 --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf, mcscf
mol = gto.M(atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; e_hf = mf.kernel()
mc = mcscf.CASSCF(mf, 2, 2); mc.conv_tol = 1e-12
e_mc = mc.kernel()[0]
print(e_hf, e_mc, e_mc - e_hf, mc.converged)
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) (CASSCF) | −0.020524527092077 Ha | **1e-6 Ha** |
| \(E_{\mathrm{tot}}\) (CASSCF) | −1.137283834488502 Ha | **1e-6 Ha** (product gate) |

CASSCF(2,2) coincides with CASCI / FCI / CCSD to ~1e-14 Ha on this system
(full valence active space; zero non-redundant orbital rotations).

## Gates (product regression)

- \(|E_{\mathrm{tot,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\) (primary gate)
- \(|E_{\mathrm{corr,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(E_{\mathrm{RHF}}\): \(10^{-6}\) Ha
- `ncore == 0`, `ncas == 2`, `nelecas == 2`, `converged == true`
- CASSCF energy equals CASCI on the same reference to \(10^{-10}\) Ha

## Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method casscf --geom h2 --basis sto-3g --repeat 5 -- \
    ./build/regressions/reg_casscf-h2-sto3g
```

| Field | Value |
|---|---|
| Protocol | median of ≥5 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.2112 s |
| Atomiverse median wall | 0.0092 s |
| **ratio_atv_over_pyscf** | **0.0435** |
| **within_10x** | **true** |
| Bench date | 2026-07-23, login node, OMP_NUM_THREADS=1 |

H₂/STO-3G is startup-dominated; ATV is faster than PySCF on this foothold
(ratio ≪ 1). No performance optimization pass required.
