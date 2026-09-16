# H2 CASCI(2,2)/STO-3G — total energy pin

Closed-shell **CASCI** with a full valence active space on H₂ / STO-3G.
Product API: `RHF::casci(frame, /*ncas=*/2, /*nelecas=*/2)` (drives a fresh
RHF, freezes no core, dense FCI in the 2-orbital active space).

Literature / theory:

- B. O. Roos, *The Complete Active Space Self-Consistent Field Method and its
  Applications in Electronic Structure Calculations*, Adv. Chem. Phys. **69**,
  399 (1987). DOI [10.1002/9780470142943.ch7](https://doi.org/10.1002/9780470142943.ch7)
- On this two-electron / two-orbital foothold, CASCI(2e,2o) is **identical** to
  full CI in the basis (and to CCSD / CISD).

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
| CASCI | `pyscf.mcscf.CASCI(mf, ncas=2, nelecas=2)` |
| AO / MO | 2 AO; 1 occupied, 1 virtual spatial MO; ncore=0 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py casci \
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
mc = mcscf.CASCI(mf, 2, 2); mc.kernel()
print(e_hf, mc.e_tot, mc.e_tot - e_hf)
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) (CASCI) | −0.020524527092077 Ha | **1e-6 Ha** |
| \(E_{\mathrm{tot}}\) (CASCI) | −1.137283834488502 Ha | **1e-6 Ha** (product gate) |

CASCI(2,2) coincides with FCI / CCSD to ~1e-14 Ha on this system.

## Gates (product regression)

- \(|E_{\mathrm{tot,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\) (primary gate)
- \(|E_{\mathrm{corr,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(E_{\mathrm{RHF}}\): \(10^{-6}\) Ha
- `ncore == 0`, `ncas == 2`, `nelecas == 2`

## Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method casci --geom h2 --basis sto-3g --repeat 3 -- \
    ./build/regressions/reg_casci-h2-sto3g
```

| Field | Value |
|---|---|
| Protocol | median of ≥5 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.1427 s |
| Atomiverse median wall | 0.0109 s |
| **ratio_atv_over_pyscf** | **0.076** |
| **within_10x** | **true** |
| Bench date | 2026-07-23, login node, OMP_NUM_THREADS=1 |

H₂/STO-3G is startup-dominated; ATV is faster than PySCF on this foothold
(ratio ≪ 1). No performance optimization pass required.
