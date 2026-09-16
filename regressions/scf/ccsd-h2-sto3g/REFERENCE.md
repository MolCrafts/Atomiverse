# H2 CCSD/STO-3G — correlation + total energy pin

Closed-shell **CCSD** on H₂ / STO-3G. Product API: `RHF::ccsd(frame)`
(drives a fresh RHF, then spin-orbital Gauss–Stanton CCSD).

Literature / theory:

- J. F. Stanton, J. Gauss, J. D. Watts, R. J. Bartlett, *A direct product
  decomposition approach for symmetry exploitation in many-body methods.
  I. Energy calculations*, J. Chem. Phys. **94**, 4334 (1991).
  DOI [10.1063/1.460620](https://doi.org/10.1063/1.460620)
- J. Gauss, J. F. Stanton, *Coupled-cluster calculations of nuclear magnetic
  resonance chemical shifts*, J. Chem. Phys. **103**, 3561 (1995)
  (Table III intermediates).
- For two-electron systems CCSD = FCI (exact in the basis).

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
| CCSD | `pyscf.cc.CCSD(mf)`, `conv_tol=1e-12` |
| AO / MO | 2 AO; 1 occupied, 1 virtual spatial MO |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py ccsd \
    --geom h2 --basis sto-3g --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf, cc
mol = gto.M(atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; e_hf = mf.kernel()
mycc = cc.CCSD(mf); mycc.conv_tol = 1e-12
e_corr, t1, t2 = mycc.kernel()
print(e_hf, e_corr, mycc.e_tot)
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) (CCSD) | −0.020524527092106 Ha | **1e-6 Ha** (product gate) |
| \(E_{\mathrm{tot}}\) (CCSD) | −1.137283834488532 Ha | **1e-6 Ha** (product gate) |

On this minimal basis \(T_1 \approx 0\) by symmetry and CCSD coincides with FCI
to ~1e-11 Ha.

## Gates (product regression)

- \(|E_{\mathrm{corr,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(|E_{\mathrm{tot,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(E_{\mathrm{RHF}}\): \(10^{-6}\) Ha
- Amplitudes report `converged == true`

## Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method ccsd --geom h2 --basis sto-3g --repeat 3 -- \
    ./build/regressions/reg_ccsd-h2-sto3g
```

| Field | Value |
|---|---|
| Protocol | median of ≥3 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.0482 s |
| Atomiverse median wall | 0.0153 s |
| **ratio_atv_over_pyscf** | **0.317** |
| **within_10x** | **true** |
| Bench date | 2026-07-23, login node, OMP_NUM_THREADS=1 |

H₂/STO-3G is startup-dominated; ATV is faster than PySCF on this foothold
(ratio ≪ 1). No performance optimization pass required.
