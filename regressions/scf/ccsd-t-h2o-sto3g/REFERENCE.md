# H2O CCSD(T)/STO-3G — correlation + total energy pin

Closed-shell **CCSD(T)** on H₂O / STO-3G. Product API: `RHF::ccsd_t(frame)`
(drives a fresh RHF, spin-orbital Gauss–Stanton CCSD, then non-iterative (T)).

Literature / theory:

- K. Raghavachari, G. W. Trucks, J. A. Pople, M. Head-Gordon, *A fifth-order
  perturbation comparison of electron correlation theories*, Chem. Phys. Lett.
  **157**, 479 (1989). DOI [10.1016/S0009-2614(89)87395-6](https://doi.org/10.1016/S0009-2614(89)87395-6)
- J. D. Watts, J. Gauss, R. J. Bartlett, *Coupled-cluster methods with
  noniterative triple excitations for restricted open-shell Hartree–Fock and
  other general single determinant reference functions. Energies and analytical
  gradients*, J. Chem. Phys. **98**, 8718 (1993).
  DOI [10.1063/1.464480](https://doi.org/10.1063/1.464480)
- J. F. Stanton, J. Gauss, J. D. Watts, R. J. Bartlett, *A direct product
  decomposition approach for symmetry exploitation in many-body methods.
  I. Energy calculations*, J. Chem. Phys. **94**, 4334 (1991)
  (CCSD intermediates).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂O, charge 0, singlet |
| Geometry (Å) | O(0,0,0.1173); H(0,±0.7572,−0.4692) — `pin_scf` `water` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` (s/p only; engine-neutral vs spherical for STO-3G) |
| SCF | RHF, `conv_tol=1e-12` |
| CCSD | `pyscf.cc.CCSD(mf)`, `conv_tol=1e-12` |
| (T) | `mycc.ccsd_t()` |
| AO / MO | 7 AO; 5 occupied, 2 virtual spatial MO |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py ccsd-t \
    --geom water --basis sto-3g --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf, cc
mol = gto.M(atom="""
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; e_hf = mf.kernel()
mycc = cc.CCSD(mf); mycc.conv_tol = 1e-12
e_corr_ccsd, t1, t2 = mycc.kernel()
e_t = mycc.ccsd_t()
print(e_hf, e_corr_ccsd, e_t, e_corr_ccsd + e_t, e_hf + e_corr_ccsd + e_t)
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −74.963023138462930 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) (CCSD) | −0.049438563031963 Ha | 1e-6 Ha (diagnostic) |
| \(E_{(T)}\) | −0.000067409682944 Ha | 1e-6 Ha (diagnostic) |
| \(E_{\mathrm{corr}}\) (CCSD(T)) | −0.049505972714907 Ha | **1e-6 Ha** (product gate) |
| \(E_{\mathrm{tot}}\) (CCSD(T)) | −75.012529111177841 Ha | **1e-6 Ha** (product gate) |

Product observable is total correlation \(E_{\mathrm{CCSD}}+E_{(T)}\).

## Gates (product regression)

- \(|E_{\mathrm{corr,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(|E_{\mathrm{tot,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(E_{\mathrm{RHF}}\), \(E_{\mathrm{CCSD}}\), \(E_{(T)}\): \(10^{-6}\) Ha
- Amplitudes report `converged == true`

## Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method ccsd_t --geom water --basis sto-3g --repeat 3 -- \
    ./build/regressions/reg_ccsd-t-h2o-sto3g
```

| Field | Value |
|---|---|
| Protocol | median of ≥3 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.0933 s |
| Atomiverse median wall | 0.0311 s |
| **ratio_atv_over_pyscf** | **0.333** |
| **within_10x** | **true** |
| Bench date | 2026-07-23, login node, OMP_NUM_THREADS=1 |

H₂O/STO-3G is startup + small-N dominated (7 AO; spin-orbital no=10, nv=4);
ATV is faster than PySCF on this foothold (ratio ≪ 1). No performance
optimization pass required.
