# H2 CC2/STO-3G — correlation + total energy pin

Closed-shell **CC2** on H₂ / STO-3G. Product API: `RHF::cc2(frame)`
(drives a fresh RHF, then spin-orbital Christiansen–Koch–Jørgensen CC2).

Literature / theory:

- O. Christiansen, H. Koch, P. Jørgensen, *The second-order approximate
  coupled cluster singles and doubles model CC2*, Chem. Phys. Lett. **243**,
  409 (1995). DOI [10.1016/0009-2614(95)00841-Q](https://doi.org/10.1016/0009-2614(95)00841-Q)
- Full singles residual; doubles residual ≤ linear in \(T_2\) with the
  \(T_1\)-dressed Fock operator (no quadratic \(T_2\) ring / ladder).
- On this minimal H₂ basis \(T_1 \approx 0\) by symmetry, so CC2 coincides
  with MP2 to ~1e-15 Ha (distinct from full CCSD / FCI).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

Stock PySCF 2.14 has **no** top-level `cc.CC2` class; the product oracle is
`pyscf.cc.rccsd.RCCSD` with `cc2=True` (same path as `rccsd_slow` CC2).

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂, charge 0, singlet |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — `pin_scf` `h2` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` (s only; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| CC2 | `pyscf.cc.rccsd.RCCSD(mf); my.cc2=True`, `conv_tol=1e-12` |
| AO / MO | 2 AO; 1 occupied, 1 virtual spatial MO |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py cc2 \
    --geom h2 --basis sto-3g --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf
from pyscf.cc import rccsd
mol = gto.M(atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; e_hf = mf.kernel()
mycc = rccsd.RCCSD(mf); mycc.conv_tol = 1e-12; mycc.conv_tol_normt = 1e-12
mycc.cc2 = True
e_corr, t1, t2 = mycc.kernel()
print(e_hf, e_corr, mycc.e_tot)
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) (CC2) | −0.013138073589533 Ha | **1e-6 Ha** (product gate) |
| \(E_{\mathrm{tot}}\) (CC2) | −1.129897380985958 Ha | **1e-6 Ha** (product gate) |

Cross-check: on this geometry/basis, PySCF `mp.MP2` gives the same
\(E_{\mathrm{corr}}\) (T₁ vanishes; CC2 reduces to MP2). Full CCSD
\(E_{\mathrm{corr}} = -0.020524527\) Ha (FCI) is intentionally different.

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
    --method cc2 --geom h2 --basis sto-3g --repeat 3 -- \
    ./build/regressions/reg_cc2-h2-sto3g
```

| Field | Value |
|---|---|
| Protocol | median of ≥3 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.0350 s |
| Atomiverse median wall | 0.0137 s |
| **ratio_atv_over_pyscf** | **0.392** |
| **within_10x** | **true** |
| Bench date | 2026-07-23, login node, OMP_NUM_THREADS=1 |

H₂/STO-3G is startup-dominated; ATV is faster than PySCF on this foothold
(ratio ≪ 1). No performance optimization pass required.
