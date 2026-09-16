# H2 EOM-EA-CCSD/STO-3G — lowest electron-attachment energy pin

Closed-shell **EOM-EA-CCSD** on H₂ / STO-3G. Product API:
`RHF::eom_ea_ccsd(frame, n_states)` (drives RHF + spin-orbital CCSD + EOM-EA;
unique attachment roots from the spin-orbital Jacobian, spin degeneracy
de-duplicated).

Literature / theory:

- M. Nooijen & R. J. Bartlett, *Equation of motion coupled cluster method for
  electron attachment*, J. Chem. Phys. **102**, 3629 (1995).
  DOI [10.1063/1.468591](https://doi.org/10.1063/1.468591)
- J. F. Stanton & R. J. Bartlett, *The equation of motion coupled-cluster
  method…*, J. Chem. Phys. **98**, 7029 (1993).
  DOI [10.1063/1.464746](https://doi.org/10.1063/1.464746)
- Spin-orbital σ equations as coded in PySCF `eom_gccsd.eaccsd_matvec`
  (Nooijen–Bartlett Eqs. (30)–(31)).

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
| EOM | `mycc.eaccsd(nroots=1)` (spatial RCCSD EA; matches GCCSD EOMEA) |
| AO / MO | 2 AO; 1 occupied, 1 virtual spatial MO |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py eom-ea-ccsd \
    --geom h2 --basis sto-3g --nstates 1 --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf, cc
HA_TO_EV = 27.211386245988
mol = gto.M(atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
mycc = cc.CCSD(mf); mycc.conv_tol = 1e-12; mycc.kernel()
e, v = mycc.eaccsd(nroots=1)
print(float(e[0]), float(e[0]) * HA_TO_EV, mycc.e_tot)
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{CCSD}}\) | −1.137283834497130 Ha | 1e-6 Ha |
| \(\mathrm{EA}_0\) (lowest EOM-EA) | **18.821245619413340 eV** | **1e-4 eV** (product gate) |
| \(\mathrm{EA}_0\) | 0.691668019014956 Ha | (cross-check) |

H₂/STO-3G attachment is unbound (positive EA energy relative to neutral CCSD);
the LUMO Koopmans estimate is ~18.26 eV.

## Gates (product regression)

- \(|\mathrm{EA}_{0,\mathrm{ours}} - \mathrm{pin}| < 10^{-4}\,\mathrm{eV}\)
- \(|E_{\mathrm{CCSD,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- `converged == true`

## Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method eom-ea-ccsd --geom h2 --basis sto-3g --nstates 1 --repeat 3 -- \
    ./build/regressions/reg_eom-ea-ccsd
```

| Field | Value |
|---|---|
| Protocol | median of ≥3 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.9293 s |
| Atomiverse median wall | 0.0635 s |
| **ratio_atv_over_pyscf** | **0.068** |
| **within_10x** | **true** |
| Bench date | 2026-07-23, login node, OMP_NUM_THREADS=1 |

H₂/STO-3G is startup-dominated; ATV is faster than PySCF on this foothold
(ratio ≪ 1). No performance optimization pass required.
