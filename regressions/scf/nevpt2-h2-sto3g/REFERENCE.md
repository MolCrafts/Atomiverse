# H2 SC-NEVPT2 / CASSCF(2,2)/STO-3G — correlation pin

Closed-shell **strongly contracted NEVPT2** on a full-valence CASSCF(2,2)
reference for H₂ / STO-3G. Product API:
`RHF::nevpt2(frame, /*ncas=*/2, /*nelecas=*/2)` (drives RHF → CASSCF → SC-NEVPT2).

Literature / theory:

- C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, J.-P. Malrieu,
  *Introduction of n-electron valence states for multireference perturbation
  theory*, J. Chem. Phys. **114**, 10252 (2001).
  DOI [10.1063/1.1361246](https://doi.org/10.1063/1.1361246)
- C. Angeli, R. Cimiraglia, J.-P. Malrieu, *n-electron valence state
  perturbation theory: A spinless formulation and an efficient implementation
  of the strongly contracted and of the partially contracted variants*,
  J. Chem. Phys. **117**, 9138 (2002).
  DOI [10.1063/1.1515317](https://doi.org/10.1063/1.1515317)

On this two-electron / two-orbital foothold, CASSCF(2e,2o) spans **all** MOs
(`ncore = 0`, `nvirt = 0`). Every SC-NEVPT2 perturber class (Sr, Si, Sij, Srs,
Sir, Sijr, Srsi, Sijrs) requires at least one inactive or external index, so
the PT2 correction is **identically zero** and
\(E_{\mathrm{NEVPT2}} = E_{\mathrm{CASSCF}} = E_{\mathrm{FCI}}\).

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
| NEVPT2 | `pyscf.mrpt.NEVPT(mc).kernel()` (strongly contracted default) |
| AO / MO | 2 AO; ncore=0, ncas=2, nvirt=0 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-24, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py nevpt2 \
    --geom h2 --basis sto-3g --ncas 2 --nelecas 2 --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf, mcscf, mrpt
mol = gto.M(atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; e_hf = mf.kernel()
mc = mcscf.CASSCF(mf, 2, 2); mc.conv_tol = 1e-12
e_mc = mc.kernel()[0]
e_pt2 = mrpt.NEVPT(mc).kernel()
print(e_hf, e_mc, e_pt2, e_mc + e_pt2, mc.ncore)
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | 1e-6 Ha |
| \(E_{\mathrm{CASSCF}}\) | −1.137283834488502 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) (SC-NEVPT2 PT2) | **0.0** Ha | **1e-6 Ha** (product gate) |
| \(E_{\mathrm{tot}}\) (NEVPT2) | −1.137283834488502 Ha | **1e-6 Ha** |

`e_corr` is the PT2 correction vs CASSCF (PySCF `mrpt.NEVPT.e_corr`), not
vs HF. On this system it is exactly zero by construction of the active space.

## Gates (product regression)

- \(|E_{\mathrm{corr,ours}} - 0| < 10^{-6}\,\mathrm{Ha}\) (primary gate)
- \(|E_{\mathrm{tot,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(E_{\mathrm{CASSCF}}\), \(E_{\mathrm{RHF}}\): \(10^{-6}\) Ha
- `ncore == 0`, `nvirt == 0`, `ncas == 2`, `nelecas == 2`, `converged == true`
- NEVPT2 total equals CASSCF on the same reference to \(10^{-10}\) Ha

## Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method nevpt2 --geom h2 --basis sto-3g --repeat 5 -- \
    ./build/regressions/reg_nevpt2-h2-sto3g
```

| Field | Value |
|---|---|
| Protocol | median of ≥5 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.1725 s |
| Atomiverse median wall | 0.0125 s |
| **ratio_atv_over_pyscf** | **0.0723** |
| **within_10x** | **true** |
| Bench date | 2026-07-24, login node, OMP_NUM_THREADS=1 |

H₂/STO-3G is startup-dominated; ATV is faster than PySCF on this foothold
(ratio ≪ 1). No performance optimization pass required.
