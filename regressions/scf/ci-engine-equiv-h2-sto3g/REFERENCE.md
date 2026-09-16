# H2 / STO-3G — unified determinant-CI engine equivalence

`RHF::fci` and `RHF::casci(2,2)` must both reproduce the offline PySCF full-CI
pin AND agree with each other to machine precision. On H2 / STO-3G the full CI
space and the CAS(2e,2o) space are identical, so FCI == CASCI(full space) is
exact. This is the numeric equivalence gate for scf-review-03-posthf-dedup: it
proves that unifying the determinant-CI machinery (formerly forked between
`fci.cpp` and `casscf.cpp`) into one engine changed no number and that both
public entry points route the same engine.

Product API (out-of-library consumer, public headers only):

```cpp
RHF::fci(frame);                       // full CI, drives a fresh RHF
RHF::casci(frame, /*ncas=*/2, /*nelecas=*/2);  // full valence CAS, drives a fresh RHF
```

Literature / theory:

- B. O. Roos, *The Complete Active Space Self-Consistent Field Method and its
  Applications in Electronic Structure Calculations*, Adv. Chem. Phys. **69**,
  399 (1987). DOI [10.1002/9780470142943.ch7](https://doi.org/10.1002/9780470142943.ch7)
- Szabo & Ostlund, *Modern Quantum Chemistry*, Dover (1996), §2.3 (Slater–Condon
  rules), ISBN 0-486-69186-1.
- On the two-electron / two-orbital H2 foothold, CASCI(2e,2o) is **identical** to
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
| FCI | `pyscf.fci.FCI(mf)` (full space) |
| CASCI | `pyscf.mcscf.CASCI(mf, ncas=2, nelecas=2)` |
| AO / MO | 2 AO; 1 occupied, 1 virtual spatial MO; ncore=0 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` (same pin as `casci-h2-sto3g`) |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py casci \
    --geom h2 --basis sto-3g --ncas 2 --nelecas 2 --conv-tol 1e-12
```

Equivalent (FCI == CASCI(full) on this system):

```python
from pyscf import gto, scf, mcscf, fci
mol = gto.M(atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; e_hf = mf.kernel()
e_fci = fci.FCI(mf).kernel()[0]
mc = mcscf.CASCI(mf, 2, 2); e_cas = mc.kernel()[0]
print(e_hf, e_fci, e_cas)   # e_fci == e_cas == -1.137283834488502
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{FCI}}\) (total) | −1.137283834488502 Ha | 1e-6 Ha |
| \(E_{\mathrm{CASCI}(2,2)}\) (total) | −1.137283834488502 Ha | 1e-6 Ha |
| \(|E_{\mathrm{FCI}} - E_{\mathrm{CASCI}}|\) | 0 | **1e-12 Ha** (equivalence gate) |

The full-CI total −1.137283834488502 Ha is the same offline PySCF 2.14.0 literal
pinned by `regressions/scf/casci-h2-sto3g` (`mc.e_tot`); FCI coincides with
CASCI(2,2) to ~1e-14 Ha on this system.

## Gates (product regression)

- \(|E_{\mathrm{FCI}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(|E_{\mathrm{CASCI}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(|E_{\mathrm{FCI}} - E_{\mathrm{CASCI}}| < 10^{-12}\,\mathrm{Ha}\) (unified-engine equivalence)
- `ncore == 0`, `ncas == 2`, `nelecas == 2`
- runs in < 60 s; no live third-party call.
