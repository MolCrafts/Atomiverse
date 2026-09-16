# O₂ EOM-SF-CCSD/STO-3G — spin-flip excitation energies

Open-shell **EOM-SF-CCSD** on triplet O₂ / STO-3G. Product API:
`UHF::eom_sf_ccsd(frame, n_states, guess)` (drives high-spin UHF + UCCSD +
spin-flip EOM; ROHF density warm-start for the multi-minima UHF surface).

Literature / theory:

- A. I. Krylov, *Size-consistent wave functions for bond-breaking: the
  equation-of-motion spin-flip model*, Chem. Phys. Lett. **338**, 375 (2001).
  DOI [10.1016/S0009-2614(01)00287-1](https://doi.org/10.1016/S0009-2614(01)00287-1)
- S. V. Levchenko & A. I. Krylov, *Equation-of-motion spin-flip coupled-cluster
  model with single and double substitutions: Theory and application to
  cyclobutadiene*, J. Chem. Phys. **120**, 175 (2004).
  DOI [10.1063/1.1630018](https://doi.org/10.1063/1.1630018)
- PySCF implementation: `pyscf.cc.eom_uccsd.EOMEESpinFlip` /
  `UCCSD.eomsf_ccsd`.

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | O₂, charge 0, triplet (`spin=2` = 2S) |
| Geometry (Å) | O(0,0,0); O(0,0,1.2075) — `pin_scf` `o2` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` |
| SCF | UHF, `conv_tol=1e-12`, ROHF warm-start in ATV |
| CCSD | `pyscf.cc.UCCSD(mf)`, `conv_tol=1e-10` |
| EOM | `mycc.eomsf_ccsd(nroots=15)` then take the **lowest 5** real roots |
| Electrons | n_α=9, n_β=7 |
| AO / MO | 10 spatial AO/MO |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

**Davidson note:** `eomsf_ccsd(nroots=5)` alone can miss intermediate SF roots
(jumps from 1.59 eV to 8.81 eV). Requesting ≥10–15 roots recovers the full
lowest ladder; ATV uses a dense non-symmetric eigen-solve of the spin-orbital
Jacobian with majority-channel (α→β) SF selection, which matches that full
lowest ladder.

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py eom-sf-ccsd \
    --geom o2 --basis sto-3g --spin 2 --nstates 15 --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf, cc
import numpy as np
HA_TO_EV = 27.211386245988
mol = gto.M(atom="""
O  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.207500
""", basis="sto-3g", unit="Angstrom", cart=True, spin=2, verbose=0)
mf = scf.UHF(mol); mf.conv_tol = 1e-12; mf.kernel()
mycc = cc.UCCSD(mf); mycc.conv_tol = 1e-10; mycc.kernel()
e, v = mycc.eomsf_ccsd(nroots=15)
e = np.sort(np.atleast_1d(np.asarray(e, dtype=float)))
print(mycc.e_tot, (e[:5] * HA_TO_EV).tolist())
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{UHF}}\) | −147.633946785450 Ha | 1e-6 Ha |
| \(E_{\mathrm{UCCSD}}\) | −147.741936323434 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) | −0.107989537984 Ha | 1e-6 Ha |
| \(\omega_0\) (SF, near-zero multiplet) | **0.004154770937 eV** | **1e-4 eV** |
| \(\omega_1\) | **1.024028019354 eV** | **1e-4 eV** |
| \(\omega_2\) | **1.024028354230 eV** | **1e-4 eV** |
| \(\omega_3\) | **1.586396740693 eV** | **1e-4 eV** |
| \(\omega_4\) | **6.184403293079 eV** | **1e-4 eV** |

Product gate focuses on \(\omega_0\ldots\omega_4\) at 1e-4 eV and UCCSD energy
at 1e-6 Ha.

## Gates (product regression)

- \(|\omega_{k,\mathrm{ours}} - \mathrm{pin}| < 10^{-4}\,\mathrm{eV}\) for \(k=0..4\)
- \(|E_{\mathrm{UCCSD,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- `converged == true`

## Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method eom-sf-ccsd --geom o2 --basis sto-3g --spin 2 --nstates 5 --repeat 3 -- \
    ./build/regressions/reg_eom-sf-ccsd-o2
```

| Field | Value |
|---|---|
| Protocol | median of ≥3 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.317 s |
| Atomiverse median wall | 0.988 s |
| **ratio_atv_over_pyscf** | **3.12** |
| **within_10x** | **true** |
| Bench date | 2026-07-23, login node, OMP_NUM_THREADS=1 |

ATV path: UHF (ROHF warm-start) + UCCSD + majority-SF dense Jacobian
(factored matvec intermediates). PySCF twin: UHF + UCCSD + `eomsf_ccsd(nroots=5)`
(Davidson; misses some intermediate SF roots that dense recovers — pins use
`nroots=15` lowest 5 for the accuracy gate).
