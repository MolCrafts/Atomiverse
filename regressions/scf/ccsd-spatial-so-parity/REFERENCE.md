# H2O / STO-3G — closed-shell CCSD spatial-vs-spin-orbital parity + DIIS/Jacobi

The permanent oracle cross-check for **scf-review-08-ccsd-spatial-diis**. Closed-shell
CCSD on H₂O / STO-3G is solved three ways and all three must agree; the spatial energy
must also match the offline PySCF `cc.CCSD` pin.

1. **Spatial-orbital + amplitude DIIS** — `RHF::ccsd(frame)` (closed-shell default
   routes to `ccsd_energy_spatial`, n_o²·n_v² storage, `CcAmplitudeDiis`) → `E_corr_spatial`.
2. **Spin-orbital oracle** — `RHF::ccsd_t(frame)` keeps the spin-orbital Gauss–Stanton
   kernel (spec out-of-scope; `(T)` still consumes the `CCSDSolution` SO blocks). Its
   `e_corr_ccsd` field is the SO CCSD-only correlation → `E_corr_so`.
3. **Plain Jacobi (DIIS off)** — `RHF::ccsd(frame)` with `cc_diis = false` → `E_corr_jacobi`.

Invariants (permanent gates, both paths converged to `max|Δt| < 1e-8`):

- `|E_corr_spatial − E_corr_so|   ≤ 1e-9 Ha` — spatial vs spin-orbital (the
  spin-summation-coefficient error gate).
- `|E_corr_spatial − E_corr_jacobi| ≤ 1e-9 Ha` — DIIS and Jacobi reach the same fixed
  point at equal residual tolerance.
- `|E_corr_spatial − E_corr_golden| ≤ 1e-6 Ha` — vs the offline PySCF `cc.CCSD` literal.

## Public-API note (deviation flag)

`regressions/` are public-API-only consumers (README rule 1). The dual-path precedent
`regressions/scf/ci-engine-equiv-h2-sto3g` reaches its two engines through two PUBLIC
entries; this case mirrors that:

| Leg | Public entry | Engine reached |
|---|---|---|
| spatial + DIIS | `RHF::ccsd(frame)` | closed-shell spatial fast path (post-impl) |
| spin-orbital oracle | `RHF::ccsd_t(frame).e_corr_ccsd` | spin-orbital Gauss–Stanton (unchanged) |
| Jacobi | `RHF::ccsd(frame)` with `cc_diis = false` | spatial, DIIS disabled |

The DIIS-vs-Jacobi leg has **no existing public toggle**. It is pinned against a
**proposed** minimal public member `RHF::cc_diis` (`bool`, default `true`; `false` →
plain Jacobi) — a plain public data member in the house style (parameters are members,
no `*Config` bag, no factory), the **single invented symbol** in this case. Until the
impl adds `cc_diis` this regression is RED (compile-fail on that member). No
internal-include precedent exists in `regressions/` (grep: only `README.md` and
`h2_rhf_stability/REFERENCE.md` mention `internal/`, both in prose), so the case stays a
pure public consumer rather than reaching inside.

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂O, charge 0, singlet |
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) — `pin_scf` `water` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` (s/p only; engine-neutral for STO-3G) |
| SCF | RHF, `conv_tol = 1e-12` |
| CCSD | `pyscf.cc.CCSD(mf)`, `conv_tol = 1e-12` |
| AO / MO | 7 AO; 5 occupied, 2 virtual spatial MO |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Golden provenance

`E_corr` (CCSD) golden is the same offline PySCF 2.14.0 literal pinned by
`regressions/scf/ccsd-t-h2o-sto3g/REFERENCE.md` (CCSD-only piece, `conv_tol = 1e-12`).
No live third-party call at CMake/ctest/runtime.

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
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
mycc = cc.CCSD(mf); mycc.conv_tol = 1e-12
e_corr, t1, t2 = mycc.kernel()
print(e_corr)   # -0.049438563031963
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{corr}}\) (CCSD, H₂O/STO-3G) | −0.049438563031963 Ha | **1e-6 Ha** (vs PySCF) |
| \(|E_{\mathrm{corr,spatial}} - E_{\mathrm{corr,SO}}|\) | 0 | **1e-9 Ha** (oracle parity) |
| \(|E_{\mathrm{corr,DIIS}} - E_{\mathrm{corr,Jacobi}}|\) | 0 | **1e-9 Ha** (same fixed point) |

## Why the tolerances

- **1e-9 Ha** (parity): spatial and spin-orbital CCSD solve the identical closed-shell
  amplitude equations in different bases; at `max|Δt| < 1e-8` residual they must land on
  the same correlation energy to well under 1e-9 Ha. A spin-summation weight error
  (`2J−K`, τ vs t2) shifts `E_corr` by ≫ 1e-6 Ha, so the gate catches it decisively.
- **1e-9 Ha** (DIIS vs Jacobi): DIIS only accelerates convergence to the same fixed
  point; equal residual tolerance ⇒ equal `E_corr` to numerical noise.
- **1e-6 Ha** (golden): matches the sibling CCSD regressions' product gate; leaves margin
  for SCF/CCSD residual noise under different thread counts without masking a wrong method.

## Gates (product regression)

- \(|E_{\mathrm{corr,spatial}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(|E_{\mathrm{corr,spatial}} - E_{\mathrm{corr,SO}}| < 10^{-9}\,\mathrm{Ha}\)
- \(|E_{\mathrm{corr,spatial}} - E_{\mathrm{corr,Jacobi}}| < 10^{-9}\,\mathrm{Ha}\)
- all three paths report `converged == true`; runs in < 60 s; no live third-party call.
