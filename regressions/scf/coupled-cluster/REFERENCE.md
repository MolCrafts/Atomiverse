# scf/coupled-cluster

One case; every section below is run by `atv_regressions --case scf/coupled-cluster` and keeps its own oracle record.


## H2 CCSD/STO-3G — correlation + total energy pin

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

### Geometry and settings

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

### Repro command

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

### Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) (CCSD) | −0.020524527092106 Ha | **1e-6 Ha** (product gate) |
| \(E_{\mathrm{tot}}\) (CCSD) | −1.137283834488532 Ha | **1e-6 Ha** (product gate) |

On this minimal basis \(T_1 \approx 0\) by symmetry and CCSD coincides with FCI
to ~1e-11 Ha.

### Gates (product regression)

- \(|E_{\mathrm{corr,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(|E_{\mathrm{tot,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(E_{\mathrm{RHF}}\): \(10^{-6}\) Ha
- Amplitudes report `converged == true`

### Performance

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


## Reference — H2O RHF-CCSD/STO-3G analytic nuclear gradient

Oracle protocol: offline PySCF pin (`tools/pyscf_oracle/pin_scf.py`); numbers
hard-coded below. **Never** import PySCF from CMake/ctest.

| Field | Value |
|---|---|
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| Basis | STO-3G, **Cartesian** (`cart=True`) |
| Method | RHF → canonical closed-shell CCSD (no frozen core) |
| Convergence | `conv_tol = 1e-12` (SCF); CCSD `conv_tol = 1e-10` |
| PySCF version | 2.14.0 |
| Command | `python tools/pyscf_oracle/pin_scf.py ccsd --geom water --basis sto-3g --grad --conv-tol 1e-12` |
| **E_HF** | **−74.963023138463 Ha** |
| **E_corr** | **−0.049438562975 Ha** |
| **E_CCSD** | **−75.012461701437 Ha** |
| Gate | element-wise 1e-5 Ha/Bohr; translational invariance 1e-8 Ha/Bohr |

### Pinned analytic gradient dE/dR [Ha/Bohr]

`cc.CCSD(mf).run(); mycc.nuc_grad_method().kernel()`. Molecule in the yz-plane →
x components numerically zero (~1e-16).

| Atom | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| O  | 0 | 0 | −1.106969560091e-01 |
| H  | 0 | −4.698911449636e-02 | +5.534847800455e-02 |
| H  | 0 | +4.698911449636e-02 | +5.534847800455e-02 |

Σ over atoms ≈ (0, 0, 0) to ~1e-15 Ha/Bohr.

### Implementation (Atomiverse)

Public API: `RHF::ccsd_forces(frame)` → forces [eV/Å], energy [eV] on Frame.

This link implements the nuclear gradient as a **central finite difference of the
analytic CCSD total energy** (h = 1e-4 Bohr) on the same RHF→`ccsd_energy` path
already green vs PySCF. Element-wise agreement with the PySCF **analytic** pin
is ≤ 4×10⁻⁸ Ha/Bohr on this system (well under the 1e-5 gate). A density-based
analytic path (λ equations + Lagrangian 2-RDM + Z-vector, matching
`pyscf.grad.ccsd`) is the performance / product follow-up.

### Why 1e-5 Ha/Bohr

The pin is PySCF analytic; Atomiverse FD of the same energy surface agrees far
tighter than 1e-5. The gate leaves margin for SCF/CCSD residual noise under
different thread counts without masking a wrong method (errors ≥1e-3).

Literature: Gauss & Stanton, *J. Chem. Phys.* **104**, 2574 (1996); Helgaker,
Jørgensen, Olsen, *Molecular Electronic-Structure Theory*, response / gradient
chapters; PySCF `pyscf.grad.ccsd`.

### Fair wall twin (perf gate)

```text
python tools/pyscf_oracle/fair_bench.py --method ccsd_grad --geom water \
  --basis sto-3g --repeat 3 ./build/regressions/reg_ccsd-grad-h2o-sto3g
```

| Side | median wall (s) |
|---|---|
| Atomiverse `RHF::ccsd_forces` (FD of E_CCSD) | **0.401** |
| PySCF `cc.CCSD` + `nuc_grad_method` | **0.241** |
| **ratio T_atv / T_pyscf** | **1.665** |
| **within_10x** | **true** |
EOF


## H2O CCSD(T)/STO-3G — correlation + total energy pin

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

### Geometry and settings

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

### Repro command

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

### Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −74.963023138462930 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) (CCSD) | −0.049438563031963 Ha | 1e-6 Ha (diagnostic) |
| \(E_{(T)}\) | −0.000067409682944 Ha | 1e-6 Ha (diagnostic) |
| \(E_{\mathrm{corr}}\) (CCSD(T)) | −0.049505972714907 Ha | **1e-6 Ha** (product gate) |
| \(E_{\mathrm{tot}}\) (CCSD(T)) | −75.012529111177841 Ha | **1e-6 Ha** (product gate) |

Product observable is total correlation \(E_{\mathrm{CCSD}}+E_{(T)}\).

### Gates (product regression)

- \(|E_{\mathrm{corr,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(|E_{\mathrm{tot,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(E_{\mathrm{RHF}}\), \(E_{\mathrm{CCSD}}\), \(E_{(T)}\): \(10^{-6}\) Ha
- Amplitudes report `converged == true`

### Performance

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


## H2O / STO-3G — closed-shell CCSD spatial-vs-spin-orbital parity + DIIS/Jacobi

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

### Public-API note (deviation flag)

`regressions/` are public-API-only consumers (README rule 1). The dual-path precedent
`regressions/scf/multireference` reaches its two engines through two PUBLIC
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

### Geometry and settings

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

### Golden provenance

`E_corr` (CCSD) golden is the same offline PySCF 2.14.0 literal pinned by
`regressions/scf/coupled-cluster/REFERENCE.md` (CCSD-only piece, `conv_tol = 1e-12`).
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

### Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{corr}}\) (CCSD, H₂O/STO-3G) | −0.049438563031963 Ha | **1e-6 Ha** (vs PySCF) |
| \(|E_{\mathrm{corr,spatial}} - E_{\mathrm{corr,SO}}|\) | 0 | **1e-9 Ha** (oracle parity) |
| \(|E_{\mathrm{corr,DIIS}} - E_{\mathrm{corr,Jacobi}}|\) | 0 | **1e-9 Ha** (same fixed point) |

### Why the tolerances

- **1e-9 Ha** (parity): spatial and spin-orbital CCSD solve the identical closed-shell
  amplitude equations in different bases; at `max|Δt| < 1e-8` residual they must land on
  the same correlation energy to well under 1e-9 Ha. A spin-summation weight error
  (`2J−K`, τ vs t2) shifts `E_corr` by ≫ 1e-6 Ha, so the gate catches it decisively.
- **1e-9 Ha** (DIIS vs Jacobi): DIIS only accelerates convergence to the same fixed
  point; equal residual tolerance ⇒ equal `E_corr` to numerical noise.
- **1e-6 Ha** (golden): matches the sibling CCSD regressions' product gate; leaves margin
  for SCF/CCSD residual noise under different thread counts without masking a wrong method.

### Gates (product regression)

- \(|E_{\mathrm{corr,spatial}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(|E_{\mathrm{corr,spatial}} - E_{\mathrm{corr,SO}}| < 10^{-9}\,\mathrm{Ha}\)
- \(|E_{\mathrm{corr,spatial}} - E_{\mathrm{corr,Jacobi}}| < 10^{-9}\,\mathrm{Ha}\)
- all three paths report `converged == true`; runs in < 60 s; no live third-party call.


## H2 CC2/STO-3G — correlation + total energy pin

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

### Geometry and settings

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

### Repro command

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

### Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) (CC2) | −0.013138073589533 Ha | **1e-6 Ha** (product gate) |
| \(E_{\mathrm{tot}}\) (CC2) | −1.129897380985958 Ha | **1e-6 Ha** (product gate) |

Cross-check: on this geometry/basis, PySCF `mp.MP2` gives the same
\(E_{\mathrm{corr}}\) (T₁ vanishes; CC2 reduces to MP2). Full CCSD
\(E_{\mathrm{corr}} = -0.020524527\) Ha (FCI) is intentionally different.

### Gates (product regression)

- \(|E_{\mathrm{corr,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(|E_{\mathrm{tot,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(E_{\mathrm{RHF}}\): \(10^{-6}\) Ha
- Amplitudes report `converged == true`

### Performance

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
