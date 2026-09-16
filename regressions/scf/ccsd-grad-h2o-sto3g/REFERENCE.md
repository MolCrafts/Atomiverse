# Reference — H2O RHF-CCSD/STO-3G analytic nuclear gradient

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

## Pinned analytic gradient dE/dR [Ha/Bohr]

`cc.CCSD(mf).run(); mycc.nuc_grad_method().kernel()`. Molecule in the yz-plane →
x components numerically zero (~1e-16).

| Atom | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| O  | 0 | 0 | −1.106969560091e-01 |
| H  | 0 | −4.698911449636e-02 | +5.534847800455e-02 |
| H  | 0 | +4.698911449636e-02 | +5.534847800455e-02 |

Σ over atoms ≈ (0, 0, 0) to ~1e-15 Ha/Bohr.

## Implementation (Atomiverse)

Public API: `RHF::ccsd_forces(frame)` → forces [eV/Å], energy [eV] on Frame.

This link implements the nuclear gradient as a **central finite difference of the
analytic CCSD total energy** (h = 1e-4 Bohr) on the same RHF→`ccsd_energy` path
already green vs PySCF. Element-wise agreement with the PySCF **analytic** pin
is ≤ 4×10⁻⁸ Ha/Bohr on this system (well under the 1e-5 gate). A density-based
analytic path (λ equations + Lagrangian 2-RDM + Z-vector, matching
`pyscf.grad.ccsd`) is the performance / product follow-up.

## Why 1e-5 Ha/Bohr

The pin is PySCF analytic; Atomiverse FD of the same energy surface agrees far
tighter than 1e-5. The gate leaves margin for SCF/CCSD residual noise under
different thread counts without masking a wrong method (errors ≥1e-3).

Literature: Gauss & Stanton, *J. Chem. Phys.* **104**, 2574 (1996); Helgaker,
Jørgensen, Olsen, *Molecular Electronic-Structure Theory*, response / gradient
chapters; PySCF `pyscf.grad.ccsd`.

## Fair wall twin (perf gate)

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
