# Reference — H2O RMP2/STO-3G analytic nuclear gradient

Oracle protocol: offline PySCF pin (`tools/pyscf_oracle/pin_scf.py`); numbers
hard-coded below. **Never** import PySCF from CMake/ctest.

| Field | Value |
|---|---|
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| Basis | STO-3G, **Cartesian** (`cart=True`) |
| Method | RHF → canonical RMP2 (no frozen core) |
| Convergence | `conv_tol = 1e-12` |
| PySCF version | 2.14.0 |
| Command | `python tools/pyscf_oracle/pin_scf.py mp2 --geom water --basis sto-3g --grad --conv-tol 1e-12` |
| **E_HF** | **−74.963023138463 Ha** |
| **E_corr** | **−0.035545651647 Ha** |
| **E_MP2** | **−74.998568790110 Ha** |
| Gate | element-wise 1e-5 Ha/Bohr; translational invariance 1e-8 Ha/Bohr |

## Pinned analytic gradient dE/dR [Ha/Bohr]

`mp.MP2(mf).run(); m2.nuc_grad_method().kernel()`. Molecule in the yz-plane →
x components numerically zero (~1e-16).

| Atom | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| O  | 0 | 0 | −9.528798092138e-02 |
| H  | 0 | −3.796971346463e-02 | +4.764399046069e-02 |
| H  | 0 | +3.796971346463e-02 | +4.764399046069e-02 |

Σ over atoms ≈ (0, 0, 0) to ~1e-15 Ha/Bohr.

## Why 1e-5 Ha/Bohr

PySCF and Atomiverse both deliver **analytic** RMP2 gradients (Lagrangian
1-RDM + non-separable 2-RDM + Z-vector on ¹(A+B)). Agreement to ~1e-6–1e-8 is
expected on this minimal system; the regression gate is a looser 1e-5 Ha/Bohr
to cover packing / AO-slice assembly noise without masking a missing response
or 2-RDM term (those errors are typically ≥1e-3 Ha/Bohr).

Literature: Pople, Krishnan, Schlegel, Binkley, *Int. J. Quantum Chem. Symp.*
**13**, 225 (1979); Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure
Theory*, response / gradient chapters; PySCF `pyscf.grad.mp2`.
