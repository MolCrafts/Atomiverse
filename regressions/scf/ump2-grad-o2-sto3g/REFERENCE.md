# Reference — O₂ UMP2/STO-3G analytic nuclear gradient

Oracle protocol: offline PySCF pin (`tools/pyscf_oracle/pin_scf.py`); numbers
hard-coded below. **Never** import PySCF from CMake/ctest.

| Field | Value |
|---|---|
| Geometry (Å) | O(0, 0, 0); O(0, 0, 1.2075) — same as `ump2-o2-sto3g` / `o2_rohf_sto3g` |
| Basis | STO-3G, **Cartesian** (`cart=True`) |
| Method | UHF (spin \(2S=2\) triplet) → canonical UMP2 (no frozen core) |
| Convergence | `conv_tol = 1e-12` |
| PySCF version | 2.14.0 |
| Command | `python tools/pyscf_oracle/pin_scf.py ump2 --geom o2 --basis sto-3g --spin 2 --conv-tol 1e-12 --grad` |
| **E_UHF** | **−147.633946785450 Ha** |
| **E_corr** | **−0.096393144130 Ha** |
| **E_UMP2** | **−147.730339929580 Ha** |
| Gate | element-wise 1e-5 Ha/Bohr; translational invariance 1e-8 Ha/Bohr |

## Pinned analytic gradient dE/dR [Ha/Bohr]

`mp.UMP2(mf).run(); m2.nuc_grad_method().kernel()`. Collinear diatomic on z →
x,y components numerically zero (~1e-16).

| Atom | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| O  | 0 | 0 | +1.665132663664e-01 |
| O  | 0 | 0 | −1.665132663664e-01 |

Σ over atoms ≈ (0, 0, 0) to ~1e-15 Ha/Bohr.

## Why 1e-5 Ha/Bohr

PySCF and Atomiverse both deliver **analytic** UMP2 gradients (spin-resolved
Lagrangian 1-RDM + non-separable 2-RDM + unrestricted Z-vector on (A+B)).
Agreement to ~1e-6–1e-8 is expected on this minimal system; the regression gate
is a looser 1e-5 Ha/Bohr to cover packing / AO-slice assembly noise without
masking a missing response or 2-RDM term (those errors are typically ≥1e-3
Ha/Bohr).

Literature: Pople, Krishnan, Schlegel, Binkley, *Int. J. Quantum Chem. Symp.*
**13**, 225 (1979); Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure
Theory*, response / gradient chapters; PySCF `pyscf.grad.ump2`.

## Multi-minima note

Cold UHF Hcore/MINAO often lands on a higher local solution for O₂. The
regression warm-starts from ROHF densities (same path as `ump2-o2-sto3g`) so
UHF lands on the PySCF minimum used for the pin.

## Fair wall twin (perf gate)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/pyscf_oracle/fair_bench.py --method ump2_grad --geom o2 \
    --basis sto-3g --spin 2 --repeat 3 -- ./build/regressions/reg_ump2-grad-o2-sto3g
```

| Side | median wall (s) |
|---|---|
| PySCF 2.14.0 UHF+UMP2+grad | 0.296 |
| Atomiverse `reg_ump2-grad-o2-sto3g` | 0.404 |
| **ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\)** | **1.37** |
| within_10x | **true** |
