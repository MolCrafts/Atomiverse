# Reference — equilateral H₃ GHF/STO-3G nuclear gradient

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).
Energy sibling: regressions/scf/h3_ghf_sto3g (same geometry / basis / Néel guess).

| Field | Value |
|---|---|
| Geometry (Å) | equilateral side 1.0: H(0,0,0); H(1,0,0); H(0.5, √3/2, 0) |
| Basis | STO-3G (Cartesian) |
| Method | real two-component GHF, N = 3 (spinor aufbau) |
| Convergence | `conv_tol = 1e-13`, 120° per-AO Néel guess |
| PySCF version | 2.14.0 |
| Oracle note | PySCF `scf.GHF.nuc_grad_method()` raises `NotImplementedError` — pin is the **analytic GHF gradient evaluated with PySCF AO / dAO integrals** at the pinned noncollinear density (cross-checked vs central FD of `scf.GHF.kernel`, h = 1e-4 Bohr, max |Δ| ≈ 1e-7 Ha/Bohr) |
| Compare tol | element-wise **1e-5 Ha/Bohr**; translational invariance 1e-8 Ha/Bohr |

## Commands (verbatim, offline)

```python
import numpy as np
from pyscf import gto, scf

a = 1.0
mol = gto.M(atom=f"H 0 0 0; H {a} 0 0; H {a/2} {a*np.sqrt(3)/2} 0",
            basis="sto-3g", spin=1, charge=0, cart=True, unit="Angstrom")
n = mol.nao
dm0 = np.zeros((2*n, 2*n)); dm0[:n, :n] = np.eye(n)

def rot(dm, theta):
    G = np.zeros((2*n, 2*n))
    for mu in range(n):
        th = theta * mu; c, s = np.cos(th/2), np.sin(th/2)
        G[mu, mu] = c; G[mu, n+mu] = -s; G[n+mu, mu] = s; G[n+mu, n+mu] = c
    return G @ dm @ G.T

mf = scf.GHF(mol); mf.conv_tol = 1e-13
e = mf.kernel(dm0=rot(dm0, 2*np.pi/3))   # -1.34044034354119 Ha
# Analytic pin: dE/dR = Tr[Pc dH] - Tr[Wc dS] + ½ Pc:dERI:Pc
#              - ½ Σ_{σσ'} P^{σσ'}:dERI:P^{σ'σ} + dV_nn
# at fixed converged spinor density (dAO via central FD of PySCF integrals).
# Helper: tools/pyscf_oracle/pin_scf.py ghf --geom h3 --basis sto-3g --grad
```

## Pinned analytic gradient dE/dR [Ha/Bohr]

| Atom | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| H (0) | +9.473114183e-02 | +5.469305020e-02 | 0 |
| H (1) | −9.473114183e-02 | +5.469305020e-02 | 0 |
| H (2) | 0 | −1.093861004e-01 | 0 |

Σ_A g_A ≈ 0 (translational invariance). Planar molecule → z components identically zero.

## Physics

Without spin–orbit every UHF solution is a GHF stationary point; on the
spin-frustrated equilateral H₃ triangle the GHF minimum is the coplanar 120°
Néel arrangement (energy sibling pin E = −1.34044034354119 Ha). The nuclear
gradient of that noncollinear solution is the quantity pinned here — a pure
UHF gradient at a collinear saddle is a different surface and must not be
used as the oracle.

## Citations

- H. Fukutome, Int. J. Quantum Chem. **20**, 955 (1981), DOI 10.1002/qua.560200502
- T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure Theory*, §9–10
- P. Pulay, Mol. Phys. **17**, 197 (1969) (Pulay forces)
