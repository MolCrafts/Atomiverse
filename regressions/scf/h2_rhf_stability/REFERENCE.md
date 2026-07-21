# H2/STO-3G RHF stability + broken-symmetry UHF — reference values

## Literature basis

- R. Seeger, J. A. Pople, *Self-consistent molecular orbital methods. XVIII.
  Constraints and stability in Hartree-Fock theory*, J. Chem. Phys. **66**,
  3045 (1977). DOI 10.1063/1.434318 — the internal/external classification:
  a real RHF solution is **internally** stable iff the lowest eigenvalue of
  the singlet ¹(A+B) orbital-Hessian block is ≥ 0, and **externally**
  (triplet, RHF→UHF) stable iff the lowest eigenvalue of ³(A+B) is ≥ 0.
- A. Szabo, N. S. Ostlund, *Modern Quantum Chemistry*, §3.8.7 — stretched
  H2: beyond the Coulson-Fischer point the RHF solution is a saddle point
  in the unrestricted parameter space and UHF dissociates below it.
- Closed-shell spatial-orbital blocks (real orbitals, chemist notation):
  ¹A = δδ(ε_a−ε_i) + 2(ai|jb) − (ij|ab), ¹B = 2(ai|bj) − (aj|bi);
  ³A = δδ(ε_a−ε_i) − (ij|ab),            ³B = −(aj|bi).

## Oracle: PySCF 2.14.0 (external run — nothing here reads PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf 2.14.0). STO-3G is run with
`cart=True` for consistency with the engine convention (Pople bases =
Cartesian), which is a no-op for the s-only H2 case.

The eigenvalue pins were produced by building (A+B) per channel **from the
formulas above** with converged-RHF MO integrals (`mol.intor('int2e')` +
einsum quarter transforms), then `numpy.linalg.eigvalsh` — i.e. an
independent implementation of the same matrices this regression's kernel
builds. PySCF's own `mf.stability(external=True, return_status=True)`
verdicts were recorded as corroboration of the sign classification (PySCF's
internal hessian operators carry different overall scale factors, so the
booleans — not its raw hop values — are the cross-check).

Command (verbatim):

```bash
~/work/.x86_64/bin/python - <<'EOF'
import numpy as np
from pyscf import gto, scf
for R in (0.735, 1.5):
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {R}", basis="sto-3g", cart=True,
                unit="Angstrom", verbose=0)
    mf = scf.RHF(mol); mf.conv_tol = 1e-12
    e_rhf = mf.kernel()
    C, eps, no = mf.mo_coeff, mf.mo_energy, mol.nelectron // 2
    n = C.shape[1]; nv = n - no
    Co, Cv = C[:, :no], C[:, no:]
    eri = mol.intor("int2e")
    ovov = np.einsum("ua,vi,uvls,lb,sj->aibj", Cv, Co, eri, Cv, Co, optimize=True)
    oovv = np.einsum("ui,vj,uvls,la,sb->ijab", Co, Co, eri, Cv, Cv, optimize=True)
    de = eps[no:, None] - eps[None, :no]
    nov = no * nv
    A1 = np.zeros((nov, nov)); B1 = np.zeros((nov, nov))
    A3 = np.zeros((nov, nov)); B3 = np.zeros((nov, nov))
    for a in range(nv):
        for i in range(no):
            for b in range(nv):
                for j in range(no):
                    p, q = a * no + i, b * no + j
                    dd = de[a, i] if (a == b and i == j) else 0.0
                    A1[p, q] = dd + 2 * ovov[a, i, b, j] - oovv[i, j, a, b]
                    B1[p, q] = 2 * ovov[a, i, b, j] - ovov[a, j, b, i]
                    A3[p, q] = dd - oovv[i, j, a, b]
                    B3[p, q] = -ovov[a, j, b, i]
    w1 = np.linalg.eigvalsh(A1 + B1); w3 = np.linalg.eigvalsh(A3 + B3)
    st = mf.stability(external=True, return_status=True)
    print(R, e_rhf, w1[0], w3[0], st[2], st[3])
    if R == 1.5:
        th = np.pi / 4
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Ca = C.copy(); Cb = C.copy()
        Ca[:, :2] = C[:, :2] @ rot; Cb[:, :2] = C[:, :2] @ rot.T
        mfu = scf.UHF(mol); mfu.conv_tol = 1e-12
        e_uhf = mfu.kernel(dm0=(Ca[:, :1] @ Ca[:, :1].T, Cb[:, :1] @ Cb[:, :1].T))
        print("UHF", e_uhf, mfu.spin_square())
EOF
```

## Pinned values (Hartree)

| Quantity | R = 0.735 Å | R = 1.5 Å |
|---|---|---|
| E_RHF | −1.1169989968 | −0.9108735546 |
| λ_min ¹(A+B) (internal) | +1.1351770407 | +0.7088965814 |
| λ_min ³(A+B) (external) | +0.4114522416 | −0.2092471629 |
| PySCF `mf.stability` verdict | stable / stable | internal stable / external **unstable** |
| E_UHF (broken symmetry) | — | −0.9577067934 (⟨S²⟩ = 0.6949) |
| E_UHF − E_RHF | — | −0.0468332388 |

## What the regression checks

1. R = 1.5 Å: engine RHF energy, internal and external lowest eigenvalues
   match the pins to 1e-6 Ha; external < 0 while internal ≥ 0 (the
   Seeger-Pople trap: internal stability does NOT imply external stability).
2. UHF seeded from `StabilityResult::rotated_C` (α density from the rotated
   coefficients, β from the unrotated RHF density, via the public
   `SCFInitialGuess`) converges to the pinned broken-symmetry energy and
   lies below RHF by ≥ 1e-4 Ha (actual margin ≈ 0.047 Ha).
3. R = 0.735 Å: both channels stable with pinned positive eigenvalues.

Tolerances: 1e-6 Ha on energies and eigenvalues (engine vs PySCF integral
parity); the UHF-below-RHF gate is the spec's 1e-4 Ha. Runtime ≪ 60 s.
