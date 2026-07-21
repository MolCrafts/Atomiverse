# H3 GHF/STO-3G — pinned reference values

Real two-component generalized Hartree-Fock (GHF) on the spin-frustrated
equilateral H3 triangle, plus the H2 collapse sentinel.

## Oracle

PySCF **2.14.0** (`~/work/.x86_64/bin/python`), run offline 2026-07-21.
Cartesian AO convention (`cart=True`), `conv_tol <= 1e-12`, Angstrom units.

### Geometry (used verbatim in the C++ case)

Equilateral H3, side a = 1.0 Å, in the xy-plane:

```
H 0.0 0.0 0.0
H 1.0 0.0 0.0
H 0.5 0.8660254037844386 0.0        # (a/2, a*sqrt(3)/2, 0)
```

H2 collapse sentinel: `H 0 0 0; H 0.74 0 0`.

### Commands (verbatim)

```python
import numpy as np, scipy.linalg as sla
from pyscf import gto, scf

a = 1.0
mol = gto.M(atom=f"H 0 0 0; H {a} 0 0; H {a/2} {a*np.sqrt(3)/2} 0",
            basis="sto-3g", spin=1, charge=0, cart=True, unit="Angstrom")
n = mol.nao

# 1) Collinear references
mf_u = scf.UHF(mol); mf_u.conv_tol = 1e-13
e_bs = mf_u.kernel()                       # -1.33598005401819 (MINAO guess)
mf_u.stability(external=True)              # reports UHF -> GHF instability

S = mol.intor('int1e_ovlp'); h = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
e0, C0 = sla.eigh(h, S)
Pa = C0[:,:2] @ C0[:,:2].T; Pb = C0[:,:1] @ C0[:,:1].T
mf_s = scf.UHF(mol); mf_s.conv_tol = 1e-13
e_sym = mf_s.kernel(dm0=(Pa, Pb))          # -1.32817575538128, <S2> = 0.75

# 2) Noncollinear GHF from a 120-degree per-atom Neel moment guess
dm0 = np.zeros((2*n, 2*n)); dm0[:n, :n] = np.eye(n)   # localized up moments
def rot(dm, theta):                         # real spin rotation per AO
    G = np.zeros((2*n, 2*n))
    for mu in range(n):
        th = theta*mu; c, s = np.cos(th/2), np.sin(th/2)
        G[mu,mu]=c; G[mu,n+mu]=-s; G[n+mu,mu]=s; G[n+mu,n+mu]=c
    return G @ dm @ G.T
mf_g = scf.GHF(mol); mf_g.conv_tol = 1e-13
e_g = mf_g.kernel(dm0=rot(dm0, 2*np.pi/3)) # -1.34044034354119
dm = mf_g.make_rdm1()
np.linalg.norm(dm[:n, n:])                 # ||P_ab||_F = 0.9780588
mf_g.spin_square()                         # <S2> = 0.840667839
mf_g.stability()                           # "stable in the internal
                                           #  stability analysis"

# 3) Collinear trap: any collinear guess stays at a collinear energy
mf_c = scf.GHF(mol); mf_c.conv_tol = 1e-12
pa, pb = mf_u.make_rdm1()
dmc = np.zeros((2*n, 2*n)); dmc[:n,:n] = pa; dmc[n:,n:] = pb
mf_c.kernel(dm0=dmc)                       # stays at -1.33598005401819
```

H2 sentinel:

```python
molh2 = gto.M(atom="H 0 0 0; H 0.74 0 0", basis="sto-3g", cart=True,
              unit="Angstrom")
scf.UHF(molh2).kernel()                    # -1.11675930739643
scf.GHF(molh2).kernel()                    # -1.11675930739643  (collapse)
```

## Certifying the noncollinear minimum (not a collinear saddle)

Three independent checks were run:

1. `mf_u.stability(external=True)` on the collinear UHF reports an
   **UHF/UKS -> GHF/GKS instability** — a lower GHF solution must exist.
2. `mf_g.stability()` on the converged GHF reports **stable in the internal
   stability analysis** — the pinned point is a true local minimum.
3. Guess-robustness scan: per-atom moment angles θ ∈ {1.0, 1.5, 2π/3, 2.5}
   from the localized Néel construction all converge to the same
   E = −1.34044034354119 Ha with ‖P_αβ‖_F = 0.9780588; collinear guesses
   (θ = 0) and weakly rotated delocalized guesses (θ ≤ 0.3 applied to the
   aufbau density) converge back to collinear energies. The pinned value is
   therefore the noncollinear minimum, not a saddle.

## Pinned values

| Quantity | Value | Tolerance |
|---|---|---|
| E(GHF, H3, noncollinear) | **−1.34044034354119 Ha** | 1e-8 Ha |
| E(UHF, H3, broken-symmetry collinear) | −1.33598005401819 Ha | 1e-8 Ha |
| E(UHF, H3, symmetric doublet, ⟨S²⟩ = 0.75) | −1.32817575538128 Ha | 1e-8 Ha |
| ‖P_αβ‖_F (GHF, H3) | 0.9780588 | 1e-3 |
| ⟨S²⟩ (GHF, H3) | 0.840667839 | informational |
| E(UHF = GHF, H2, R = 0.74 Å) | −1.11675930739643 Ha | 1e-8 Ha |

Frustration gain: E_GHF − E_UHF(best collinear) = **−4.460 mHa**.

## Physics

Without spin–orbit coupling every UHF solution is a stationary point of the
GHF functional; on spin-unfrustrated systems the GHF minimum is collinear
and E_GHF = E_UHF (the H2 collapse sentinel). On the equilateral H3
triangle the three spins cannot pairwise anti-align (spin frustration): the
GHF minimum is the coplanar 120° Néel-like arrangement, which real
two-component spinors represent exactly (magnetization confined to the x-z
spin plane in the real algebra).

## Citations

- H. Fukutome, *Unrestricted Hartree-Fock theory and its applications*,
  Int. J. Quantum Chem. **20**, 955 (1981), DOI 10.1002/qua.560200502
  (the eight classes of broken-symmetry HF solutions; GHF = "torsional
  spin density wave" class on frustrated rings).
- T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure
  Theory*, Wiley (2000), §10 (general spin orbitals / spinor mean field).
- R. Seeger, J. A. Pople, J. Chem. Phys. **66**, 3045 (1977),
  DOI 10.1063/1.434318 (constraint hierarchy and stability analysis).
