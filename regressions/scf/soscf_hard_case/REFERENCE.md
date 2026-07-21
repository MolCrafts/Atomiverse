# SOSCF hard cases — CIAH second-order SCF vs PySCF `mf.newton()`

End-to-end case for `hf-capability-gaps-09-soscf`:

1. **Case A (RHF pin)** — stretched N₂, R = 2.5 Å, 6-31G (Cartesian), Hcore
   guess. `soscf = true` must converge onto the PySCF `mf.newton()` solution
   (energy ≤ 1e-8 Ha, HOMO ≤ 1e-6 Ha — stretched N₂ RHF has multiple SCF
   solutions, the HOMO pins the basin), with a Newton-phase macro count
   ≤ 2× the PySCF pin, and fewer total iterations than plain CDIIS.
2. **Case B (UHF stall contrast)** — 2.6×-stretched H₂O, def2-SVP, UHF from
   the Hcore guess (the 07-link hard case, used per the spec's fallback
   allowance after stretched-N₂ RHF turned out to converge fine under the
   engine's CDIIS — no stall contrast there). Plain CDIIS oscillates around
   the unstable spin-symmetric solution for the whole 60-iteration budget;
   the DIIS-early/SOSCF-finish escalation converges onto the pinned
   broken-symmetry minimum.

## Literature

- Q. Sun, *Co-iterative augmented Hessian method for orbital optimization*,
  [arXiv:1610.08423](https://arxiv.org/abs/1610.08423) (2016) — the CIAH
  augmented-Hessian eigenproblem [[0, gᵀ],[g, H]](1; κ) = λ(1; κ) solved by
  σ-vector co-iteration; PySCF's `mf.newton()` implements it.
- T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure
  Theory* (Wiley, 2000), ch. 10 — orbital-rotation parameterization E(κ),
  gradient g_ai = (∂E/∂κ_ai) ∝ F_ai, second-order convergence conditions.
- R. Seeger, J. A. Pople, *Self-consistent molecular orbital methods.
  XVIII*, J. Chem. Phys. **66**, 3045 (1977).
  DOI [10.1063/1.434318](https://doi.org/10.1063/1.434318) — the stability
  rotation used to seed the PySCF broken-symmetry reference run.

## Oracle (PySCF 2.14.0, external — never a build/test dependency)

Interpreter: `~/work/.x86_64/bin/python` (pyscf 2.14.0). 6-31G is a Pople
basis → `cart=True` (engine convention); def2-SVP is spherical → default.
SOSCF macro counts are parsed from the `macro= N` lines of the verbose log:

```python
import io, re
from pyscf import gto, scf

def run_newton(mf, dm0):
    buf = io.StringIO()
    mf.verbose = 4
    mf.stdout = buf
    e = mf.kernel(dm0=dm0)
    macros = re.findall(r"macro= *(\d+)", buf.getvalue())
    return e, mf.converged, (int(macros[-1]) + 1 if macros else -1)
```

### Case A — N₂ R = 2.5 Å, 6-31G (cart), RHF, hcore guess

```python
mol = gto.M(atom="N 0 0 0; N 0 0 2.5", basis="6-31G", cart=True, verbose=0)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
dm0 = mf.get_init_guess(key="hcore")
nt = scf.RHF(mol).newton()
nt.conv_tol = 1e-12
e, conv, n_macro = run_newton(nt, dm0)
homo = nt.mo_energy[mol.nelectron // 2 - 1]
print(conv, n_macro, f"{e:.11f}", f"{homo:.9f}")
```

Output (2026-07-21):

```
A: N2 R=2.5 6-31G cart RHF newton(hcore): conv=True macro=4 E=-108.21122307939 HOMO=-0.304093695
```

Pinned: **E = −108.21122307939 Ha, HOMO = −0.304093695 Ha, 4 macro
cycles**. (Solution multiplicity note: PySCF DIIS from the *minao* guess
lands on a different RHF solution, −108.10642431699 Ha / HOMO −0.36332151 —
the HOMO assertion is what pins the basin.)

### Case B — H₂O 2.6×, def2-SVP, UHF broken-symmetry minimum

```python
stretch = 2.6
hx = 0.7572 * stretch
oz = 0.1173
hz = oz + (-0.4692 - oz) * stretch
mol2 = gto.M(atom=f"O 0 0 {oz}; H 0 {hx} {hz}; H 0 {-hx} {hz}",
             basis="def2-SVP", cart=False, verbose=0)

# The broken-symmetry state + its energy: newton from PySCF's own
# stability-rotated density (Seeger-Pople).
umf = scf.UHF(mol2); umf.conv_tol = 1e-12; umf.max_cycle = 200
umf.kernel()                       # -75.35728186304 (unstable, <S2>=0)
mo1 = umf.stability()[0]
dm_bs = umf.make_rdm1(mo1, umf.mo_occ)
un = scf.UHF(mol2).newton(); un.conv_tol = 1e-12
e2, conv2, n_macro2 = run_newton(un, dm_bs)   # 4 macros, warm seed

# Escalation-equivalent macro pin: engine-like mixed-Hcore guess (20-degree
# alpha HOMO/LUMO mixing = the engine's apply_homo_lumo_mixing), 6 DIIS
# warmup cycles, then newton — the protocol comparable to the engine's
# DIIS-early/SOSCF-finish escalation.
import numpy, scipy.linalg
h = mol2.intor("int1e_kin") + mol2.intor("int1e_nuc")
s_ao = mol2.intor("int1e_ovlp")
e_h, c_h = scipy.linalg.eigh(h, s_ao)
na = nb = 5
theta = 0.3490658503988659
ca = c_h.copy()
homo_c, lumo_c = ca[:, na - 1].copy(), ca[:, na].copy()
ca[:, na - 1] = numpy.cos(theta) * homo_c + numpy.sin(theta) * lumo_c
ca[:, na] = -numpy.sin(theta) * homo_c + numpy.cos(theta) * lumo_c
dm_a = ca[:, :na] @ ca[:, :na].T
dm_b = c_h[:, :nb] @ c_h[:, :nb].T
w = scf.UHF(mol2); w.conv_tol = 1e-12; w.max_cycle = 6
w.kernel(dm0=(dm_a, dm_b))
nk = scf.UHF(mol2).newton(); nk.conv_tol = 1e-12
ek, ck, mk = run_newton(nk, w.make_rdm1())    # 8 macros to the BS minimum
```

Output (2026-07-21):

```
B:   UHF newton(stability dm0):      conv=True macro=4 E=-75.71896632262 S2=1.9710
B'': UHF 6xDIIS-warmup + newton:     conv=True macro=8 E=-75.71896632262 S2=1.9710
```

Pinned: **E_BS = −75.71896632262 Ha, ⟨S²⟩ = 1.9710; escalation-equivalent
newton macro count = 8** (the regression asserts the engine's Newton-phase
count ≤ 2×8).

Seed-sensitivity record (why the escalation-equivalent pin is the honest
one — PySCF newton from colder or shallower seeds does NOT reach the BS
minimum at all):

```
newton(hcore dm0, no mixing)          : macro –, E=-75.35474626683  S2=0.00  (wrong state)
newton(mixed-hcore dm0, no warmup)    : macro 20, E=-75.44063850496 S2=1.01  (wrong state)
newton(3xDIIS warmup from mixed dm0)  : macro 6,  E=-75.35474626683 S2=0.00  (wrong state)
newton(4xDIIS warmup from mixed dm0)  : macro 7,  E=-75.39162489736 S2=0.00  (wrong state)
newton(5xDIIS warmup from mixed dm0)  : macro 4,  E=-75.35728186304 S2=0.00  (wrong state)
newton(6xDIIS warmup from mixed dm0)  : macro 8,  E=-75.71896632262 S2=1.9710 (BS minimum)
newton(stability-rotated converged dm): macro 4,  E=-75.71896632262 S2=1.9710 (BS minimum)
```

The ⟨S²⟩ check (not per-channel HOMOs) pins solution identity because the
BS minimum is a mirror-degenerate pair (α/β swap): the two members have
identical E and ⟨S²⟩ but swapped channel HOMOs (−0.60595 / −0.50190 Ha).

## Engine result (this regression, tol = 1e-9, budget 60)

```
Case A: CDIIS 13 iters conv → -108.21122307947; SOSCF 6 iters (newton 4) → -108.21122307941, HOMO -0.30409370
Case B: CDIIS 60 iters NOT converged (oscillating ~ -75.357);
        SOSCF 13 iters (newton 9) → -75.71896632262, <S2> = 1.9710
```
