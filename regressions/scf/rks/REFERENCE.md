# scf/rks

One case; every section below is run by `atv_regressions --case scf/rks` and keeps its own oracle record.


## Reference — H₂O / STO-3G restricted BLYP total energy

### Method

Restricted Kohn–Sham (RKS) with the pure GGA functional **BLYP**
(Becke-1988 exchange + Lee–Yang–Parr correlation), minimal STO-3G Gaussian
basis, single-point total electronic + nuclear-repulsion energy on a fixed
water geometry. Driven by the concrete `cpu::scf::RKS` method with
`XCType::BLYP` through the public Frame API.

BLYP is pure semi-local GGA: `hybrid_exchange_fraction(BLYP) = 0` (no exact
HF exchange). Building blocks `b88_exchange_spin` / `lyp_correlation` are
shared with the B3LYP semi-local mix; BLYP is the unit-weight composite
(B88 + LYP), not the hybrid recipe.

### Reference value

**E_BLYP/STO-3G(H₂O) = −75.277447482665 Hartree**

at the pin geometry below (cart=True, conv_tol=1e-12).

Source: offline PySCF **2.14.0** pin (`dft.RKS`, `xc='blyp'`). This is an
external-engine oracle value, not a golden snapshot of Atomiverse's own
output. The STO-3G basis is from W. J. Hehre, R. F. Stewart, J. A. Pople,
*J. Chem. Phys.* **51**, 2657 (1969), doi:10.1063/1.1672392. B88:
A. D. Becke, *Phys. Rev. A* **38**, 3098 (1988). LYP: C. Lee, W. Yang,
R. G. Parr, *Phys. Rev. B* **37**, 785 (1988); B. Miehlich et al.,
*Chem. Phys. Lett.* **157**, 200 (1989).

### Geometry

PySCF oracle preset `water` from `tools/pyscf_oracle/pin_scf.py` [Å]:

| Atom | x | y | z |
|---|---|---|---|
| O | 0.000000 | 0.000000 | 0.117300 |
| H | 0.000000 | 0.757200 | −0.469200 |
| H | 0.000000 | −0.757200 | −0.469200 |

Derived metrics: R(O–H) ≈ 0.9578 Å, ∠H–O–H ≈ 104.48°. The molecule sits in
a 20 Å cubic box (gas-phase; the box only supplies a non-degenerate cell
and does not enter the isolated-molecule SCF).

### Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True` in PySCF; STO-3G has no d shells, so
  cart/sph is a no-op here but is recorded for protocol consistency)

### Convergence

- Ground-state SCF: `conv_tol = 1e-12` (oracle); Atomiverse uses
  `tol = 1e-10` with default MatchPySCF grid.

### Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

#### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py rks \
  --geom water --basis sto-3g --xc blyp --conv-tol 1e-12
```

#### Pinned JSON excerpt (2026-07-23)

```json
{
  "basis": "sto-3g",
  "cart": true,
  "conv_tol": 1e-12,
  "converged": true,
  "e_tot_ha": -75.27744748266491,
  "geom": "water",
  "method": "rks",
  "pyscf": "2.14.0",
  "xc": "blyp"
}
```

### Tolerance and its rationale

**1 × 10⁻⁶ Ha.** Job gate for this pure-GGA total-energy case: with the
MatchPySCF grid and the same geometry/basis, Atomiverse's native B88+LYP
path is expected to track the PySCF total energy to micro-Hartree. The
window is the regression acceptance band, not a claim of machine-precision
identity of every intermediate.

### What is checked

`cpu::scf::RKS{XCType::BLYP, basis = STO3G}.run(frame)` converges
(`rks.converged()`) and the total energy `rks.energy()` [Hartree] (or
equivalently `frame.meta.get(kEnergy)` [eV] converted via
`units::kHartreeInEv`) lies within 1 × 10⁻⁶ Ha of the pinned
−75.277447482665 Ha. Non-convergence, a non-finite energy, or an
out-of-tolerance value exits non-zero with `got / want / tol`.


## Reference — H₂O / def2-SVP restricted B3LYP total energy (fused-JK path)

### Method

Closed-shell restricted Kohn–Sham **B3LYP** global hybrid on a fixed water
geometry in the def2-SVP Gaussian basis (spherical 5d). The restricted-hybrid
Fock matrix is

\[
F = H_{\text{core}} + J[P] - \tfrac{1}{2}\,\alpha_{\text{hf}}\,K[P] + V_{xc}[\rho],
\qquad \alpha_{\text{hf}} = 0.20,
\]

which is exactly the branch spec **scf-review-06-scf-perf** fuses into a single
`ShellBlockPairList` sweep (`jk_rhf`, one-pass bare J + bare K). This
regression runs TODAY on the two-pass `g_rhf` + `k` path and must keep
reproducing the pinned energy after the fused path lands — the fusion is
algebraically exact (≤1e-12 Ha per-element Fock invariant, ac-003).

def2-SVP places a spherical **d-shell on oxygen**, so the run exercises d-block
exchange quartets — the exact 8-fold permutation multiplicity-unpack the
≤1e-12 fused-JK gate guards against.

Concrete public API (no Config bag, no factory, Frame I/O only):

```cpp
cpu::scf::RKS rks(cpu::scf::BasisSet::Def2SVP, cpu::scf::XCType::B3LYP);
rks.tol = 1e-10;
rks.max_iter = 200;
rks.run(frame);
double e_ha = rks.energy();   // Hartree
```

### Reference value

| Quantity | Value [Hartree] |
|---|---|
| **E_B3LYP total** | **−76.35814183099154** |

### Geometry

PySCF oracle preset `water` from `tools/pyscf_oracle/pin_scf.py` [Å]:

| Atom | x | y | z |
|---|---|---|---|
| O | 0.000000 | 0.000000 | 0.117300 |
| H | 0.000000 | 0.757200 | −0.469200 |
| H | 0.000000 | −0.757200 | −0.469200 |

20 Å cubic box (gas-phase cell only; does not enter the isolated-molecule SCF).

### Basis / angular convention

- Basis: **def2-SVP**
- Angular: **spherical** (`cart=False`; def2 bases are defined with real
  spherical harmonics — 5d on oxygen). Atomiverse `BasisSet::Def2SVP` is
  spherical and runs the c2s sandwich + Fock-build bridge, matching PySCF's
  `cart=False`.

### Convergence / grid

- Oracle SCF: `conv_tol = 1e-12`
- Atomiverse: `tol = 1e-10`, default RKS grid **MatchPySCF** (reproduces
  PySCF's default DFT grid, level = 3).

### Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**). Captured
**2026-07-24**.

#### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py rks \
  --geom water --basis def2-svp --xc b3lyp --conv-tol 1e-12
```

(`rks` builds `dft.RKS(mol)` with `mf.xc = 'b3lyp'`; def2-SVP ⇒ `cart=False`.)

#### Verbatim twin of the call sequence

```bash
~/work/.x86_64/bin/python - <<'EOF'
from pyscf import gto, dft
mol = gto.M(
    atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
    basis="def2-svp",
    unit="Angstrom",
    cart=False,
    verbose=0,
)
mf = dft.RKS(mol, xc="B3LYP")
mf.conv_tol = 1e-12
e = mf.kernel()
print("E_RKS = %.14f Ha  converged=%s" % (e, mf.converged))
EOF
```

#### Pinned JSON (2026-07-24)

```json
{
  "basis": "def2-svp",
  "cart": false,
  "charge": 0,
  "conv_tol": 1e-12,
  "converged": true,
  "e_tot_ha": -76.35814183099154,
  "geom": "water",
  "method": "rks",
  "nao": 24,
  "natm": 3,
  "ok": true,
  "pyscf": "2.14.0",
  "spin": 0,
  "xc": "b3lyp"
}
```

### Tolerance and its rationale

**|E − (−76.35814183099154)| ≤ 1 × 10⁻⁶ Ha** (ac-006).

Rationale: with the MatchPySCF grid and the same geometry / basis / functional,
Atomiverse tracks the PySCF B3LYP total energy to **micro-Hartree** — the pure
HF def2-SVP link already matches PySCF to 1e-8 Ha (`h2o_rhf_def2svp_sph`), and
the B3LYP grid XC term adds no cross-code drift beyond machine noise on this
grid. **Verified before hard-coding:** the Atomiverse run reproduces the pin at
`|diff| = 1.78e-12 Ha` (2026-07-24, worktree build). The 1e-6 Ha gate is a
conservative honest band; the fused-JK path must reproduce it byte-compatibly
(algebraic identity, not a re-derivation).

### What is checked

`cpu::scf::RKS{XCType::B3LYP}` with `BasisSet::Def2SVP` converges the restricted
hybrid KS reference, reports a finite energy, and matches the hard-coded PySCF
`e_tot_ha` within 1e-6 Ha. Runs ≤ 60 s single-CPU. No live PySCF call / no
subprocess at ctest time.

### Literature pointers (method, not numbers)

- A. D. Becke, *J. Chem. Phys.* **98**, 5648 (1993) — B3LYP hybrid form,
  α_hf = 0.20 (PySCF `xc='B3LYP'` parameterization).
- P. J. Stephens et al., *J. Phys. Chem.* **98**, 11623 (1994) — B3LYP with
  VWN correlation as used in practice.
- F. Weigend, R. Ahlrichs, *Phys. Chem. Chem. Phys.* **7**, 3297 (2005) —
  def2-SVP basis.
- G. B. Bacskay, *Chem. Phys.* **61**, 385 (1981) — direct-SCF J+K assembly.
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF.


## `cam-b3lyp-h2-sto3g`

H₂ / STO-3G Cartesian closed-shell **CAM-B3LYP** energy foothold.

### Geometry

| Atom | x / Å | y / Å | z / Å |
| ---- | ----- | ----- | ----- |
| H    | 0     | 0     | 0     |
| H    | 0     | 0     | 0.74  |

Same as `tools/pyscf_oracle/pin_scf.py` preset `h2`.

### Oracle

- **Code:** PySCF 2.14.0 + libcint (offline; never imported by ctest)
- **Date:** 2026-08-17
- **Command:**

```bash
python3 - <<'PY'
from pyscf import gto, dft
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g',
            unit='Angstrom', cart=True, verbose=0)
mf = dft.RKS(mol)
## Same mix as ATV v1: B3LYP semi-local + Yanai (α,β,μ)=(0.19,0.46,0.33)
## on K[1/r] and K[erf(μ r)/r]. LR_HF is erf (libxc/PySCF convention).
mf.xc = ('0.19*HF + 0.46*LR_HF(0.33) + 0.08*LDA_X + 0.72*GGA_X_B88 '
         '+ 0.19*LDA_C_VWN_RPA + 0.81*GGA_C_LYP')
print(mf.kernel())
PY
```

- **Printed energy:** `-1.3067931936345114` Ha
- **Tolerance:** \(1\times 10^{-6}\) Ha

This is **not** stock libxc `cam-b3lyp` (that uses range-separated ITYH
B88 and prints `-1.1561638628742086` Ha on the same geometry). v1 reuses
the existing B3LYP semi-local kernels and only adds the erf-attenuated
exact-exchange operator.

### Product path

`cpu::scf::RKS{XCType::CAMB3LYP}` + `run(frame)`.


## Reference — H₂O / STO-3G restricted B2PLYP double-hybrid total energy

### Method

Restricted Kohn–Sham double hybrid **B2PLYP** (Grimme, *J. Chem. Phys.*
**124**, 034108 (2006), doi:10.1063/1.2148954) on a fixed water geometry in
the STO-3G Gaussian basis:

\[
E_{\mathrm{xc}}
  = (1-a_x)\,E_x^{\mathrm{B88}}
  + a_x\,E_x^{\mathrm{HF}}
  + (1-a_c)\,E_c^{\mathrm{LYP}}
  + a_c\,E_c^{\mathrm{MP2}},
\qquad a_x = 0.53,\quad a_c = 0.27.
\]

Concrete public API (no Config bag, no factory):

```cpp
cpu::scf::RKS rks(cpu::scf::BasisSet::STO3G, cpu::scf::XCType::B2PLYP);
const double a_c = cpu::scf::hybrid_pt2_fraction(cpu::scf::XCType::B2PLYP);
rks.mp2_os_scale = a_c;
rks.mp2_ss_scale = a_c;
auto dh = rks.mp2(frame);   // dh.e_tot [Ha] = E_KS + a_c * E_MP2
```

- Hybrid KS semi-local part: `0.47 * B88 + 0.73 * LYP` via `evaluate_xc`.
- Exact exchange fraction: `hybrid_exchange_fraction(B2PLYP) = 0.53`.
- PT2 scale: `hybrid_pt2_fraction(B2PLYP) = 0.27`, applied through
  `mp2_os_scale` / `mp2_ss_scale` so both spin channels share a_c.
- MP2 is the closed-shell canonical RMP2 kernel on the converged KS orbitals
  and orbital energies (same composition as offline PySCF `mp.MP2(mf)`).

### Reference values

| Quantity | Value [Hartree] |
|---|---|
| E_KS (hybrid only) | −75.198832360592 |
| E_MP2 (unscaled) | −0.044726379786 |
| a_c · E_MP2 | −0.012076122542 |
| **E_B2PLYP total** | **−75.210908483134** |

at the pin geometry below (cart=True, conv_tol=1e-12).

Source: offline PySCF **2.14.0** pin. Hosted libxc in this environment does
not register a named `B2PLYP` composite, so the hybrid-KS part is composed as
`0.53*HF + 0.47*GGA_X_B88 + 0.73*GGA_C_LYP` and PT2 is
`a_c * mp.MP2(mf).e_corr` offline. External-engine oracle — not a golden
snapshot of Atomiverse output.

### Geometry

PySCF oracle preset `water` from `tools/pyscf_oracle/pin_scf.py` [Å]:

| Atom | x | y | z |
|---|---|---|---|
| O | 0.000000 | 0.000000 | 0.117300 |
| H | 0.000000 | 0.757200 | −0.469200 |
| H | 0.000000 | −0.757200 | −0.469200 |

20 Å cubic box (gas-phase cell only; does not enter the isolated-molecule SCF).

### Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True`; STO-3G has no d shells)

### Convergence

- Oracle SCF: `conv_tol = 1e-12`
- Atomiverse: `tol = 1e-10`, default MatchPySCF grid

### Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

#### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py b2plyp \
  --geom water --basis sto-3g --conv-tol 1e-12
```

#### Pinned JSON excerpt (2026-07-23)

```json
{
  "a_c": 0.27,
  "a_x": 0.53,
  "basis": "sto-3g",
  "cart": true,
  "conv_tol": 1e-12,
  "converged": true,
  "e_corr_ha": -0.012076122542341092,
  "e_corr_mp2_ha": -0.044726379786448486,
  "e_corr_pt2_ha": -0.012076122542341092,
  "e_ks_ha": -75.19883236059151,
  "e_tot_ha": -75.21090848313385,
  "geom": "water",
  "method": "b2plyp",
  "pyscf": "2.14.0",
  "xc": "0.53*HF + 0.47*GGA_X_B88 + 0.73*GGA_C_LYP"
}
```

### Tolerance and its rationale

**1 × 10⁻⁶ Ha** on e_ks, e_mp2 (unscaled), a_c·E_MP2, and e_tot. Job gate for
this double-hybrid total-energy case: with the MatchPySCF grid and the same
geometry/basis, Atomiverse is expected to track the composed PySCF B2PLYP
energy to micro-Hartree.

### What is checked

`cpu::scf::RKS{XCType::B2PLYP}` with `mp2_os_scale = mp2_ss_scale = 0.27` and
`mp2(frame)` converges the hybrid KS reference, reports finite energies, and
matches the pinned e_ks / unscaled MP2 / PT2 / e_tot within 1e-6 Ha.
Reconstructs `e_corr == a_c * (e_corr_os + e_corr_ss)` and
`e_tot == e_ks + e_corr`.

### Performance

Fair wall-time vs offline PySCF (same geom/basis/method class; median of ≥3
runs). Protocol: `tools/pyscf_oracle/fair_bench.py`.

```bash
## OMP_NUM_THREADS=1 for a fair single-thread A/B on this tiny system
export OMP_NUM_THREADS=1 OMP_PROC_BIND=close
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method b2plyp --geom water --basis sto-3g --repeat 3 --max-ratio 10 \
  -- ./build/regressions/reg_b2plyp-h2o
```

| T_pyscf median [s] | T_atv median [s] | ratio T_atv/T_pyscf | within_10x |
|---|---|---|---|
| 0.291 | 0.509 | **1.75** | ✓ |

Worktree bench (2026-07-23): `ratio_atv_over_pyscf = 1.7475`, `within_10x = true`.
No optimize pass required (ratio ≤ 10).


## Reference — H₂O RKS-LSDA/STO-3G nuclear Hessian

### Method

Nuclear Hessian of the restricted Kohn–Sham energy via central finite
difference of the **analytic** RKS nuclear forces (`RKS::forces` / KS
gradient with grid-response XC), LSDA (Slater–VWN5) / STO-3G on water.
Public API: `cpu::scf::RKS{XCType::LSDA}.hessian(frame)`. Observable is the
dense Cartesian Hessian in eV/Å² (Frame/vib contract).

Full analytic CPKS (PySCF `hessian.RKS` closed form with nuclear XC second
derivatives) is a follow-up; the offline product pin uses the **same
algorithm class** as Atomiverse (FD of analytic RKS gradients), matching
the ROHF Hessian product path. PySCF's analytic `mf.Hessian().kernel()`
differs from this FD twin by \(\lesssim 4\cdot 10^{-2}\) eV/Å² at
\(h=10^{-4}\) Bohr (recorded below for cross-check only).

### Geometry

PySCF oracle preset `water` from `tools/pyscf_oracle/pin_scf.py` [Å]:

| Atom | x | y | z |
|---|---|---|---|
| O | 0.000000 | 0.000000 | 0.117300 |
| H | 0.000000 | 0.757200 | −0.469200 |
| H | 0.000000 | −0.757200 | −0.469200 |

20 Å cubic box (gas phase; the box does not enter the molecular SCF).

### Pinned PySCF oracle (FD of analytic RKS gradients)

PySCF **2.14.0** (`~/work/.x86_64/bin/python`), run 2026-07-23. Central FD
of `mf.nuc_grad_method().kernel()`, step \(h = 10^{-4}\) Bohr, symmetrized,
converted to eV/Å² with Atomiverse CODATA factors
(`kHartreeInEv = 27.211386245988`, `kBohrInAngstrom = 0.529177210903`):

```python
import numpy as np
from pyscf import gto, dft

HA_TO_EV = 27.211386245988
BOHR_IN_A = 0.529177210903
to_ev_a2 = HA_TO_EV / (BOHR_IN_A**2)

mol0 = gto.M(atom="""
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
""", basis="sto-3g", cart=True, unit="Angstrom", verbose=0)
mf0 = dft.RKS(mol0); mf0.xc = "LDA,VWN"; mf0.conv_tol = 1e-12
e0 = mf0.kernel()                       # -74.732038853874 Ha
coords = mol0.atom_coords().copy()
h = 1e-4
H = np.zeros((3, 3, 3, 3))
for A in range(3):
    for i in range(3):
        def g_at(delta, A=A, i=i):
            c = coords.copy(); c[A, i] += delta
            mol = mol0.copy(); mol.set_geom_(c, unit="Bohr"); mol.build(0, 0)
            mf = dft.RKS(mol); mf.xc = "LDA,VWN"; mf.conv_tol = 1e-12
            mf.kernel()
            return mf.nuc_grad_method().kernel()
        gp, gm = g_at(+h), g_at(-h)
        H[A, :, i, :] = (gp - gm) / (2 * h)
flat = np.zeros((9, 9))
for A in range(3):
    for i in range(3):
        for B in range(3):
            for j in range(3):
                flat[3 * A + i, 3 * B + j] = H[A, B, i, j]
flat = 0.5 * (flat + flat.T)
H_ev = flat * to_ev_a2
```

Cross-check (not the product gate): PySCF analytic `mf.Hessian().kernel()`
at the same geometry differs from this FD twin by at most
\(4.34\cdot 10^{-2}\) eV/Å² (max element abs).

Reference energy (not the product gate): \(E_{\mathrm{RKS}} = -74.732038853874\) Ha.

Gated Cartesian elements (eV/Å²) — FD twin pin:

| (row, col) | Value [eV/Å²] |
|---|---|
| \(H_{xx}(0,0)\) | −9.632120712096 |
| \(H_{yy}(1,1)\) | +89.527387528595 |
| \(H_{zz}(2,2)\) | +54.829425918251 |
| \(H_{xx}(0,3)\) | +4.820104284662 |
| \(H_{xx}(0,6)\) | +4.820104284678 |
| \(H_{yy}(1,4)\) | −44.758907247881 |
| \(H_{yz}(1,5)\) | +38.397479554108 |
| \(H_{yy}(1,7)\) | −44.758906500277 |
| \(H_{yz}(1,8)\) | −38.397479301244 |
| \(H_{zz}(2,5)\) | −27.417403943596 |
| \(H_{yy}(4,4)\) | +47.406667576594 |
| \(H_{zz}(5,5)\) | +26.895192808433 |
| \(H_{yy}(7,7)\) | +47.406666805220 |
| \(H_{zz}(8,8)\) | +26.895192389626 |

### Tolerance and its rationale

**2e-2 eV/Å² per gated element.** Atomiverse and the oracle share the same
algorithm class (FD of analytic RKS gradients/forces, \(h = 10^{-4}\) Bohr),
but the KS integration grids are not bit-identical: Atomiverse
`MatchPySCF` builds 33632 points on this water/STO-3G case while PySCF
`level=3` builds 33704. That residual is visible as \(\sim 2\cdot 10^{-3}\)
eV/Å on the analytic forces and propagates through the FD step into
\(\sim 10^{-2}\) eV/Å² on Hessian elements. The product gate is set at
\(2\cdot 10^{-2}\) (2× that floor) so it still catches sign/factor errors
on stiff blocks (\(\sim 10^2\) eV/Å²) without demanding bit-identical grids.
A full CPKS twin of PySCF `hessian.RKS` at 1e-5 eV/Å² is a follow-up.

### Fair wall-time (offline; not ctest)

Fair twin: Atomiverse `RKS::hessian` (18 force evaluations) vs PySCF loop of
18 RKS analytic gradients (same FD skeleton, \(h=10^{-4}\) Bohr). Analytic
`hessian.RKS` is a different algorithm (closed form, median ~0.094 s) and is
**not** the fair wall twin for this product path.

Recorded 2026-07-23 (`OMP_NUM_THREADS=8`, Release build):

| Side | Median wall [s] |
|---|---|
| Atomiverse `reg_rks-hessian-h2o-sto3g` | 3.138 |
| PySCF FD twin (18× RKS + `nuc_grad`) | 1.019 |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf,FD}}\) | **3.08** |
| within_10x | **true** |
| PySCF analytic `Hessian().kernel()` (cross-check only) | 0.094 |
| ratio vs analytic kernel (unfair) | 33.4 |

### References

- B. G. Johnson, J. A. Pople, P. M. W. Gill, M. J. Frisch,
  *Chem. Phys. Lett.* **220**, 377 (1994) — analytic DFT second derivatives
  (target closed form; this product ships FD of analytic forces).
- PySCF `hessian.rks` / `grad.rks` — external oracle engines.

### What is checked

`cpu::scf::RKS{XCType::LSDA}` with `basis = STO3G`, `tol = 1e-12` →
`rks.hessian(frame)` returns a 9×9 Hessian in eV/Å². Product gate: gated
elements above within **2e-2 eV/Å²** of the pinned PySCF FD twin; failure
prints got / want / tol and exits non-zero. ≤ 60 s, single node.

## `custom-xc-water` — external XC energy and all force components

- **oracle:** external — PySCF 2.14.0, actual output in
  `tools/pyscf_oracle/inputs/xc-extension/water.json`.
- **input / reproduction:** [external XC example](../../../examples/extensions/xc/README.md),
  `0.8*LDA_X,`, STO-3G Cartesian, grid level 3, SCF tolerance 1e-12 Ha.
- **quantity:** total energy and complete nuclear gradient, pins copied verbatim
  into `examples/extensions/xc/pins.h`, F = −gradient converted to eV/Å.
- **tolerances (declared before ATV run):** 1e-6 Ha energy; 2e-5 Ha/Bohr per
  force component for the independently known grid pruning/response difference.
- **consumer:** external `ScaledSlater` subclass; public headers only, same
  molecular grid/Fock/analytic-force implementation as built-in XC. One CPU thread.
