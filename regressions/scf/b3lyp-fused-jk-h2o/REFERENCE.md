# Reference — H₂O / def2-SVP restricted B3LYP total energy (fused-JK path)

## Method

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
cpu::scf::RKS rks(cpu::scf::XCType::B3LYP);
rks.basis = cpu::scf::BasisSet::Def2SVP;
rks.tol = 1e-10;
rks.max_iter = 200;
rks.run(frame);
double e_ha = rks.energy();   // Hartree
```

## Reference value

| Quantity | Value [Hartree] |
|---|---|
| **E_B3LYP total** | **−76.35814183099154** |

## Geometry

PySCF oracle preset `water` from `tools/pyscf_oracle/pin_scf.py` [Å]:

| Atom | x | y | z |
|---|---|---|---|
| O | 0.000000 | 0.000000 | 0.117300 |
| H | 0.000000 | 0.757200 | −0.469200 |
| H | 0.000000 | −0.757200 | −0.469200 |

20 Å cubic box (gas-phase cell only; does not enter the isolated-molecule SCF).

## Basis / angular convention

- Basis: **def2-SVP**
- Angular: **spherical** (`cart=False`; def2 bases are defined with real
  spherical harmonics — 5d on oxygen). Atomiverse `BasisSet::Def2SVP` is
  spherical and runs the c2s sandwich + Fock-build bridge, matching PySCF's
  `cart=False`.

## Convergence / grid

- Oracle SCF: `conv_tol = 1e-12`
- Atomiverse: `tol = 1e-10`, default RKS grid **MatchPySCF** (reproduces
  PySCF's default DFT grid, level = 3).

## Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**). Captured
**2026-07-24**.

### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py rks \
  --geom water --basis def2-svp --xc b3lyp --conv-tol 1e-12
```

(`rks` builds `dft.RKS(mol)` with `mf.xc = 'b3lyp'`; def2-SVP ⇒ `cart=False`.)

### Verbatim twin of the call sequence

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

### Pinned JSON (2026-07-24)

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

## Tolerance and its rationale

**|E − (−76.35814183099154)| ≤ 1 × 10⁻⁶ Ha** (ac-006).

Rationale: with the MatchPySCF grid and the same geometry / basis / functional,
Atomiverse tracks the PySCF B3LYP total energy to **micro-Hartree** — the pure
HF def2-SVP link already matches PySCF to 1e-8 Ha (`h2o_rhf_def2svp_sph`), and
the B3LYP grid XC term adds no cross-code drift beyond machine noise on this
grid. **Verified before hard-coding:** the Atomiverse run reproduces the pin at
`|diff| = 1.78e-12 Ha` (2026-07-24, worktree build). The 1e-6 Ha gate is a
conservative honest band; the fused-JK path must reproduce it byte-compatibly
(algebraic identity, not a re-derivation).

## What is checked

`cpu::scf::RKS{XCType::B3LYP}` with `BasisSet::Def2SVP` converges the restricted
hybrid KS reference, reports a finite energy, and matches the hard-coded PySCF
`e_tot_ha` within 1e-6 Ha. Runs ≤ 60 s single-CPU. No live PySCF call / no
subprocess at ctest time.

## Literature pointers (method, not numbers)

- A. D. Becke, *J. Chem. Phys.* **98**, 5648 (1993) — B3LYP hybrid form,
  α_hf = 0.20 (PySCF `xc='B3LYP'` parameterization).
- P. J. Stephens et al., *J. Phys. Chem.* **98**, 11623 (1994) — B3LYP with
  VWN correlation as used in practice.
- F. Weigend, R. Ahlrichs, *Phys. Chem. Chem. Phys.* **7**, 3297 (2005) —
  def2-SVP basis.
- G. B. Bacskay, *Chem. Phys.* **61**, 385 (1981) — direct-SCF J+K assembly.
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF.
