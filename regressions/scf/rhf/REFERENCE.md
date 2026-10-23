# scf/rhf

One case; every section below is run by `atv_regressions --case scf/rhf` and keeps its own oracle record.


## Reference — H₂O / STO-3G restricted Hartree–Fock total energy

### Method

Restricted Hartree–Fock (RHF), minimal STO-3G Gaussian basis, single-point
total electronic + nuclear-repulsion energy on a fixed water geometry. Driven
by the concrete `RHF` method through the public Frame API.

**One regression case, multi-device** (PyTorch-style): run
`reg_h2o_rhf_sto3g --device=cpu|cuda|both` (default `both` when built with
CUDA). Same geometry and literature pin on every device — not a separate
`*_cuda` sibling case.

### Reference value

**E_RHF/STO-3G(H₂O) = −74.961754 Hartree** at R(O–H) = 0.95 Å, ∠H–O–H = 104.5°.

Source: LibreTexts, *Live Textbook of Physical Chemistry* (R. Peverati),
§28.2 "Hartree–Fock Calculation for Water", which reports −74.961754063 Ha at
exactly this geometry. Cross-checked against PySCF 2.13 (`RHF`/`STO-3G` at the
same geometry reproduces the value to six decimals). The STO-3G basis itself is
from W. J. Hehre, R. F. Stewart, J. A. Pople, *J. Chem. Phys.* **51**, 2657
(1969), doi:10.1063/1.1672392. This is a cited literature value at the case
geometry, not a golden snapshot of Atomiverse's own output.

### Geometry

O at the origin; the two H symmetric about the +z axis in the xz-plane, so the
O–H bonds subtend the reference angle:

- R(O–H) = 0.95 Å
- ∠H–O–H = 104.5°

Derived Cartesian coordinates [Å], with the half-angle θ = 52.25° and
x_H = R sin θ, z_H = R cos θ — the same two formulas `reg_h2o_rhf_sto3g.cpp`
evaluates:

| Atom | x | y | z |
|---|---|---|---|
| O | 0.000000 | 0.0 | 0.000000 |
| H | +0.751155 | 0.0 | 0.581606 |
| H | −0.751155 | 0.0 | 0.581606 |

These are rounded for reading only. The case computes them in double
precision from R and the angle (0.7511550950566511 / 0.5816064160327268);
do not retype the rounded table into a consumer.

The molecule sits in a 20 Å cubic box (gas-phase; the box only supplies a
non-degenerate cell and does not enter the isolated-molecule SCF).

### Tolerance and its rationale

**1 × 10⁻³ Ha.** The citation geometry **coincides** with the case geometry, so
a tight milli-Hartree sanity window is honest: this checks that the public RHF
path reproduces the reference water energy at the very geometry the reference
quotes. (The engine in fact matches PySCF 2.13 at this geometry to below
1 × 10⁻⁵ Ha; the 1 × 10⁻³ window is a deliberately generous end-to-end sanity
band, not a high-precision benchmark — that belongs to an acceptance-time
external-engine comparison run outside this tree.)

### Provenance note (do not re-pair the old figure)

An earlier draft of this case cited **−74.9420 Ha** as a "Szabo & Ostlund"
water value. That pairing is wrong on two counts, recorded here so it does not
come back:

1. Szabo & Ostlund's worked minimal-basis STO-3G SCF example is **HeH⁺**, not
   water — the water number was misattributed.
2. −74.9420 Ha is the **Crawford Programming Project #3** water energy at a
   *stretched* geometry, R(O–H) = 1.1 Å, ∠H–O–H = 104.0° (reproduced to the
   quoted digits by PySCF 2.13). It is not an equilibrium-geometry literature
   value and does not match this case's 0.95 Å / 104.5° geometry.

The accepted equilibrium-geometry RHF/STO-3G water energy is ≈ −74.96 Ha, as
used above.

### What is checked

For each selected device, `RHF{STO3G}.run(frame)` converges and
the Frame energy `frame.meta.get(kEnergy)` [eV], converted to Hartree via
`units::kHartreeInEv`, lies within 1 × 10⁻³ Ha of the cited −74.961754 Ha.
Non-convergence, a non-finite energy, or an out-of-tolerance value exits
non-zero with `[device] got / want / tol`.

No public integral pack / solver bag: Frame + concrete RHF only.


## Reference — H2O RHF/STO-3G analytic gradient (device s/p force path)

**One case, multi-device:** `reg_h2o_rhf_sto3g_grad --device=cpu|cuda|both`
(default `both` when built with CUDA). Same PySCF pin on every device — not a
`*_cuda` sibling. Public RHF + Frame only.

**Why this case exists:** the device analytic s/p force path
(`launch_rhf_forces_sp`, `cuda::scf::RHF::forces`) is a delivered feature that
previously had only unit coverage (`tests/cuda/scf/test_rhf.cu`). This is its
e2e regression. The CUDA leg is **fail-closed**: it asserts
`last_scf_on_device()` *and* `last_force_on_device()`, so a silent host
fallback (grading `cpu::scf::RHF::forces` under a "cuda" banner) fails the
case instead of passing it. The out-of-coverage counterpart
(`scf/h2o_rhf_def2svp_grad` CUDA leg) asserts that `forces()` **throws**.

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).

| Field | Value |
|---|---|
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| Basis | STO-3G — H2O carries s/p shells only, so Cartesian ≡ spherical |
| Convergence | `conv_tol = 1e-12`, RHF |
| PySCF version | 2.14.0 |
| Command | `~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py rhf --geom water --basis sto-3g --grad --conv-tol 1e-12` |
| **E_RHF** | **−74.963023138463 Ha** (same run as the `mp2-grad-h2o-sto3g` E_HF) |

### Pinned analytic gradient dE/dR [Ha/Bohr]

`mf.nuc_grad_method().kernel()`. Molecule lies in the yz-plane, so all x
components are numerically zero (~1e-16).

| Atom | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| O      | 0 | 0 | −6.142775911808e-02 |
| H (+y) | 0 | −2.364133873480e-02 | +3.071387955904e-02 |
| H (−y) | 0 | +2.364133873480e-02 | +3.071387955904e-02 |

Σ over atoms ≈ (0, 0, 0) to ~1e-15 Ha/Bohr.

### Gates and their rationale

| Leg | Energy [Ha] | Element-wise vs pin [Ha/Bohr] | ‖Σ_A g_A‖ [Ha/Bohr] | Extra |
|---|---|---|---|---|
| cpu | 1e-8 | 1e-6 | 1e-8 | — |
| cuda | 1e-6 | 5e-6 | 1e-7 | `last_scf_on_device()` ∧ `last_force_on_device()`; CPU parity ≤ 2e-6 Ha/Bohr |

Both engines deliver **analytic** gradients, so the CPU leg reproduces PySCF
to well below 1e-6 Ha/Bohr; a wrong/missing Pulay or operator-derivative term
shows up at ≥ 1e-3 Ha/Bohr. The CUDA leg computes the same closed-form
quantities with device kernels whose accumulation order is not bit-identical
to the host, hence the looser 5e-6/1e-7 windows. On top of the shared external
pin the CUDA leg grades **same-settings CPU parity** at 2e-6 Ha/Bohr — the
Ha/Bohr equivalent of the 1e-4 eV/Å CUDA↔CPU force tolerance stated in
`tests/cuda/scf/test_rhf.cu` — per the sanctioned CUDA-ES rule that
same-settings CPU results are the correctness oracle.

Literature: Pulay, *Mol. Phys.* **17**, 197 (1969) (energy-weighted density /
Pulay forces); Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure
Theory* §10 (molecular gradients); Hehre, Stewart, Pople, *J. Chem. Phys.*
**51**, 2657 (1969), doi:10.1063/1.1672392 (STO-3G).

### Measurement log

| Item | Value |
| ---- | ----- |
| GH200 job id | **1039997** (node `n188`, NVIDIA GH200 120GB) — the committed tree |
| Toolchain | `GPU/buildenv-gcccuda/2026.03-cu13.0`, nvcc **13.0.88** / GNU 14.3.0; cubins sm_75 + runtime PTX JIT to ISA 90 |
| GH200 log path | `.claude/reg-cuda-grad-honesty-1039997.log` |
| cpu leg | PASS — max element-wise err **3.870e-11** Ha/Bohr (gate 1e-6); ‖Σ_A g_A‖ 1.133e-14 (gate 1e-8) |
| cuda leg | PASS — `path=device` (both residency flags asserted true); max element-wise err **3.867e-11** Ha/Bohr (gate 5e-6); ‖Σ_A g_A‖ 4.430e-14 (gate 1e-7) |
| CUDA↔CPU parity | **3.200e-14** Ha/Bohr (gate 2e-6) |
| Job verdict | `=== ALL CUDA LEGS PASS ===` (this case + `h2o_rhf_def2svp_grad`, `--device=both`) |


## Reference — H2O RHF/def2-SVP analytic gradient, spherical harmonics (5d)

**One case, multi-device:** `reg_h2o_rhf_def2svp_grad --device=cpu|cuda|both`
(default `both` when built with CUDA). Same PySCF pin on every device — not a
`*_cuda` sibling. Public RHF + Frame only (no integral pack bag).

### Device legs — what `--device=cuda` actually exercises

The CUDA `--device=cuda` leg is a **negative envelope gate, not a gradient
result**. def2-SVP carries d shells, outside the device s/p STO-3G envelope
(`device_sp_sto3g_ok`, `src/cuda/scf/rhf.cu`), so `cuda::scf::RHF::forces`
**throws** `std::invalid_argument` with a how-to pointing at
`cpu::scf::RHF{basis}.forces(frame)`. There is no silent host-force fallback.

If the device envelope ever grows to cover def2-SVP this leg must fail
loudly — rewrite with device tolerances. The fail-closed *device* s/p
gradient e2e is `scf/h2o_rhf_sto3g_grad` (asserts the flags are true).
The PySCF pin below is the **CPU** leg.

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).

| 字段 | 值 |
|---|---|
| 几何 (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| 基组 | def2-SVP, **spherical** (`cart=False`; the set's defining convention) |
| 收敛 | `conv_tol = 1e-12`, RHF |
| PySCF 版本 | 2.14.0 |
| 命令 | `gto.M(atom=..., unit='Angstrom', basis='def2-svp', cart=False)`; `mf=scf.RHF(mol); mf.conv_tol=1e-12; mf.kernel()`; `g = mf.nuc_grad_method().kernel()` |
| **E_RHF** | **−75.96098398708 Ha** (identical to the energy sibling — same run) |
| 对照容差 | 元素级 1e-6 Ha/Bohr;平移不变性 1e-8 Ha/Bohr |

### Pinned analytic gradient dE/dR [Ha/Bohr]

`mf.nuc_grad_method().kernel()` (PySCF has **no** `kick_start`; the entry point
is `.kernel()`). Molecule lies in the yz-plane, so all x components are
numerically zero (~1e-16).

| 原子 | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| O  | 0            | 0             | +1.804450607022e-02 |
| H  | 0            | +1.107031890619e-02 | −9.022253035110e-03 |
| H  | 0            | −1.107031890619e-02 | −9.022253035110e-03 |

Σ over atoms = (−3e-30, +4.2e-15, −4.2e-15) Ha/Bohr — translationally invariant
to ~1e-15 (the regression gate is a looser 1e-8, covering the code's assembly).

`max|∂E/∂R| = 1.804450607022e-02 Ha/Bohr`.

### 为什么 1e-6 是判别阈值

PySCF 与 Atomiverse 都给**解析**梯度,同一 c2s 约定几何无关。正确的 c2s
sandwich(1e 路径)+ 密度反变换(2e 路径)复现 PySCF 到机器精度量级;错误的
约定/缺失的 Pulay 或算符导数项会产生 ≥ 1e-3 Ha/Bohr 的偏差,远高于本阈值。
d 壳层(l=2,6→5)在此几何上被激活,是约定错误最先暴露之处。

文献:Weigend & Ahlrichs, *PCCP* **7**, 3297 (2005), DOI 10.1039/B508541A
(def2-SVP 定义,球谐约定);Schlegel & Frisch, *Int. J. Quantum Chem.* **54**,
83 (1995), DOI 10.1002/qua.560540202(c2s 闭式系数);Pulay(能量加权密度 /
Pulay 力);Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure
Theory* §10(分子梯度)。

### Measurement log

GH200 job **1039997** (node `n188`, `GPU/buildenv-gcccuda/2026.03-cu13.0`,
nvcc 13.0.88; log `.claude/reg-cuda-grad-honesty-1039997.log`): cpu leg PASS
(max element-wise err 1.681e-09 Ha/Bohr, gate 1e-6; ‖Σ_A g_A‖ 3.195e-14);
cuda leg was then a documented host-fallback. That fallback is **deleted**:
out-of-coverage `cuda::scf::RHF::forces` now throws. Re-gate the throw on
the next GH200 CUDA job.


## Reference — H2O RHF/def2-SVP, spherical harmonics (5d)

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).

| 字段 | 值 |
|---|---|
| 几何 (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| 基组 | def2-SVP, **spherical** (`cart=False`; the set's defining convention) |
| 收敛 | `conv_tol = 1e-12`, RHF |
| PySCF 版本 | 2.14.0 |
| 命令 | `gto.M(atom=..., unit='Angstrom', basis='def2-svp', cart=False)`; `scf.RHF(mol).kernel()` |
| **E_RHF** | **−75.96098398708 Ha** |
| 对照容差 | 1e-8 Ha |

对照锚点(同几何、同版本):cartesian(6d)运行 `cart=True` 给
E = −75.9622197549911 Ha,nao 25 → 24;两者之差 ~1.2 mHa 即 6d 的 s-污染
分量的变分贡献 — 教程 ch04「笛卡尔与球谐高斯基」一节引用本数据。

文献:Weigend & Ahlrichs, *PCCP* **7**, 3297 (2005), DOI 10.1039/B508541A
(def2-SVP 定义,球谐约定);Schlegel & Frisch, *Int. J. Quantum Chem.* **54**,
83 (1995), DOI 10.1002/qua.560540202(c2s 闭式系数)。


## Reference — 弥散函数与阴离子:OH⁻ ε_HOMO 束缚性 + Cl⁻/H2O 总能量

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md(外部运行、钉定;
PySCF 绝不进 CMake/ctest)。

### 物理背景与文献

- 阴离子的多余电子结合松散,紧凑基组缺少慢衰减(弥散)函数,HOMO 呈非
  束缚或勉强束缚伪像;增广弥散函数恢复束缚:Kendall, Dunning, Harrison,
  *J. Chem. Phys.* **96**, 6796 (1992), DOI 10.1063/1.462569(aug-cc 增广
  弥散基组的定义文献)。
- 6-31G* 极化函数:Hariharan, Pople, *Theor. Chim. Acta* **28**, 213
  (1973), DOI 10.1007/BF00533485;6-31G 分裂价基:Ditchfield, Hehre,
  Pople, *J. Chem. Phys.* **54**, 724 (1971), DOI 10.1063/1.1674902。
- 本几何(r(OH) = 0.9640 Å)下的实测(PySCF)符号序:STO-3G 的 ε_HOMO
  为 **+0.2519 Ha(非束缚伪像)**,6-31G* 已勉强束缚(−0.0378 Ha),
  6-31+G* 弥散 sp 使其加深近 3 倍(−0.1036 Ha)。回归据此断言:STO-3G
  伪像为正、6-31+G* 严格低于 6-31G* 且与钉值同号,逐值对钉 ≤1e-6 Ha。

### 钉定值(全部 PySCF 2.14.0,`conv_tol = 1e-12`)

| 体系 | 基组(约定)| E_RHF [Ha] | ε_HOMO [Ha] | 回归容差 |
|---|---|---|---|---|
| OH⁻ | STO-3G (cart) | −74.056328130466 | **+0.251911715713** | ε ≤1e-6 |
| OH⁻ | 6-31G* (cart) | −75.326596547338 | −0.037774390226 | ε ≤1e-6 |
| OH⁻ | 6-31+G* (cart) | −75.376293363750 | −0.103579773338 | E ≤1e-7, ε ≤1e-6 |
| Cl⁻ | 6-31+G* (cart) | −459.539661044387 | (−0.150252200261) | E ≤1e-7 |
| H2O | aug-cc-pVDZ (sph) | −76.041393519987 | (−0.509406243328) | E ≤1e-7 |

约定:Pople 系(6-31G*/6-31+G*、STO-3G)在 PySCF 侧跑 `cart=True`
(笛卡尔 6d,匹配引擎的 per-set AngularConvention::Cartesian);
aug-cc-pVDZ 跑默认 `cart=False`(球谐,匹配 AngularConvention::Spherical)。

### 几何(Å;C++ 回归与 PySCF 钉定逐字相同)

- OH⁻:O(0, 0, 0); H(0, 0, 0.9640);charge = −1。
- Cl⁻:单原子 Cl(0, 0, 0);charge = −1。
- H2O:O(0, 0, 0.1173); H(0, ±0.7572, −0.4692)(与
  regressions/scf/properties / h2o_rhf_def2svp_sph 同一几何)。

### 钉定命令(逐字;venv `~/work/.x86_64/bin/python`,pyscf 2.14.0)

```python
from pyscf import gto, scf

def run(atom, basis, charge, cart):
    mol = gto.M(atom=atom, basis=basis, charge=charge, cart=cart,
                unit="Angstrom")
    mf = scf.RHF(mol); mf.conv_tol = 1e-12
    e = mf.kernel()
    homo = max(mf.mo_energy[mf.mo_occ > 0])
    return e, homo

oh = "O 0 0 0; H 0 0 0.9640"
run(oh, "sto-3g",  -1, True)   # E=-74.056328130466, eps=+0.251911715713
run(oh, "6-31g*",  -1, True)   # E=-75.326596547338, eps=-0.037774390226
run(oh, "6-31+g*", -1, True)   # E=-75.376293363750, eps=-0.103579773338
run("Cl 0 0 0", "6-31+g*", -1, True)   # E=-459.539661044387
h2o = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
run(h2o, "aug-cc-pvdz", 0, False)      # E=-76.041393519987
```

### 已知偏差(有意为之,已文档化)

spec 叙事预期"6-31G* 下 ε_HOMO ≥ 0 伪像";实测该伪像出现在 STO-3G
(+0.2519 Ha),6-31G* 在本几何下已勉强束缚(−0.0378 Ha)。回归遵循
acceptance ac-006 的字面判据(逐基组对钉、6-31+G* 严格更低、与钉值同
号),并加钉 STO-3G 行以保留真实的"非束缚→束缚"符号翻转演示。见 spec
Design 的实现审计注记。


## H2/STO-3G RHF stability + broken-symmetry UHF — reference values

### Literature basis

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

### Oracle: PySCF 2.14.0 (external run — nothing here reads PySCF)

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

### Pinned values (Hartree)

| Quantity | R = 0.735 Å | R = 1.5 Å |
|---|---|---|
| E_RHF | −1.1169989968 | −0.9108735546 |
| λ_min ¹(A+B) (internal) | +1.1351770407 | +0.7088965814 |
| λ_min ³(A+B) (external) | +0.4114522416 | −0.2092471629 |
| PySCF `mf.stability` verdict | stable / stable | internal stable / external **unstable** |
| E_UHF (broken symmetry) | — | −0.9577067934 (⟨S²⟩ = 0.6949) |
| E_UHF − E_RHF | — | −0.0468332388 |

### What the regression checks

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


## SCF convergence layer — initial guesses + far-field mixers (H2O/def2-SVP)

End-to-end case for `hf-capability-gaps-07-guess`:

1. **Guess quality** (H2O/def2-SVP RHF, equilibrium geometry): iteration
   counts must order `Hueckel < Hcore` and `SAP < Hcore`, and every guess
   must converge to the pinned RHF energy.
2. **Far-field mixer contrast** (2.6×-stretched H2O/def2-SVP UHF): plain
   CDIIS stalls within a 60-iteration budget on the *unstable*
   spin-symmetric solution, while `ADIIS→CDIIS` converges to the pinned
   broken-symmetry UHF minimum within 1e-6 Ha.

### Literature

- S. Lehtola, *Assessment of Initial Guesses for Self-Consistent Field
  Calculations. Superposition of Atomic Potentials: Simple yet Efficient*,
  J. Chem. Theory Comput. **15**, 1593 (2019).
  DOI [10.1021/acs.jctc.8b01089](https://doi.org/10.1021/acs.jctc.8b01089)
  — extended-Hückel and SAP initial guesses.
- K. N. Kudin, G. E. Scuseria, E. Cancès, *A black-box self-consistent
  field convergence algorithm: One step closer*, J. Chem. Phys. **116**,
  8255 (2002). DOI
  [10.1063/1.1470195](https://doi.org/10.1063/1.1470195) — EDIIS.
- X. Hu, W. Yang, *Accelerating self-consistent field convergence with the
  augmented Roothaan–Hall energy function*, J. Chem. Phys. **132**, 054109
  (2010). DOI [10.1063/1.3304922](https://doi.org/10.1063/1.3304922)
  — ADIIS.

### Oracle (PySCF 2.14.0, external — never a build/test dependency)

Interpreter: `~/work/.x86_64/bin/python` (pyscf 2.14.0, def2-SVP spherical).

#### Part 1 — equilibrium H2O iteration counts + energy pin

```python
from pyscf import gto, scf
mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
            basis="def2-svp", verbose=0)
for ig in ["hcore", "minao", "huckel", "sap"]:
    mf = scf.RHF(mol)
    mf.init_guess = ig
    mf.conv_tol = 1e-9
    e = mf.kernel()
    print(f"init_guess={ig:7s} cycles={mf.cycles:3d} converged={mf.converged} E={e:.11f}")
mf = scf.RHF(mol); mf.conv_tol = 1e-12
print("RHF equilibrium E(conv_tol=1e-12) =", f"{mf.kernel():.11f}")
```

Output (2026-07-21):

```
init_guess=hcore   cycles= 10 converged=True E=-75.96098398708
init_guess=minao   cycles=  8 converged=True E=-75.96098398708
init_guess=huckel  cycles=  8 converged=True E=-75.96098398708
init_guess=sap     cycles=  8 converged=True E=-75.96098398708
RHF equilibrium E(conv_tol=1e-12) = -75.96098398708
```

Pinned: **E_RHF(eq) = −75.96098398708 Ha**; PySCF guess ordering
hcore(10) > minao(8) = huckel(8) = sap(8). (PySCF's `huckel` is Lehtola's
parameter-free on-the-fly variant; Atomiverse implements the static-VSIP
extended Hückel from the same paper, so only the *ordering vs. hcore* is
comparable, not the exact count.)

#### Part 2 — stretched-H2O UHF broken-symmetry pin

Geometry = regression geometry at stretch 2.6:
`hx = 0.7572·2.6 = 1.96872`, `hz = 0.1173 − 0.5865·2.6 = −1.40760` Å.

```python
from pyscf import gto, scf
mol = gto.M(atom="O 0 0 0.1173; H 0 1.96872 -1.40760; H 0 -1.96872 -1.40760",
            basis="def2-svp", spin=0, verbose=0)
mf = scf.UHF(mol)
mf.conv_tol = 1e-12
mf.max_cycle = 500
mf.kernel()
print("first solution:", f"{mf.e_tot:.11f}", mf.converged)
for it in range(8):
    mo1 = mf.stability()[0]
    if mo1 is mf.mo_coeff:
        print("stable after", it, "rotations"); break
    mf.kernel(dm0=mf.make_rdm1(mo1, mf.mo_occ))
print("final UHF E =", f"{mf.e_tot:.11f}", "converged:", mf.converged)
print("<S^2> =", mf.spin_square()[0])
```

Output (2026-07-21):

```
first solution: -75.35728186304 True
  restarted -> -75.71896632261 True
stable after 1 rotations
final UHF E = -75.71896632261 converged: True
<S^2> = 1.9709626405956904
```

Pinned: **E_UHF(broken symmetry) = −75.71896632261 Ha** (⟨S²⟩ ≈ 1.97).
Note that PySCF's own default-guess UHF also lands first on the *unstable*
spin-symmetric solution (−75.35728186304) — the very state the plain-CDIIS
engine run oscillates around without converging, which is what makes this a
genuine far-from-convergence case rather than an engine artifact.

### Engine evidence (Atomiverse, this machine, 2026-07-21)

```
H2O/def2-SVP RHF (tol=1e-9), iterations per guess:
  hcore    iters= 16   minao iters= 16   hueckel iters= 14   sap iters= 14
2.6x-stretched H2O/def2-SVP UHF (tol=1e-9, hcore guess, damping 0,
diis_start_iter 1, budget 60):
  CDIIS alone : 60 iterations, NOT converged, E ~ -75.357 (symmetric state)
  ADIIS->CDIIS: 38 iterations, converged, E = -75.71896632 Ha
```

### Acceptance

- Every equilibrium guess converges to −75.96098398708 Ha within 1e-7.
- `iters(Hueckel) < iters(Hcore)` and `iters(SAP) < iters(Hcore)`.
- Stretched UHF: CDIIS-alone **not** converged in 60 iterations;
  ADIIS→CDIIS converged with |E − (−75.71896632261)| ≤ 1e-6 Ha.
- Runtime ≤ 60 s, single process.


## SOSCF hard cases — CIAH second-order SCF vs PySCF `mf.newton()`

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

### Literature

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

### Oracle (PySCF 2.14.0, external — never a build/test dependency)

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

#### Case A — N₂ R = 2.5 Å, 6-31G (cart), RHF, hcore guess

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

#### Case B — H₂O 2.6×, def2-SVP, UHF broken-symmetry minimum

```python
stretch = 2.6
hx = 0.7572 * stretch
oz = 0.1173
hz = oz + (-0.4692 - oz) * stretch
mol2 = gto.M(atom=f"O 0 0 {oz}; H 0 {hx} {hz}; H 0 {-hx} {hz}",
             basis="def2-SVP", cart=False, verbose=0)

## The broken-symmetry state + its energy: newton from PySCF's own
## stability-rotated density (Seeger-Pople).
umf = scf.UHF(mol2); umf.conv_tol = 1e-12; umf.max_cycle = 200
umf.kernel()                       # -75.35728186304 (unstable, <S2>=0)
mo1 = umf.stability()[0]
dm_bs = umf.make_rdm1(mo1, umf.mo_occ)
un = scf.UHF(mol2).newton(); un.conv_tol = 1e-12
e2, conv2, n_macro2 = run_newton(un, dm_bs)   # 4 macros, warm seed

## Escalation-equivalent macro pin: engine-like mixed-Hcore guess (20-degree
## alpha HOMO/LUMO mixing = the engine's apply_homo_lumo_mixing), 6 DIIS
## warmup cycles, then newton — the protocol comparable to the engine's
## DIIS-early/SOSCF-finish escalation.
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

### Engine result (this regression, tol = 1e-9, budget 60)

```
Case A: CDIIS 13 iters conv → -108.21122307947; SOSCF 6 iters (newton 4) → -108.21122307941, HOMO -0.30409370
Case B: CDIIS 60 iters NOT converged (oscillating ~ -75.357);
        SOSCF 13 iters (newton 9) → -75.71896632262, <S2> = 1.9710
```


## Reference — staged CUDA RHF driver, H₂/STO-3G and H₂O/6-31G

### Method

Restricted Hartree–Fock total energy through the public Frame API.

- CPU: `cpu::scf::RHF{basis, charge, tol, max_iter}.run(frame)`.
- CUDA: `cuda::scf::RHF` is `cuda::Driver<scf::Phase>` (fock, mix,
  diagonalize, density, check). The H₂/STO-3G leg must report
  `last_scf_on_device()` and `last_force_on_device()`. A host ERI fill
  under a CUDA banner is not a pass.

**One regression, multi-device.** `reg_es-cuda-rhf-driver --device=cpu|cuda|both`
(default `both` when built with CUDA). The `.cpp` is the CUDA-off path.

### Oracle

Offline PySCF 2.14.0, `tools/pyscf_oracle/pin_scf.py`, `conv_tol=1e-12`,
Cartesian (`cart=True`; neither basis has a d shell, so pure vs Cartesian
does not change the energy).

```
python tools/pyscf_oracle/pin_scf.py rhf --geom h2 --basis sto-3g
python tools/pyscf_oracle/pin_scf.py rhf --geom water --basis 6-31g
```

| System | Basis | Geometry [Å] | E_RHF [Ha] |
| --- | --- | --- | --- |
| H₂ | STO-3G | H (0,0,0), H (0,0,0.74) | −1.1167593073964255 |
| H₂O | 6-31G | O (0,0,0.1173), H (0, ±0.7572, −0.4692) | −75.98397447272204 |

`oracle: pyscf` 2.14.0 (`pyscf.scf.RHF.kernel`). Not an Atomiverse measurement.

### Tolerance

Same bands as `scf/h2o_rhf_sto3g_grad` for the RHF total energy:

| Device | \|E − E_PySCF\| |
| --- | --- |
| cpu | 1e-8 Ha |
| cuda (H₂/STO-3G only) | 1e-6 Ha |

H₂O/6-31G is outside the device s/p STO-3G integral kernel (`device_sp_sto3g_ok`).
The CUDA leg does **not** run it. Grading the transitional host-ERI fill as a
device result is forbidden. The 6-31G pin is the CPU leg only.

### What is checked

Both systems converge and `energy()` [Hartree] and `kEnergy` [eV] / Ha agree
with the pin inside the device band. The CUDA H₂ leg additionally requires
the device SCF and analytic-force flags. Non-convergence, a host-integral
CUDA path, or an out-of-tolerance energy exits non-zero.


## Reference — H₂O RHF, Cartesian d (6-31G*) and Cartesian f (def2-TZVP)

**One case, multi-device:** `reg_hf-cuda-lgt1 --device=cpu|cuda|both`
(default `both` when built with CUDA). Same PySCF pins on every device.
Public `RHF` + Frame only.

### Systems

Geometry [Å], the `water` preset of `tools/pyscf_oracle/pin_scf.py`:

| Atom | x | y | z |
|---|---|---|---|
| O | 0 | 0 | 0.1173 |
| H | 0 | 0.7572 | −0.4692 |
| H | 0 | −0.7572 | −0.4692 |

20 Å cubic box (gas phase; the cell is not used by the isolated-molecule SCF).

| System | Basis | What l > 1 it exercises | PySCF `cart` | nao | E_RHF [Ha] |
|---|---|---|---|---|---|
| H₂O | 6-31G* | Cartesian d on O (6 functions) | true | 19 | −76.01050498825705 |
| H₂O | def2-TZVP | Cartesian f on O (10 functions), then the basis's spherical sandwich (7f / 5d) | false | 43 | −76.05900476057711 |

def2-TZVP is a spherical basis. The device (and the CPU) still **evaluates**
the oxygen f shell as ten Cartesian GTOs (`fill_cart_comps`, libcint order)
and the oxygen d shells as six, then applies the same `CartToSph` sandwich
as `IntegralWorkspace` (`integrals.cpp`). A Cartesian-only PySCF run
(`--cart`, nao 48) prints **−76.05941121596425 Ha** and is **not** this pin:
that number is the unsandwiched basis, which the public `BasisSet::Def2TZVP`
does not publish.

H₂ / STO-3G and H₂O / 6-31G never put l > 1 on an atom, so they are not
this case.

### Oracle

PySCF 2.14.0, `conv_tol = 1e-12`, offline:

```
python tools/pyscf_oracle/pin_scf.py rhf --geom water --basis 6-31g*
python tools/pyscf_oracle/pin_scf.py rhf --geom water --basis def2-tzvp
```

`cart` follows the script default (Pople Cartesian, def2 spherical).

CPU tolerance **1e-8 Ha**, the same window as H₂O RHF/def2-SVP
(`regressions/scf/rhf`). The CPU leg matches the pin to
~1e-12 Ha.

CUDA tolerance **1e-6 Ha**, the same window as the device H₂/STO-3G energy
pin. The device McMurchie–Davidson ERI is not bit-identical to the CPU Rys
engine; on this geometry the device is 3e-8 Ha (6-31G*) and 4e-7 Ha
(def2-TZVP) from the pin.

### Device leg

`cuda::scf::RHF` with `with_gradient(false)` (analytic forces outside
STO-3G l ≤ 1 still throw `launch_rhf_forces_lgt1`). The SCF must report
`last_scf_on_device()` and must not report `last_force_on_device()`.
There is no host ERI fill. l ≥ 4 throws `launch_eri_g`.


## Reference — H₂O dimer HF/6-31G* Boys–Bernardi counterpoise

### Method

Restricted Hartree–Fock (RHF), Pople 6-31G\* (Cartesian), single-point
Boys–Bernardi counterpoise (CP) correction on a fixed water-dimer geometry.
Driven by the public `cpu::scf::Counterpoise` orchestrator over
`cpu::scf::RHF` (five serial SCFs: dimer, free A, free B, A-with-B-ghost,
B-with-A-ghost). Frame is the sole geometry currency; ghost markers live on
the method (`ghost_atoms`), never on Frame.

### Literature

Boys, S. F. & Bernardi, F. *Mol. Phys.* **19**, 553 (1970).
DOI [10.1080/00268977000101561](https://doi.org/10.1080/00268977000101561).

Sign convention (locked in code, tests, and tutorial):

\[
\mathrm{bsse} = (E_A^{AB} - E_A^{A}) + (E_B^{AB} - E_B^{B}) \le 0
\]

\[
\Delta E_{\mathrm{CP}} = E_{AB}^{AB} - E_A^{AB} - E_B^{AB}
\]

Monomer energy lowers in the larger dimer basis, so BSSE is non-positive;
\(|\mathrm{bsse}|\) is the magnitude of the basis-set superposition error.

### Geometry

Fixed water-dimer Cartesian coordinates [Å]:

| Atom | role | x | y | z |
|---|---|---:|---:|---:|
| O | A | −1.551007 | −0.114520 | 0.000000 |
| H | A | −1.934259 | 0.762503 | 0.000000 |
| H | A | −0.599677 | 0.040712 | 0.000000 |
| O | B | 1.350625 | 0.111469 | 0.000000 |
| H | B | 1.680398 | −0.373741 | −0.758561 |
| H | B | 1.680398 | −0.373741 | 0.758561 |

Monomer A = indices `{0,1,2}`; monomer B = `{3,4,5}`. Gas-phase 30 Å cubic
box (cell does not enter the isolated-molecule SCF).

### Externally pinned PySCF values

Oracle engine: **PySCF 2.14.0**, `scf.RHF`, basis `'6-31g*'`, **`cart=True`**
(matching Atomiverse Pople Cartesian convention; PySCF's default is spherical
and would disagree by ~0.01 mHa on the CP energy), `unit='Angstrom'`, ghost
atoms via the `ghost:O` / `ghost:H` element prefix. Convergence default.
Dimer AO count = 38 (Cartesian 6-31G\*).

```python
from pyscf import gto, scf

geom = """
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
"""
e_dimer = scf.RHF(gto.M(atom=geom, basis="6-31g*", unit="Angstrom",
                        cart=True, verbose=0)).kernel()
## free monomers, then ghost: prefixes for A-in-AB and B-in-AB
## → e_a, e_b, e_a_ghost, e_b_ghost  (all with cart=True)
e_int_cp = e_dimer - e_a_ghost - e_b_ghost
bsse = (e_a_ghost - e_a) + (e_b_ghost - e_b)
```

Pinned scalars [Hartree] (cart=True):

| Quantity | PySCF 2.14.0 |
|---|---:|
| \(E_{AB}^{AB}\) | −152.02982896658477 |
| \(E_A^{A}\) | −76.01034691284323 |
| \(E_B^{B}\) | −76.01045593149992 |
| \(E_A^{AB}\) (B ghost) | −76.01072014609187 |
| \(E_B^{AB}\) (A ghost) | −76.01156637927771 |
| \(\Delta E_{\mathrm{raw}}\) | −0.00902612224162 |
| \(\Delta E_{\mathrm{CP}}\) | **−0.00754244121519** |
| bsse | **−0.00148368102643** (≈ 1.48 mHa) |

These are **external** pins, not golden snapshots of Atomiverse output.

### Tolerance and its rationale

**1 × 10⁻⁶ Ha** on `e_int_cp` and `bsse`. Same-geometry, same Cartesian
basis, pure HF integral path — micro-Hartree agreement with PySCF is the
honest acceptance window for this chain link. The CP correction magnitude
(~1.5 mHa) sits in the published milli-Hartree class for HF/6-31G\* water
dimer BSSE.

### What is checked

`cpu::scf::Counterpoise({0,1,2},{3,4,5}).run(dimer, rhf)` with
`cpu::scf::RHF rhf{Pople631Gs}` yields `e_int_cp` and `bsse` within 1e-6 Ha of the
PySCF pins above; `bsse ≤ 0`; `|bsse|` in [0.1, 50] mHa. Non-finite or
out-of-tolerance values exit non-zero with `got / want / tol`.


## H2O / def2-TZVP RI-JK — external oracle pins

Case: RHF with RI-JK density fitting over the def2 universal JKFIT
auxiliary basis, double-gated against (1) our own exact-ERI RHF and
(2) the externally pinned PySCF `density_fit()` energy.

### Literature basis

- Robust Coulomb-metric fit (error second order in the fitting
  residual): O. Vahtras, J. Almlöf, M. W. Feyereisen,
  *Chem. Phys. Lett.* **213**, 514 (1993),
  DOI [10.1016/0009-2614(93)89151-7](https://doi.org/10.1016/0009-2614(93)89151-7).
- RI-K and the universal JKFIT auxiliary basis: F. Weigend,
  *Phys. Chem. Chem. Phys.* **4**, 4285 (2002),
  DOI [10.1039/B204199P](https://doi.org/10.1039/B204199P).
- Accuracy of the Coulomb fit / auxiliary basis matching: F. Weigend,
  *Phys. Chem. Chem. Phys.* **8**, 1057 (2006),
  DOI [10.1039/b515623h](https://doi.org/10.1039/b515623h).

### Oracle runs (PySCF 2.14.0, `~/work/.x86_64/bin/python`, 2026-07-21)

Command, verbatim:

```python
import numpy as np
from pyscf import gto, scf, df

mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
            basis='def2-tzvp')  # spherical (PySCF default, cart=False)
mf = scf.RHF(mol); mf.conv_tol = 1e-12
e_exact = mf.kernel()
mfd = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit')
mfd.conv_tol = 1e-12
e_df = mfd.kernel()
print('H2O/def2-TZVP  E_exact  = %.12f' % e_exact)
print('H2O/def2-TZVP  E_DF-JK  = %.12f' % e_df)
print('DF - exact = %.3e Ha' % (e_df - e_exact))
```

Output:

```
H2O/def2-TZVP  E_exact  = -76.059004760577
H2O/def2-TZVP  E_DF-JK  = -76.058999347289
DF - exact = 5.413e-06 Ha
```

### Pinned values and gates

| Quantity | Value | Gate |
|---|---|---|
| PySCF exact-ERI RHF | −76.059004760577 Ha | (context only) |
| PySCF DF (`density_fit`, def2-universal-jkfit) | **−76.058999347289 Ha** | our E(RI-JK) within **1e-6 Ha** |
| PySCF fit error (DF − exact) | +5.413e-6 Ha | our \|E(RI-JK) − E(own exact)\| ≤ **5e-5 Ha** |

Gate 1 (≤ 50 μHa vs our own exact-ERI HF) directly tests the
second-order-error claim of the Coulomb-metric robust fit with a
matched auxiliary basis; PySCF's own fit error (5.4 μHa) shows the
expected order of magnitude. Gate 2 (≤ 1e-6 Ha vs the PySCF DF pin)
proves the fit lives in the same *spherical* def2-universal-jkfit
auxiliary span — a Cartesian-span fit would pass gate 1 but drift off
gate 2.

Geometry (Å): O(0, 0, 0.1173), H(0, ±0.7572, −0.4692) — identical to
regressions/scf/h2o_rhf_sto3g.


## C10H22 / STO-3G CFMM-J — reference

### What is under test

Continuous fast multipole method (CFMM) for the HF Coulomb matrix J on
n-decane / STO-3G, with multipole order `cfmm_order = 10` and extent
threshold `cfmm_threshold = 1e-9`. Exchange (K) stays on the exact
Schwarz path.

### Oracle

**In-tree Schwarz-screened direct J** (the vacuum path already pinned by
PySCF on earlier chain links). **PySCF does not ship an FMM/CFMM** for
molecular Coulomb — its default is direct SCF + Schwarz (optional DF).
There is therefore no external FMM numerical oracle; correctness is
defined against our own Direct path.

### Citations

- White, Johnson, Gill, Head-Gordon, *Chem. Phys. Lett.* **230**, 8
  (1994), DOI [10.1016/0009-2614(94)01128-1](https://doi.org/10.1016/0009-2614(94)01128-1)
  — Gaussian extent classification and continuous FMM.
- Greengard & Rokhlin, *J. Comput. Phys.* **73**, 325 (1987) — FMM
  skeleton (tree operators from chain link 20).

### Gates

| Gate | Quantity | Tolerance |
|------|----------|-----------|
| Energy | \|E_CFMM − E_Direct\| | ≤ 1e-7 Ha |
| J matrix (unit test) | max_ij \|J_CFMM − J_direct\| on H2O/STO-3G | ≤ 1e-8 Ha |

The elementwise J hard gate lives in `tests/cpu/scf/test_cfmm_j.cpp`
(out-of-library regressions cannot reach the private J matrix). This
regression pins the **converged total energy** dual gate on C10H22.

### Geometry

All-trans n-decane skeleton (C–C 1.54 Å, C–H 1.09 Å), 32 atoms, cubic
box 80 Å. Coordinates generated in `reg_c10h22_cfmm_j.cpp`.


## Reference — RHF `run(const Frame&)`, H₂/STO-3G and H₂O/STO-3G

**One section, multi-device:** `sto3g_const --device=cpu|cuda|both`
(default `both` when built with CUDA). The `.cpp` is the CUDA-off path.
Public `RHF` + Frame only.

`run(const Frame&)` is the geometry-read-only entry. It runs the same SCF
as `run(Frame&)` — PySCF `scf.hf.kernel`, with the total
`energy_elec` + nuclear repulsion (`energy_tot`) — and writes nothing.
`energy()` holds that total [Hartree]. `kEnergy`, `kEnergyElectronic`,
positions, and `kAtom_f*` keep the caller's values. On CUDA,
`compute_force` stays true and the const entry still does not call
`launch_rhf_forces_sp`.

### Oracle

Offline PySCF 2.14.0, `tools/pyscf_oracle/pin_scf.py`, `conv_tol=1e-12`
(the script default), Cartesian (`cart=True`; STO-3G has no d shell).

```
python tools/pyscf_oracle/pin_scf.py rhf --geom h2 --basis sto-3g
python tools/pyscf_oracle/pin_scf.py rhf --geom water --basis sto-3g
```

| System | Geometry [Å] | E_RHF [Ha] |
| --- | --- | --- |
| H₂ | H (0,0,0), H (0,0,0.74) — preset `h2` | −1.1167593073964255 |
| H₂O | O (0,0,0.1173), H (0, ±0.7572, −0.4692) — preset `water` | −74.96302313846289 |

`oracle: pyscf` 2.14.0 (`pyscf.scf.RHF.kernel` / `e_tot_ha`). Not an
Atomiverse measurement. Printed by PySCF 2.14.0 as `"e_tot_ha": -1.1167593073964255` (H₂) and
`"e_tot_ha": -74.96302313846289` (H₂O), both `"converged": true`.
The H₂ figure matches the `es-cuda-rhf-driver` pin.

### Tolerance

Same energy windows as `h2o_rhf_sto3g_grad` and `es-cuda-rhf-driver`:

| Device | \|E − E_PySCF\| |
| --- | --- |
| cpu | 1e-8 Ha |
| cuda | 1e-6 Ha |

The CUDA leg also requires `last_scf_on_device()` and
`last_force_on_device() == false` on both molecules. Both are STO-3G
l ≤ 1, so a host ERI fill or a force-kernel trip fails the section.

### What is checked

Both systems converge. `energy()` [Hartree] matches the pin. Sentinels
`kEnergy = 42` eV, `kEnergyElectronic = −7` eV, positions, and `kAtom_f*`
are unchanged. Non-convergence, a device-path miss, a mutated Frame, or
an out-of-tolerance energy exits non-zero with `got / want / tol`.
