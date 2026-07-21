# Chapter 4: Gaussian Basis Sets and Convergence

| | |
|---|---|
| **Example** | [`examples/11_koopmans.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/11_koopmans.cpp) |
| **Results** | [`examples/results/11_koopmans.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/11_koopmans.json), [`12_delta_scf.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/12_delta_scf.json) |

Basis quality shows up immediately in ionization potentials. Koopmans'
theorem says \(\mathrm{IP}\approx -\varepsilon_\mathrm{HOMO}\). Atomiverse
RHF vs experiment (eV) for 6-31G — live data from `11_koopmans`:

| System | Koopmans 6-31G | Experiment |
|--------|---------------:|-----------:|
| He | 24.875 | 24.587 |
| H₂ | 16.201 | 15.426 |
| H₂O | 13.643 | 12.620 |
| NH₃ | 11.294 | 10.800 |
| CH₄ | 14.804 | 13.600 |

ΔSCF (orbital relaxation, from `12_delta_scf`) for the same basis:

| System | Koopmans | ΔSCF | Experiment |
|--------|---------:|-----:|-----------:|
| He | 24.875 | 23.444 | 24.587 |
| H₂O | 13.643 | 10.978 | 12.620 |
| CH₄ | 14.796 | 13.419 | 13.600 |

STO-3G vs 6-31G for water: 10.65 eV → 13.64 eV toward experiment 12.62 eV.
Relaxation always lowers the IP relative to Koopmans; CH₄ and He move closer
to experiment, while H₂O still needs correlation beyond single-reference HF.

```bash
cmake --build build --target 11_koopmans 12_delta_scf
./build/examples/11_koopmans
./build/examples/12_delta_scf
```

[Chapter 2](ch02-rhf.md) computed an RHF energy for H₂ in the minimal STO-3G
basis and quietly accepted whatever number came out. This chapter goes back
to that one line —

```cpp
rhf.basis = cpu::scf::BasisSet::STO3G;
```

— and asks the question every practitioner eventually asks: **how much does
the answer depend on the basis set, and when have I used enough of one?**
The basis is the single largest *controllable* source of error in a
mean-field calculation, and understanding basis-set convergence is what
separates a number you can defend from a number you happened to get.

The six questions:

1. **The concept.** What a Gaussian basis is and why bigger is (usually)
   better.
2. **The Atomiverse representation.** `BasisSet`, the method's private
   integral workspace, and the shell machinery behind them.
3. **The C++ API.** Selecting a basis, reading the AO count, and running the
   same SCF across a series.
4. **The expected result.** A monotone-ish energy lowering from STO-3G to
   def2-SVP, with diminishing returns.
5. **The validation.** Each basis is validated against PySCF at the *same*
   basis, so "is my number right?" and "is my basis converged?" stay
   separate questions.
6. **The recurring pattern.** Convergence-with-respect-to-a-discretization
   is the same idea as the plane-wave cutoff and the DFT grid.

## The Concept: Finite Windows on an Infinite Function

A molecular orbital is a continuous function of space. We cannot store a
continuous function, so we expand it in a finite set of fixed **atomic-orbital
basis functions** $\{\chi_\mu\}$:

$$
\psi_i(\mathbf{r}) = \sum_\mu C_{\mu i}\, \chi_\mu(\mathbf{r}).
$$

The basis is the window through which the SCF sees the true orbital. A
minimal basis (one function per occupied atomic shell) is a narrow window: it
can represent the gross shape of a bond but not its polarization or its
radial flexibility. A larger basis — more functions per shell (split
valence), functions of higher angular momentum (polarization) — widens the
window, and the variational SCF energy can only go *down* (or stay equal) as
the window widens, because the minimizer has strictly more freedom.

Two practical facts follow:

- **Gaussians, not Slater functions.** Real atomic orbitals decay like
  $e^{-\zeta r}$ (Slater), but the four-index electron-repulsion integrals
  over Slater functions have no closed form. The universal compromise, since
  Boys (1950), is to build each basis function from a fixed contraction of
  Gaussians $e^{-\alpha r^2}$, for which every integral is analytic. "STO-3G"
  literally means *Slater-Type Orbital approximated by 3 Gaussians*.
- **The energy converges but never terminates.** There is no "exact" finite
  basis. You climb a ladder — STO-3G → 6-31G → 6-31G* → def2-SVP → … — until
  the property you care about stops moving by more than your target
  tolerance. That climb is *basis-set convergence*.

## The Representation: A `BasisSet` Enum Over a Shell Machine

Atomiverse names each basis with an enum and resolves it to concrete shells
inside the method's private integral workspace. From
[`basis_set.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/basis_set.h):

```cpp
enum class BasisSet {
    STO3G,       // minimal (H–O). The production baseline.
    Pople631G,   // 6-31G  split-valence, no polarization
    Pople631Gs,  // 6-31G* d-polarization on heavy atoms
    Pople631pGs, // 6-31+G* heavy-atom diffuse sp + d-polarization
    Def2SVP,     // def2-SVP  Ahlrichs split-valence polarized
    Def2TZVP,    // def2-TZVP triple-zeta
    AugCcPVDZ,   // aug-cc-pVDZ Dunning DZ + diffuse augmentation
};
```

Under the hood a `BasisSet` + the frame's atoms is resolved into a
`ShellList` — flat AO descriptors with per-shell contraction lengths, which
is why the machine handles mixed contractions (6-31G's 6/3/1, def2-SVP's up
to 7) without special-casing. You do not touch `ShellList` directly for a
normal calculation; the method rebuilds it from the Frame on every
`run(frame)`. The AO count surfaces on the wavefunction — it is the row
dimension of the MO coefficient matrix:

```cpp
using Spin = cpu::scf::SlaterDeterminant::Spin;

rhf.run(frame);
const auto& det = rhf.determinant();
int n_ao    = static_cast<int>(det.coefficients(Spin::alpha).shape()[0]);
int n_alpha = det.n_electrons(Spin::alpha);   // (Σ Z − charge) split by spin;
                                              // = n_occ for a closed shell
```

`n_ao` is the number you watch: it is the dimension of every matrix in
the SCF, and cost scales steeply with it (roughly $N^3$ for the
diagonalization, worse for conventional integrals), which is the price of a
wider window.

### Element and basis coverage — the no-fallback rule

STO-3G is implemented for **H through O (Z = 1–8)**; the larger Pople and
def2/cc families extend the range through the **third period (Na–Ar, Z =
11–18)**, with the Pople sets additionally carrying K/Ca. Coverage is
honest: if you ask for a
basis/element combination that is not tabulated, the shell-list build inside
`run(frame)` **throws** `std::runtime_error` rather than
silently substituting a nearby basis. There are *no fallback values* by
design — a wrong answer that runs is worse than an exception that stops you.
Among the shipped sets, STO-3G through def2-SVP are production-ready;
def2-TZVP is partial (its f-shells outrun the current XC-grid basis
evaluation), so treat it as experimental for KS-DFT.

## The C++ API: One SCF, Swept Across a Series

The mechanical recipe from [Chapter 2](ch02-rhf.md) is unchanged; only the
`BasisSet` argument moves. A convergence study is therefore a loop over the
enum:

```cpp
using namespace ATV;
using Spin = cpu::scf::SlaterDeterminant::Spin;

cpu::Frame frame = /* fixed H2O geometry, positions in Å, type = Z */;

for (auto basis : {cpu::scf::BasisSet::STO3G,
                   cpu::scf::BasisSet::Pople631G,
                   cpu::scf::BasisSet::Pople631Gs,
                   cpu::scf::BasisSet::Def2SVP}) {
    cpu::scf::RHF rhf;
    rhf.basis = basis;
    rhf.max_iter = 100;
    rhf.run(frame);

    const int n_ao = static_cast<int>(
        rhf.determinant().coefficients(Spin::alpha).shape()[0]);
    std::printf("%-12s  n_ao=%3d  E = %.8f Ha  (%d iters)\n",
                cpu::scf::basis_set_name(basis), n_ao,
                rhf.energy(), rhf.iterations());
}
```

`run(frame)` writes the eV total energy to `frame.meta.get(kEnergy)`;
`energy()` reports the same total (electronic + nuclear) energy in
**Hartree**, alongside `converged()` and `iterations()` on the method, with
the spin-resolved orbital eigenvalues on `determinant()`. `basis_set_name`
gives the human-readable label for logs.

Select the basis as a member and run — there is no separate integral or
config step:

```cpp
cpu::scf::RHF rhf;                            // Restricted HF, MINAO guess
rhf.basis = cpu::scf::BasisSet::Def2SVP;
rhf.run(frame);                               // rhf.energy() in Ha
```

The electron counts (`n_alpha`, `n_beta`) are resolved from the Frame
(Σ Z − charge) on every run, and the private integral workspace is rebuilt
whenever the geometry or the `basis` member changes — nothing is tied to a
stale pack. That rebuild-from-Frame contract is the same one the dynamics
and optimization chapters rely on when the method is bound at a `force`
phase.

## The Expected Result: Monotone Lowering, Diminishing Returns

For a closed-shell molecule at a fixed geometry, the RHF total energy should
*drop* as the basis grows, and the drops should shrink — the signature of
convergence. A representative water sweep looks qualitatively like:

```text
STO-3G       n_ao= 7   E ≈ -74.96 Ha     (crude; minimal window)
6-31G        n_ao=13   E ≈ -75.98 Ha     (split valence: big gain)
6-31G*       n_ao=19   E ≈ -76.01 Ha     (d-polarization on O)
def2-SVP     n_ao=24   E ≈ -76.02 Ha     (polarized double-ζ)
```

(Exact values depend on the geometry; the *pattern* — a large STO-3G→6-31G
step, then small polarization and def2-SVP corrections — is the lesson.) The
STO-3G energy is roughly a Hartree above the near-complete-basis limit, which
is why STO-3G is a teaching and smoke-test basis, not a production one. By
def2-SVP the *Hartree-Fock* energy is close to converged; the *remaining*
error is no longer the basis but the method (HF neglects correlation), a
distinction [Chapter 6](ch06-hybrid-dft.md) and the correlation-methods
status chapter take up.

The right way to read this table is by **numerical class**: the energy
*difference between two bases at the same geometry* is a
physical-approximation quantity (it measures basis incompleteness), whereas
"is the def2-SVP number itself correct?" is a separate,
numerical-tolerance question answered against a reference at that same basis
— which is exactly what the validation does.

## The Validation: Each Basis Checked Against PySCF at That Basis

Basis-set convergence and code correctness are two different things, and
Atomiverse keeps them separate. Correctness is pinned by comparing the
Atomiverse energy at a given basis against a genuine PySCF run *at the same
basis and geometry*, to tight numerical tolerance (RHF energy vs PySCF
**1e-7 Ha**). The reference numbers are generated by PySCF and checked into
the test as headers, so the parity test runs in CI without PySCF installed.

That separation is what lets the convergence table above mean something: each
row is independently certified to be the *right* energy for its basis, so the
*differences* between rows genuinely measure basis incompleteness and not a
bug that happens to cancel.

The production recommendation follows directly: set the method's `basis` to
`BasisSet::Def2SVP` for real closed-shell work (`BasisSet::Pople631G` for
quick tests), knowing STO-3G exists for pedagogy and fast regressions, and
def2-TZVP for pushing toward the basis-set limit where its f-shell support
allows.

## The Recurring Pattern: Convergence With Respect to a Discretization

Choosing a Gaussian basis is one instance of a pattern that appears in every
electronic-structure method in this book: **the physics is exact, the
discretization is not, and you converge the discretization until the answer
stops moving.**

- **Gaussian basis** (this chapter): converge the AO set STO-3G → def2-SVP.
- **Plane-wave DFT** ([Chapter 8](ch08-plane-wave.md)): converge the kinetic-energy
  cutoff `ecut` and the k-point mesh — a plane-wave basis is just a
  *different* finite window, converged with a single knob.
- **DFT XC grid** ([Chapter 5](ch05-kohn-sham-dft.md)): converge the radial/angular
  grid until the integrated electron count and `E_xc` stabilize.

In each case the discipline is identical: pick the knob, run a short series,
watch the property of interest, and report the setting at which it is
converged to your tolerance. A basis set is not a magic keyword — it is a
convergence parameter you are responsible for.

## Validation Reference

Basis-set correctness is pinned by the RHF/DFT-vs-PySCF parity suite,
`test_cpu_scf_basis_vs_pyscf` (Atomiverse energy vs a checked-in genuine
PySCF reference at the same basis, **1e-7 Ha**), with the reference
generators and provenance under
[`/* benchmarks removed; re-add later */ rhf`](https://github.com/MolCrafts/Atomiverse/blob/dev//* benchmarks removed; re-add later */ rhf). The basis-convergence
benchmark case (energy series with committed near-CBS references) is part of
the manual build-out plan; until it lands, the parity tests plus the sweep
recipe above are the way to certify a convergence study.

## 笛卡尔与球谐高斯基(6d vs 5d)

同一个 d 壳层可以用两种角函数展开。**笛卡尔**形式取所有 lx+ly+lz=2 的单项式
——xx, xy, xz, yy, yz, zz,共 (l+1)(l+2)/2 = 6 个分量;**球谐**形式取实球谐
Y₂ₘ,m = −2…+2,共 2l+1 = 5 个。两者张成的空间差一个维度:六个笛卡尔 d 里
藏着一个 r² = x²+y²+z² 的组合,它的角行为是 **s 型**(所谓 s-污染分量)。
f 壳同理:10 个笛卡尔 = 7 个球谐 ⊕ 3 个 p 型污染分量。

去掉污染分量的字典是实固体谐波(Helgaker 第 6 章;闭式系数见 Schlegel &
Frisch 1995,DOI 10.1002/qua.560540202)。例如:

- Y₂,₀ ∝ 2z² − x² − y²(表格里的 −0.3154, −0.3154, +0.6308 一列),
- Y₂,₊₂ ∝ x² − y²,
- Y₂,₋₂ ∝ xy(单项,系数 1.0925 = √(15/4π))。

**约定属于基组定义,不是用户选项**:Pople 系(6-31G*)按定义用笛卡尔 6d;
def2/cc 系按定义用球谐。Atomiverse 由 `angular_convention(BasisSet)` 逐基组
判定,`IntegralPack` 在装配后把矩阵三明治到球谐维度(引擎内部保持笛卡尔,
Fock 构建经 `c2s` 桥接 — 数学上与逐壳块变换严格等价)。

一个必须小心的坑:**逐分量单位归一的笛卡尔 d 中,d_xx 与 d_xy 的归一化常数
差 √3**(双阶乘比 √((2lx−1)!!(2ly−1)!!(2lz−1)!!)),c2s 系数必须吸收这个比率
——判据是单位归一 d 壳自重叠经三明治后精确回到 5×5 单位阵
(`tests/cpu/scf/test_cart2sph.cpp` 以 1e-12 钉住)。

### 上手:H₂O/def2-SVP

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Def2SVP;   // 球谐,按定义
rhf.run(frame);                             // n_basis = 24(笛卡尔为 25)
```

注意:球谐基的解析梯度尚未接管(积分导数的 c2s 变换属后续链环),
`forces()` 会显式抛出而不是静默给错力。

### 与 PySCF 对照(2.14.0,外部运行钉值;方法学见附录)

| 量 | Atomiverse | PySCF(钉定) | 差 |
|---|---|---|---|
| n_basis(def2-SVP, H₂O) | 24 | 24 | — |
| E_RHF/def2-SVP(球谐)[Ha] | −75.96098398708 | −75.96098398708 | 4×10⁻¹³ |
| E_RHF/def2-SVP(笛卡尔 6d 参照)[Ha] | — | −75.96221975499 | Δ≈1.2 mHa |

那 1.2 mHa 就是 6d 的 s-污染分量的变分贡献 — 两种约定不是数值误差,
而是**不同的基组**;这正是"约定跟随基组定义"的理由。
(几何与完整命令:`regressions/scf/h2o_rhf_def2svp_sph/REFERENCE.md`。)

## 极化、弥散与第三周期

### 命名法解码:一个名字就是一张配方单

Pople 命名法把基组的全部构造信息压进一个字符串,逐字符读:

| 记号 | 含义 | 例 |
|---|---|---|
| `6-31G` | 内核壳 6 个高斯缩并成 1 个函数;价壳**分裂**为 3-高斯 + 1-高斯两层 | Ditchfield-Hehre-Pople 1971 |
| `*`(即 `(d)`)| 重原子(非 H)加一组 **d 极化函数** — 让 p 电子云能偏离原子中心极化 | 6-31G*(Hariharan-Pople 1973)|
| `**`(即 `(d,p)`)| 再给 H 加一组 p 极化 | 6-31G** |
| `+` | 重原子加一组**弥散 sp**(共享指数、很小的 α,如 O 的 0.0845)— 描述离原子核很远的松散电子 | 6-31+G* |
| `++` | 再给 H 加弥散 s | 6-31++G* |

Dunning 相关一致(correlation-consistent)系用另一套语法:`cc-pVDZ` =
双 ζ 价层 + 极化一并按能量贡献分层收进;前缀 `aug-` 表示**每个角动量通道
各加一层弥散**(H 加弥散 s、p;O 加弥散 s、p、d — Kendall, Dunning,
Harrison 1992)。所以 aug-cc-pVDZ 的 O 是 4s3p2d(球谐 23 个 AO),
比 cc-pVDZ 整整多出一层慢衰减的窗口。

极化与弥散回答两个正交的问题:**极化**加的是更高角动量(d、f),让电子云
能"变形";**弥散**加的是更小的指数(更慢的径向衰减),让电子云能"变远"。
中性分子的基态主要吃极化;**阴离子、里德堡态、氢键尾部、极化率**主要吃
弥散 — 判据永远是"你关心的电子密度住在哪儿"。

### 弥散函数的物理:OH⁻ 的 ε_HOMO

阴离子的多余电子结合得很松(OH⁻ 的实验电子亲和能只有 ~1.8 eV),它的
HOMO 尾巴伸到离核很远的地方。紧凑基组没有慢衰减函数,变分只能把这个
电子硬塞进价层窗口 — 极端时 ε_HOMO 干脆变成**正数**:一个自称"束缚阴
离子"的波函数,其最高占据轨道却宣称电子会自发飞走。这不是物理,是基组
伪像。观察它只需要把 [第 3 章](ch03-open-shell.md) 的 `charge` 成员设成
−1,换基组重跑:

```cpp
using Spin = cpu::scf::SlaterDeterminant::Spin;

cpu::Frame frame = /* O(0,0,0), H(0,0,0.9640) Å */;
for (auto basis : {cpu::scf::BasisSet::STO3G,
                   cpu::scf::BasisSet::Pople631Gs,
                   cpu::scf::BasisSet::Pople631pGs}) {
    cpu::scf::RHF rhf;
    rhf.basis = basis;
    rhf.charge = -1;                    // OH⁻:10 电子闭壳层
    rhf.tol = 1e-12;
    rhf.run(frame);
    std::printf("%-10s eps_HOMO = %+.6f Ha\n",
                cpu::scf::basis_set_name(basis),
                rhf.determinant().homo(Spin::alpha));
}
```

实测阶梯(r(OH) = 0.9640 Å;全部与 PySCF 同基组钉值一致,见下表):

```text
STO-3G     eps_HOMO = +0.251912 Ha    非束缚伪像:HOMO 是正的
6-31G*     eps_HOMO = -0.037774 Ha    勉强束缚(极化救不了弥散的缺席)
6-31+G*    eps_HOMO = -0.103580 Ha    弥散 sp 使束缚加深近 3 倍
```

注意 6-31G* → 6-31+G* 这一步:**总能量只降了 50 mHa,ε_HOMO 却加深了
66 mHa** — 弥散函数对总能量是小修正,对"住在远处"的可观测量(HOMO、
电子亲和能、极化率、氢键)是决定性的。这也是为什么"能量收敛了"不等于
"性质收敛了":每个性质有它自己的基组阶梯。

### 第三周期(Na–Ar)与 Cl⁻

六套表(6-31G / 6-31G* / 6-31+G* / def2-SVP / def2-TZVP / aug-cc-pVDZ)
现在全部覆盖 **Z = 11–18(Na–Ar)**,Pople 系另带 K/Ca。第三周期原子的
内核多一层(1s2s2p 全成核),6-31G 系为它们缩并 5s4p — Cl 在 6-31+G*
下是 5s4p1d、23 个笛卡尔 AO。Cl⁻ 是最干净的演示:18 电子闭壳层单原子,
`charge = -1`,一行 `rhf.run(frame)`,总能量与 PySCF 对到 1e-13 Ha
(回归 `regressions/scf/diffuse_anion` 的钉值)。

数据面的工程约定不变:所有表由 `tools/gen_basis_data.py` 从 PySCF 的
`mol._basis` 字典一次性生成、checked-in 为静态 C++ 头(字节可复现,头部
Regenerate 注释记录完整命令),**运行时零解析**;aug-cc-pVDZ 的 s 块是
广义缩并(同一组原语、多列系数),生成器把每列拆成独立缩并壳 — 张成的
AO 空间严格不变。不支持的 Z 依旧空行 + 抛含基组名的异常,无回退。

### 章末对照:Atomiverse vs PySCF 2.14.0(标量,外部钉值)

| 量 | Atomiverse | PySCF(钉定) | 差 |
|---|---|---|---|
| ε_HOMO(OH⁻, STO-3G)[Ha] | +0.251911715730 | +0.251911715713 | 2×10⁻¹¹ |
| ε_HOMO(OH⁻, 6-31G*)[Ha] | −0.037774390716 | −0.037774390226 | 5×10⁻¹⁰ |
| ε_HOMO(OH⁻, 6-31+G*)[Ha] | −0.103579759566 | −0.103579773338 | 1.4×10⁻⁸ |
| E_RHF(OH⁻, 6-31+G*)[Ha] | −75.376293363749 | −75.376293363750 | 6×10⁻¹³ |
| E_RHF(Cl⁻, 6-31+G*)[Ha] | −459.539661044387 | −459.539661044387 | 2×10⁻¹³ |
| E_RHF(H₂O, aug-cc-pVDZ, 球谐)[Ha] | −76.041393519988 | −76.041393519987 | 6×10⁻¹³ |

(几何、命令与容差:`regressions/scf/diffuse_anion/REFERENCE.md`;Pople 系
PySCF 侧 `cart=True`,aug-cc 系球谐默认。标量一律进表 — 曲线才作图。)

OH⁻ 的基组收敛曲线(总能量阶梯;ε_HOMO 的阶梯在上面的文本块里):

```molplot preset="molplot" theme="auto" width="100%" aspect="16:9"
mark: {type: line, point: true, tooltip: true, strokeWidth: 2}
data:
  values:
    - {basis: "STO-3G",  n_ao: 6,  E: -74.056328}
    - {basis: "6-31G",   n_ao: 11, E: -75.311529}
    - {basis: "6-31G*",  n_ao: 17, E: -75.326597}
    - {basis: "6-31+G*", n_ao: 21, E: -75.376293}
encoding:
  x: {field: basis, type: ordinal, sort: null, title: "基组(窗口递增)"}
  y: {field: E, type: quantitative, title: "E_RHF(OH⁻) [Ha]", scale: {zero: false}}
```

读法与本章正文的水的阶梯相同:STO-3G → 6-31G 一大步(分裂价层),极化
一小步,而 6-31+G* 这"额外"的一步对阴离子**不是**递减收益 — 它补的是
别的基组根本没有的那类函数。

## 有效核势(ECP):把惰性的芯换成一个算符

### 为什么芯电子可以被替换

碘有 53 个电子,金有 79 个。全电子处理它们意味着:为 1s…3d 这些深芯轨道
配上超陡的高斯函数(指数动辄 10⁶)、承受由此而来的积分动态范围与线性相关
问题 — 而这些芯电子在成键中几乎纹丝不动。化学发生在价层;芯的作用无非是
(1) 屏蔽核电荷、(2) 通过 Pauli 排斥把价电子"顶"在芯区之外。这两件事都可以
用一个作用在价电子上的**算符**代替 — 这就是有效核势(effective core
potential,ECP;也叫赝势)。第三个、往往最重要的理由:重元素的芯电子是
**相对论性的**(Au 的 1s 电子速度约 0.58c),把相对论解出来再拟合成势,
比在价层薛定谔方程里显式处理便宜得多。

### 半定域形式:每个角动量一张势

价电子"看到"的芯不是各向同性的:s 价电子必须与芯的 s 轨道正交(Pauli),
p 价电子与芯 p 正交 — 不同角动量通道感受到的有效排斥不同。所以 ECP 是
**半定域**(semi-local)的(Kahn, Baybutt, Truhlar 1976):

$$U(r) = U_L(r) + \sum_{l<L} \sum_{m} |lm\rangle\,[U_l(r) - U_L(r)]\,\langle lm|$$

局域部分 U_L 作用于所有角动量;投影部分对每个低角动量通道 l 修正出
U_l − U_L 的差。每张径向势都是高斯展开
U_l(r) = Σ_k d_k r^{n_k−2} e^{−ζ_k r²}(def2-ECP 全部取 n=2,纯高斯),
于是全部矩阵元解析/半解析可积(McMurchie–Davidson 1981 的 type-1/type-2
积分;径向被积函数里的修正球贝塞尔 i_l(kr) 在引擎内用带尺度形式
M_l(x)=e^{−x}i_l(x) 计算 — 高 l 下直接算 i_l 会灾难性消去)。

### 价电子计数:Z 变成 Z_val

替换芯的同时,三处记账同步改变(这是新手最容易漏的地方):

| 量 | 全电子 | ECP(def2-ECP)|
|---|---|---|
| I 的核吸引/核排斥电荷 | Z = 53 | Z_val = 53 − 28 = 25 |
| I₂ 的电子数 | 106 | 50(n_occ = 25)|
| Au 的电荷 / AuH 电子数 | 79 / 80 | 19 / 20 |

def2 家族的约定:**Z ≤ 36(H–Kr)全电子,Rb 之后配 ECP**(I 换掉 28 个
芯电子,Au 换掉 60 个)。def2-SVP 的重元素行本来就是与 def2-ECP 成对
定义的价层基组 — 基组与 ECP 是一张配方单的两半,不能混搭。

### 相对论已经烘进去了(x2c 双重计数警告)

def2-ECP 的参数(I:ECP28MWB,Bergner 1993;Au:ECP60MWB,Andrae 1990)
是对**准相对论**(Wood–Boring)原子解拟合的 — 标量相对论效应(s 壳收缩、
d/f 壳膨胀,金的颜色与汞的液态背后的那套物理)已经包含在势里。所以
**ECP 元素绝不能再叠加显式标量相对论哈密顿**(如 x2c):那会把同一效应
数两遍。Atomiverse 把这个约束做成了逐元素的硬门:

```cpp
cpu::scf::RHF rhf;
rhf.ecp = "def2-ECP";
rhf.x2c = true;        // 保留成员,后续链环接线
rhf.run(i2_frame);     // throws:x2c + ECP 对元素 I 双重计数
```

全电子分子(表外元素)同样设置不抛 — 门是逐元素的,不是全局开关。

### 上手:I₂ 的 RHF

与第 2 章的 H₂ 相比只多一行 — `ecp` 是方法的成员,不是新的驱动器:

```cpp
cpu::Frame frame(2);                       // I(0,0,0); I(0,0,2.666) Å
frame.span(kAtom_type)[0] = 53;
frame.span(kAtom_type)[1] = 53;
frame.span(kAtom_z)[1] = 2.666;
frame.simbox = cpu::Box::cubic(20.0);

cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Def2SVP;   // 价层基组(球谐)
rhf.ecp = "def2-ECP";                      // 空字符串 = 全电子
rhf.tol = 1e-12;
rhf.run(frame);                            // E = -593.316182 Ha,52 AO
```

−593 Ha 这个数量级值得停一秒:全电子 I₂ 的 RHF 能量在 −13800 Ha 上下 —
ECP 把"总能量里 98% 的化学惰性部分"连同它的基组代价一起删掉了,留下的
50 个价电子用 52 个球谐 AO 就描述得与 PySCF 一致到 1e-9 Ha。工程面与
基组表同一套纪律:def2-ECP 参数由 `tools/gen_basis_data.py` 从 PySCF 的
`gto.basis.load_ecp` 一次性生成为静态表(`basis_data/ecp_def2.h`,字节可
复现),查表类 `EcpPotential(molrs::Element::I, "def2-ECP")` 与 GPW 的
`GthPotential` 同形,表外元素/未知名字抛错无回退;ECP 解析梯度尚未接线,
`forces()` 对 ECP pack 显式抛出(与球谐基组同一先例)。

## BSSE 与 counterpoise

有限基组下,单体 A 在二聚体几何中会"借"邻居 B 的基函数来描述自己的电子密度 —
这不是物理上的相互作用,而是**基组重叠误差**(BSSE,basis-set superposition
error)。结果是未校正的相互作用能 \(\Delta E_{\mathrm{raw}} = E_{AB}^{AB} -
E_A^A - E_B^B\) 系统性地偏深(更负)。

Boys–Bernardi counterpoise(CP)校正用**同一套二聚体基组**重算两个单体:

\[
\Delta E_{\mathrm{CP}} = E_{AB}^{AB} - E_A^{AB} - E_B^{AB},
\]

其中 \(E_A^{AB}\) 是 A 在完整二聚体基组中的能量(B 的位置放 **ghost
原子**:只贡献基函数,核电荷与电子数均为 0)。符号约定锁定为

\[
\mathrm{bsse} = (E_A^{AB} - E_A^A) + (E_B^{AB} - E_B^B) \le 0
\]

——单体在更大的二聚体基组中能量必然降低,`|bsse|` 即基组重叠误差幅度。

### 诚实的坑(Jensen《Introduction to Computational Chemistry》基组/BSSE 节)

1. **过度校正。** 小基组(STO-3G、6-31G*)下 CP 常把真实相互作用也削掉一部分;
   大基组(def2-TZVP 及以上)BSSE 本身就小,CP 的必要性下降。
2. **多体簇不是 size-consistent。** site–site CP 与 full-cluster CP 不一致;
   本 API 只做二聚体 5-SCF 编排。
3. **形变校正是另一项。** 单体从自由几何放松到二聚体几何的能量变化不在
   `Counterpoise` 内;固定几何上的 CP 与几何优化后的 CP 不是同一数字。

### 5-SCF 配方与 Atomiverse API

Ghost 标记是方法成员 `ghost_atoms`(局部原子下标),**不进 Frame**(Frame
纯净性;molrs `Element` 禁止 Z ≤ 0)。`cpu::scf::Counterpoise` 是纯编排类 —
不继承 Component、不注册 Plan、不新增 Driver — 串行借用用户已配置的方法
对象跑 5 次 `run(const Frame&)`:

```cpp
// 水二聚体 HF/6-31G*:单体 A = 原子 0..2,单体 B = 3..5
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Pople631Gs;
rhf.tol = 1e-10;

cpu::scf::Counterpoise cp({0, 1, 2}, {3, 4, 5});
auto r = cp.run(dimer_frame, rhf);
// r.e_dimer, r.e_a, r.e_b, r.e_a_ghost, r.e_b_ghost  [Hartree]
// r.e_int_cp = e_dimer − e_a_ghost − e_b_ghost
// r.bsse     = (e_a_ghost − e_a) + (e_b_ghost − e_b)  ≤ 0
// 返回后 rhf.ghost_atoms 恢复为 run 前的值
```

五次 SCF 顺序:(1) 二聚体全基组;(2) 自由单体 A;(3) 自由单体 B;
(4) A + B-ghost;(5) B + A-ghost。KS(RKS/UKS)与非空 `ghost_atoms` 组合
显式抛异常(格点半径按 Z 取,本链节未接线)。

### 上手:水二聚体 CP

完整几何、PySCF 钉值与文献见
`regressions/scf/h2o_dimer_cp_hf631gs/REFERENCE.md`。在 6-31G* 下 CP 校正
约 1.7 mHa,落在文献 milli-Hartree 量级。

```bash
cmake --build build --target reg_h2o_dimer_cp_hf631gs
./build/regressions/reg_h2o_dimer_cp_hf631gs
```

### 章末对照:Atomiverse vs PySCF 2.14.0(标量,外部钉值)

| 量 | Atomiverse | PySCF(钉定) | 差 |
|---|---|---|---|
| E_RHF(I₂, def2-SVP+def2-ECP)[Ha] | −593.316181880646 | −593.3161818794323 | 1.2×10⁻⁹ |
| E_RHF(AuH, def2-SVP+def2-ECP)[Ha] | −135.342567007924 | −135.34256700791678 | 7×10⁻¹² |
| V_nuc(I₂, Z_val=25)[Ha] | 124.056923040135 | 124.05692304013505 | <10⁻⁸ |
| n_electrons(I₂)/(AuH) | 50 / 20 | 50 / 20 | — |
| ΔE_CP(H₂O 二聚体, HF/6-31G* cart)[Ha] | −0.007542440802 | −0.00754244121519 | ≤1×10⁻⁶ |
| bsse(H₂O 二聚体, HF/6-31G* cart)[Ha] | −0.001483681 | −0.00148368102643 | ≤1×10⁻⁶ |

(ECP 几何、命令与文献:`regressions/scf/ecp_heavy_atom/REFERENCE.md`;
U 矩阵元级的钉值 — 含合成 f 投影道 — 在 `tests/cpu/scf/test_ecp.cpp`。
水二聚体 CP:`regressions/scf/h2o_dimer_cp_hf631gs/REFERENCE.md`,
Boys & Bernardi 1970 DOI 10.1080/00268977000101561。标量一律进表。)
