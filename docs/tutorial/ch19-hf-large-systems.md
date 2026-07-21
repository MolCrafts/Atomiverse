# Chapter 19 — 大体系 HF:从 Schwarz 筛选到线性标度库仑

> **状态:部分成文**(19.1–19.2 由 12-df 填充;19.3–19.5 仍为骨架)。各节由标注的链环填充;
> 本章对应 Jensen《Introduction to Computational Chemistry》§3.8 后半
> ("the SCF bottleneck" / linear scaling):方法为主干,把 direct SCF、密度
> 拟合与多极方法编成一条"库仑瓶颈如何被逐级击破"的叙事。

## 19.1 SCF 的瓶颈:四指标 ERI

Jensen(§3.8)把大体系 HF 的困境总结为一句话:Fock 矩阵本身只有
O(N²) 个元素,但喂给它的双电子积分张量 (μν|λσ) 有 **O(N⁴)** 个。存不
下,就每步重算(direct SCF);重算太贵,就用 Cauchy–Schwarz 上界
√(μν|μν)·√(λσ|λσ) 在壳对层面把指数衰减的贡献整批筛掉——这正是
Atomiverse 的现役路径(`ShellBlockPairList::g_rhf`,壳块级 Schwarz 筛
选 + Rys 求积),也是 PySCF 的分子路线(诚实注明:PySCF 分子代码同样
没有 FMM,它的快来自积分内核与迭代数,不是更好的标度)。

Schwarz 筛选把渐进标度压到 ~O(N²)(远壳对指数消失),但**前置系数**
仍由每步全量重算的四指标积分决定。本章逐级击破这个库仑瓶颈:先用密度
拟合把 4-指标压成 3-指标(本节),再把远场交给多极展开(19.3–19.5)。

## 19.2 密度拟合 RI

### 4-指标 → 3-指标:为什么能压缩

四指标张量贵,是因为它把**乘积密度** ρ_μν(r) = χ_μ(r)χ_ν(r) 的所有
O(N²) 个成员两两配对。密度拟合(RI,resolution of the identity)的观
察是:这 O(N²) 个乘积密度挤在同一段分子空间里,线性相关得厉害——用一
组 O(N) 的**辅助基** {P} 就能把它们展开:

```
ρ_μν(r) ≈ Σ_P c^P_μν P(r)
```

在**库仑度规**下求拟合系数(Vahtras–Almlöf–Feyereisen 1993,
DOI 10.1016/0009-2614(93)89151-7),得到经典的三中心分解:

```
(μν|λσ) ≈ Σ_PQ (μν|P) [V⁻¹]_PQ (Q|λσ),   V_PQ = (P|Q)
```

四指标张量从此不再出现:存的是 O(N²·N_aux) 的 (μν|P)。库仑度规的妙处
在于它让 J 的能量误差是拟合残差的**二阶小量**——残差 δρ 对能量的一阶
项 (δρ|ρ) 恰好被拟合条件消掉,剩下 (δρ|δρ)。配对好的辅助基下,这是每
原子几十 μHa 的量级(Weigend 2006,DOI 10.1039/b515623h)。

工程实现(`internal::DensityFitting`):V = L·Lᵀ 做 Cholesky,取白化
因子 W = L⁻¹(W·V·Wᵀ = I),把 B_{μν,P} = Σ_Q (μν|Q)·W_QP 一次算好。
之后每步 SCF:

- **RI-J(γ_P 路线)**:γ = Bᵀ·vec(P),J = B·γ——两次 GEMV;
- **RI-K(B 半变换路线,Weigend 2002,DOI 10.1039/B204199P)**:
  B^P_μi = Σ_ν B^P_μν C_νi,K_μν = Σ_{P,i} B^P_μi B^P_νi。

### 单表取舍:只有 JKFIT,没有 JFIT

文献里辅助基分两族:只拟 J 的 JFIT 和拟 J+K 的 JKFIT。Atomiverse 只
收 **def2 万能 JKFIT** 一张表(`basis_data/def2_jkfit.h`,含 g 壳,
l = 4):交换拟合的误差比库仑大一个量级,**必须** JKFIT——JFIT 不够;
反过来用 JKFIT 兼任纯 J 拟合只是辅助维度略大,精度不降。一表两用,少
一张表少一类失配错误。配套的陷阱也直接编码:轨道基元素在 JKFIT 表无
行时**抛异常**,没有静默回退。

### 动手:一行成员开关

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Def2TZVP;
rhf.tol   = 1e-12;
rhf.ri    = cpu::scf::RI::JK;                     // None | J | JK
rhf.aux_basis = cpu::scf::AuxBasis::Def2JKFIT;    // 默认值,可省略
rhf.run(frame);          // frame.meta[kEnergy];rhf.energy() [Ha]
```

`RI::None`(默认)与旧路径逐位一致;`RI::J` 只拟库仑(交换仍走精确壳
块 K);`RI::JK` 全拟。未接线的组合(ROHF/GHF/SOSCF/解析力)一律抛异
常说明原因。

### 诚实的误差预算(μHa)

对照 regressions/scf/h2o_ri_jk 与 benchmarks/scf/validation/ri_speed
的实测(def2 万能 JKFIT,球谐辅助跨度):

| 体系 | E(RI-JK) − E(精确 ERI) | 每原子 | 与 PySCF `density_fit()` 之差 |
|---|---|---|---|
| H2O / def2-TZVP | +5.41 μHa | 1.8 μHa | 3.3e-13 Ha(同一拟合空间) |
| 甘氨酸 / def2-SVP | +258 μHa | 26 μHa | ~1e-9 Ha |

两点提醒:(1) 误差随基组配对质量浮动——SVP 配万能 JKFIT 比 TZVP 粗;
(2) 拟合误差是**能量的系统偏移**,不是噪声,对反应能差还会进一步抵
消。注意辅助跨度必须取**球谐**(与 def2 家族的定义一致):笛卡尔 g 壳
会多张出低阶函数,拟合虽仍变分,但数值就对不上 PySCF 的
def2-universal-jkfit 了。

## 19.3 长程库仑 I:Ewald 与 PME

周期体系中的库仑和是**条件收敛**的:裸的 ∑_{i<j} q_i q_j / r_{ij} 在
PBC 下按求和顺序发散(正/负表面偶极层)。Ewald 求和把 1/r 劈成短程
(在实空间用 erfc 截断)与长程(在倒空间用高斯平滑的 Poisson 解):

\[
\frac{1}{r}
  = \frac{\mathrm{erfc}(\alpha r)}{r}
  + \frac{\mathrm{erf}(\alpha r)}{r}
\]

实空间半截 `CoulLong(α, cutoff)` 只走邻居列表;倒空间半截由
`KSpaceEwald` 或 `KSpacePME` 提供。两边**必须共用同一个 α**
(计划在 `build()` 时 fail-closed 校验,见
[`docs/developer/kspace-long-range.md`](../developer/kspace-long-range.md))。

### 为何 PME 是 O(N log N)

直接 Ewald 对每个 k 向量累加结构因子 S(k)=∑_j q_j e^{ik·r_j},
k 点数随精度与体积增长 → 整体 O(N²)(固定精度、密度时对 N)。
**光滑粒子网格 Ewald**(SPME; Essmann *et al.*, *J. Chem. Phys.*
**103**, 8577 (1995), DOI: [10.1063/1.470117](https://doi.org/10.1063/1.470117))
把电荷用 B 样条铺到规则网格,一次 3D FFT 完成所有 k 模的卷积,再
样条插回原子力:

| 步骤 | 复杂度 |
|---|---|
| 电荷铺展(B 样条 stencil) | O(N · n³)(n = 样条阶,常数) |
| 正/逆 FFT | O(K log K), K ∝ N(固定网格密度) |
| 力插值 | O(N · n³) |

于是总标度 O(N log N)。Ewald 1921 原始公式见 *Ann. Phys.* **369**,
253, DOI: [10.1002/andp.19213690304](https://doi.org/10.1002/andp.19213690304)。

### 代码居所(facade + 薄包装)

| 层 | 路径 | 内容 |
|---|---|---|
| 纯算子数学 | `include/atomiverse/longrange/` | `EwaldSum`, `bspline` / `spline_moduli` / `good_fft_size`, 影响函数与力插值 — **无** Frame / Component / MPI / FFT |
| CPU 包装 | `include/atomiverse/cpu/longrange/` | `KSpaceEwald` / `KSpacePME` — Frame span、Allreduce、FFTW 计划、meta 写回 |
| 开关 | `-DATV_BUILD_KSPACE=ON` | 门控 FFTW3 / PME(名字保留) |

规则(为后续 FMM/CFMM 立此存照):**SCF 侧只许 include
`atomiverse/longrange/*`,永不 `cpu/longrange/*`。**

```cpp
// Classical MD: real-space CoulLong + reciprocal KSpacePME
cpu::md::MD sim;
const double alpha = 0.3;   // 1/Å — must match both halves
const double cutoff = 8.0;  // Å
sim.add_potential(cpu::CoulLong{alpha, cutoff});
sim.add(cpu::KSpacePME{alpha, /*grid_spacing=*/1.0, /*order=*/6});
// … integrator, neighbor, build, run …
```

### 动手:NaCl Madelung 常数

岩盐结构(交替 ±1 电荷的简单立方)每离子对的 Madelung 常数

\[
M = 1.74756459\ldots
\quad\Rightarrow\quad
E = -\tfrac12 N\, M\, k_C / d
\]

(`k_C = units::coulomb`, d = 最近邻间距)。回归
`regressions/md/npt_pme_nacl` 与单元
`tests/cpu/longrange/test_{ewald,pme}_coul.cpp` 钉住 M ≈ 1.7476
(相对 1e-4 级);构建需 `-DATV_BUILD_KSPACE=ON`。

### 章末标量表(长程 I)

| 方法 | 标度 | 边界 | 何时用 |
|---|---|---|---|
| 直接 Ewald (`KSpaceEwald`) | O(N²) | PBC | 小盒、对照、无 FFTW |
| SPME (`KSpacePME`) | O(N log N) | PBC | 生产 MD / NPT |
| FMM / CFMM | O(N) | 开边界或 PBC | 下两节(环 20/21) |

## 19.4 长程库仑 II:快速多极方法(骨架)

<!-- 填充:20-fmm — 多极压缩远场、M2M/M2L/L2L、与 PME 的开边界对偶;
     误差-阶数收敛表(标量表格);molplot 标度曲线 -->

## 19.5 长程库仑 III:CFMM 加速 HF(骨架)

<!-- 填充:21-cfmm — 高斯分布延伸分类、CFMM-J 接入、硬门(逐元素复现
     direct J)、交叉点规模表;direct→RI→CFMM 叙事收束 -->

## 章末:与 PySCF 对照(累积表)

精度(能量,Ha;PySCF 2.14.0,conv_tol=1e-12,球谐):

| 体系/设置 | 量 | Atomiverse | PySCF(钉定) | 容差 | 环 |
|---|---|---|---|---|---|
| H2O / def2-TZVP, RI-JK | E(RHF) | −76.058999347289 | −76.058999347289 | 1e-6(实测 3.3e-13) | 12 |
| H2O / def2-TZVP, 精确 ERI | E(RHF) | −76.059004760577 | −76.059004760577 | 1e-8 | 12 |
| NaCl rock-salt / Ewald+PME | Madelung M | 1.74756… | 文献 −1.7476/离子对 | 1e-4 rel | 19 |
| _待 21-cfmm 填充_ | CFMM-J 精度 / 交叉点 | — | — | — | 21 |

速度(甘氨酸 / def2-SVP RHF,tol 1e-9,8 线程,3 次取中位;
方法学与机器环境钉于 benchmarks/scf/validation/ri_speed/README.md,
2026-07-21):

| 列 | 墙钟 [s] | E(RHF) [Ha] |
|---|---|---|
| ours-RI-JK(`rhf.ri = RI::JK`) | 1.617 | −282.615514331 |
| ours-exact(`RI::None`) | 2.516 | −282.615772070 |
| PySCF-DF(`density_fit()`) | 0.104 | −282.615514330 |

RI-JK 在 10 原子的甘氨酸上已胜精确 ERI 1.56×,且随体系增大而扩大;与
PySCF-DF 的 ~15× 差距是引擎整体速度(迭代数与积分内核),不是 RI 算法
差异——两者能量在同一拟合空间里对齐到 1e-9 Ha。

对照方法学见[附录:PySCF 对照方法学](appendix-pyscf-oracle.md)。
