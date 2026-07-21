# Chapter 20 — HF 性质与环境:从偶极矩到相对论

> **状态:骨架章**(hf-capability-gaps 链 01 环创建)。各节由标注的链环填充;
> 本章把"从收敛的波函数读出物理"讲成一条弧:期望值 → 布居 → 响应(频率/激发)
> → 环境(外场/嵌入/溶剂)→ 相对论,对应 Jensen 性质诸章的 HF 侧。

## 20.1 分子多极矩

收敛的波函数里能读出的第一个物理量是**偶极矩**。对单行列式,任何单电子
算符的期望值都是一个迹:⟨Φ|Ô|Φ⟩ = Tr(P·O) — 这正是 `SlaterDeterminant::expect`
的契约(双电子算符没有这种形式;把 Fock 矩阵塞进去得到的 Σε 会双计电子互斥)。
于是

μ = Σ_A Z_A R_A − Tr(P·r),  Θ_ab = ½ Σ q (3 r_a r_b − r² δ_ab)(无迹 Buckingham)。

**原点问题**:把原点平移 d,偶极变换为 μ′ = μ − q_tot·d — 中性分子原点无关,
带电体系则不然;四极矩同理带上偶极交叉项。所以约定必须随数值一起报告:
Atomiverse 取**核电荷中心**为原点并写入 `kMultipoleOrigin`,这样全局平移下
报告值不变(原点随体系移动)。

### 上手

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Pople631Gs;
rhf.run(frame);

cpu::MultipoleObserver obs{rhf.result_handle()};
// 生产用法:挂到宿主驱动的 end_of_step;单点后也可直接 launch 一次。
// frame.meta 里出现 kDipole [Debye]、kQuadrupole [D·Å]、kMultipoleOrigin [Å]。
```

molvis 渲染:把 `kDipole` 矢量画在 `kMultipoleOrigin` 上即得教科书的
"偶极箭头"图(场景 API 见 molvis 文档)。

积分内核是移位中心闭式(Obara & Saika 1986, DOI 10.1063/1.450106),
平移恒等式 M¹(C′) = M¹(C) + (C−C′)·S 由单元测试在 1e-13 钉住。

## 20.2 布居分析与 ESP 电荷

"这个原子带多少电荷"不是可观测量:密度矩阵是整体的,任何按原子的划分都
是一种**约定**。本节把三种约定排成一条弧 —— Mulliken(重叠对半分)→
Löwdin(先正交化再划分)→ ESP 拟合(放弃轨道空间,直接拟合分子外的
静电势)—— 各自的适用域和陷阱都不同。

### Mulliken 回顾与 Löwdin

Mulliken 布居 N_A = Σ_{μ∈A}(PS)_{μμ} 把重叠布居对半劈给两个中心 —— 任意,
且对基组出了名地敏感(ch13 已见)。Löwdin 的修正是先把基组对称正交化:

q_A = Z_A − Σ_{μ∈A} (S^{1/2} P S^{1/2})_{μμ}   (Löwdin, 1950)

S^{1/2} 把每个 AO 变成"最像原来自己"的正交组合(对称正交化在最小二乘意
义下离原基组最近),重叠布居随之自然归属。两个警告:

- **Löwdin 对逐 AO 缩放不不变**(Mulliken/ESP 则不变)。笛卡尔 d/f 函数的
  工程归一化各码不同(PySCF cart 裸 libcint 与 Atomiverse 引擎的 d 自重叠
  差一个常数因子),所以 Atomiverse 在**单位归一化的 AO 基**(S_ii = 1)里
  做 Löwdin —— 数值与工程约定无关,神谕脚本同一步骤。
- 它仍是"轨道空间划分":基组越弥散,"这个函数属于哪个原子"越没意义。

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Pople631Gs;
rhf.tol = 1e-12;
rhf.run(frame);                                   // H2O

auto q  = cpu::lowdin_charge(rhf.result_handle(), frame);   // [e] 按原子
auto sz = cpu::lowdin_spin(rhf.result_handle(), frame);     // P^α−P^β 版本
```

### ESP 拟合电荷:MK 与 CHELPG

力场要的电荷有一个明确的物理目标:**再现分子外的静电势**。ESP 拟合直接
以此为目标函数 —— 在分子 vdW 面外采样 QM 势

V(r_k) = Σ_A Z_A/|R_A−r_k| − Σ_{μν} P_μν ⟨μ|1/|r−r′||ν⟩

(电子项即核吸引积分内核,单位点电荷放在格点上),再解带 Lagrange 乘子的
约束最小二乘(Σq = 分子净电荷,KKT 线性系统)。两种采样约定:

- **MK**(Besler-Merz-Kollman 1990):1.4/1.6/1.8/2.0 × Bondi 半径的嵌套壳
  层,每壳黄金角 Fibonacci 球面 ~1 点/Å²,落入他原子壳内的点弃掉。
- **CHELPG**(Breneman-Wiberg 1990):0.3 Å 立方格点,vdW 面外、距任一原子
  vdW 面 ≤2.8 Å 的带内取点。点多(H₂O ~8500 vs MK ~360),对分子取向更钝。

```cpp
cpu::MKCharges mk;                 // 参数即成员:mk.shell_scales, mk.density_per_ang2
auto q_mk = mk.fit(rhf.result_handle(), frame);        // [e]

cpu::ChelpgCharges chelpg;         // chelpg.spacing_ang = 0.3, chelpg.band_ang = 2.8
auto q_chelpg = chelpg.fit(rhf.result_handle(), frame);
```

**埋藏原子陷阱**:ESP 只在分子外采样,深埋原子(大分子内部的 sp³ 碳)的
1/r 列几乎与近邻的列共线 —— 最小二乘病态,电荷数值大而无物理意义,微小
的密度变化引起电荷大摆。Atomiverse 在 KKT 矩阵奇异值坍缩时自动切 SVD 最小
范数解(不发散,但**共线组内的电荷划分是任意的**——总和仍可靠)。RESP 的
分阶段拟合与等价约束就是为此发明的(本环未实现)。对称等价原子(H₂O 的两
个 H)的拟合电荷只在格点离散化水平相等(~1e-3 e):确定性格点不严格镜像
对称,这是约定的一部分,不是 bug。

molvis 可视化:`mk.grid(frame)` / `chelpg.grid(frame)` 返回 Å 单位的采样
点,直接作为点云叠在分子上,按 V(r_k) 上色即得教科书的 ESP 采样图(场景
API 见 molvis 文档)。

三法对照(H₂O RHF/6-31G*,PySCF 2.14.0 钉定,`regressions/scf/h2o_charges_pyscf`):
Löwdin 给 q_O ≈ −0.73 e,MK/CHELPG 都给 ≈ −0.81 e —— 轨道划分与"再现外部
静电势的有效电荷"本来就是两个目标,数值见章末表。

## 20.3 解析 Hessian 与振动频率

一阶导数(力)有一个奢侈的性质:变分波函数让密度弛豫项全部消失
(Hellmann-Feynman + Pulay 项即可,ch19 的梯度就是这么装配的)。二阶导数
没有这个运气。**2n+1 定则**说 n 阶波函数响应能给出 2n+1 阶能量导数——
零阶波函数给到梯度(2·0+1 = 1),而 Hessian(二阶)必须知道**轨道如何
随核位移弛豫**:一阶响应 U^x。这正是耦合微扰 Hartree-Fock(CPHF)方程
(Pople-Krishnan-Schlegel-Binkley 1979):

¹(A+B)·U^x = −B^x,每个核坐标 x 一个右端,共 3N 个。

- 左端 ¹(A+B)_{ai,bj} = δδ(ε_a−ε_i) + 4(ai|bj) − (ab|ij) − (aj|ib) ——
  **就是 ch03 稳定性分析的轨道 Hessian 单重态通道**。这是该共享内核的
  第三个配置:08 环求它的最低本征值(稳定性),09 环用它做 Newton 步
  (SOSCF),本环把它当线性方程组的系数矩阵。三个物理问题,一份 σ-向量
  代码(`OrbitalHessian::sigma_plus`),没有第二份积分变换。
- 右端 B^x_{ai} = F^{(x)}_{ai} − ε_i S^{(x)}_{ai} − Σ_kl S^{(x)}_{kl}
  [2(ai|kl) − (ak|il)],骨架(skeleton)导数 Fock 与重叠的 MO 块——
  历史上解析 Hessian 的两大 bug 源(重叠导数项、能量加权密度项)都在
  这里,所以 Atomiverse 对每一件都有独立的有限差分门禁单测。

装配式为(全部 Hartree/Bohr 内部):

E^{xy} = Tr[P h^{(xy)}] + ½ΣΣ P P g^{(xy)} + V_nuc^{xy} − Tr[W S^{(xy)}]
       + Tr[P^y F^{(x)}] − Tr[W^y S^{(x)}]

其中 W = 2Σ_i ε_i C_i C_iᵀ,响应项用密度响应 P^y 与能量加权密度响应
W^y 写成**约定无关**的形式(等价于教科书的 4Σ U^y B^x 耦合加 occ-occ
伴随项)。二阶 ERI 缩并逐原始四元组即时进行——四指标二阶导数张量从不
落地。

**第一道闸永远是自洽**:解析 Hessian 先对"解析梯度的有限差分"吻合
(两者都在库内,频率 ~0.01 cm⁻¹),过了这道闸才谈外部 oracle。有限差分
生产线(`cpu::vib::Hessian` 驱动)原样保留——两条生产线汇于同一个
`vib::HessianResult` 消费端。

### 上手:H₂O 的解析谐振频率

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Pople631Gs;   // 6-31G*,笛卡尔 d
rhf.tol = 1e-12;
cpu::vib::HessianResult H = rhf.hessian(frame);   // [3N,3N] eV/Å²

// 既有 vib 流水:质量加权 → TR 投影 → 对角化 → 频率 [cm⁻¹]
cpu::vib::HessianWorkspace ws;
ws.allocate(H.n_atoms);
ws.hessian = H.hessian;
// ws.r0 ← 平衡几何(Å);见 regressions/scf/h2o_rhf_freq
cpu::vib::MassWeighting{&ws}.launch(fin);
cpu::vib::HessianDiagonalizer{&ws}.launch(fin);
// ws.frequencies == [1843.71, 4036.17, 4146.38] cm⁻¹(0.95 Å/104.5° 几何)
```

`hessian()` 是与 `forces()` 并列的**方法输出**——没有"频率驱动器"这种
东西;振动分析属于 vib 组件,RHF 只负责把 d²E/dR² 算对。RHF-only:
UHF/RKS 的解析 Hessian(开壳层 CPHF、XC 二阶核)是后续链环;He/STO-3G
这类没有虚轨道的体系直接 throw(不存在 occ→virt 响应)。

### molvis:简正模可视化

`HessianDiagonalizer` 把简正模本征向量升回 3N 笛卡尔基
(`ws.normal_modes` 的列)。把某一列作为每原子位移矢量叠加在平衡几何上,
即得教科书的"振动箭头"图;`NormalModeWriter` 也已把每个模态写成
`mode_k.xyz` 位移轨迹,molvis 直接播放即可(molplot 场景 API 见其文档)。
弯曲模(1843 cm⁻¹)里氢原子沿角平分线相向运动、对称/反对称伸缩
(4036/4146 cm⁻¹)的 O-H 键相位差,一眼可辨。

频率对照见章末表:解析值与 PySCF `hessian.RHF` 的逐模偏差 ~3×10⁻⁴ cm⁻¹,
与自家有限差分生产线 ~10⁻² cm⁻¹(差分截断)——`regressions/scf/
h2o_rhf_freq` 把这两道闸都钉在 1 cm⁻¹。

## 20.4 激发态:TDHF/RPA 与 TDA

到目前为止本章读出的都是**基态**的物理:期望值(20.1/20.2)和核位移响应
(20.3)。激发能藏在另一种响应里:给分子加一个频率为 ω 的振荡电场,问极
化率 α(ω) 如何随 ω 变化——当 ω 扫过一个激发能时 α(ω) **发散**。激发能不
是"算出来的能量差",而是**频率相关极化率的极点**;在极点处,无穷小的微
扰引起有限的密度振荡,这个自持振荡就是激发态。把"HF 密度对振荡场的一阶
响应存在自持解"写成方程,就是时间相关 Hartree-Fock(TDHF,历史上也叫
RPA,随机相近似):

[[A, B], [−B, −A]] (X; Y) = ω (X; Y)。

X 是 occ→virt 激发振幅,Y 是 virt→occ **退激发**振幅——负频率极点的伴生
分量,超矩阵因此非对称,本征值成 ±ω 对出现。而 A、B 两个块正是老朋友:

A_ai,bj = δδ(ε_a−ε_i) + 2(ai|jb) − (ij|ab),  B_ai,bj = 2(ai|bj) − (aj|bi)。

这是链环 08 那份轨道 Hessian A/B 内核的**第四个配置**。同一份 σ-向量代码
(`OrbitalHessian::sigma_plus/sigma_minus`),四个物理问题:

| 环 | 配置 | 问的问题 |
|---|---|---|
| 08 | (A+B) 最低本征值 | 这个 SCF 解是极小还是鞍点?(ch03 稳定性节) |
| 09 | (A+B) 上的 Newton 步 | 从鞍点/远处如何二阶收敛?(SOSCF) |
| 13 | ¹(A+B)·U^x = −B^x | 轨道如何随核位移弛豫?(CPHF → Hessian) |
| 14 | [[A,B],[−B,−A]] 本征对 | 密度自持振荡的频率是多少?(激发能) |

四份物理,一份积分变换——这就是把内核抽成共享私有件的全部回报。

### 求解:Hermitian 约化与 TDA

实轨道下非对称超矩阵可以约化成对称问题(Atomiverse 走的路):

(A−B)^{1/2} (A+B) (A−B)^{1/2} Z = ω² Z,

解出 ω² 后回代归一化的 (X, Y)(归一条件 XᵀX − YᵀY = 1——注意不是普通的
单位范数,Y 分量带负号计入,这是非 Hermitian 结构留下的痕迹)。小体系直
接稠密解;大体系用与 08/13 环同款的确定性 Davidson,(A−B) 只在子空间内
本征分解(Stratmann-Scuseria-Frisch 1998),σ± 乘积之外从不落地任何
n_ov × n_ov 矩阵。

**TDA(Tamm-Dancoff 近似)**:直接扔掉 B(即扔掉 Y),剩下 Hermitian 的
A X = ω X。在 HF 参考上这**就是 CIS**——单激发组态相互作用。少了退激发
耦合,TDA 激发能系统性偏高:对每个最低根有 **ω_RPA ≤ ω_TDA**(变分性
质,单元测试逐根钉住)。

### 诚实的对比:RPA 更"好"吗?

教科书弧线会说 RPA 比 TDA 多包含了关联(基态隐含 RPA 关联),但代价在
一个地方爆发:**triplet instability**。约化形式良定的前提是 (A+B)、(A−B)
正定——而"³(A+B) 出现负本征值"恰恰是 ch03 稳定性节的 RHF→UHF 外部不稳
定判据。参考态在三重态方向是鞍点时,³ω² < 0,**三重态 RPA 激发能是虚数**:
不是数值 bug,是数学在告诉你"这个参考态正沿该方向塌缩,微扰理论失效"。
TDA 没有这个病:A 是 Hermitian 的,谱永远是实的,不稳定参考上照样给出
(定性尚可的)激发能——这正是 Hirata-Head-Gordon 论文让 TDA 流行起来的
原因之一。

Atomiverse 把这个陷阱做成**前置检查而不是事后惊吓**:`tdhf()` 先跑链环
08 的稳定性诊断,请求通道的 (A+B) 不稳定就直接 throw,消息里带最低本征
值和出路("改用 `tda()`,或先沿 `stability().rotated_C` 重收敛");
`tda()` 在同一参考上保持实数。拉伸 H₂(R = 1.5 Å,ch03 的老例子)就是
单元测试里的活教材。

### 跃迁性质:亮态与暗态

激发能只回答"在哪儿吸收";"吸收多强"由长度规范跃迁偶极给出:

μ_0n = √2 Σ_ia (X+Y)_ia ⟨φ_i|r|φ_a⟩,  f = (2/3)·ω[Ha]·|μ_0n|²[a.u.]。

AO 偶极矩阵复用 20.1 的移位中心 OS 内核(链环 03),occ-virt MO 变换取自
行列式的 C——又一次没有第二份积分代码。三重态通道自旋禁阻:μ ≡ 0、
f ≡ 0,不是数值上的小,是严格为零(自旋积分正交)。

### 上手:甲醛的 n→π* 与 π→π*

甲醛是激发态教学的果蝇:HOMO 是氧的面内孤对 n(b₂),LUMO 是 C=O 的
π*(b₁),最低单重激发 n→π* 属 ¹A₂ ——**偶极禁阻**(A₂ 不含任何偶极分量
的不可约表示),f = 0,实验上只以微弱振动借强出现;而 π→π*(¹A₁)是沿
C=O 轴偏振的亮态。

```cpp
using cpu::scf::ExcitationChannel;

cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Pople631Gs;   // 6-31G*,笛卡尔 d
rhf.tol = 1e-12;
rhf.run(frame);                               // CH2O,几何见 regressions

auto tda  = rhf.tda(3);                       // TDA(=CIS),默认 singlet
auto tdhf = rhf.tdhf(3);                      // 完整 TDHF/RPA(先过稳定性检查)

for (int s = 0; s < 3; ++s)
    std::printf("root %d: TDA %.4f eV (f=%.4f)   TDHF %.4f eV (f=%.4f)\n",
                s + 1, tda.energy_ev[s], tda.oscillator_strength[s],
                tdhf.energy_ev[s], tdhf.oscillator_strength[s]);
// root 1:  4.6690 eV (f=0.0000)    4.4999 eV (f=0.0000)   ← n→π* 暗态
// root 2:  9.9505 eV (f=0.0013)    9.7075 eV (f=0.0008)   ← σ→π*
// root 3: 10.3002 eV (f=0.2166)    9.7559 eV (f=0.1796)   ← π→π* 亮态

auto t3 = rhf.tda(3, ExcitationChannel::triplet);  // 三重态梯:μ=0, f=0
```

读表三件事:每根 ω_RPA ≤ ω_TDA;n→π* 在两种理论下都严格 f = 0(对称性,
不是巧合);π→π* 的跃迁偶极 >99% 落在 z(C=O 轴)分量上。CIS/TDHF 的
n→π* ~4.5 eV 对实验 ~4.0 eV 偏高约半 eV——单激发理论的典型精度,Jensen
的告诫原样适用:这是**定性归属 + 半定量能级**的工具,不是光谱仪。

### molvis:轨道对与跃迁的可视化

暗态/亮态的判据可以用眼睛验证:`OrbitalCubeWriter` 把 HOMO(n,面内)、
HOMO−1(π,面外)与 LUMO(π*,面外)写成 cube(ch13 的密度/轨道管线原
样复用),molvis 等值面叠加即见——n 与 π* 处处正交于同一平面,偶极积分
逐点相消(A₂ 禁阻的实空间版本);π 与 π* 同为面外、沿键轴反相,乘积沿
C=O 不变号,亮态一目了然。把 `transition_dipole_au` 的行向量画成
20.1 式的箭头(置于核电荷中心),方向即偏振方向(场景 API 见 molvis
文档)。

对照与门禁:`regressions/scf/ch2o_tdhf` 把 TDA/TDHF 各前 3 根钉在外部运
行的 PySCF `tdscf.TDA`/`tdscf.TDHF` 上(1e-4 eV,实差 ~3e-7 eV;亮态 f
1e-3,实差 ~1e-8),数值见章末表。理论脉络:Dreuw & Head-Gordon 2005
(DOI 10.1021/cr0505627)、Hirata & Head-Gordon 1999
(DOI 10.1016/S0009-2614(99)01149-5)、Szabo & Ostlund §7。

## 20.5 外场与点电荷嵌入

环境不是第三种势能函数,而是**单电子 Fock 贡献**:点电荷嵌入把 MM 区的
静电势写进核哈密顿量,均匀外场把偶极算符写进核哈密顿量。两者都不改
动双电子算符,也不碰 Frame——成员设好就跑:

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;

// QM/MM 静电嵌入(QM 侧):位置 [Å],电荷 [e]
rhf.point_charges = {
    {0.0, 0.0,  3.0,  0.5},
    {0.0, 0.0, -3.0, -0.5},
};

// 均匀外场 [a.u.];原点 = 核电荷中心(与 20.1 偶极同一约定)
rhf.external_field = {0.0, 0.0, 1e-4};

rhf.run(frame);   // e_total 自动含 QM–MM / MM–MM / 核–场经典项
// 解析力尚未接线:forces() 在环境非空时抛异常(无静默错误力)
```

### 环境如何进 Fock

- 点电荷:h_μν += −Σ_C q_C ⟨μ|1/|r−R_C||ν⟩,积分就是"放在 R_C 的单位电荷核
  吸引",复用核吸引原语(与 20.2 ESP 势核同一机器)。经典项
  Σ_A Σ_C Z_A q_C/r + Σ_{C<C'} q_C q_C'/r 进 v_nuc。
- 均匀电场:h_μν += F·⟨μ|(r−R₀)|ν⟩(电子电荷 −1 a.u.),核项
  −Σ_A Z_A F·(R_A−R₀);R₀ 必须与偶极原点一致,否则有限场恒等式破掉。

接线在 `drive_scf_core_` 的积分序幕:IntegralPack 装完 Hcore 之后、Fock
循环之前一次加上——hamiltonian_builder 无感知,e_total 自动含环境。

### 有限场偶极 vs 解析偶极

一阶恒等式 E(F) = E(0) − μ·F 给出交叉核验:中心差分 dE/dF 在 F→0 时等于
−μ(解析,20.1 节)。H₂O/STO-3G、F = 1e-4 a.u. 的对照:

| 量 | 有限场 dE/dF_z | 解析 μ_z | 容差 |
|---|---|---|---|
| μ_z [a.u.] | ≈ −0.678787 | −0.678787 | 1e-5 a.u. |

门禁:`regressions/scf/h2o_embedding_field`。

### ⚠ 有限场变分坍缩

静电场下能量**无下界**:电子可以沿 −F 方向无限远离、能量无限低。弥散
基下 SCF 真的会"电离"——波函数跑到箱子边缘,偶极发散。纪律:

- 场强保持小(~**1e-4 a.u.**);不要用 1e-2 "看更清楚"
- 有限场超极化率/高阶响应另论;本链路只钉一阶偶极恒等式
- 工程上无自动防护——调用方负责,教程与 Doxygen 都写了这条警告

### 点电荷嵌入 vs PySCF

H₂O/STO-3G + 上列两个 ±0.5 e 电荷,Atomiverse 总能量对外部钉定的
PySCF 2.14.0 `qmmm.mm_charge` 在 **1e-8 Ha** 内一致。注意 PySCF 的
`mm_charge` **不含** MM–MM 自能,Atomiverse 的 `v_nuc` **含**——断言时把
经典 MM–MM 常数显式加回去(见 `REFERENCE.md`)。

## 20.6 隐式溶剂:C-PCM

把溶剂当成介电常数为 ε 的**连续介质**,溶质坐在一个由原子中心球构成的
**空腔**里——这是导体类 PCM(Polarizable Continuum Model)的基本图景。
空腔表面的**表观表面电荷**(apparent surface charge, ASC)响应溶质静电势,
自洽地写回 Fock 矩阵;与 ch20.5 的静态点电荷嵌入不同,溶剂电荷**随密度
更新**(每次 SCF 迭代解一次导体方程)。

### 导体方程与缩放约定

在 tessera 点电荷离散下:

S q = −f(ε) V,   f(ε) = (ε − 1)/(ε + x),   ΔG_el = ½ Σ_k q_k V_k

- **x = 0** → **C-PCM**(Barone–Cossi 1998;PySCF `solvent.PCM` 默认) —
  `f = (ε−1)/ε`
- **x = 0.5** → 原始 **COSMO**(Klamt 1993) — `f = (ε−1)/(ε+½)`

低 ε 时两约定可差几个百分点;高 ε(水 78.4)下几乎重合。Atomiverse 成员
`solvent_x` 默认 0(C-PCM)。S 的非对角是(高斯模糊的)1/r,对角用
Lange–Herbert / PySCF 的 ξ 宽度表;Lebedev 默认 302 点/球。

### 空腔:Bondi × 缩放,硬剔除

空腔球半径 = **Bondi 半径 × `solvent_radii_scale`**(默认 1.2,Q-Chem/PySCF
惯例)。氢用 PCM 常见的 **modified Bondi 1.1 Å**(经典 Bondi 为 1.20 Å)。
落在**其他球内部**的 Lebedev 点直接丢掉(硬开关)——空腔表面在核坐标上
**不连续**,几何扫描会出现能量折角。

> **诚实声明(范围裁定)**
>
> 1. **只含静电** ΔG_el。空穴化、色散、排斥等**非静电项一律不含** —
>    与实验水合自由能差几个 kcal/mol 是预期的,不要拿本环数字当 ΔG_solv。
> 2. **表面未做 SWIG 平滑**(York–Karplus / Lange–Herbert 开关函数)。
>    PySCF 默认 SWIG;两者对小分子差 ≲ 0.1 kcal/mol,大分子/几何扫描时
>    不连续更明显。平滑与 ddCOSMO 线性标度求解器是后续路线。
> 3. **`forces()` 在 `solvent_eps > 0` 时 throw**——解析 PCM 梯度未实现;
>    需要力时用有限差分或先关溶剂。

### 上手:H₂O 溶剂化能

```cpp
cpu::scf::RHF gas;
gas.basis = cpu::scf::BasisSet::Pople631Gs;
gas.tol = 1e-10;
gas.run(frame);                          // solvent_eps 默认 0 = 气相

cpu::scf::RHF sol;
sol.basis = cpu::scf::BasisSet::Pople631Gs;
sol.tol = 1e-10;
sol.solvent_eps = 78.4;                  // 水
sol.solvent_radii_scale = 1.2;           // Bondi × 1.2
sol.solvent_x = 0.0;                     // C-PCM
sol.run(frame);

double dg_el_ha   = sol.e_solvent();     // ½ q·V [Hartree]
double dg_el_kcal = dg_el_ha * 627.509474;
// 期望: dg_el < 0, sol.energy() < gas.energy()
```

`solvent_eps == 0` 路径与完全不碰溶剂成员的气相**逐位相同**(零开销关断)。

### molvis:空腔与表面电荷

收敛后,ASC 与 tessera 坐标是 `internal::Pcm` 的私有工作区(不写 Frame)。
教程可视化路线:

1. 用 `solvent_eps > 0` 跑一次 SCF,从回归/调试钩子导出 tessera 位置与 q_k
   (或在本机用 PySCF `mf.with_solvent.surface` 对照);
2. molvis 把每个 tessera 画成带符号着色的小球(正电荷红 / 负电荷蓝),
   原子 vdW 球半透明叠加即得教科书的"表面电荷云"图。

生产 API 若要把 ASC 挂到 Frame meta,是后续观测器环的事——本环只保证
能量自洽。

### 与 PySCF 对照

H₂O RHF/6-31G*、ε=78.4、Bondi×1.2、x=0、Lebedev 302:

| 量 | Atomiverse | PySCF 2.14.0 C-PCM | 容差 |
|---|---|---|---|
| ΔG_el [kcal/mol] | −8.018 | −7.960 | 0.1 kcal/mol |

完整钉定见 `regressions/scf/h2o_rhf_631gs_cpcm/REFERENCE.md`。

## 20.7 标量相对论:x2c

### 重元素为什么需要相对论

电子在重核附近局域加速,Dirac 方程的动能修正使 **s/p 轨道径向收缩**
(有效核电荷升高 → 能量降低、半径变小),而 d/f 因壳层屏蔽变化反而
**径向膨胀**。对第三周期及以后(Cl、Ar、Br、I …)的全电子计算,忽略
标量相对论会把内层能量系统抬高 ~eV–Ha 量级;外层性质(键长、电离能)
也会通过屏蔽间接偏移。

自旋–轨道耦合(SOC)是另一条相对论通道,本环**不**处理——我们只做
无自旋(spin-free)的一电子精确二分量修正。

### x2c-1e 一页讲清

**精确二分量(exact two-component, x2c)** 把四分量 Dirac 方程用
一步 Foldy–Wouthuysen 型变换退耦到正能二分量空间
(Kutzelnigg–Liu 2005; Peng–Reiher 2012)。**x2c-1e** 只对一电子哈密顿
做这件事:

1. 在收缩 AO 基里组装 {S, T, V, W},其中
   \(W_{\mu\nu}=\langle\nabla\mu|V|\nabla\nu\rangle\)
   (PySCF `int1e_pnucp`);
2. 动能平衡基下的无自旋 Dirac 矩阵
   \(h=\begin{pmatrix}V&T\\T&W/(4c^2)-T\end{pmatrix}\)、
   度规 \(M=\begin{pmatrix}S&0\\0&T/(2c^2)\end{pmatrix}\);
3. 取正能解得退耦矩阵 \(X=C_S C_L^{-1}\),重整化 \(R\),得到
   \(h^{\mathrm{X2C}}\);
4. SCF 的 **Hcore ← h^X2C**,S / ERI / Fock 循环逐字节不动
   (二电子积分保持非相对论——这就是 x2c-1e 的定义)。

光速取 PySCF 同值 \(c=137.03599967994\) a.u.。退耦在**收缩** AO 基内
完成;PySCF 默认 `xuncontract=True`,对账时必须显式
`xuncontract=False`。

### ECP vs x2c:怎么选

| | ECP(第 06 环) | x2c-1e(本环) |
|---|---|---|
| 覆盖 | 重元素价层(def2-ECP: Rb–Rn) | 全电子基组覆盖的元素(当前 Z≤18) |
| 效应 | 参数化进有效核势(含标量相对论) | 显式改造一电子哈密顿 |
| 基组 | 价层基(def2-SVP/TZVP + ECP) | 全电子基(def2-SVP/TZVP 等) |
| 力 | `forces()` throw | `forces()` throw |
| 互斥 | **与 x2c 永久互斥** | **与 ECP 永久互斥** |

**规则:同一原子绝不能 ECP + x2c。** 相对论 ECP 已经把标量相对论烘进
参数,再叠 x2c 就是双重计数——引擎在 `run()` 入口按元素 THROW,错误
信息带元素符号。

- 有 def2-ECP 覆盖的重原子 → 用 ECP,`x2c=false`(默认)。
- 轻/中等元素全电子、想看相对论内层收缩 → `x2c=true`,`ecp=""`。
- 混合体系(轻原子全电子 + 重原子 ECP)→ 只开 ECP,不要开 x2c
  (门是 per-element 的:只要有一个 ECP 命中原子且 `x2c=true` 就抛)。

### 诚实警告:picture-change 误差(PCE)

x2c 轨道是在变换后的算符下变分得到的。若用**未变换**的非相对论算符
(偶极 \(\mathbf{r}\)、ESP、多极)去算期望值,结果含
**picture-change error**。本环**只替换 Hcore,不变换属性算符**——
偶极/ESP/多极(第 03/04 环)在 `x2c=true` 时仍走非相对论算符路径,
数值可用作粗估,但不得当作 picture-change 修正后的真值。属性算符
的 PCE 变换是后续环。

`forces()` 在 `x2c=true` 时 throw(h^X2C 核梯度未推导;无静默错误力)。

### 上手:Ar 原子 x2c-1e RHF

```cpp
cpu::Frame ar(1);
ar.span(kAtom_type)[0] = 18;
ar.span(kAtom_x)[0] = ar.span(kAtom_y)[0] = ar.span(kAtom_z)[0] = 0.0;
ar.simbox = cpu::Box::cubic(30.0);

cpu::scf::RHF nr;
nr.basis = cpu::scf::BasisSet::Def2TZVP;
nr.tol = 1e-12;
nr.run(ar);
double e_nr = ar.meta.get(kEnergy) / units::kHartreeInEv;

cpu::scf::RHF x2c;
x2c.basis = cpu::scf::BasisSet::Def2TZVP;
x2c.tol = 1e-12;
x2c.x2c = true;                 // 标量相对论 x2c-1e
x2c.run(ar);
double e_x2c = ar.meta.get(kEnergy) / units::kHartreeInEv;
// 期望: e_x2c < e_nr, |e_x2c - e_nr| ≈ 1.84 Ha (Ar/def2-TZVP)
```

端到端钉定见 `regressions/scf/x2c_argon/`。

### 与 PySCF 对照

Ar RHF/def2-TZVP、`xuncontract=False`、c = 137.03599967994:

| 量 | Atomiverse | PySCF 2.14.0 x2c1e | 容差 |
|---|---|---|---|
| E_x2c [Ha] | (回归断言) | −528.6442231685685 | 1e-5 Ha |
| E_x2c − E_nonrel [Ha] | (回归断言) | −1.84146283604 | 符号 + 幅值 5% |

## 章末:与 PySCF 对照(累积表占位)

| 体系/设置 | 量 | Atomiverse | PySCF(钉定) | 容差 | 环 |
|---|---|---|---|---|---|
| H2O RHF/6-31G* | μ_z [a.u.] | −0.8757858700 | −0.8757858708 | 1e-6 a.u.(实差 8e-10) | 03 |
| H2O RHF/6-31G* | μ_z [D] | −2.226026 | −2.2260255980 | —(HF 高估 vs 1.85 D 实验) | 03 |
| H2O RHF/6-31G* | Löwdin q_O [e](归一化 AO) | −0.72840153 | −0.7284015337 | 1e-6 e(实差 5e-9) | 04 |
| H2O RHF/6-31G* | MK q_O [e](文档化格点) | −0.80709214 | −0.8070921421 | 1e-3 e(实差 8e-10) | 04 |
| H2O RHF/6-31G* | CHELPG q_O [e](文档化格点) | −0.80500612 | −0.8050061239 | 1e-3 e(实差 9e-10) | 04 |
| N2 RHF/6-31G* | −ε_HOMO [eV](Koopmans IP) | 16.646730 | 16.64672961 | 1e-3 eV(实差 4e-7) | 04 |
| N2(+) UHF/6-31G* | ΔSCF IP [eV] | 15.692600 | 15.69256822 | 1e-3 eV(实差 3e-5) | 04 |
| H2O RHF/6-31G* | ν₁ 弯曲 [cm⁻¹] | 1843.7086 | 1843.7084 | 1 cm⁻¹(实差 2e-4) | 13 |
| H2O RHF/6-31G* | ν₂ 对称伸缩 [cm⁻¹] | 4036.1734 | 4036.1737 | 1 cm⁻¹(实差 3e-4) | 13 |
| H2O RHF/6-31G* | ν₃ 反对称伸缩 [cm⁻¹] | 4146.3779 | 4146.3782 | 1 cm⁻¹(实差 3e-4) | 13 |
| CH2O RHF/6-31G* | TDA root 1 n→π* [eV] | 4.6689702 | 4.6689705 | 1e-4 eV(实差 3e-7) | 14 |
| CH2O RHF/6-31G* | TDA root 2 σ→π* [eV] | 9.9504661 | 9.9504663 | 1e-4 eV(实差 2e-7) | 14 |
| CH2O RHF/6-31G* | TDA root 3 π→π* [eV] | 10.3001674 | 10.3001676 | 1e-4 eV(实差 2e-7) | 14 |
| CH2O RHF/6-31G* | TDHF root 1 n→π* [eV] | 4.4999396 | 4.4999399 | 1e-4 eV(实差 3e-7) | 14 |
| CH2O RHF/6-31G* | TDHF root 2 σ→π* [eV] | 9.7075259 | 9.7075262 | 1e-4 eV(实差 3e-7) | 14 |
| CH2O RHF/6-31G* | TDHF root 3 π→π* [eV] | 9.7558617 | 9.7558620 | 1e-4 eV(实差 3e-7) | 14 |
| CH2O RHF/6-31G* | π→π* f(TDA)(长度规范) | 0.2165662 | 0.2165661719 | 1e-3(实差 ~1e-8) | 14 |
| CH2O RHF/6-31G* | π→π* f(TDHF)(长度规范) | 0.1795989 | 0.1795988961 | 1e-3(实差 ~1e-8) | 14 |
| H2O RHF/STO-3G + 2×±0.5e @ ±3 Å | E_embed − E_MM-MM [Ha] | −74.9841893192 | −74.9841893192 | 1e-8 Ha | 15 |
| H2O RHF/STO-3G | μ_z 有限场 vs 解析 [a.u.] | ≈ −0.678787 | −0.678787 | 1e-5 a.u. | 15 |
| H2O RHF/6-31G* C-PCM ε=78.4 Bondi×1.2 x=0 | ΔG_el [kcal/mol] | −8.018 | −7.960 | 0.1 kcal/mol | 16 |
| Ar RHF/def2-TZVP x2c-1e (xuncontract=False) | E_x2c [Ha] | (reg) | −528.6442231685685 | 1e-5 Ha | 18 |
| Ar RHF/def2-TZVP x2c-1e (xuncontract=False) | E_x2c−E_nr [Ha] | (reg) | −1.84146283604 | 符号+5% | 18 |

对照方法学见[附录:PySCF 对照方法学](appendix-pyscf-oracle.md)。
