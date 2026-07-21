# Chapter 18 — SCF 技巧:初猜、混合与二阶方法

> **状态:全章**(初猜与 DIIS 家族由 hf-capability-gaps-07-guess 填充;
> 18.4 二阶方法由 09-soscf 环填充)。读者读完本章应相当于读完 Jensen
> 《Introduction to Computational Chemistry》§3.8 前半("SCF techniques"):
> 以**方法为主干**,把算法与技巧织进收敛叙事,而不是罗列功能。

## 18.1 SCF 为什么会发散

Roothaan-Hall 迭代 \(F(P_k)\,C = S C \varepsilon \to P_{k+1}\) 是一个不动点
迭代,而不动点迭代天生不保证收敛。发散的典型病理只有一种,却以两副面孔出现:

- **电荷来回倒(charge sloshing)**:两个近简并的电子构型 A、B 之间,
  \(P_A\) 生成的 Fock 算符偏爱构型 B,\(P_B\) 生成的又偏爱 A。密度在两个
  构型之间做周期 2 的振荡,能量曲线呈锯齿。小 HOMO-LUMO 能隙的体系
  (拉伸的共价键、双自由基、过渡金属)几乎必然表现出这一行为。
- **过冲(overshoot)**:每步走得太远,新的密度比旧的离自洽点更远,
  振幅逐步放大直至数值溢出。裸 Hcore 初猜的第一步常是重灾区——没有任何
  电子屏蔽的核哈密顿量会把全部电子塞进最深的几个轨道。

最古老的两副镇痛剂在第 2、3 章已经用过,这里回顾其机理与成员位置:

- **密度阻尼(damping)**:\(P \leftarrow (1-\alpha)P_{\text{new}} +
  \alpha P_{\text{old}}\),成员 `damping_factor`(PySCF 约定,α 是**保留旧
  密度**的比例;`UHF` 默认 0.7,DIIS 激活后自动关闭)。它压制振荡,但把
  收敛速度也一并压扁——纯阻尼是线性收敛。
- **能级移位(level shift)**:把虚轨道整体抬高 λ(成员 `level_shift`
  [Hartree];Saunders & Hillier 1973)。人为拉大能隙可以强行止住电荷
  来回倒,代价同样是变慢:λ 越大越稳、越慢,且移位下的"收敛"要在去掉
  移位后重新验证(引擎在收敛后自动用未移位 Fock 重解一次轨道)。

现代的标准配方是**三段式**:好初猜把起点放进收敛域附近(18.2),
远域混合器(EDIIS/ADIIS)负责把远离自洽的迭代拖进近域(18.3),
近域再交给 CDIIS 收尾。三段各司其职,任何一段用错位置都会失速。

## 18.2 初猜家族

SCF 是局部迭代,初猜决定落进哪个收敛盆、以及离盆底多远。四种初猜按
"物理含量"递增排列(Lehtola, *JCTC* **15**, 1593 (2019), DOI
[10.1021/acs.jctc.8b01089](https://doi.org/10.1021/acs.jctc.8b01089) 对
它们做了系统评测):

- **Hcore**(`InitGuess::Hcore`):对角化裸核哈密顿量 \(T + V_{ne}\)。
  没有电子-电子屏蔽,轨道被过度压向核,是所有初猜里最差的——它存在的
  意义是永远可用(不依赖任何表格)。
- **MINAO / SAD**(`InitGuess::MINAO`,默认):原子密度叠加——把预先算好
  的原子块密度投影到分子基组,得到一个不错的分子密度,但它不是任何
  Fock 算符的本征态。PySCF 的默认路线。
- **扩展 Hückel**(`InitGuess::Hueckel`):组装一个经验 Fock 矩阵
  \(F_{\mu\mu} = h_{\mu\mu}\)(VSIP,价态电离势静态表,Hartree),
  \(F_{\mu\nu} = \tfrac12 K (h_{\mu\mu}+h_{\nu\nu}) S_{\mu\nu}\),
  \(K = 1.75\)(Wolfsberg-Helmholz),对角化一次即得初猜轨道。与 MINAO
  相反:密度粗糙,但轨道**有能级结构**——占据模式从一开始就合理。
  表内元素 H–Ar;表外元素直接抛异常(无回退值)。
- **SAP**(`InitGuess::SAP`):原子势叠加——有效单电子势取
  \(V_{\text{SAP}}(\mathbf r) = \sum_A V_A(|\mathbf r - \mathbf R_A|)\),
  \(V_A\) 是 Lehtola 表列的交换泛函原子径向有效势(本引擎采用其
  高斯拟合 erf 展开形式,数据转录自 PySCF 的 `sap_grasp_large`;
  Lehtola-Visscher-Engel, DOI
  [10.1063/5.0004046](https://doi.org/10.1063/5.0004046)),
  \(F = T + V_{\text{SAP}}\) 在 DFT 求积网格上装配,对角化一次,
  **完全不需要密度**。全周期表 Z ≤ 118。

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Def2SVP;
rhf.init_guess = cpu::scf::InitGuess::Hueckel;   // 或 InitGuess::SAP
rhf.run(frame);
```

**开壳陷阱(占据平均化)**:Hückel 与 SAP 都是自旋对称的算符,单次对角化
给不出自旋极化。开壳层运行沿用现有规则——同一组初猜轨道分别填 α/β,
并做 HOMO-LUMO 混合以便打破对称。寻找碎对称解时,初猜只负责"起点合理",
对称破缺仍靠混合角与后续迭代(见 18.3 的困难例)。

H2O/def2-SVP RHF(tol=1e-9)的实测迭代数(PySCF 2.14.0 同几何对照;
PySCF 的 `huckel` 是 Lehtola 的免参数即时原子计算变体,本引擎是同文献的
静态 VSIP 表变体,故只应比较**相对 Hcore 的次序**,不比较绝对值):

| 初猜 | Atomiverse 迭代数 | PySCF 迭代数 |
|---|---:|---:|
| Hcore | 16 | 10 |
| MINAO | 16 | 8 |
| Hückel | **14** | 8 |
| SAP | **14** | 8 |

两条断言 `iters(Hückel) < iters(Hcore)`、`iters(SAP) < iters(Hcore)` 由
`regressions/scf/h2o_scf_convergence` 长期钉住。

## 18.3 DIIS 家族:CDIIS、EDIIS、ADIIS

**CDIIS 回顾**(Pulay 1980/1982):以对易子
\(e_i = X^\top(F_iP_iS - SP_iF_i)X\) 为误差向量,在历史子空间里做
无约束最小二乘,再以同一组系数外推 Fock:\(\tilde F = \sum_i c_i F_i\),
\(\sum_i c_i = 1\) 但 **c 可以为负、可以远大于 1**。自洽点附近 \(e\) 与
密度误差近似线性,这一外推近乎二次收敛——CDIIS 是无可争议的近域冠军。
但在远离自洽处,线性化失效,外推系数动辄 ±10,一步把密度甩出物理区域。

**EDIIS**(Kudin-Scuseria-Cancès 2002, DOI
[10.1063/1.1470195](https://doi.org/10.1063/1.1470195))换了目标函数:
在**单纯形约束**(\(\sum c_i = 1\), \(c_i \ge 0\))下最小化能量的二次模型

\[
\min_{c}\; \sum_i c_i E_i \;-\; \tfrac12 \sum_{ij} c_i c_j
\langle D_i - D_j \,|\, F_i - F_j \rangle .
\]

系数非负意味着新密度永远是历史密度的**凸组合**——绝不过冲,单调把
能量往下压。**ADIIS**(Hu-Yang 2010, DOI
[10.1063/1.3304922](https://doi.org/10.1063/1.3304922))用增广
Roothaan-Hall 能量替代显式能量,同一单纯形约束:

\[
\min_{c}\; 2\sum_i c_i \langle D_i - D_n | F_n \rangle
+ \sum_{ij} c_i c_j \langle D_i - D_n | F_j - F_n \rangle ,
\]

只需要 (D, F) 历史、不需要能量通道,是 PySCF 推荐的默认远域混合器。

**适用域陷阱——远域专用。** 凸组合既是它们的美德也是它们的天花板:
接近自洽时最优解退化到"全部权重压在最新一步",EDIIS/ADIIS 在
DIIS 误差 ~1e-4 附近失速,永远磨不出 1e-9。**用到底是错误用法。**
标准配方是升级切换:远域用 ADIIS/EDIIS,一旦 max|DIIS 误差| 降到阈值
以下就**永久**切回 CDIIS——

```cpp
cpu::scf::UHF uhf(5, 5);
uhf.mixer = cpu::scf::MixerKind::ADIIS_CDIIS;  // 默认 CDIIS,行为零变化
uhf.mixer_switch_tol = 1e-2;                   // max|DIIS err| 低于即切换
uhf.run(frame);
```

困难例:2.6 倍拉伸的 H2O/def2-SVP UHF(Hcore 初猜,无阻尼)。这个几何
的自旋对称解是不稳定驻点(PySCF 稳定性分析确认其存在内禀不稳定性),
真正的 UHF 极小是碎对称双自由基解(⟨S²⟩ ≈ 1.97)。纯 CDIIS 在 60 步预算
内始终绕着不稳定对称解振荡;ADIIS→CDIIS 则在 38 步收敛到 PySCF 钉定的
碎对称能量 −75.71896632 Ha(≤1e-6 Ha):

```molplot preset="molplot" theme="auto" width="100%" aspect="16:9"
mark: {type: line, point: true, tooltip: true}
data:
  values:
    - {iter: 2, series: CDIIS, E: -62.204}
    - {iter: 4, series: CDIIS, E: -80.771}
    - {iter: 6, series: CDIIS, E: -73.833}
    - {iter: 8, series: CDIIS, E: -76.205}
    - {iter: 10, series: CDIIS, E: -75.478}
    - {iter: 14, series: CDIIS, E: -75.357}
    - {iter: 18, series: CDIIS, E: -75.358}
    - {iter: 22, series: CDIIS, E: -75.358}
    - {iter: 26, series: CDIIS, E: -75.357}
    - {iter: 30, series: CDIIS, E: -75.358}
    - {iter: 34, series: CDIIS, E: -75.357}
    - {iter: 38, series: CDIIS, E: -75.357}
    - {iter: 42, series: CDIIS, E: -75.357}
    - {iter: 2, series: ADIIS-CDIIS, E: -62.204}
    - {iter: 4, series: ADIIS-CDIIS, E: -71.732}
    - {iter: 6, series: ADIIS-CDIIS, E: -75.354}
    - {iter: 8, series: ADIIS-CDIIS, E: -75.213}
    - {iter: 10, series: ADIIS-CDIIS, E: -75.365}
    - {iter: 14, series: ADIIS-CDIIS, E: -75.704}
    - {iter: 18, series: ADIIS-CDIIS, E: -75.718}
    - {iter: 22, series: ADIIS-CDIIS, E: -75.719}
    - {iter: 26, series: ADIIS-CDIIS, E: -75.720}
    - {iter: 30, series: ADIIS-CDIIS, E: -75.719}
    - {iter: 34, series: ADIIS-CDIIS, E: -75.71897}
    - {iter: 38, series: ADIIS-CDIIS, E: -75.71897}
    - {iter: 42, series: ADIIS-CDIIS, E: -75.71897}
encoding:
  x: {field: iter, type: quantitative, title: "SCF iteration"}
  y: {field: E, type: quantitative, title: "E (Hartree)", scale: {zero: false}}
  color: {field: series, type: nominal, title: null}
```

CDIIS(上方平台)被不稳定对称解捕获;ADIIS 的凸组合先把能量单调压过
对称平台,切换回 CDIIS 后二次收敛锁定碎对称极小(下方平台)。

工程注记:EDIIS 需要能量通道 \(E_i\),本引擎经 `MixStage` 的能量引用
传入,而能量在 density 阶段才结算,故 \(E_i\) 滞后其 (F, D) 对一次迭代
——这是远域启发式可接受的近似,但也是上面困难例中 EDIIS 表现不如
ADIIS 的原因之一;ADIIS 完全不用能量,无此近似。二者失速的另一面
(近域必须切回 CDIIS)由 `mixer_switch_tol` 负责,阈值默认 1e-2。

## 18.4 二阶方法:SOSCF / CIAH

前三节的所有技巧——初猜、阻尼、移位、DIIS 家族——都在同一个框架内打转:
Roothaan-Hall 不动点迭代。它们改善迭代的**路径**,却改不了迭代的**本性**:
不动点迭代至多线性收敛,病态体系里连线性都保不住。要质变,得换问题本身。

### 18.4.1 把 SCF 写成一个优化问题

SCF 的真身不是"解不动点",而是**在轨道空间里求能量极小**。参数化用轨道
旋转:给定当前系数 \(C\),任何保持正交归一(\(C^{\mathsf T} S C = I\))的
新系数都可写成

\[ C(\kappa) = C\,e^{\kappa},\qquad \kappa^{\mathsf T} = -\kappa , \]

其中反对称生成元 \(\kappa\) 只需占据-虚拟块 \(\kappa_{ai}\)(占据-占据与
虚拟-虚拟旋转不改变行列式)。能量对 \(\kappa\) 作二阶展开
(Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure Theory*,
Wiley 2000, ch. 10):

\[ E(\kappa) = E_0 + g^{\mathsf T}\kappa
   + \tfrac12\,\kappa^{\mathsf T} H \kappa + \cdots \]

闭壳层实轨道下梯度就是 Fock 矩阵的占据-虚拟块,\(g_{ai} = 4F_{ai}\)
(Hartree;引擎单元测试用旋转能量的有限差分逐分量钉住这个 4 的约定);
Hessian \(H = 4\,{}^1(A+B)\) 恰是第 15 章稳定性分析里那对 A/B 矩阵的单重态
组合——**能量极小化的 Hessian 与内稳定性判据是同一个算符**,这不是巧合:
"是否存在下降方向"与"Newton 往哪走"本来就是一个问题的两问。UHF 则是
α/β 两道拼接的旋转向量,\(g^{\sigma}_{ai} = 2F^{\sigma}_{ai}\),同自旋块含
库仑-交换响应、异自旋块只有库仑项。

Newton-Raphson 步 \(\kappa = -H^{-1}g\) 在极小点附近**二次收敛**:误差
平方地缩小,三四步内从 1e-3 干到 1e-12。DIIS 用历史差分**近似**这条曲率
信息,Newton 直接**拥有**它——这就是质变的来源。

### 18.4.2 CIAH:增广 Hessian + 矩阵自由 σ-vector

直接 Newton 有两个工程障碍。其一,\(H\) 是 \(n_{occ}n_{virt} \times
n_{occ}n_{virt}\) 的庞然大物,显式构造与求逆都不可接受;其二,远离极小点
时 \(H\) 未必正定,裸 Newton 步会奔向鞍点。CIAH(co-iterative augmented
Hessian;Sun, [arXiv:1610.08423](https://arxiv.org/abs/1610.08423),PySCF
`mf.newton()` 的内核)一石二鸟——解**增广 Hessian** 本征问题

\[ \begin{pmatrix} 0 & g^{\mathsf T} \\ g & H \end{pmatrix}
   \begin{pmatrix} 1 \\ \kappa \end{pmatrix}
   = \lambda \begin{pmatrix} 1 \\ \kappa \end{pmatrix} , \]

取最低根。展开即 \((H-\lambda)\kappa = -g\):最低根保证 \(\lambda \le 0\),
于是 \(H-\lambda\) 在极小点附近正定,\(\lambda\) 自动扮演能级移位的角色;
收敛时 \(g \to 0\)、\(\lambda \to 0\),回到精确 Newton 步。本征问题用
Davidson 迭代解,而 Davidson 只需要**矩阵-向量积** \(H\cdot v\)——这正是
第 15 章稳定性分析交付的 σ-vector 内核(O(N⁵) 的 MO 积分变换一次,之后每个
σ 积只花 O(n_{ov}^2)),\(H\) 从头到尾不显式存在。每宏迭代的 Davidson
展开数由成员 `soscf_max_micro`(默认 12)封顶:预算内拿不到收敛根就用
当前最优 Ritz 向量走步——这正是 CIAH"共迭代"的本意,近似 Newton 步
照样超线性。

两道保险负责全局收敛。**信赖域**:\(\lVert\kappa\rVert_2\) 超过
`soscf_trust`(默认 0.4,无量纲)就按比例缩回;**升能回退**:一步之后
能量不降反升,就恢复上一个被接受的迭代点、把信赖域减半重走(引擎在
diagonalize 阶段做旋转,回退只需把系数换回快照,下一轮 Fock/密度自动
重建)。最后,\(C \leftarrow C\,e^{\kappa}\) 的指数用缩放-平方 Taylor 求值:
反对称生成元的精确指数是正交阵,\(C^{\mathsf T} S C = I\) 保持到 1e-12
(单元测试逐位钉住)。

### 18.4.3 何时划算:DIIS 早期,SOSCF 收尾

Newton 步不是免费的:每宏迭代一次 O(N⁵) 的占据-虚拟 MO 变换,外加
Davidson 的若干 σ 积;而一次 CDIIS 外推只花一次矩阵求逆。健康体系里
CDIIS 五六步就进 1e-9,SOSCF 纯属浪费。二阶方法的用武之地是 18.1 那种
病理:CDIIS 在两个构型之间永久振荡,而 Newton 只要落进正确的收敛盆,
二次收敛就是铁律。

所以引擎的配方与 18.3 的远域-近域切换一脉相承——**升级链**,而不是
替换:

```cpp
cpu::scf::UHF uhf(5, 5);
uhf.basis = BasisSet::Def2SVP;
uhf.soscf = true;          // DIIS 早期,CIAH 收尾
uhf.soscf_start = 0.25;    // max|DIIS 误差| 低于此阈值即切换(∞-范数)
uhf.soscf_max_micro = 12;  // 每宏迭代的 Davidson 展开预算
uhf.soscf_trust = 0.4;     // ‖κ‖₂ 信赖域
uhf.run(frame);
uhf.soscf_iterations();    // Newton 相的宏迭代数(诊断)
```

`soscf_start` 的角色和 `mixer_switch_tol` 对称:DIIS 把迭代拖到误差
∞-范数 0.25 以内(收敛盆边缘),Newton 接管后二次收敛锁定。切换从第 1
迭代起才有资格发生——第 0 迭代必须先做一次 Roothaan 对角化,把密度型
初猜(MINAO / 外部密度)与系数同步,否则 Newton 旋转的是与密度无关的
轨道(实测发散约 7 Hartree,引擎以 iter ≥ 1 门槛根除)。收敛后引擎自动
用收敛密度的未移位 Fock 补一次终末对角化,行列式携带**正则**轨道能——
SOSCF 迭代中间的轨道能只是半正则的(occ-occ/virt-virt 块对角化的副产品),
不能拿去做 Koopmans。

18.3 的困难例——2.6× 拉伸 H₂O/def2-SVP UHF,Hcore 初猜——现在有了
第三条曲线。CDIIS 在不稳定对称解附近振荡满 60 步预算;`soscf = true`
在第 4 迭代切换,9 个 Newton 宏迭代后锁定碎对称极小(总 13 步),
终能量与 PySCF `mf.newton()` 钉值 −75.71896632262 Ha 差 < 1e-11:

```molplot preset="molplot" theme="auto" width="100%" aspect="16:9"
mark: line
data:
  values:
    - {iter: 2, series: CDIIS, E: -62.204}
    - {iter: 4, series: CDIIS, E: -80.771}
    - {iter: 6, series: CDIIS, E: -73.833}
    - {iter: 8, series: CDIIS, E: -76.205}
    - {iter: 10, series: CDIIS, E: -75.478}
    - {iter: 14, series: CDIIS, E: -75.357}
    - {iter: 18, series: CDIIS, E: -75.358}
    - {iter: 22, series: CDIIS, E: -75.358}
    - {iter: 26, series: CDIIS, E: -75.357}
    - {iter: 30, series: CDIIS, E: -75.358}
    - {iter: 2, series: DIIS→SOSCF, E: -62.204}
    - {iter: 4, series: DIIS→SOSCF, E: -80.771}
    - {iter: 5, series: DIIS→SOSCF, E: -74.848}
    - {iter: 6, series: DIIS→SOSCF, E: -74.939}
    - {iter: 7, series: DIIS→SOSCF, E: -75.217}
    - {iter: 8, series: DIIS→SOSCF, E: -75.466}
    - {iter: 9, series: DIIS→SOSCF, E: -75.655}
    - {iter: 10, series: DIIS→SOSCF, E: -75.7175}
    - {iter: 11, series: DIIS→SOSCF, E: -75.71897}
    - {iter: 13, series: DIIS→SOSCF, E: -75.71897}
encoding:
  x: {field: iter, type: quantitative, title: "SCF iteration"}
  y: {field: E, type: quantitative, title: "E (Hartree)", scale: {zero: false}}
  color: {field: series, type: nominal, title: null}
```

第 9–11 迭代之间能量从 −75.655 一步到 −75.7175、再一步到小数点后五位
不动——这三步就是教科书上的二次收敛:误差每步平方。对照组里 PySCF
`mf.newton()` 从等价升级点(混合 Hcore 初猜 + 6 步 DIIS 预热)到同一
碎对称极小要 8 个宏迭代,与引擎的 9 个同量级;而从更冷的种子出发,
PySCF 的纯 Newton 会停在**别的**驻点上(−75.44,⟨S²⟩≈1.0)——这正是
"DIIS 早期、SOSCF 收尾"胜过"从头 Newton"的实验证据:远域让凸组合
混合器选盆,近域让曲率收割。

拉伸 N₂(R = 2.5 Å,6-31G,RHF)给出另一面:那里 CDIIS 并不失败
(13 步收敛),但 SOSCF 总共 6 步、其中 Newton 相 4 步——与 PySCF
`mf.newton()` 的 4 个宏迭代逐一对应,终能量差 2e-11 Ha,HOMO 差
< 1e-8 Ha(拉伸 N₂ 的 RHF 有多个自洽解,HOMO 是盆的指纹)。注意本环
的 SOSCF 只覆盖 HF(RHF/UHF):Kohn-Sham 的轨道 Hessian 还需要 XC
二阶核,`RKS/UKS` 上开 `soscf` 会直接抛异常,而不是悄悄 Newton 一个
错误的曲面。

## 章末:与 PySCF 对照(累积表)

| 体系/设置 | 量 | Atomiverse | PySCF(钉定) | 容差 | 环 |
|---|---|---|---|---|---|
| H2O/def2-SVP RHF, Hcore | 迭代数 | 16 | 10 | 次序断言 | 07 |
| H2O/def2-SVP RHF, MINAO | 迭代数 | 16 | 8 | 次序断言 | 07 |
| H2O/def2-SVP RHF, Hückel | 迭代数 | 14 | 8 | 次序断言 | 07 |
| H2O/def2-SVP RHF, SAP | 迭代数 | 14 | 8 | 次序断言 | 07 |
| H2O/def2-SVP RHF | E (Ha) | −75.96098399 | −75.96098398708 | 1e-7 | 07 |
| 2.6×H2O/def2-SVP UHF, ADIIS→CDIIS | E (Ha) | −75.71896632 | −75.71896632261 | 1e-6 | 07 |
| 2.6×H2O/def2-SVP UHF, 纯 CDIIS | 60 步内收敛? | 否(失速) | —(对称解不稳定) | — | 07 |
| N₂ R=2.5 Å/6-31G RHF, SOSCF | E (Ha) | −108.21122307941 | −108.21122307939 | 1e-8 | 09 |
| N₂ R=2.5 Å/6-31G RHF, SOSCF | Newton 宏迭代数 | 4 | 4(`mf.newton()`) | ≤ 2× | 09 |
| 2.6×H2O/def2-SVP UHF, SOSCF | E (Ha) | −75.71896632262 | −75.71896632262 | 1e-8 | 09 |
| 2.6×H2O/def2-SVP UHF, SOSCF | ⟨S²⟩ | 1.9710 | 1.9710 | 1e-3 | 09 |
| 2.6×H2O/def2-SVP UHF, SOSCF | Newton 宏迭代数 | 9 | 8(升级等价种子) | ≤ 2× | 09 |
| 2.6×H2O/def2-SVP UHF, 纯 CDIIS | 60 步内收敛? | 否(失速) | —(SOSCF 13 步收敛) | — | 09 |

对照方法学见[附录:PySCF 对照方法学](appendix-pyscf-oracle.md)。
