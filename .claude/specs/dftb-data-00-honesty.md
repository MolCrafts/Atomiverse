---
title: dftb-data-00-honesty — remove fabricated DFTB constants, require published parameters as explicit input
status: code-complete
created: 2026-09-04
revised: 2026-09-04
grilled: true
---

# dftb-data-00-honesty — remove fabricated DFTB constants, require published parameters as explicit input

## Summary

DFTB 路径上有两组被凭空编造的科学常数：`DftbElementRow::hubbard_du`（DFTB3 的 Hubbard 导数 U^d）与 `DftbElementRow::spin_w`（共线自旋常数 W），它们由 `tools/gen_dftb_mio_tables.py` 的 `SKETCH_DU` / `SKETCH_SPIN_W` 两个字典注入到一个其余部分都如实转录自 mio-1-1 的头文件里，因此在使用者眼中与真实转录数据无法区分。本 spec 按 DFTB+ 的做法处理：`.skf` 只携带 onsite / Hubbard U / 占据数 / SK 网格 / 排斥项；U^d 与 W 一律是**显式用户输入**，缺失即拒绝运行，永不默认。删除两组捏造值，在 `DFTB3` 与 `DFTB2` 上各增加一个公开成员承接用户输入（原子单位 Hartree，与 DFTB+ 输入一致以保证复制粘贴安全），把已发表数值作为可引用的静态表随代码提供但**绝不隐式套用**；同时把两处**静默近似**提升成 API 表面上被声明的近似——`DFTB3::sk_set`（程序不再替用户挑参数集：未声明即抛，用户自己声明的 mio+3ob 跨集配对只 warning）与 `DFTB2::shell_resolved_spin`（分壳 W 表不再静默塌缩成 HOAO 标量）；补上 `dftb_3ob_tables.h` 缺失的生成器使其可复现、把该表从 `float` 改为 `double` 并按全精度重新发射，最后按铁律 9 修正九个 `REFERENCE.md` 中自相矛盾的 provenance 标签并重取受数值变更影响的 pin。交付后，DFTB 路径上不再存在任何"看起来像转录、实际上是编的"数字。

## Domain basis

**Hubbard 导数 U^d（DFTB3）。** 定义 U^d_A = dU_A/dq_A，经由 Γ_AB = ∂γ_AB/∂q_A 进入三阶项：

\[
E^{(2)}=\tfrac12\sum_{AB}\gamma_{AB}\Delta q_A\Delta q_B,\qquad
E^{(3)}=\tfrac13\sum_{AB}\Gamma_{AB}\,\Delta q_A^2\,\Delta q_B,
\]
\[
V^{(3)}_C=\tfrac23\Delta q_C\sum_B\Gamma_{CB}\Delta q_B+\tfrac13\sum_B\Delta q_B^2\,\Gamma_{BC},
\qquad \gamma_{AA}=U_A\Rightarrow\Gamma_{AA}=U^{\mathrm d}_A,\quad \Gamma_{AB}\neq\Gamma_{BA}.
\]

ATV 现有实现（`src/cpu/semiempirical/dftb3_scc.cpp:104-113, :129-144`）已逐项核对与 Christensen, Elstner, Cui, arXiv:1507.00370 Eq. (1)/(3) 一致：**DFTB3 机制本身正确，唯有 U^d 是编的**。

已发表 3ob-3-1 数值（原子单位，Hartree/电子，每元素一个标量；仅在 `ShellResolvedSCC = Yes` 下才分壳，3ob 不用该模式）：

| 元素 | U^d [Ha] | 捏造值 [eV] | 已发表值换算 [eV] |
| --- | --- | --- | --- |
| H | −0.1857 | −1.36 | −5.05315 |
| C | −0.1492 | −2.50 | −4.05994 |
| N | −0.1535 | −3.20 | −4.17695 |
| O | −0.1575 | −3.81 | −4.28579 |

与之强制配套的还有 ζ = 4.00（γ^h 阻尼指数，DFTB+ `DampXHExponent`）。

**对"数量级错误"说法的明确更正（必须写进本 spec，不得含糊）：** 捏造值并非差几个数量级——符号相同、量级相同（10⁰ eV）。真正的定性缺陷是**次序**：已发表 |U^d| 满足 H (0.1857) > O (0.1575) > N (0.1535) > C (0.1492)，氢**最大**；捏造值随 Z 单调、氢**最小**，即对 DFTB3 当初专为修正的那类原子（质子亲和能、氢键）编码了相反的化学趋势。

来源：3ob-3-1 参数发行版 README，(c) 2017 M. Elstner (KIT)，CC-BY-SA-4.0，https://github.com/dftbparams/3ob，"List of all atomic Hubbard derivatives (atomic units)"；单位由 DFTB+ 手册 `doc/dftb+/manual/dftbp.tex` DFTB3 节确认（"HubbardDerivs … must be specified (in atomic units)"）。原始方法/参数文献：Gaus, Goez, Elstner, *J. Chem. Theory Comput.* **9**, 338–354 (2013), DOI 10.1021/ct300849w（3OB 参数集）；DFTB3 方法本身为 Gaus, Cui, Elstner, *JCTC* **7**, 931–948 (2011), DOI 10.1021/ct100684s。**两者是不同论文，作者名单不得合并。**
**未能验证项（必须原样记入头文件 provenance）：** ct300849w 为付费墙文献，未能打开；上表数值取自官方参数发行版（CC-BY-SA，Elstner/KIT），即 DFTB+ 用户实际喂给 `HubbardDerivs` 的那份产物，**不是**从论文表格读出的。头文件中不得声称某个论文表号；若日后有人有期刊访问权，需另行确认。

**自旋常数 W。** 经 Janak 定理定义为每元素的**分壳矩阵** W^A_{ll'}，不是原子标量：

\[
W^A_{ll'}=\tfrac12\Bigl(\frac{\partial\varepsilon_{l\uparrow}}{\partial n_{l'\uparrow}}-\frac{\partial\varepsilon_{l\uparrow}}{\partial n_{l'\downarrow}}\Bigr),\qquad
E_{\mathrm{spin}}=\tfrac12\sum_A\sum_{l\in A}\sum_{l'\in A}p_{Al}\,W^A_{ll'}\,p_{Al'},\qquad
p_{Al}=n_{Al\uparrow}-n_{Al\downarrow}.
\]

DFTB+ 输入次序为第二个壳指标变化最快（w_ss, w_sp, w_ps, w_pp）；sp 元素 4 个数，H 1 个数；单位 Hartree。已发表 PBE 列（mio/3ob 的 SK 由 PBE 参数化，故取 PBE 列；DFTB+ 手册仅要求"使用与 SK 文件相同的泛函计算的常数"，此即本 spec 的选择依据，LDA 列一并提供但不作默认）：

| 元素 | W_ss | W_sp | W_ps | W_pp |
| --- | --- | --- | --- | --- |
| H | −0.072 | — | — | — |
| C | −0.031 | −0.025 | −0.025 | −0.023 |
| N | −0.033 | −0.027 | −0.027 | −0.026 |
| O | −0.035 | −0.030 | −0.030 | −0.028 |

（LDA 列：H −0.064；C −0.028/−0.024/−0.024/−0.022；N −0.030/−0.026/−0.026/−0.025；O −0.032/−0.028/−0.028/−0.027。原文注明"只有前两位小数在数值上有意义"。）
来源：DFTB+ 手册附录 `app:spinconst`（`doc/dftb+/manual/spin_constants.tex`），引 C. Köhler 博士论文, Univ. Paderborn (2004)；同行评审表述见 Köhler, Frauenheim, Hourahine, Seifert, Sternberg, *J. Phys. Chem. A* **111**, 5622–5629 (2007), DOI 10.1021/jp068802p。**未能验证项：** jp068802p 未能打开；上述方程取自 Moradi et al., arXiv:2405.05761 eqs. (2)–(3) 与 DFTB+ 实现，须原样记入 provenance。

**ATV 现有自旋实现是一个合法的、有文献依据的近似，不是错误形状（对简报的明确更正）。** ATV（`dftb2.cpp:490-509, :1105, :1193`）计算
\(\delta V^{\mathrm s}_A=W_A m_A\)、\(\varepsilon^\sigma_A=\sum_B\gamma_{AB}\Delta q_B+\sigma W_A m_A\)、\(E_{\mathrm{spin}}=\tfrac12\sum_A W_A m_A^2\)。令 W^A_{ll'}≡W_A 对所有 l,l' 成立，则
\(E_{\mathrm{spin}}=\tfrac12\sum_A W_A(\sum_l p_{Al})^2=\tfrac12\sum_A W_A m_A^2\)，且 \(\partial E_{\mathrm{spin}}/\partial m_A=W_Am_A\) 恰为 ATV 的移位——**自旋块内部自洽**。这正是 DFTB+ 自己的 `ShellResolvedSpin = No` 模式。真正的缺陷是三条：近似未记录、数值是编的、标量取值约定非标准（文献约定是取 HOAO 对角 W_{ll}，DFTB+ 自己的输入范例写 `N = {-0.026} # HOMO Wpp`）。

塌缩标量（HOAO 约定）已发表值与捏造值对比：H W_ss −0.072 Ha = −1.95922 eV vs 捏造 −0.50（3.92×）；C W_pp −0.023 Ha = −0.62586 vs −0.80（0.78×）；N W_pp −0.026 Ha = −0.70750 vs −0.90（0.79×）；O W_pp −0.028 Ha = −0.76192 vs −1.00（0.76×）。

**具体 pin 位移（O₂ 三重态，`regressions/semiempirical/spin-dftb2-o2`）：** m_A = 1.0 e/原子（由对称性与电子数固定），故 E_spin = ½ Σ_A W m_A² = W。捏造 W = −1.00000 eV → E_spin = −1.00000 eV；已发表 W_pp(O,PBE) = −0.028 Ha = −0.761918814888 eV → E_spin = −0.761918814888 eV。总能 pin −142.1382039484 eV 位移 **+0.238081185112 eV** → −141.900122763288 eV。O₂ 处标量塌缩是**精确的**（三重态自旋密度为 π*，纯 p，p_s ≈ 0）；塌缩仅在原子同时带 s 与 p 自旋密度时才损失精度（σ 自由基、H 原子），C 的 W_ss/W_pp = 1.35 即该误差尺度。同一塌缩也适用于 `td_casida` 三重态核 \(K^{\mathrm T}_{ia,jb}=\sum_A q^{ia}_A W_A q^{jb}_A\)（`dftb2.cpp:3591`）——同一近似、同一取值问题、同一处修复。

**总能位移为何精确等于 ΔE_spin（逐行核对源码后给出的强论证，取代先前"对称性固定 m_A"的说法）。** 旧理由只说明对称性把 m_A 钉在 1.0、因而 E_spin = W——那**只证明了自旋项本身**，并未证明自洽密度不变，因此不足以预测**总能**位移。正确的论证是四步，每步都可在源码上指认：

1. **总能装配不含能带和。** `src/cpu/semiempirical/dftb2.cpp:1205-1207` 写的是
   `out.energy = trace_product(Pa, H0) + trace_product(Pb, H0) + e_scc + e_spin + e_rep`，
   即 \(E=\mathrm{Tr}(\rho H^0)+E^{(2)}+E_{\mathrm{spin}}+E_{\mathrm{rep}}\)。用的是 **Tr(ρH⁰) 而不是 Σ_o ε_o**，所以自旋移位根本不进入第一项。
2. **移位在两个原子上完全相同。** O₂ 同核 ⇒ Δq_A ≡ 0 ⇒ 库仑移位 dVc = 0（`dftb2.cpp:1098-1107`）；自旋移位 \(\varepsilon^\sigma_A=\sigma W_A m_A\) 于是是同一个常数 \(c=\sigma W m\)。
3. **常数移位等价于 H⁰ + cS。** `apply_shift`（`dftb2.cpp:490-509`）对每个块取 `half = ½(ε_a + ε_b)`；两原子相等时 half ≡ c，故 \(H^\alpha=H^0+cS\)、\(H^\beta=H^0-cS\)。
4. **广义本征问题同解。** \((H^0+cS)C=\varepsilon SC \iff H^0C=(\varepsilon-c)SC\)：**本征向量逐一相同，本征值整体平移 c**。因此 ρ^α、ρ^β 不变 ⇒ m_A 仍自洽地等于 1.0、Tr(ρH⁰) 不变、E^{(2)} 不变（Δq ≡ 0 本就为零）、E_rep 不变。

**结论：总能位移精确等于 ΔE_spin = +0.238081185112 eV。** 这是一个等式，不是一个估计——1e-4 eV 的容差只是数值噪声余量，不是论证的软度。**若实测位移偏离超过该容差，说明上述四步中有一步在实现里不成立，必须按铁律 9 停下并报告 `path:line` + 实测值，不得放宽容差蒙混。**

**3ob 表 float 假说已被证伪（实测，非论证）。** 用本机上的真实 3ob-3-1 `.skf`（与 ATV 元素行逐字节一致）复现 ATV 自己的 8 点 Neville 窗口（`dftb2.cpp:2322-2447`）测得：(a) 全通道全原子对上 float32 往返误差 ≤ 1.09e-07 Ha = 2.95e-06 eV；(b) CH₄ 几何下经 7 次 Neville 窗传播，C–H ssσ 误差 9.4e-09 Ha、H–C spσ 7.3e-09 Ha、重叠 ≤ 6.4e-09；(c) 收敛密度下一阶能量响应 \(\delta E=\sum\rho\,\delta H^0-\sum W\,\delta S\) 的**完全相干最坏情形** ≤ 2.03e-06 Ha = 5.53e-05 eV。实测残差 2.6e-02 eV = 9.4e-04 Ha，是最悲观 float 上界的 ~470 倍。窗口跨越处的不连续为 −4.6e-10 Ha 且 float/double 相同（是 8 点窗方案的性质，与存储无关）；排斥项独立复现到 5.3e-08 Ha。**结论：本 spec 中的 float→double 是可复现性修复，不是精度修复；不得声称 float 截断解释了 2.6e-2 eV 残差。预期 `dftb2-3ob-organic` 数值移动 < 1e-4 eV。** 残差实际位于电子学装配（H⁰/S 构造或 SCC 核），列为已发现的阻塞性欠债（见 Out of scope）。

## Design

### 用户输入面（铁律 3：公开成员，非 Config 包、非构造参数）

- **`DFTB3::hubbard_du_ha`** —— `std::array<double, 96>`，brace-init 全 0，声明在 `include/atomiverse/cpu/semiempirical/dftb3.h` 的 `with_forces`（:70）之后。`DFTB3` 派生自 `cpu::Semiempirical` 而非 `DFTB2`，无共享基类，故必须独立声明。索引 = 原子序数 Z。
- **`DFTB2::spin_w_ha`** —— `std::vector<basis_data::DftbSpinConstant>`，默认空，声明在 `include/atomiverse/cpu/semiempirical/dftb2.h` 的 `u_eff`（:261）之后，复用 `u_eff` 的文档块形状（`@brief` + 单位 + "空/零 = 未设置" + LaTeX 公式 + 哪些路径会抛 + 指向 `.claude/notes/` 的链接）。
- **`DFTB2::shell_resolved_spin`** —— `bool`，默认 `false`，**紧挨 `spin_w_ha` 声明**，名字直取 DFTB+ 的同名关键字 `ShellResolvedSpin`。`true` → `std::runtime_error`，消息点名后续 slug `dftb-shell-resolved-spin`（完整分壳收缩未实现）；`false` → 照常运行并施加 HOAO 标量塌缩，`@brief` 里写明"本路径把分壳 W 表塌缩为 HOAO 对角标量，等价于 DFTB+ `ShellResolvedSpin = No`"。
  **理由（必须写进 spec，不能只写进代码注释）：** 没有这个 bool，一个提供了**分壳** W 表的用户会**静默**拿到 HOAO 标量塌缩的结果——那是一个**静默回退**，恰恰是本 spec 存在的理由所在的那一类缺陷。一个 bool 把"只写在散文里的未声明近似"变成"API 表面上被声明、被拒绝的近似"，代价是一个默认值。
- **输入面是分壳的**（W^A_{ll'}），与已发表表格的真实形状和 DFTB+ `SpinConstants` 块一致：

```cpp
/// @brief One element's shell-resolved collinear spin constants W^A_{ll'} [Ha].
struct DftbSpinConstant {
    int z;             ///< atomic number Z (dimensionless)
    int n_shell;       ///< 1 (s only) or 2 (s+p)
    double w_ha[2][2]; ///< W_{ll'} [Ha]; l,l' index {s,p}; l' varies fastest (DFTB+ order)
};
```

- **本 spec 的消费者把它塌缩为标量**，按 W_HOAO 约定（C/N/O → `w_ha[1][1]`（W_pp），H → `w_ha[0][0]`（W_ss）），既有自旋数学**一字不改**。完整分壳收缩 \(\text{shift}_l=\sum_{l'}W_{ll'}p_{l'}\) 是被点名的后续工作（`dftb-shell-resolved-spin`），不在本 spec。

### 单位边界（必须刻意设计，否则静默 27×）

DFTB+ 的输入是原子单位，ATV 公开面是 eV。若用户把 `HubbardDerivs { O = -0.1575 }` 抄进一个 eV 字段，将静默偏差 27 倍。**决定：在这一个边界上取原子单位（Hartree），与 DFTB+ 一致，使复制粘贴安全**，并且——

1. 单位写进**成员名本身**（`hubbard_du_ha` / `spin_w_ha` / `w_ha`），前例为 `GpwDft::plus_u` 的 `u_minus_j_ev`；调用点自带单位。
2. `@brief` 第一句写明 `[Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV)`。
3. 只在边界转换一次，使用 `units::kHartreeInEv`（`core/units.h:51`，`dftb2.cpp:3367` 已在用），转换后内部一律 eV。

### 校验：惰性、在 `launch()` 顶部，抛出带 how-to

严格照 `u_eff` 的形状（`dftb2.cpp:3097-3116`、`:3236-3245`）：**不在构造函数**、**不在 `build()`**（`DFTB2::build` 在 `dftb2.cpp:3088` 是空的），而在 `launch()` 顶部按实际取用的代码路径惰性校验。这正是让"未设置"成为合法默认态的机制。

- `DFTB3::launch()`：**先校验 `sk_set`（见下一节），再校验 U^d**——`sk_set` 决定了 U^d 该属于哪个参数集，所以它先说话；两者同时缺失时用户看到的是 `sk_set` 的 how-to。帧中出现的每个 Z 若 `hubbard_du_ha[z] == 0.0` 或非有限 → `std::invalid_argument`。`internal::Dftb3Scc` 的构造函数（`dftb3_scc.h:40-49`，已对 size 不匹配抛出）追加"向量必须被填充且全非零"的要求，作为第二道闸。
- `DFTB2::launch()`：仅当自旋通道真正被取用时才要求 `spin_w_ha`（`multiplicity != 1` 或开壳层电子数）——否则所有现存闭壳层 DFTB2 回归会无谓地断掉。`shell_resolved_spin == true` 的拒绝与之同址、同惰性（闭壳层运行不受影响）。`DFTB2::td_casida()` 是独立的公开方法，其三重态通道必须**同样**校验这两项（它不经 `launch()`）。
- 共享 how-to 采用 `src/cpu/multiscale/internal/qm_subsystem.cpp:24-31` 的 `kMaskHowto` 形状：文件内 `constexpr const char* kDftbSpinHowto` / `kDftbHubbardDerivHowto` / `kDftbSkSetHowto` 多行字面量，各抛出点前缀自己的一行诊断。消息形状照 `src/cpu/planewave/internal/dfpt.cpp:1800-1802`：`"<Type>::<method>: <what is wrong> — <what to do>"`，how-to 里给出可直接粘贴的一行（`d.spin_w_ha = basis_data::kDftbSpinConstantsPbeHa;`）。

### 已发表数值：可引用的静态头，但绝不隐式套用

两份新头文件（不同来源、不同许可 → 各自独立 provenance 前言，照 `include/atomiverse/cpu/potentials/eam_data/eam_tables.h:1-18` 的形状，完整度照 `src/cpu/communication/comm_brick.cpp:1-19` 的署名块，因为 3ob 发行版是 CC-BY-SA-4.0 需署名）：

- `basis_data/dftb_3ob_hubbard_derivs.h` → `inline constexpr std::array<double, 96> k3obHubbardDerivsHa`（仅 H/C/N/O 非零）。
- `basis_data/dftb_spin_constants.h` → `struct DftbSpinConstant` + `inline const std::vector<DftbSpinConstant> kDftbSpinConstantsPbeHa` 与 `…LdaHa`。选 `std::vector` 使 `d.spin_w_ha = basis_data::kDftbSpinConstantsPbeHa;` 恰好是一行。

**没有任何东西被静默套用** —— 用户必须自己写这一行。这就是"不发明默认值"在保持常用路径为一行的同时仍然成立的原因。两份头文件的 provenance 前言必须包含：来源产物（不是论文表号）、DOI、许可与署名、单位、ζ = 4.00 尚未实现的显式警告、以及上文的"未能验证项"两条。

### DFTB3 与 mio SK 的参数集混配：用户选的只 warning，程序选的必须报错

`src/cpu/semiempirical/dftb3.cpp:22-23,38,318-319` 硬绑 mio-1-1 SK 表；`DftbSkSet`（`dftb2.h:24`）是 **DFTB2 的**枚举，`DFTB3` 上**没有** `sk_set` 旋钮（`dftb3.h:21-22` 明文把 DFTB3 记载为恒用 mio 表）。把已发表的 3ob U^d 放进该路径，等于把 3ob Hubbard 导数与 mio SK/排斥项配对——3ob-3-1 README 明言 3ob "supersedes the MIO parameter set"。

**用户指令（取代本节先前"不因此拒绝运行 + 只记档"的决定）：** 如果 mio SK 与 3ob U^d 这个配对是**用户在自己的运行脚本里设置的**，就不报错、只给 warning；如果是**程序自己选的**，必须报错。

两个事实决定它如何落地：

- (a) **警告通道已存在**：`ATV_WARN` / `ATV_WARN_CAT`（`include/atomiverse/core/log.h:104`、`:124`，spdlog 后端）。不发明新机制。仓库既有用法是裸 `ATV_WARN("[tag] …")`，DFTB 没有注册子系统 logger，故用 `ATV_WARN("[dftb3] …")`。
- (b) **今天 DFTB3 没有任何 SK 选择器**，因此每一次 DFTB3 运行的 SK 选择都是**程序做的**，根本不存在"用户选过"的分支可供检查。

还有一条是承重的：**"用户给的是不是 3ob 的 U^d"在数值上不可判定**——成员是一个普通的 `double` 数组，程序无法把 `k3obHubbardDerivsHa` 与手敲的数字区分开。判别器**必须是结构性的**（用户有没有声明 SK 集），不能是值比对。

因此用户的规则把 `DFTB3::sk_set` **部分地**拉进本 spec。满足它的最小形状：

- **`DftbSkSet` 增加 `Unset = -1` 哨兵，留在 `dftb2.h`。** 决定并说明：DFTB2 与 DFTB3 是**同一个方法域**、同一份 SK 发行物，为一个枚举新开一个共享头属于"给段落起名字"的抽取，CLAUDE.md 的"第二个真实调用点前不抽取"不支持它。现有 `MioSketch = 0` / `ThreeOb = 1` 的值不动；`DFTB2::sk_set = DftbSkSet::MioSketch`（`dftb2.h:181`）默认**不变**、DFTB2 行为**不变**，只补一条对 `Unset` 的显式抛出（枚举现在可被用户赋成 `Unset`，DFTB2 的 switch 必须穷尽，不得走进默认分支）。
- **`DFTB3::sk_set`** —— 公开成员，类型 `DftbSkSet`，默认 `DftbSkSet::Unset`，与本 spec 对 U^d / W 采用的"空/零 = 未设置，未设置即抛"是同一个模式。
- **校验分支（`launch()` 顶部，惰性，与 U^d 校验同一个块，`sk_set` 在先）：**
  1. `sk_set == Unset` → `std::invalid_argument`，how-to 直说 `"DFTB3 will not pick an SK set for you — declare d3.sk_set = cpu::semiempirical::DftbSkSet::MioSketch;"`。**这就是"程序自己选的必须报错"**：选择权交还用户，选择因此变成脚本里可见的一行。
  2. `sk_set == ThreeOb` → `std::runtime_error`，指向后续 slug `dftb-gamma-h-damping`。3ob SK/排斥项接进 `dftb3.cpp` **不在本 spec**。
  3. `sk_set == MioSketch` → 通过 U^d 完整性校验后 `ATV_WARN` 一次，**随后照常运行**。警告文本点名 3ob-3-1 README 的 "supersedes the MIO parameter set" 约束、点名该配对越出参数集、并声明由此产生的 pin 不得标 `oracle: dftb+`。这是用户亲手声明的跨参数集配对。
- **每对象只响一次：** 私有 `bool warned_mixed_set_ = false;`，`ATV_WARN` 后置位。`launch()` 每步都跑，MD 循环里不能刷屏——但也**不能**降级成 `ATV_INFO` 或只在 `build()` 里发：`build()` 早于用户可能的赋值，而降级会把它变回"没人看见的近似"。
- **`oracle: dftb+` 禁令不变**：mio SK + 3ob U^d 仍然不是 DFTB+ 可复现的哈密顿量，**DFTB3 的 pin 一律不得标 `oracle: dftb+`**。
- **是否存在已发表的 mio-1-1 U^d 集：经查没有**（3ob 取代 mio 并自带一套），此结论仍写入头文件 provenance。
- **命名：`MioSketch` 在本 spec 不改。** sketch 常数删除后这个名字确实误导，但重命名是公开 API 变更、牵动 DFTB2 全部调用点，与本 spec 的诚实性目标正交；只更新 `dftb2.h:24-38` 的枚举文档块，并把重命名列入 Out of scope 以免被顺手改掉。

### 3ob 表的可复现性（生成器 + double）

`dftb_3ob_tables.h:4` 声称"Transcribed offline from DFTB.org 3ob-3-1 `.skf`"，但仓库中**不存在任何生成器**（已独立经 `git ls-files` 确认）——该声明无据。新增 `tools/gen_dftb_3ob_tables.py`，仅手工运行、**永不接入 CMake/ctest**，以 `tools/gen_basis_data.py` 为模板（唯一带 `--check` 的生成器：重新生成并 diff、字节确定、把完整重跑命令戳进发射的头文件，落地样例 `include/atomiverse/cpu/scf/basis_data/sto3g.h:1-13`），**而非**以 `tools/gen_dftb_mio_tables.py` 为模板——后者就地 patch `dftb_mio_tables.h`（`emit_elements_patch:285-364`），正是让捏造常数混进一份其余部分如实转录的头文件的那个机制。SKF 语法/通道下标/`nGrid = fileN − 1` 约定/排斥项布局仍照 `gen_dftb_mio_tables.py:164-281` 的结构。

`Dftb3obSkPoint`（`:31-34`）的 8 个字段 `float → double`，**并在同一个任务里按全精度重新发射**：现存值已按 9 位有效数字存储（`{-5.85619077e-01f, …}`），只改类型什么也挽回不了。消费者（`dftb2.cpp` 的 `pair_3ob:2314`、`sk_col_3ob:2365`、`grid_value_3ob:2403`、`interp_3ob_sk:2456`、`interp_3ob_sk_dr:2462`、`rep_3ob:2473`、`rep_3ob_dr:2499`）本就提升到 double，**无消费者数学改动**；`dftb2.cpp:2364` 的注释 "tables store float" 随之更新。

**两表布局分歧的自觉决定：** 3ob 为点主序结构体（`Dftb3obSkPoint`），mio 为通道主序（每通道一个 `const double*`）。**本 spec 明确选择"推迟"**：统一布局会重写 mio 与 3ob 双方的全部读取路径，属于纯结构变更，与本 spec 的诚实性目标无关，且 spec 1（SKF 运行时读取器）会一次性重定该布局。此处只把决定与理由记入 `dftb_3ob_tables.h` 的前言，避免它被"顺手"改掉。

### Reuse decision（逐条结清 librarian 报告）

- **`DFTB2::u_eff`（dftb2.h:261，PATTERN）→ pattern.** 新成员的声明位置、文档块形状、"零/空 = 未设置"、`launch()` 顶部惰性校验三段式全部照抄；不复用符号本身（U_eff 与 U^d、W 是不同物理量）。
- **`GpwDft::plus_u` / `struct PlusU`（gpw_solver.h:73-81，GENERALIZE）→ new — `DftbSpinConstant`.** 不构成重实现：`PlusU` 是**单**壳指标记录 `{element, l, value}`，自旋 W 需要壳**对** (l,l')，二者元数不同；且它位于 `cpu/gpw`，跨方法域共享用户记录类型会把 `cpu/gpw` 与 `cpu/semiempirical` 耦合起来，而 CLAUDE.md 明令共享代码只能落在对应的 pillar 目录、不得寄放在某个域内，"用户参数记录"没有这样的 pillar 目录；按"第二个真实调用点前不抽取"，此处仍是各自一个调用点。**继承其"空 = 关闭"的约定，显式拒绝其校验形状**——`pack_plus_u`（`src/cpu/gpw/gpw_solver.cpp:134-147`）在 `u_minus_j_ev <= 0 || l < 0` 时**静默 continue** 而不抛，违反"无回退值"；这是本 spec 发现但**不在本 spec 修复**的越界欠债（见 Out of scope 的 found-defect 清单）。因此**不产生 "Generalize …" 任务**。
- **`DFTB3` 参数面（dftb3.h:55-70，PATTERN）→ pattern.** 确认 `DFTB3 : cpu::Semiempirical`，与 `DFTB2` 无继承关系，两个成员各自独立声明，不引入共享基类（铁律 6）。
- **`tools/gen_basis_data.py`（PATTERN）→ pattern.** 新 3ob 生成器照其 `--check`、字节确定性、`/// Regenerate:` 戳记、不支持元素硬报错、"手工运行、绝不接入 CMake" 的 docstring。
- **`tools/gen_dftb_mio_tables.py:164-281 emit_sk_header`（PATTERN）→ pattern（仅结构）.** 沿用 SKF 语法解析与排斥项布局；**显式拒绝**其就地 patch 机制。
- **`Gfn1Element::lgam[2]` / `Gfn2ElementRow` / `IpeaElement`（PATTERN）→ pattern.** `DftbSpinConstant` 采用同样的"定长 C 数组 + `int n_shell` 伴随字段"形状。
- **`dftd3_params.h:51 r0ab`（PATTERN）→ pattern.** 元素×元素矩阵的唯一前例，为 `w_ha[2][2]` 的二维定长写法背书；该符号是 internal，不复用。
- **`eam_tables.h:1-18`（PATTERN）→ pattern.** 两份新数据头的 `/// Provenance:` 前言形状。其中"五个数字读自印刷表格，生成器将是没有第二个调用者的程序"这一段同样适用于 4 元素 × 2×2 自旋矩阵 —— 因此**本 spec 不为这两份新表写生成器**，只为 3ob SK 网格（数千个数、必须从 `.skf` 机器转录）写生成器。
- **`dftd3_tables.h:1-8`（PATTERN）→ pattern.** `/// PROVENANCE:` + 上游文件 + 唯一消费者的写法。
- **`comm_brick.cpp:1-19`（PATTERN）→ pattern（完整度基准）.** CC-BY-SA-4.0 署名块的完整度照此，但**不复用 LAMMPS 移植授权**：LAMMPS 移植授权只覆盖 GPL-2 的 LAMMPS，不延伸到 LGPL-3 的 DFTB+。
- **`qm_subsystem.cpp:24-31 kMaskHowto`（PATTERN）→ pattern.** 共享 how-to 字面量。
- **`dfpt.cpp:1800-1802` / `dftb2.cpp:143-144`（PATTERN）→ pattern.** 消息措辞形状。
- **仓库根 `data/` 目录（librarian 已考虑并否决的备选）→ 否决，本 spec 不创建 `data/`。** 树中不存在 `data/`；每一个科学参数集（GPW、DFT-D3、EAM、SW、mio SK、3ob SK、GFN0/1/2、IPEA、SAP、全部基组）都以 `basis_data/`（或 `*_data/`）下的已提交 `.h` 存在，CLAUDE.md 将其列为具约束力的先例。U^d 与 W **根本不在 `.skf` 里**，DFTB+ 自己也把它们当输入**文本**接收，按本链条自身的判据（"是否存在用户会合理替换的外部标准发行物？"）答案为否，因此它们与 AM1/RM1 的 NDDO 常数同类，留在编译期。`data/` 属于 spec 1（4 MB SK 表）。
- **新符号 `internal::collapse_spin_w_ev`（新）→ new.** 有**两个**真实调用点（`DFTB2::launch` 与 `DFTB2::td_casida`），满足"第二个调用点才抽取"，且铁律 1 要求单元测试必须能瞄准一个单元；此外 `element_row(int z)` 在三个匿名命名空间中重复了三次（`dftb0.cpp:29`、`dftb2.cpp:139`、`dftb3.cpp:45`），把用户值穿过静态行需要触及其中两处（`dftb0` 两个字段都不读），集中到一个可测函数里正是为了不把这份重复放大。
- **`ATV_WARN`（`core/log.h:104`）→ reuse.** 参数集混配的 warning 直接用既有 spdlog 通道，不发明第二套警告机制、不新增返回码、不新增 `warnings()` 访问器。类别宏 `ATV_WARN_CAT`（`:124`）不用：DFTB 没有注册子系统 logger。
- **`DftbSkSet`（dftb2.h:24）→ reuse（DFTB3 复用 DFTB2 的枚举，仅加 `Unset` 哨兵）.** 不为一个枚举新开共享头；不新建 `Dftb3SkSet` 平行枚举——那会立刻产生两个必须同步的参数集清单。

### 数据流改动（不改物理）

`spin_w` 的 8 个读取点全部形如 `el[a]->spin_w`（`dftb2.cpp:1105, 1193, 1319, 1595, 1898, 1994, 2062, 3591`，分布在 5 个外层函数）。改为：在 `launch()` / `td_casida()` 顶部由 `internal::collapse_spin_w_ev(spin_w_ha, z_per_atom)` 一次性构造 `std::vector<double> spin_w_ev(n)`（HOAO 塌缩 + 单次 Ha→eV），沿既有 `el` 向量的传参路径下发。`hubbard_du` 的 5 个读取点（`dftb3.cpp:196, 312, 318-319, 553, 561`）同样改为读取从 `hubbard_du_ha[z] * units::kHartreeInEv` 装配的每原子 eV 向量。`DftbElementRow` 随之删去 `hubbard_du`（:36）与 `spin_w`（:38）两个字段及四行元素值中的第 8、10 列。

## Files to create or modify

- `include/atomiverse/cpu/semiempirical/basis_data/dftb_3ob_hubbard_derivs.h` (new)
- `include/atomiverse/cpu/semiempirical/basis_data/dftb_spin_constants.h` (new)
- `include/atomiverse/cpu/semiempirical/internal/dftb_spin.h` (new)
- `src/cpu/semiempirical/dftb_spin.cpp` (new)
- `tests/cpu/semiempirical/test_dftb_published_constants.cpp` (new)
- `tools/gen_dftb_3ob_tables.py` (new)
- `tools/tb_oracle/inputs/dftbplus/spin-dftb2-o2/dftb_in.hsd` (new), `.../geo.gen` (new), `.../dftb.out.txt` (new)
- `tools/tb_oracle/inputs/dftbplus/dftb2-spin-analytic-forces/dftb_in.hsd` (new), `.../geo.gen` (new), `.../dftb.out.txt` (new)
- `tools/tb_oracle/inputs/dftbplus/dftb2-spin-analytic-stress/dftb_in.hsd` (new), `.../geo.gen` (new), `.../dftb.out.txt` (new)
- `include/atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h`
- `include/atomiverse/cpu/semiempirical/basis_data/dftb_3ob_tables.h`
- `include/atomiverse/cpu/semiempirical/dftb2.h`
- `include/atomiverse/cpu/semiempirical/dftb3.h`
- `include/atomiverse/cpu/semiempirical/internal/dftb3_scc.h`
- `src/cpu/semiempirical/dftb2.cpp`
- `src/cpu/semiempirical/dftb3.cpp`
- `src/cpu/semiempirical/dftb3_scc.cpp`
- `tools/gen_dftb_mio_tables.py`
- `tools/tb_oracle/README.md`
- `CMakeLists.txt`
- `tests/CMakeLists.txt`
- `CLAUDE.md`
- `regressions/semiempirical/spin-dftb2-o2/REFERENCE.md`, `.../reg_spin-dftb2-o2.cpp`
- `regressions/semiempirical/dftb2-spin-analytic-forces/REFERENCE.md`, `.../reg_dftb2-spin-analytic-forces.cpp`
- `regressions/semiempirical/dftb2-spin-analytic-stress/REFERENCE.md`, `.../reg_dftb2-spin-analytic-stress.cpp`
- `regressions/semiempirical/dftb2-multi-k-forces/REFERENCE.md`, `.../reg_dftb2-multi-k-forces.cpp`
- `regressions/semiempirical/td-dftb-h2/REFERENCE.md`, `.../reg_td-dftb-h2.cpp`
- `regressions/semiempirical/dftb3-water-mio/REFERENCE.md`, `.../reg_dftb3-water-mio.cpp`
- `regressions/semiempirical/dftb3-analytic-forces/REFERENCE.md`, `.../reg_dftb3-analytic-forces.cpp`
- `regressions/semiempirical/dftb3-periodic-forces/REFERENCE.md`, `.../reg_dftb3-periodic-forces.cpp`
- `regressions/semiempirical/dftb2-3ob-organic/REFERENCE.md`, `.../reg_dftb2-3ob-organic.cpp`
- `regressions/semiempirical/dftb2-analytic-forces/REFERENCE.md`
- `regressions/semiempirical/dftb2-analytic-stress/REFERENCE.md`
- `regressions/semiempirical/dftb2-analytic-forces-threeob/REFERENCE.md`
- `regressions/semiempirical/dftb2-analytic-stress-threeob/REFERENCE.md`

## Tasks

- [x] Write failing unit tests for the published-constant tables, the HOAO spin collapse, the unset-parameter refusal, and the two declared-approximation guards (`DFTB3::sk_set` unset/ThreeOb/MioSketch-warn, `DFTB2::shell_resolved_spin = true`) in `tests/cpu/semiempirical/test_dftb_published_constants.cpp`; register in `tests/CMakeLists.txt`; hard-coded expected values, one function/method per test
- [x] Add `basis_data/dftb_3ob_hubbard_derivs.h` and `basis_data/dftb_spin_constants.h` with `/// Provenance:` prologues (source artifact, DOI, CC-BY-SA-4.0 attribution, Hartree units, ζ=4.00 not-implemented warning, no-published-mio-U^d finding, unverified-source flags) and Doxygen `@brief` on every public symbol, and amend `CLAUDE.md` iron law 1 in the same change (in-tree scientific parameter data is first-party code carrying provenance; reading it from `tests/` and `regressions/` is not a third-party read; reading anything outside the repo at build or test time stays forbidden)
- [x] Implement `DFTB2::spin_w_ha` and `DFTB2::shell_resolved_spin` in `dftb2.h` plus `internal::collapse_spin_w_ev` (`internal/dftb_spin.h`, `src/cpu/semiempirical/dftb_spin.cpp`, `CMakeLists.txt`), thread the collapsed W through the 8 read sites and both lazy validation points in `src/cpu/semiempirical/dftb2.cpp`, and delete `DftbElementRow::spin_w` from `basis_data/dftb_mio_tables.h` plus `SKETCH_SPIN_W` from `tools/gen_dftb_mio_tables.py`
- [x] Implement `DFTB3::hubbard_du_ha` in `dftb3.h`, thread it through the 5 read sites and `launch()` validation in `src/cpu/semiempirical/dftb3.cpp`, tighten the `internal::Dftb3Scc` ctor guard (`internal/dftb3_scc.h`, `src/cpu/semiempirical/dftb3_scc.cpp`), and delete `DftbElementRow::hubbard_du` plus `SKETCH_DU` from `tools/gen_dftb_mio_tables.py` and the stale sketch prose in `basis_data/dftb_mio_tables.h`
- [x] Implement `DFTB3::sk_set` (`DftbSkSet::Unset = -1` sentinel and refreshed enum docs in `include/atomiverse/cpu/semiempirical/dftb2.h`; unset→throw, `ThreeOb`→throw at `dftb-gamma-h-damping`, `MioSketch`→one-shot `ATV_WARN` then run, ordered before the U^d check in `src/cpu/semiempirical/dftb3.cpp`) and the `shell_resolved_spin = true` refusal in `src/cpu/semiempirical/dftb2.cpp` (both `launch()` and `td_casida()`)
- [x] Write `tools/gen_dftb_3ob_tables.py` with `--check`, widen `Dftb3obSkPoint` to `double`, and re-emit `basis_data/dftb_3ob_tables.h` at 17 significant digits with the `/// Regenerate:` stamp and the deferred-layout decision recorded; record the measured energy movement in `regressions/semiempirical/dftb2-3ob-organic/REFERENCE.md`
- [x] Retake the value-moving pins against real offline DFTB+ 24.1 twins archived under `tools/tb_oracle/inputs/dftbplus/` — `spin-dftb2-o2`, `dftb2-spin-analytic-forces`, `dftb2-spin-analytic-stress` (upgrade to `oracle: dftb+` when the measured residual ≤ 1e-3 eV, with `SpinConstants` + `ShellResolvedSpin = No`), plus the value-only re-takes `td-dftb-h2` and `dftb2-multi-k-forces` whose labels stay as they are; update `tools/tb_oracle/README.md`
- [x] Correct every REFERENCE.md whose provenance label its own body contradicts — the DFTB3 trio (`dftb3-water-mio`, `dftb3-analytic-forces`, `dftb3-periodic-forces`: honest labels, the two named blockers, `formula-level` → `literature`, drop the stale "no DFTB+ binary" claim, restate the self-seeded absolute pin as an ATV twin snapshot) and the four contradictory `oracle: dftb+` labels on `dftb2-analytic-forces`, `dftb2-analytic-stress`, `dftb2-analytic-forces-threeob`, `dftb2-analytic-stress-threeob` (found defect, iron law 9)
- [x] Add regression example `regressions/semiempirical/spin-dftb2-o2/` (public API only: one-line `d.spin_w_ha = basis_data::kDftbSpinConstantsPbeHa;`; hard-coded goldens, no third-party at test time)
- [x] Run full check + test suite (SE-only single-domain config, then the default CPU config)
- [x] `/mol:simplify` ran clean on the spec surface (clang-format on touched C++ except generated tables; ruff format on the two generators; dropped a new `[[maybe_unused]]`; 3ob `--check` still byte-identical). Dead `pair_rep` / `h_radial` left in place per the comm-maint unused-symbol ticket.

## Testing strategy

单元测试（`tests/cpu/semiempirical/test_dftb_published_constants.cpp`，镜像 `src/cpu/semiempirical/`，每个测试瞄准**一个**函数/方法，全部硬编码期望值；仓库既有布局为 `tests/cpu/<dir>/test_<mod>.cpp` + `tests/CMakeLists.txt` 中的 `atv_add_cpp_test`）：

- **Happy path — 表值。** `k3obHubbardDerivsHa[1] == -0.1857`、`[6] == -0.1492`、`[7] == -0.1535`、`[8] == -0.1575`（精确相等，字面量）；其余 Z 全为 0.0。`kDftbSpinConstantsPbeHa` 中 O 行 `w_ha[1][1] == -0.028`、`w_ha[0][1] == w_ha[1][0] == -0.030`、`w_ha[0][0] == -0.035`、`n_shell == 2`；H 行 `n_shell == 1` 且 `w_ha[0][0] == -0.072`。LDA 表同法抽查 H `-0.064`。
- **Happy path — 塌缩与单位。** `internal::collapse_spin_w_ev` 对 Z=8 返回 `-0.761918814888` eV（tol 1e-12），Z=1 返回 `-1.959219809711` eV，Z=6 返回 `-0.625861883658` eV，Z=7 返回 `-0.707496042396` eV —— 同时钉住 HOAO 约定与 `kHartreeInEv` 的单次换算。
- **Edge case — 未设置即拒绝。** `DFTB2` 在 `multiplicity = 3` 且 `spin_w_ha` 为空时 `launch()` 抛 `std::invalid_argument`，消息中出现 `spin_w_ha` 与可粘贴的一行 how-to；`DFTB2::td_casida` 在三重态通道下同样抛。`DFTB3::launch()` 在 `sk_set = MioSketch` 已声明、`hubbard_du_ha` 未设置时抛；帧中 Z 无对应条目（如 Z=9）时也抛。
- **Edge case — SK 集选择器（三个分支各一个测试）。** (i) `sk_set` 保持默认 `Unset`、U^d 已填 → `launch()` 抛 `std::invalid_argument`，消息含 `sk_set` 与那一行可粘贴的 how-to（**这是"程序不替用户选"的证据**，且它先于 U^d 校验发生：U^d 也为空时抛的仍是 `sk_set` 消息）。(ii) `sk_set = DftbSkSet::ThreeOb` → 抛 `std::runtime_error`，消息含后续 slug 名 `dftb-gamma-h-damping`。(iii) `sk_set = DftbSkSet::MioSketch` + U^d 已填 → **不抛**、正常返回能量，且日志中出现 `supersedes the MIO parameter set`。日志断言照 `tests/cpu/dynamics/test_timing_report.cpp:122-144` 的形状：`LogConfig::logfile` + `log_init` → `logger().flush()` → 读回文件断言子串；连续两次 `launch()` 只出现一次该子串（`warned_mixed_set_` 的一次性证明）。
- **Edge case — 分壳自旋开关。** `DFTB2::shell_resolved_spin = true` + 三重态 → `launch()` 抛 `std::runtime_error`，消息含后续 slug 名 `dftb-shell-resolved-spin`；默认 `false` 的同一构型不抛并施加 HOAO 塌缩（与上面 `collapse_spin_w_ev` 的期望值一致）。
- **Edge case — DFTB2 不受 `Unset` 波及。** 默认构造的 `DFTB2` 仍为 `sk_set == DftbSkSet::MioSketch` 并正常运行；显式赋 `Unset` 后 `launch()` 抛（switch 穷尽性的证明，而不是走进静默默认分支）。
- **Edge case — 闭壳层不受影响。** `multiplicity = 1` 的 `DFTB2::launch()` 在 `spin_w_ha` 为空时**不抛**（惰性校验的正确性证明）；`shell_resolved_spin = true` 在闭壳层同样不抛（自旋通道未取用）。
- **Edge case — `Dftb3Scc` 构造闸。** 传入全零 U^d 向量抛出；size 不匹配继续抛出（既有行为不回退）。
- **Domain validation — 3ob 全精度。** `dftb_3ob_tables.h` 的 H–H 第 1 个网格点在 double 下等于 `.skf` 原值（硬编码 17 位字面量），且 `sizeof(Dftb3obSkPoint) == 8 * sizeof(double)`。
- **Domain validation — 无捏造残留。** `DftbElementRow` 不再有 `hubbard_du` / `spin_w` 成员（编译期 SFINAE/`static_assert` 形式的字段缺失断言，不是 grep 闸——铁律 1 禁止把源码扫描注册为测试）。

**回归示例（本 spec 的交付级端到端）：** `regressions/semiempirical/spin-dftb2-o2/`。仓库的铁律 1 规定端到端一律落在 `regressions/<domain>/<case>/` 并配 `REFERENCE.md`，**该布局取代通用的 `regressions/<slug>.<ext>` 扁平形状**；该案已在 `regressions/CMakeLists.txt:1820` 注册，无需新增注册。它是唯一同时演示公开 API 一行赋值、并因本次数值变更而必须重取的案例：

- 公开 API only：构造 `DFTB2`、`d.multiplicity = 3`、`d.spin_w_ha = basis_data::kDftbSpinConstantsPbeHa;`、`Evaluate` 驱动、读 `frame.meta` —— 无 internal 头、无 Context。`sk_set` 保持 DFTB2 的默认 `MioSketch`（本 spec 不改 DFTB2 默认），`shell_resolved_spin` 保持默认 `false`。
- 硬编码金标：`E_spin == -0.761918814888 eV`（tol 1e-9；解析式 E_spin = W_pp(O,PBE)，因 O₂ 三重态 m_A = 1.0 e 由对称性固定，标量塌缩在此**精确**）；总能相对被取代的 `-142.1382039484 eV` 位移 `+0.238081185112 eV`（tol 1e-4 eV）。
- **该总能位移为什么是等式而不是估计（`REFERENCE.md` 必须原样记录这四步，否则日后会有人以为容差可以放宽）：** (i) `src/cpu/semiempirical/dftb2.cpp:1205-1207` 用 `Tr(ρH⁰)` 而非能带和 Σε 装配总能，自旋移位不进入该项；(ii) O₂ 同核 ⇒ Δq_A ≡ 0 ⇒ 移位 ε^σ_A = σ W m 在两原子上相同，记作常数 c；(iii) `apply_shift`（`dftb2.cpp:490-509`）因此给出 H^σ = H⁰ ± cS；(iv) (H⁰ + cS)C = εSC 与 H⁰C = (ε−c)SC **本征向量逐一相同**，故 ρ 不变 ⇒ Tr(ρH⁰)、E^{(2)}、E_rep 全部不变，m_A 自洽地仍为 1.0。**唯一变化的是 E_spin，位移精确等于 ΔE_spin。** 1e-4 eV 只是数值噪声余量。**若实测位移偏离超过该容差，说明上述某一步在实现中不成立，必须按铁律 9 停下并报告 `path:line` + 实测值，不得改容差蒙混。**
- 离线 DFTB+：以 24.1（`tools/tb_oracle/setup_dftbplus_env.sh`）跑同一几何 + `SpinConstants` 块 + `ShellResolvedSpin = No`，input twin 与逐字输出归档到 `tools/tb_oracle/inputs/dftbplus/spin-dftb2-o2/`，`REFERENCE.md` 记录实测 ATV−DFTB+ 残差与 DFTB+ 版本。**残差 ≤ 1e-3 eV 才可标 `oracle: dftb+`**；否则标 `oracle: literature` + `oracle_version:` + 独立的 `family_oracle: dftb+` 行（`spin-dftb2-o2:4-6` 与 `td-dftb-h2:4` 的诚实形状），并把残差列为阻塞欠债。**ctest 永不启动 DFTB+**：值一律硬编码进 `reg_*.cpp`。

**pin 重取矩阵（哪些可升级、哪些不可，本 spec 必须明说而不是许诺做不到的 pin）：**

| 案例 | 数值变更 | 本 spec 可达标签 |
| --- | --- | --- |
| `spin-dftb2-o2` | W | `oracle: dftb+` 24.1（残差 ≤ 1e-3 eV 时） |
| `dftb2-spin-analytic-forces` | W | `oracle: dftb+` 24.1（同上；否则 literature + family_oracle） |
| `dftb2-spin-analytic-stress` | W | 同上 |
| `dftb2-multi-k-forces` | W（开壳层 dVa/dVb） | `oracle: literature` + `family_oracle: dftb+`；闸为 analytic-vs-FD |
| `td-dftb-h2` | W（三重态核） | `oracle: literature`（既有标签已诚实，仅重取数值） |
| `dftb3-water-mio` | U^d | `oracle: literature` + `family_oracle: dftb+`；**不得**标 `oracle: dftb+` |
| `dftb3-analytic-forces` | U^d | 同上 |
| `dftb3-periodic-forces` | U^d | 同上；同时把非法词汇 `oracle: formula-level` 改为 `oracle: literature` |
| `dftb2-3ob-organic` | 3ob double 重发射 | 既有 pin 保持，记录实测移动（预期 < 1e-4 eV） |

**该矩阵在任务上分成两半，因为它们的交付物不同类：** 上半（`spin-*` 三例，加上仅重取数值的 `td-dftb-h2` 与 `dftb2-multi-k-forces`）要**真的跑一遍离线 DFTB+ 24.1** 并可能升级标签；下半（DFTB3 三例，加上四份自相矛盾的 `dftb2-analytic-*` 标签）**没有任何数值可以升级**，只得到诚实标签与被点名的阻塞点。把两者塞进一个任务会让"没跑成 oracle"和"本来就跑不成"混为一谈。

**DFTB3 绝对身份为何在本 spec 内不可达（两个阻塞点，必须写进每份 DFTB3 `REFERENCE.md`）：** (i) 参数集混配——本 spec 之后 `DFTB3::sk_set` 存在了，但只有 `MioSketch` 被接线（`dftb3.cpp:22-23,38,318-319` 仍是 mio-1-1 SK/排斥项），所以这些 pin 是**用户显式声明的** mio SK + 3ob 类 U^d 配对，运行时带 `ATV_WARN`；配对越出参数集这一事实不因它变成用户声明而消失，**因此仍不得标 `oracle: dftb+`**；(ii) DFTB3 路径与 `internal/dftb_gamma.h` 中**完全没有** γ^h X–H 阻尼（全仓 grep `damp|zeta|DampXH` 只命中电荷混合器的 "undamped Picard"），而 DFTB3/3ob 在 ζ = 4.00 阻尼缺失时是未定义的。后续 slug 名：`dftb-gamma-h-damping`（含把 3ob SK/排斥项接进 `dftb3.cpp` 的 `ThreeOb` 分支）。

**已发现但本 spec 不修的缺陷（铁律 9，须在交付摘要中点名）：**
- `src/cpu/gpw/gpw_solver.cpp:134-147` `pack_plus_u` 对非法 `u_minus_j_ev <= 0 || l < 0` 静默 `continue` 而非抛出，违反"无回退值"（越界：不同方法域，本 spec 不依赖该表面）。
- 2.6e-02 eV 的 3ob 残差位于电子学装配（H⁰/S 构造或 SCC 核），float 假说已被实测证伪；未测候选为 SK 网格索引约定、spσ 通道下标/符号约定、Slater–Koster 旋转。建议 A/B：把 ATV 的 CH₄ 8×8 H⁰/S 与 DFTB+ 的 `hamsqr1.dat` / `oversqr.dat` 对拍。
- `dftb3-water-mio/REFERENCE.md:76` 记录的 DFTB2 水分子 ATV −110.996433406106 eV vs DFTB+ −110.9590 eV 存在 0.037 eV 缺口，与上一条同源。
- `dftb3-water-mio/REFERENCE.md:114-117` 声称"本机不存在 DFTB+ 二进制"，该陈述已过期（24.1 与 25.1 均已验证可用）。
- 该案的绝对能量 pin 是 ATV 自测值，违反 `.claude/notes/oracle-pin-provenance.md`"pin 不得由 ATV 自身测量播种"：重取时须改述为显式的 ATV twin 快照 + `oracle: literature`，并保留真正 oracle 级的闸（对 DFTB+ H₂ pin 的归约、analytic-vs-FD）。

**构建/运行（单域配置，铁律 8）：** SE-only（`ATV_BUILD_SEMIEMPIRICAL=ON`、`ATV_BUILD_EVALUATE=ON`、`TESTS`/`REGRESSIONS` ON，HF/PW/GPW/KSPACE/CUDA OFF）先跑 `ctest -R dftb`，再跑仓库默认 CPU 配置全量套件。

## Out of scope

- **γ^h X–H 阻尼（ζ = 4.00, DFTB+ `DampXHExponent`）** —— DFTB3 路径完全缺失；后续 slug `dftb-gamma-h-damping`。
- **`DFTB3::sk_set` 的 `ThreeOb` 接线** —— 本 spec 只交付**选择器本身**（`Unset` 哨兵 + 三个分支的抛/警语义），让"程序替用户挑参数集"这件事不再发生；把 3ob SK/排斥项真正接进 `dftb3.cpp`（从而消除参数集混配）仍属 `dftb-gamma-h-damping`。
- **`DftbSkSet::MioSketch` 的重命名** —— sketch 常数删除后该名字误导，但重命名是公开 API 变更、牵动 DFTB2 全部调用点，与本 spec 的诚实性目标正交；本 spec 只更新它的文档块，明确不改名，避免被顺手改掉。
- **完整分壳自旋收缩** \(\text{shift}_l=\sum_{l'}W^A_{ll'}p_{Al'}\) —— 本 spec 只交付分壳**输入面** + HOAO 标量塌缩消费 + `shell_resolved_spin = true` 的**拒绝**；`ShellResolvedSpin = Yes` 语义是后续 slug `dftb-shell-resolved-spin`。
- **2.6e-02 eV 3ob 电子学装配残差的定位** —— 已确认不是存储精度问题；作为已发现的阻塞欠债点名，修复另开 slug。
- **`pack_plus_u` 静默 continue 的修复** —— 不同方法域、不在本 spec 触及或依赖的表面，按铁律 9 报告 `path:line` 而不顺手改。
- **3ob 与 mio 两表布局的统一** —— 自觉推迟；spec 1 的 SKF 运行时读取器会一次性重定布局。
- **仓库根 `data/` 目录** —— 属于 spec 1（4 MB SK 表），本 spec 不发明该约定。U^d / W 根本不在 `.skf` 中、DFTB+ 自己也把它们当输入文本，故它们按本链条自身判据永不进入 `data/`；这与"合并 `ATV_DATA_DIR` 与 `ATV_REG_DATA_DIR`"的既有决定不冲突，后者对 spec 1 仍具约束力。
- **链条后续 spec 1–3 的全部内容** —— DFTB SKF 运行时读取器与 `data/` 根（spec 1）、SCF 基组 / D3 / D4 / SAP（spec 2）、GPW GTH/MOLOPT（spec 3）。
- **python 绑定** —— `u_eff` 未暴露，新成员同样不暴露；绑定面不变。
- **CUDA** —— DFTB 路径为 CPU-only，`cuda/` 无读取点。
- **新增 CMake 产品开关** —— `ATV_BUILD_SEMIEMPIRICAL` 已存在（`cmake/AtvProductModules.cmake:29`），新头文件位于其 DAG 闭包内，铁律 8 无需新开关。
- **从 DFTB+ 吸收任何代码** —— DFTB+ 为 LGPL-3.0、ATV 为 GPL-2.0-only，不可吸收；CLAUDE.md 的 LAMMPS 移植授权只覆盖 GPL-2 的 LAMMPS。DFTB+ 在此**只作数值 oracle**，算法一律出自公开文献；`ShellResolvedSpin` / `HubbardDerivs` / `SpinConstants` 这些**关键字名**的沿用是接口惯例，不是代码吸收。
