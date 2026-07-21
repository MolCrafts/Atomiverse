---
title: HF 能力补全 · 06 半定域 ECP — def2-ECP 静态表 + 核哈密顿积分
slug: hf-capability-gaps-06-ecp
status: code-complete
chain: hf-capability-gaps
link: 06/21
created: 2026-07-20
---

# HF 能力补全 · 06 半定域有效核势(ECP)

## Summary

为 Gaussian SCF 核哈密顿接入半定域有效核势(ECP):def2-ECP 家族(Rb–Rn,Z=37–86)作为生成一次、checked-in 的静态 C++ 表落在 `basis_data/ecp_def2.h`(`tools/gen_basis_data.py` 扩展生成,无任何运行时解析),查表类 `EcpPotential(molrs::Element, name)` 完全镜像 `GthPotential` 的查表构造器先例(`cpu/gpw/gth_potential.h`),不支持的 (元素, 名字) 抛错无回退。积分核作为 `cpu/scf/internal/ecp_integrals.h` 里工作区类 `EcpIntegrals` 的**成员**实现(cpu/scf 禁 `build_*` 自由函数,受 scf-slater-determinant-04-methods 的通配 grep 门约束):Kahn 形式高斯展开径向部分,McMurchie–Davidson type-1/type-2 径向积分,只改一电子核哈密顿(h += ⟨μ|U|ν⟩,核吸引与核排斥 Z→Z_valence,电子数扣除 n_core);S/T/ERI 逐字节不动。同时在方法层落下 ECP↔x2c 互斥 THROW 门(相对论 ECP 已烘入标量相对论,叠加 x2c 即双重计数)。用户可见成果:含 I、Au 等重元素的分子第一次能以 def2-SVP+def2-ECP 跑 RHF,并与外部钉死的 PySCF 值对账到 1e-6 Ha。依赖:05-basis(生成器与 def2 表管线已就位;`tools/gen_basis_data.py` 由 01 环恢复)。

## Domain basis

- 半定域 ECP 形式与高斯展开:U(r) = U_L(r) + Σ_{l<L} Σ_m |lm⟩[U_l(r) − U_L(r)]⟨lm|,U_l(r) = Σ_k d_kl r^{n_kl−2} e^{−ζ_kl r²}:Kahn, Baybutt, Truhlar, *J. Chem. Phys.* **65**, 3826 (1976), DOI 10.1063/1.432900。
- type-1/type-2 积分的 Hermite/径向分解:McMurchie, Davidson, *J. Comput. Phys.* **44**, 289 (1981), DOI 10.1016/0021-9991(81)90053-X。径向被积函数含修正球贝塞尔函数 i_l(kr),高 l 下数值娇气(上行递推灾难性消去):实现必须用带尺度形式 M_l(x)=e^{−x}i_l(x)(小 x 级数、大 x 递推),单元测试钉住高 l 投影道矩阵元。
- def2-ECP 参数谱系:Weigend, Ahlrichs, *PCCP* **7**, 3297 (2005), DOI 10.1039/B508541A;Au ECP60MWB:Andrae 等, *Theor. Chim. Acta* **77**, 123 (1990), DOI 10.1007/BF01114537;I ECP28MWB:Bergner 等, *Mol. Phys.* **80**, 1431 (1993), DOI 10.1080/00268979300103121。
- 教科书锚点:Jensen, *Introduction to Computational Chemistry*, ECP 一节(价电子近似、半定域形式、相对论烘入)。
- 验证靶(oracle):RHF I₂(R=2.666 Å)与 AuH(R=1.524 Å),def2-SVP+def2-ECP,总能量 vs 外部运行 PySCF(`ecp='def2-svp'`,钉值注明版本与命令)|ΔE| ≤ 1e-6 Ha。单位:表内指数/系数原子单位;能量对账 Hartree。

## Design

数据面沿用 05 环模式:`basis_table.h` 增 `EcpGaussianTerm{n, alpha, coeff}` / `EcpChannel{l, terms}` / `AtomEcp{n_core, channels}`(空 `AtomEcp` = 该 Z 无 ECP,即全电子);`ecp_def2.h` 声明 `inline const std::vector<AtomEcp>& ecp_def2()` 按 Z 索引,由生成器字节确定地发出(头部 Regenerate 注释记录命令)。查表类 `cpu::scf::EcpPotential`:成员 `element / name / n_core / channels`(局域道 l=L 排末位),双构造器与 `GthPotential` 同形 —— 显式供参构造 + `(molrs::Element, std::string_view)` 查表构造,查表失败抛含支持项清单的 `std::runtime_error`。积分面:`internal/ecp_integrals.h` 的 `EcpIntegrals` 工作区类,构造器吃 `(shells, 带 ECP 的原子列表)`,成员方法产出 U 矩阵(type-1 局域 + type-2 半定域投影),径向积分用带尺度贝塞尔 + 高斯求积;类私有、绝不进公共 API(Frame 纯净规则:方法私有工作区)。接线:`IntegralPack` 构造器增 `ecp` 名字参数(默认空 = 全电子),命中表的原子做 Z_val=Z−n_core(核吸引 V、核排斥 V_nuc)与 n_electrons 扣减,`Hcore += U`;表外元素维持全电子(def2 约定 Z≤36 全电子)。方法层:`MeanFieldMethod` 增成员 `std::string ecp;`(空=全电子)与保留成员 `bool x2c = false;`(18 环接线),`drive_scf_` 序幕内联校验:x2c && 任一原子命中 ECP 表 → 抛含元素名的 `std::runtime_error`(逐元素互斥,单次使用不抽函数)。`def2_svp.h` 由生成器补 Z=37–86 价基行(def2-SVP 重元素价基本就只与 def2-ECP 成对定义)。本环不动 SlaterDeterminant/run() 签名等并行链表面,无 external-depends。

**实现审计注记(2026-07-21)**:

1. **f 投影道钉值改走合成 ECP**:PySCF 2.14.0 的 def2-ECP 数据全族最高投影道为
   d(l=2),不存在 f 投影道(逐元素核实 Rb–La、Hf–Rn)。ac-002 的"高 l 投影道"
   钉值因此经由 PySCF **用户自定义 ECP**(`mol.ecp = {...}`,合成 l=3 通道 + 局域
   项)外部钉定 — 同一 oracle 协议(`mol.intor('ECPscalar')`),type-2 高 l 路径
   (λ ≤ 6、带尺度贝塞尔)被真实覆盖;真实 def2-ECP(I)矩阵元另有 15 个钉值,
   含 f AO(氢上离心)与 d 投影道。两组钉值均达 ≤3e-13,断言 1e-10。
2. **范围外必要修复(阻塞 ac-006)**:`src/cpu/scf/operators.cpp` 的
   `ShellBlockPairList::g_rhf` 与 `::k` 中 `AOpair ao_pairs[36]` 按 d 壳上限
   (6×6)定容,f 壳块对产生最多 10×10=100 个 AO 对 → **栈缓冲溢出(静默 UB)**。
   该缺陷随 f 壳支持既已存在(全电子 H₂O/def2-TZVP 即偏 PySCF 8e-9 Ha,AuH 偏
   7.7e-6 Ha 并破坏线性分子的 π/δ 轨道简并),本环 AuH 回归首次将其暴露为可测
   偏差;修为 `kMaxNCart² = 100`。修复后 AuH 对钉 7e-12 Ha。诊断链:S/T/V/U/
   稠密 ERI 全部 ≤6e-13 对 PySCF、能量分解定位到直接 Fock 路径。
3. **EcpPotential 附带两个 spec 未列的静态查表成员**:`family_table(name)`
   (ECP 家族名 → Z 索引表;IntegralPack 接线与 x2c 门的单一解析来源)与
   `element_symbol(Z)`(诊断消息:查表失败清单、x2c 门的元素名)。均为纯查表,
   非工厂函数。`IntegralPack` 增 `ecp`(名字)与 `has_ecp` 成员,供
   `ScfGradient::compute`(ECP pack 抛错,镜像球谐先例)与 `minao_supported`
   (ECP pack 回退 Hcore 初猜 — MINAO 参考密度为全电子)守卫。

## Files to create or modify

- `tools/gen_basis_data.py` — 增 def2-ECP 发射路径(`ecp_def2.h`)+ def2-SVP Z=37–86 价基行;字节确定
- `include/atomiverse/cpu/scf/basis_data/basis_table.h` — 增 `EcpGaussianTerm/EcpChannel/AtomEcp` 结构
- `include/atomiverse/cpu/scf/basis_data/ecp_def2.h` (new) — def2-ECP 静态表(Rb–Rn)
- `include/atomiverse/cpu/scf/basis_data/def2_svp.h` — 重生成,补 Z=37–86 价基行
- `include/atomiverse/cpu/scf/ecp_potential.h` (new) — `EcpPotential` 查表类声明
- `src/cpu/scf/ecp_potential.cpp` (new) — 查表构造器实现(抛错无回退)
- `include/atomiverse/cpu/scf/internal/ecp_integrals.h` (new) — `EcpIntegrals` 工作区类
- `src/cpu/scf/ecp_integrals.cpp` (new) — type-1/type-2 径向 + 带尺度贝塞尔实现
- `include/atomiverse/cpu/scf/internal/integrals.h` — `IntegralPack` 构造器 `ecp` 参数
- `src/cpu/scf/integrals.cpp` — Z_val / n_electrons / Hcore+=U 接线
- `include/atomiverse/cpu/scf/mean_field_method.h` — `ecp` / `x2c` 成员 + Doxygen
- `src/cpu/scf/mean_field_method.cpp` — 序幕互斥 THROW 门
- `CMakeLists.txt` — 注册两个新 src/cpu/scf/*.cpp
- `tests/cpu/scf/test_ecp.cpp` (new) — 表/查表/矩阵元/计数/互斥门单元测试
- `tests/CMakeLists.txt` — 注册 `test_ecp`
- `regressions/scf/ecp_heavy_atom/reg_ecp_heavy_atom.cpp` (new) — I₂/AuH oracle 回归
- `regressions/CMakeLists.txt` — `atv_add_regression(scf ecp_heavy_atom)`(`ATV_BUILD_HF` 门内)
- `docs/tutorial/ch04-basis-sets.md` — "有效核势" 一节

## Tasks

- [x] Write failing tests tests/cpu/scf/test_ecp.cpp(ecp_def2 表结构钉 PySCF:I/Au 的 n_core、通道数、首末指数;EcpPotential 表外元素/未知名字抛错;EcpIntegrals 若干 U 矩阵元含高 l 投影道钉外部 PySCF `mol.intor('ECPscalar')` 值;I₂ IntegralPack n_electrons=50、V_nuc 用 Z_val、S/T 与全电子路径逐元素相等;x2c+ecp 抛含元素名的 runtime_error)并注册 tests/CMakeLists.txt
- [x] Implement generator extension in tools/gen_basis_data.py(ecp_def2.h 发射 + def2_svp.h Z=37–86 行),生成两表并验证字节可复现(同命令再跑 `git diff` 为空)
- [x] Implement EcpPotential in ecp_potential.h / ecp_potential.cpp + basis_table.h 结构(查表构造器镜像 GthPotential,Doxygen 带单位)
- [x] Implement EcpIntegrals in internal/ecp_integrals.h / src/cpu/scf/ecp_integrals.cpp(type-1/type-2 McMurchie–Davidson 径向、带尺度贝塞尔 M_l(x)=e^{−x}i_l(x)、成员而非 build_* 自由函数)并注册 CMakeLists.txt
- [x] Wire ECP into IntegralPack + MeanFieldMethod(internal/integrals.h、integrals.cpp:ecp 参数、Z_val、n_electrons、Hcore+=U;mean_field_method.h/.cpp:`ecp`/`x2c` 成员与序幕互斥 THROW)
- [x] Add regression regressions/scf/ecp_heavy_atom(RHF I₂ 与 AuH,def2-SVP+def2-ECP,vs 钉住的 PySCF 值 ≤1e-6 Ha,≤60 s)并注册 regressions/CMakeLists.txt
- [x] Extend docs/tutorial/ch04-basis-sets.md:"有效核势" 一节(为何可替换芯电子、半定域形式、价电子计数;I₂ 上手示例;章末 PySCF 对照表行 —— 标量进表,不用柱状图)
- [x] Run full check + test suite

## Testing strategy

- **快乐路径(单元,秒级)**:`test_ecp` 只做表结构断言与小体系单个矩阵元/计数断言,绝不跑完整 SCF;PySCF 钉值写死在测试源码注释(版本+命令),ctest 无 Python 依赖。
- **边界**:表外 Z(如 O)的 `EcpPotential` 查表抛错;未知 ECP 名字抛错;`ecp` 为空时全路径与现状逐字节等价(S/T/ERI 及轻元素分子能量不变);高 l(f 投影道)矩阵元在带尺度贝塞尔下与钉值一致(数值娇气点被单测钉死)。
- **互斥门**:`rhf.ecp="def2-ECP"; rhf.x2c=true;` 对含 I 的 Frame `run()` 抛含元素名的 `std::runtime_error`;全电子分子 + x2c 不抛(门是逐元素的)。
- **领域验证(regressions/,≤60 s)**:`reg_ecp_heavy_atom` 以公开 API(`cpu::scf::RHF`,`basis`/`ecp` 成员)跑 I₂ 与 AuH,对钉死的 PySCF oracle 断言 ≤1e-6 Ha;非零退出即失败;只挂 `regressions` target,不进 ctest。

## Out of scope

- ECP 解析梯度(几何优化/AIMD 走 ECP 的 dU/dR)——后续环;本环 ECP 仅能量路径。
- x2c 本身(18 环);自旋-轨道 ECP(SO-ECP);镧系 4f-in-core。
- def2-SVP 之外重元素基组扩容(def2-TZVP 重元素行等)与 CUDA SCF 的 ECP 适配。
- 运行时 ECP 文件解析器 —— 永久禁区,非"暂缓"。
