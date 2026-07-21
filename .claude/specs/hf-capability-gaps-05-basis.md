---
title: HF 能力补全 · 05 基组扩容 — 第三周期 + 弥散(6-31+G*) + aug-cc-pVDZ 静态表
slug: hf-capability-gaps-05-basis
status: code-complete
chain: hf-capability-gaps
link: 05/21
created: 2026-07-20
---

# HF 能力补全 · 05 基组扩容(静态表)

## Summary

把 Gaussian SCF 的基组数据面从 H–F 扩到第三周期,并新增两个基组:既有四套表(6-31G、6-31G*、def2-SVP、def2-TZVP)经 01 环恢复的 `tools/gen_basis_data.py` 重新生成、补齐 Z=11–18(Na–Ar;Pople 系若 PySCF 定义则一并带上 K/Ca),同时新增 6-31+G*(重原子弥散 sp,新表头 `pople_631pgs.h`)与 aug-cc-pVDZ(新表头 `aug_ccpvdz.h`),注册进 `BasisSet` 枚举与 `ShellList` 分派,并按 02 环的 per-set 笛卡尔/球谐旗标登记(Pople 系 = cartesian-6d,def2/cc 系 = spherical)。全程遵守"无运行时解析"铁律:数据是生成一次、checked-in 的静态 C++ 头;不支持的 Z 保持空行 + 抛错(无回退)。用户可见成果:阴离子(OH⁻、Cl⁻)与含 Na–Ar 元素的分子第一次能在带弥散/增广基组下算 RHF,教程第 4 章随之扩写弥散函数物理。

## Domain basis

- Pople 分裂价基 6-31G / 6-31G(+)(*):Ditchfield, Hehre, Pople, *J. Chem. Phys.* **54**, 724 (1971), DOI 10.1063/1.1674902;极化函数:Hariharan, Pople, *Theor. Chim. Acta* **28**, 213 (1973), DOI 10.1007/BF00533485。
- Dunning 相关一致基:Dunning, *J. Chem. Phys.* **90**, 1007 (1989), DOI 10.1063/1.456153;增广(aug-)弥散函数:Kendall, Dunning, Harrison, *J. Chem. Phys.* **96**, 6796 (1992), DOI 10.1063/1.462569。
- def2 系:Weigend, Ahlrichs, *Phys. Chem. Chem. Phys.* **7**, 3297 (2005), DOI 10.1039/B508541A。
- 弥散函数物理(教程锚点):紧凑基组里阴离子 HOMO 呈非束缚(ε_HOMO ≥ 0 伪像);加弥散后转为束缚。验证靶:OH⁻ 的 ε_HOMO 在 6-31G* vs 6-31+G* 之间符号/束缚性翻转;数值 oracle 为外部运行 PySCF 后钉死的 Cl⁻/6-31+G* 与 H2O/aug-cc-pVDZ RHF 总能量(约定 |ΔE| ≤ 1e-7 Ha,钉值时注明 PySCF 版本与命令)。
- 单位:表内指数/系数为原子单位(PySCF `mol._basis` 未归一约定,物理归一由 `ShellList` 构造器施加);能量对账用 Hartree。

## Design

数据模型不变:每套基组一个 `basis_data/*.h`,`inline const std::vector<AtomBasis>&`,按 Z 索引、空 `AtomBasis` 即不支持(`build_shell_list_from_table` 已在 `src/cpu/scf/basis_set.cpp:112-117` 抛错,本环零改动该逻辑)。角动量上限已达 l=3(`rys_eri.h`),def2-TZVP 第三周期的 f 壳与 aug-cc-pVDZ 的 d 壳均在既有引擎能力内。新符号:`BasisSet::Pople631pGs`、`BasisSet::AugCcPVDZ`(`basis_set.h` 枚举 + Doxygen),`basis_data::pople_631pgs()`、`basis_data::aug_ccpvdz()`;`basis_set_name` 返回 "6-31+G*" / "aug-cc-pVDZ";`ShellList` 构造器 switch 各加一臂;02 环引入的 per-set 纯球谐旗标登记处同步两行(Pople631pGs=cartesian-6d,AugCcPVDZ=spherical)。生成器扩展:`tools/gen_basis_data.py` 增加 "6-31+g*"→`pople_631pgs`、"aug-cc-pvdz"→`aug_ccpvdz` 的名字映射与元素清单,输出必须字节确定(同输入两次运行逐字节一致),头部 Regenerate 注释照旧记录完整命令。STO-3G 表不在本环范围。生命周期:表为 static 局部、首次调用构造,无所有权问题。依赖假设:01 环(生成器已恢复)、02 环(旗标已存在)先行落地。

### 实现审计注记(2026-07-20,实现时落定)

1. **OH⁻ 束缚性翻转的真实位置**:Domain basis 叙事预期"6-31G* 下
   ε_HOMO ≥ 0 伪像";外部 PySCF 2.14.0 实测(r(OH)=0.9640 Å, cart=True)
   6-31G* 已勉强束缚(−0.037774 Ha),真正的非束缚伪像在 STO-3G
   (+0.251912 Ha)。回归按 ac-006 字面判据实现(逐基组对钉 ≤1e-6 Ha、
   6-31+G* 严格更低且与钉值同号),并加钉 STO-3G 行保留真实的
   非束缚→束缚符号翻转演示(REFERENCE.md"已知偏差"节同步记录)。
2. **生成器新增广义缩并拆列**:PySCF `mol._basis` 里 aug-cc-pVDZ 的
   s/p 块是多列广义缩并(同一原语组、nctr 列系数);原生成器只取第一列,
   会静默丢基函数。现按列拆成独立 `ContractedShell`(张成 AO 空间严格
   不变,列序循 PySCF nctr 序);四套分段缩并表(单列)输出不受影响。
3. **单元测试目标名**:注册为 `test_basis_tables`(与 spec 文件名对齐,
   实现指令固定 `./build/tests/test_basis_tables`);同目录旧测试沿用
   `test_cpu_scf_*` 前缀,命名对齐留给后续统一环,不在本环顺手改。
4. **生成器短形式**:`SETS` 注册表(名字→表干名+元素清单)允许
   `gen_basis_data.py 6-31+g*` 短调用;表头 Regenerate 注释仍记录完整
   显式命令,`--check` 两种形式均可复现逐字节输出。

## Files to create or modify

- `tools/gen_basis_data.py` — 新基组名字映射 + 第三周期元素清单 + 字节确定性输出
- `include/atomiverse/cpu/scf/basis_data/pople_631g.h` — 重生成,补 Z=11–18
- `include/atomiverse/cpu/scf/basis_data/pople_631gs.h` — 重生成,补 Z=11–18
- `include/atomiverse/cpu/scf/basis_data/def2_svp.h` — 重生成,补 Z=11–18
- `include/atomiverse/cpu/scf/basis_data/def2_tzvp.h` — 重生成,补 Z=11–18
- `include/atomiverse/cpu/scf/basis_data/pople_631pgs.h` (new) — 6-31+G* 表
- `include/atomiverse/cpu/scf/basis_data/aug_ccpvdz.h` (new) — aug-cc-pVDZ 表
- `include/atomiverse/cpu/scf/basis_set.h` — 枚举加 `Pople631pGs` / `AugCcPVDZ`
- `src/cpu/scf/basis_set.cpp` — `basis_set_name` + switch 分派 + include + 02 环旗标登记
- `tests/cpu/scf/test_basis_tables.cpp` (new) — 表结构单元测试
- `tests/CMakeLists.txt` — 注册 `test_basis_tables`
- `regressions/scf/diffuse_anion/reg_diffuse_anion.cpp` (new) — 弥散阴离子文献/oracle 回归
- `regressions/CMakeLists.txt` — `atv_add_regression(scf diffuse_anion)`(`ATV_BUILD_HF` 门内)
- `docs/tutorial/ch04-basis-sets.md` — "极化、弥散与第三周期" 扩写

## Tasks

- [x] Write failing tests tests/cpu/scf/test_basis_tables.cpp(新表逐元素壳数/AO 数钉 PySCF `mol._basis`;四套既有表 Z=11–18 行非空;不支持 Z 经 ShellList 抛含基组名的 runtime_error)并注册 tests/CMakeLists.txt
- [x] Implement generator extension in tools/gen_basis_data.py(6-31+g*→pople_631pgs、aug-cc-pvdz→aug_ccpvdz、第三周期元素清单、字节确定输出)
- [x] Regenerate pople_631g.h / pople_631gs.h / def2_svp.h / def2_tzvp.h 补 Z=11–18,并验证字节可复现(同命令再跑一次 `git diff` 为空)
- [x] Implement new tables:生成并 check in basis_data/pople_631pgs.h 与 basis_data/aug_ccpvdz.h
- [x] Implement enum + dispatch in basis_set.h / basis_set.cpp(枚举、basis_set_name、switch、02 环 cart/sph 旗标两行、Doxygen @brief)
- [x] Add regression regressions/scf/diffuse_anion(OH⁻ ε_HOMO 束缚性翻转 6-31G* vs 6-31+G*;Cl⁻/6-31+G* 与 H2O/aug-cc-pVDZ RHF 总能量 vs 钉住的 PySCF 值 ≤1e-7 Ha;≤60 s)并注册 regressions/CMakeLists.txt
- [x] Extend docs/tutorial/ch04-basis-sets.md:命名法解码、OH⁻ 弥散演示、章末 PySCF 对照表行(标量进表)、molplot 基组收敛曲线
- [x] Run full check + test suite

## Testing strategy

- **快乐路径(单元,秒级)**:`test_basis_tables` 只做表结构断言(壳数、l 序列、AO 计数、首末指数抽查钉 PySCF),绝不跑 SCF;两新枚举经 `ShellList` 构造成功。
- **边界**:Z=0 行为空;四套既有表 Z=2(He)等原本不支持的行为保持原状;`sto3g.h` 逐字节不动;`basis_set_name` 对新枚举无 default 回退。
- **领域验证(regressions/,≤60 s)**:`reg_diffuse_anion` 以公开 API(`cpu::scf::RHF`,`charge` 成员)跑 OH⁻/Cl⁻/H2O,对 Kendall-Dunning-Harrison 弥散物理 + 钉住的 PySCF oracle 断言;非零退出即失败。
- **生成器**:字节可复现由重生成 + `git diff` 验证(开发时一次性动作;ctest 内不得引入 Python/PySCF 依赖,符合 tests/ 铁律)。

## Out of scope

- STO-3G 第三周期、K/Ca 之后的任何 Z、ECP(def2 重元素赝势路线)。
- 球谐↔笛卡尔变换本身(02 环职责);l>3(g 壳)引擎支持。
- CUDA SCF 路径的新基组适配;6-311G/cc-pVTZ 等更多基组(后续环)。
- 运行时基组文件解析器 —— 永久禁区,非"暂缓"。
