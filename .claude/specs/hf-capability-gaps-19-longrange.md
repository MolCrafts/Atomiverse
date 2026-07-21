---
title: 长程库仑重构 — 顶层 longrange/ 算子数学抽取与 cpu/kspace/ 更名
slug: hf-capability-gaps-19-longrange
status: code-complete
chain: hf-capability-gaps
link: 19/21
created: 2026-07-20
---

# 长程库仑重构 — 顶层 longrange/ 算子数学抽取与 cpu/kspace/ 更名

## Summary

行为保持的重构链环(架构裁定,用户已确认):新建顶层 facade 目录 `include/atomiverse/longrange/`(与 `potentials/` 同宪章 — backend 无关、无 Component、无 Frame、无 MPI、无 FFT 引擎),把 `cpu/kspace/` 中的纯算子数学(Ewald k 向量表、部分结构因子 S(k)、由全局 S(k) 求力、自能/净电荷/倒空间能量与逐 k 位力、`recommend_alpha`;PME 的 B 样条、样条模、`good_fft_size`、电荷铺展、影响函数+位力、力插值)抽到该目录;随后 `git mv cpu/kspace/ → cpu/longrange/`(3 头文件 + 2 cpp),薄 Component 包装保留类名 `KSpaceEwald`/`KSpacePME` 与 `ATV_BUILD_KSPACE` 开关(它门控 FFTW 依赖)。Ewald/PME 能量前后逐位一致 — 现有单元测试与 `reg_npt_pme_nacl` 回归钉死数值。本环为 20-fmm/21 铺路;并附教程第 19 章"长程库仑 I:Ewald 与 PME"节,记录既有能力的新居所。

## Domain basis

无新物理 — 行为保持。搬移携带原引文:

- Ewald 求和:P. P. Ewald, *Ann. Phys.* **369**, 253 (1921). DOI: 10.1002/andp.19213690304。
- 光滑粒子网格 Ewald(SPME,B 样条与影响函数即其附录 `fill_bspline`):U. Essmann *et al.*, *J. Chem. Phys.* **103**, 8577 (1995). DOI: 10.1063/1.470117。
- 教学锚点(回归守卫):NaCl Madelung 常数 −1.7476(每离子对,rocksalt),`regressions/md/npt_pme_nacl` 对照文献值。
- 单位:对外 eV/Å(`core/units.h` 的 `units::coulomb`),k 向量 Å⁻¹;门槛 = 现有测试容差不变,能量逐位一致。

## Design

- **`ATV::longrange`(新顶层 facade 簇)**:纯算子数学,操作裸数组/span 与 `inv_cell` 3×3,不触 `cpu::Frame`、Component、MPI、FFTW。
  - `longrange/ewald_sum.h` — `class EwaldSum`,验证式构造 `EwaldSum(double alpha, double accuracy)`(查表构造风格,禁工厂);成员:k 向量表生成(输入 `inv_cell`,源 `ewald.cpp:46-117`)、部分 S(k)(裸 x,y,z,q 数组,:152-163)、由全局 S(k) 求力(:178-195)、自能/净电荷/倒空间能量与逐 k 位力(:197-243)、静态 `recommend_alpha`(:35-38)。**MPI 缝恰在"算部分 S(k)"与"由全局 S(k) 求力"之间** — 两调用内核 API,Allreduce 留在 backend 包装内。
  - `longrange/pme_spline.h` — `bspline()`、`spline_moduli()`、`good_fft_size()`(`pme.cpp:38-108`,已是匿名命名空间自由函数,提升为带 Doxygen 的纯数学函数)与电荷铺展(:257-290,grid-view 参数)。
  - `longrange/pme_influence.h` — 影响函数+位力(:312-364)、力插值(:375-413);均基于 grid view,**不含 FFTW 调用** — FFT 留在 backend 侧。
- **`cpu/longrange/`(git mv 自 `cpu/kspace/`)**:`KSpaceEwald`/`KSpacePME` 变薄包装 — Frame span 解取、DD/MPI Allreduce、FFTW 计划与执行、meta 写回,委托 facade 数学;类名、Phase 绑定、`ATV_BUILD_KSPACE` 不变。
- **规则(为 20/21 环立此存照)**:SCF 侧今后只许 include `atomiverse/longrange/*`,永不 `cpu/longrange/*`。

## Files to create or modify

- `include/atomiverse/longrange/ewald_sum.h` (new)
- `include/atomiverse/longrange/pme_spline.h` (new)
- `include/atomiverse/longrange/pme_influence.h` (new)
- `include/atomiverse/cpu/longrange/kspace.h`、`ewald.h`、`pme.h`(git mv 自 `cpu/kspace/`,改写为薄包装)
- `src/cpu/longrange/ewald.cpp`、`src/cpu/longrange/pme.cpp`(git mv 自 `src/cpu/kspace/`)
- `tests/cpu/longrange/test_ewald_sum.cpp` (new)、`tests/cpu/longrange/test_pme_math.cpp` (new)
- `tests/cpu/longrange/test_ewald_coul.cpp`、`test_pme_coul.cpp`(git mv 自 `tests/cpu/kspace/`,仅改 include)
- `tests/mpi/test_pme_nacl_parity.cpp`、`regressions/md/npt_pme_nacl/reg_npt_pme_nacl.cpp`(仅改 include 路径)
- `tests/CMakeLists.txt`(:151/:153)、根 `CMakeLists.txt`(源列表 :222/:312 与 `ATV_BUILD_KSPACE` 块 :573 附近)
- `docs/developer/kspace-long-range.md`(路径与新居所叙述)
- `docs/tutorial/ch19-hf-large-systems.md` (new)、`docs/tutorial/index.md`(章节表加一行)
- `CLAUDE.md`(三处:分层表 `kspace/`→`longrange/` 行;规则 bullet 改为"长程求解器在 longrange/;共享算子数学只写一次,在顶层 longrange/";facade 节 `potentials/` 旁加 `longrange/` 行)

## Tasks

- [x] Write failing tests for facade 算子数学(`tests/cpu/longrange/test_ewald_sum.cpp`、`test_pme_math.cpp`):B 样条单位分解与 Essmann 附录值、`spline_moduli`/`good_fft_size` 钉值、双离子小盒 EwaldSum k 表规模与 S(k)/倒空间能量 — 期望值取自当前 `cpu/kspace/` 实现(pinning)
- [x] Implement `EwaldSum` in `include/atomiverse/longrange/ewald_sum.h`(两调用内核 API;验证式构造;`recommend_alpha` 静态成员;逐 k 位力)
- [x] Implement PME 纯数学 in `include/atomiverse/longrange/pme_spline.h` + `pme_influence.h`(grid-view;无 FFTW/MPI/Frame include)
- [x] Implement 更名与薄包装:git mv `cpu/kspace/`→`cpu/longrange/`(3 头 + 2 cpp),`KSpaceEwald`/`KSpacePME` 委托 facade,Allreduce 保持在部分 S(k) 与力调用之间;类名与 `ATV_BUILD_KSPACE` 不动
- [x] Update include 路径与构建:git mv `tests/cpu/kspace/`→`tests/cpu/longrange/`,改 `tests/mpi/test_pme_nacl_parity.cpp`、`regressions/md/npt_pme_nacl/reg_npt_pme_nacl.cpp`、`tests/CMakeLists.txt`、根 `CMakeLists.txt`
- [x] Verify 纯度与清场:grep `include/atomiverse/longrange/` 无 Component/Frame/mpi.h/fftw include;全树 grep 零 `cpu/kspace`;新公有符号 Doxygen(`@brief`/`@param` 带单位)齐全
- [x] Add docs:`docs/developer/kspace-long-range.md` 更新路径;`CLAUDE.md` 三处编辑;新写 `docs/tutorial/ch19-hf-large-systems.md` 之"长程库仑 I:Ewald 与 PME"节(条件收敛、实/倒空间劈分、PME 为何 O(N log N)、NaCl Madelung 动手、章末标量表)并在 `docs/tutorial/index.md` 登记
- [x] Run full check + test suite(含 `test_cpu_ewald_coul`/`test_cpu_pme_coul` 原容差通过)+ `reg_npt_pme_nacl` 回归

## Testing strategy

- **快乐路径(单元,秒级)**:新 facade 测试钉住 B 样条/样条模/`good_fft_size`/EwaldSum 小盒数值 = 重构前实现输出;既有 `test_ewald_coul`/`test_pme_coul` 除 include 行外零改动地通过 — 即能量逐位一致的守卫。
- **边界**:`EwaldSum` 构造对 `alpha<=0`、`accuracy∉(0,1)` 抛异常(承袭现行为);`good_fft_size` 小 n 下限 4;`spline_moduli` 偶数阶 K/2 零点邻均修复。
- **领域验证(science.required)**:`reg_npt_pme_nacl` 文献值回归(Madelung −1.7476/离子对,Ewald 解析)不动地通过;MPI 构型下 `test_pme_nacl_parity` 通过。
- 单元测试禁 e2e;不新增回归(既有 NaCl 回归即本环守卫)。

## Out of scope

- FMM 与 `LongRangeRole::FullRange`(未来 MD FMM Component)— 明确留给链环 20/21。
- `cuda/kspace/` 更名与 CUDA 侧对 facade 数学的消费 — 裁定只涉 CPU 树,正交性对齐为后续专项。
- 任何数值/算法改动(alpha 启发式、样条阶、网格选择均原样);`docs/developer/kspace-long-range.md` 文件名不改。
- SCF 侧接入 `atomiverse/longrange/*`(20/21 环消费;本环只立"永不 include cpu/longrange"规则)。
