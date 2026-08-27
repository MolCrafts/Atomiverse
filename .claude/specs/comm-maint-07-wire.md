---
title: Border wire mask, the wire_id trap, and the Comm base decision
status: approved
created: 2026-08-27
chain: comm-maint
---

# Border wire mask, the wire_id trap, and the Comm base decision

## Summary

三件较小但各自独立的事。其一，border 线宽由 `core/frame_schema.h:89` 的字段表定死为 x,y,z,vx,vy,vz,mass,charge,type,tag,image = 79 B/ghost，没有任何按运行裁剪的开关；LAMMPS 的 `default_border` 是 `{"id","type","mask","x"}`（`atom_vec.cpp:39`），速度走独立的 `default_border_vel`（`:40`）由 `comm_modify vel` 选择，per-type 的质量根本不上线。其二，`atom_pack.cpp` 有三处（`:353`、`:449`、`:490`）把**任何 `wire_id <= AtomFieldId::z` 的列**当作需要 PBC 位移的坐标——在字段表里于 `z` 之前插入一行会让新列静默收到 `shift[wire_id]`，无静态断言、无注释、无测试。其三，`cpu::Comm` 与 CUDA 侧同名不同义的 `DomainDecomposition`。

## Domain basis

不声明新物理。border 线宽变化会改变 ghost 行携带的字段集合，但不改变任何被计算的物理量——判据是能量与力逐位不变。PBC 位移 \(p_d L_d\)（\(p_d \in \{-1,0,+1\}\)，\(L_d\) 为当前盒边长，单位 Å）的正确性由既有 parity 闸门覆盖。

## Design

**(1) `wire_id <= z` 陷阱——先修这个，它是三件里唯一能造成静默错误的。**

三处按 `wire_id` 的**数值**判断"这是不是坐标"。字段表是 X-macro，`AtomFieldId` 与 `wire_id` 都由表中行序生成（`frame_schema.h:153` 的 `static_cast<std::uint16_t>(AtomFieldId::id)`），所以在 `z` 之前插入任何一行都会把 `z` 的 id 顶高，新列落进 `wire_id <= 2` 而收到坐标位移——若它不是 `Storage` 则抛 "position halo column has invalid schema"，若恰好是 `Storage` 则**静默加上一个盒长**。

顺带记录一处文档自相矛盾：`frame_schema.h:76-77` 写 "wire_id is explicit and never derived from a variant or container index"，而 `:153` 正是用容器索引生成它。

**做法**：把"这一列是否是需要位移的坐标"变成字段表上的**显式标志**（例如 `kColPbcShift`），三处改为查标志而不是比较 `wire_id`；并加一个 `static_assert` 钉住 `AtomFieldId::x/y/z == 0/1/2`，让任何行序改动在编译期爆掉而不是运行时静默。同时修正 `:76-77` 那句注释。

**(2) border 线宽的 per-run mask。**

声明式字段表要**保留**——它比 LAMMPS 的 per-style 手写 `fields_border` 好：加一个通信字段是改一行，LAMMPS 是改 30 来个 `atom_vec_*.cpp` 之一并在同一构造函数里重复 7 次。缺的只是"这次运行不需要 ghost 速度"这种裁剪。

现状 ATV 已经**部分**具备：`PackPlan` 由 frame 当前存在的列构建，所以没有 `charge` 列的 frame 不会发送它。缺的是"列存在但这次运行不需要上线"。

**机制已经存在，不要发明第三个。** `Block::resolve_columns(unsigned required_flags)`（`block.h:95-96`）就是现成的按位列选择器，由 `make_plan`（`atom_pack.cpp:122-137`）唯一调用。per-run 裁剪就是给它加第二个平铺的 `unsigned drop_flags`，并在 `atom_pack.h:45-46` 加一个 `make_border_plan(Frame&, unsigned drop_flags)` 重载。"哪些订阅者需要哪些列"的并集算法照抄 `mask_union_all`（`src/cpu/asynchronous/snapshot_publisher.cpp:29`）的**算法形状**——但**不要**移植 `CaptureMask`（`frame_snapshot.h:28`），它是五个 bool 的硬编码结构体、与 `ATV_ATOM_FIELDS` 脱钩，正是约束 (a) 要挡的形状。

**指纹交互——这是本项的主要未知数。** per-run mask 会改变 `plan.fingerprint`（`atom_pack.cpp:143`），而 `CommBrick` 集合锚定三个指纹（`comm_brick.h:87,:88,:92`）之后只做本地比较。因此 mask 必须在 `setup()` 的锚定点（`comm_brick.cpp:1037`）**之前**、以在每个 rank 上完全相同的方式导出，否则 `require_collective_halo_schema`（`:518`）会触发。导出函数必须是 rank 一致输入的纯函数——与 `derive_uniform_rounds` 在 `halo.h:266-270` 记录的纪律相同。**先测量再决定是否值得**：当前 vx/vy/vz + mass 是 32 B/ghost，在 3151 ghost × 231 次 borders 下约占 borders 的 3.5%——**如果测量结果确认收益低于 2%，本项降级为只写一条注释说明为何 border 比 LAMMPS 宽，不实现开关**。不要为了对齐 LAMMPS 而引入一个没人打开的旋钮。

**(3) `cpu::Comm` 基类去留——结论是留，但要改注释。**

我在评审里说它是"空壳、单实现"。耦合扫描纠正了这一点：`tests/cpu/driver/test_verlet_comm.cpp:21` 的 `SpyComm` 是第二个实现，接口买到了可测试性——`comm-maint-06-transport` 的时序断言正依赖它。LAMMPS 的 `Comm` 基类（175h/1245cpp）装的是 topology + policy + `ring()`/`rendezvous()`，**不是**共享的 swap 逻辑；`CommTiled` 是 `CommBrick` 的兄弟而非子类，代价是每个实现约 500 行逐字重复的 wrapper。所以"照抄 LAMMPS 的两兄弟形状"并不是自动正确的目标。

**折叠的真实代价不是虚表，是头文件。** `CommBrick` 的构造函数是私有的、只 friend `DomainRuntime`（`comm_brick.h:50,:75`），所以 `Comm` 是消费者端唯一**可拼写**的类型。折叠意味着 `ctx.md.comm` 变成 `comm::CommBrick*`，于是每个碰它的 TU 传递性地吃进 `halo.h`（25 KB，且它 include `domain.h`）。

**做法**：保留 `Comm`，在头注释里写明它现在的职责边界（契约 + 测试替身接缝，点名 `SpyComm`），以及若将来加第二个真实实现，共享逻辑应该往哪放。不做 `CommTiled` 占位。

**(4) CUDA 同名类型。**

`cuda/communication/domain_decomposition.cuh:20` 的 `DomainDecomposition` 语义是"单进程 3D 空间切分"，与 `cpu::md::mpi::DomainDecomposition`（MPI 分解）同名不同义。改名（例如 `SpatialGrid`），一并更新 `.claude/notes/architecture.md`。

## Files

- `include/atomiverse/core/frame_schema.h` — `kColPbcShift` 标志；`static_assert`；修正 `:76-77` 注释
- `src/cpu/communication/atom_pack.cpp` — 三处 `wire_id <= z` 改为查标志
- `include/atomiverse/cpu/communication/comm.h` — 基类职责注释；（若实现）border 线宽开关
- `include/atomiverse/cuda/communication/domain_decomposition.cuh` / `src/cuda/communication/domain_decomposition.cu` — 改名
- `.claude/notes/architecture.md`

## Tasks

- [ ] Add an explicit per-field flag marking the columns that take a PBC shift, and replace all three wire_id <= z comparisons with it
- [ ] Add a static_assert pinning AtomFieldId::x/y/z to 0/1/2 so a table reorder fails at compile time
- [ ] Correct the frame_schema.h comment that claims wire_id is never derived from a container index
- [ ] Add a test that inserting a non-coordinate column ahead of z no longer causes a silent shift
- [ ] Measure what the border wire costs by shipping vx/vy/vz and mass at the bench geometry; implement the narrowing only if it is worth more than 2% of borders, otherwise document why the wire is wider than LAMMPS
- [ ] If implementing: add a flat `unsigned drop_flags` to Block::resolve_columns and a make_border_plan overload; derive the mask by union-over-subscribers copying mask_union_all's algorithm, not CaptureMask's type
- [ ] If implementing: derive the mask before Comm::setup anchors the fingerprints, as a pure function of rank-consistent inputs, and add a test that all ranks agree
- [ ] Rewrite the cpu::Comm header comment to state its real job (contract plus test-double seam) and where shared logic would go if a second real implementation arrives
- [ ] Rename the CUDA DomainDecomposition to a name that does not collide with the MPI one, and update architecture.md

## Testing

`wire_id` 陷阱：新增一个单元测试，用 `Block::insert` 在表序之外造一个 `Storage` 运行时列并断言它**不**收到盒长位移；再加编译期的 `static_assert`。border mask：若实现，需要一个"关掉 ghost 速度后能量与力逐位不变、且线宽下降"的测试；若不实现，需要在报告里给出测量数字。改名是纯机械改动，判据是 CUDA 构建通过（GH200 批处理）。全程 `ctest` 全绿 + `widebox_parity.sh` dev=0.0。

## Out of scope

- 不实现 `CommTiled`。
- 不改声明式字段表的形状——它是本层做得对的部分。
- 不动 CUDA 多 GPU halo 的算法（按 mandate 它是 OpenMM 引擎，不受 LAMMPS 移植约束）。
