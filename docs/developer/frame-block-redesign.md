# CPU Frame 的 Block 终态架构

CPU Frame 已完成一次性 hard cut。当前且唯一的物理表示是：

```cpp
class cpu::Frame {
  public:
    BlockMap blocks;
    MetaMap meta;
    cpu::Box simbox;
};
```

这三个成员就是完整表示，不存在平行存储、代理 view、旧名称 alias、双读双写
或 fallback decoder。CUDA 和 Torch 的物理切换单独 deferred；它们不改变 CPU
的当前 API，也不能要求 CPU 保留过渡表面。

## 1. 两级映射

`blocks` 是 `string -> Block`，每个 `Block` 是确定性排序的
`string -> Column`。同一个 Block 的所有列共享行数；不同 Block 不要求共享
axis-0 长度。因此原子、网格和方法输出可以各自拥有合适的行语义。

当前标准原子状态位于 `"atoms"` Block：

- 坐标、速度、质量和电荷使用 `StorageReal`；
- 力使用恒为 double 的 `AccReal`；
- 类型和 topology count 使用原生 `int`；
- 全局标识使用可配置的 `TagInt`；
- image 使用三列原生 `int8_t`；
- bond、angle、dihedral 和 special-list 使用 per-row 变宽拓扑列；
- 运行时列直接插入 Block，并携带自己的 dtype、列宽、分组和通信语义。

CPU `Column` 统一拥有二维 `[rows, cols]` xtensor。标量列的 `cols == 1`；
固定向量和 topology stride 也用同一个表示。存储 variant 保留
`StorageReal`、`AccReal`、`TagInt`、`int` 和 `int8_t` 的真实类型，绝不通过
`double` 做通用 get/set。

`core/frame_schema.h` 是标准原子列的唯一语义清单。它规定显式 wire ID、
dtype、stride、topology group 和 grow/migrate/exchange/halo/write-back flags。
Block 自身拥有列和容量；Frame 没有另一份命名字段可与它失去同步。

## 2. 类型化访问和解析边界

业务代码通过编译期类型键解析连续 span：

```cpp
auto x = frame.span(kAtom_x);
auto force_x = frame.span(kAtom_fx);
auto tags = frame.span(kAtom_tag);

StorageReal* px = x.data();
AccReal* pfx = force_x.data();
for (int i = 0; i < frame.nlocal(); ++i)
    pfx[i] += static_cast<AccReal>(px[i]);
```

解析发生在 component launch、通信 plan 构建或其他明确边界。粒子、pair 和
topology 内层循环只接收已解析的 span/指针，不查字符串键。这样物理存储保持
mapping，同时生成的热循环仍是普通 SoA 指针算术。

Block 的结构 generation 在插入/删除列、改变容量和 topology re-stride 时
递增。`Span` 和 `ColumnRef` 捕获 generation；结构变化后使用旧 handle 会抛出
`stale_handle_error`，而不是访问悬空内存。长期对象可以保存类型键或 plan 的
逻辑描述，但不得跨结构变化缓存裸指针。

Frame 的 resolution window 统计 Block 和 meta 的解析次数，用于验证解析次数
随列数而不是随原子数增长。它是诊断通道，不进入每粒子循环。

## 3. 唯一的 typed meta

非列状态全部由 `meta` 拥有。标准键包括：

- backend-local：`nlocal`、`nghost`、`nmax`、topology stride、
  `has_bonded`、`next_tag` 和 tag-to-row index；
- serializable：step、各能量分量、kinetic、conserved、stress 和
  kinetic stress；
- 运行时 serializable typed key。

访问同样是类型化的：

```cpp
const double energy = frame.meta.get(kEnergy);
frame.meta.set(kStep, std::int64_t{42});
auto stress = frame.meta.ref(kStress);
```

可序列化 schema 与 molrs 使用相同的 19 个显式 dtype ID：标量
Bool/I32/I64/U32/U64/F32/F64/String，以及长度 3、6、9 的规定数组类型。
tag-to-row index 是 CPU 本地第 255 类，只能重建，不能跨 FFI、MPI Frame
broadcast 或文件边界。

`MetaRef` 也捕获 meta generation。插入运行时 key 会使旧引用失效；修改已存在
值不改变 schema。边界代码使用 key 排序的 serializable plan 和 fingerprint，
不会把 typed value 字符串化，也不会接受旧 string-meta 格式。

## 4. Box 是独立的 cell 所有者

`simbox` 是 Frame 唯一 cell。它拥有 cell matrix 以及同步更新的
`inv_cell`、`inv_L`、`half_L` 和 orthogonality cache。调用者使用当前 Box API：

```cpp
frame.simbox = cpu::Box::cubic(40.0);
frame.simbox.set_cell(cell);
const double lx = frame.simbox.length(0);
const double volume = frame.simbox.volume();
```

cell 更新必须经过 `Box`，从而同时刷新缓存。Frame 不提供另一个 cell 成员，
Box 也不提供把 `Mat3` 当作旧聚合对象操作的元素/赋值兼容运算符。

## 5. 结构操作只遍历 schema

容量、行移动和拓扑操作集中在 Block/Frame：

- `grow` / `set_capacity` 调整所有 atoms 列；
- `copy_row` 搬运声明为 migrate 的标准列和运行时列；
- `reserve_topology` / `grow_group` 统一 re-stride counted topology；
- `clone_structure(rows)` 复制完整 Block/Column schema、运行时列、
  serializable meta schema/value 和 simbox，但重建 backend-local counts/index；
- `memory_bytes` 从实际 Block/Column/meta 所有权计算。

新增列必须通过 `frame.block("atoms").insert(key, ColumnSpec{...})` 注册。
`ColumnSpec` 的 flags 明确规定它是否参与 migrate、exchange、border、comm 或
write-back；没有第二套手抄字段列表。

## 6. MPI 使用原生 schema plan

通信路径先把 Block 编译成窄 `PackPlan`，逐行只执行原生字节拷贝：

```cpp
auto plan = cpu::comm::make_exchange_plan(frame);
cpu::comm::require_collective_schema(plan, communicator, "migration");
const std::size_t written = cpu::comm::pack_exchange(
    frame, plan, row, bytes, capacity);
```

wire fingerprint 包含当前 wire version、排序后的 key/wire ID、dtype、原生元素
宽度、stride/group 和语义 flags。运行时列的插入顺序不影响 wire 顺序；真正的
schema 差异在 payload 交换前 collective fail。

`comm`、`border` 和 `exchange` plan 都保留配置后的原生宽度。尤其是
`StorageReal=float`、`smallbig TagInt=int32` 和大于 `2^53` 的 64-bit tag 不会
经过 `double`。Exchange row 使用 count 驱动的变长 topology payload，解包时
先扫描 count，必要时只 re-stride 一次并刷新 generation-bound plan。

全 Frame broadcast/gather 也使用同一 exchange schema，加上显式 typed meta 和
cell 协议；禁止 raw-memory 发送 C++ Frame、Block、Box 或 variant 对象。

## 7. molrs 边界

`materialize()` 是 `molrs::Frame -> cpu::Frame` 的唯一入口，`write_back()` 是
反向边界。两边都直接读写 Block 列和 exact typed meta；不存在临时
SystemScalars Block、string key/value metadata 或旧 decoder。

边界职责保持清楚：

- molrs flat bonds/angles/dihedrals 在 materialize 时转换成 CPU per-row
  global-tag topology；
- 元素身份只有 `molrs::Element` 一个公开 owner。Atomiverse 不声明别名、第二个
  enum 或隐式转换；ExtXYZ 专用入口消费格式规定的 `species`，用 canonical
  `molrs::Element` 生成 `atoms.type` 后删除格式列，再进入严格 materialize；
  通用边界不做 `type`/`element`/`species` fallback；
- write-back 写当前边界契约中的结果列和 serializable meta，tag narrowing
  做显式范围检查；
- CPU 本地 population、capacity、stride 和 tag index 永不序列化；
- Box 通过显式 cell 数据转换，不复制 C++ 对象内存。

molpy 直接使用 `molrs.Frame` / `molrs.Block`。Atomiverse、molrs 和 molpy 共享
一个当前格式版本；非当前版本必须拒绝，不能隐式迁移。

## 8. 性能纪律

Block 化不改变 kernel 的计算布局：CPU 仍是连续 SoA，xsimd/OpenMP 内层循环
仍接收裸指针。性能风险只来自错误地在内层重复解析 mapping，因此验收关注：

- launch 解析次数对 N 和 pair 数为 O(1)；
- 稳态 MD/neighbor/comm 不因 Frame 表示引入分配；
- LJ neighbor reuse 与 forced-renighbor 分别对比 cutover 基线；
- materialize、write-back、pack/unpack、broadcast、halo、migration 和 gather
  分项测量；
- `StorageReal`、`WorkReal` 和 `TagInt` 配置矩阵做位精确 round trip。

早期 1M-row 微基准测得“launch 一次解析后使用指针”和直接 SoA 指针同速；
每次字符串解析约为纳秒级，而在每行重复 map lookup 会造成数量级退化。因此
类型化边界解析是架构约束，不是代码风格偏好。

## 9. CPU 当前表面的验收

CPU hard cut 的 source/compile-negative gates 至少保证：

- `cpu::Frame` 只有 `blocks`、`meta`、`simbox` 三个物理成员；
- 没有独立的原子聚合存储或 SystemScalars 镜像；
- 没有旧 Frame 成员、Box 运算符、forwarding insert、proxy macro；
- CPU production、tests、regressions、examples、tools、app、Python 和文档只教授
  `span(ColKey)`、typed meta 与 `simbox`；
- MPI payload 没有 double funnel，runtime columns/topology 参与 schema plan；
- molrs typed meta ID/版本与 CXX 边界精确一致；
- CUDA/Torch 仍明确 deferred，且 CPU 不因它们保留任何兼容表面。

验收与删除清单见
`.claude/specs/frame-block-04-delete-compatibility.md`；该 spec 描述 hard-cut
合同，本文描述完成后的当前架构。
