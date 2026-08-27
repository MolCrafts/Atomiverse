---
title: Components declare their fields; build collects the schema
status: code-complete
created: 2026-08-27
chain: comm-maint
supersedes: comm-maint-08-wireplan (naming + the passed type), absorbs comm-maint-07-wire border mask
---

# Components declare their fields; the Driver holds the system Schema

## Summary

操作者 2026-08-27:*"每个 component 声明自己需要什么字段(atom 的、bond 的等等),然后 compile 的时候收集起来形成一个 schema。然后不要 wire、plan 这种语意不明的东西,直接传 schema"*。

**把声明方向反过来。** 现在 `core/frame_schema.h` 的全局表由**字段**声明自己上哪条线(`x` 那行带 `kColComm`);改成由**消费者**声明自己在 ghost 行上要读什么,`build()` 时取并集,那个并集就是这次运行的 schema。

先例已经在树里:`Runtime::cache_verlet_comm_`(`runtime.cpp:269-287`)已经遍历每个 component 对 `NeighborConsumer::halo_cutoff()` **取最大值**。本步是把同一个模式从一个标量推广到一组字段。

## Domain basis

不声明新物理。schema 决定哪些列跨边界,不改变任何被计算的量——判据是能量与力逐位不变,以及"没有 component 读的列不再上线"。

## 为什么这比现状好

现状 `kColBorder` 是**全局的、对"谁可能需要"的猜测**。后果是可测的:border 每个 ghost 带 x,y,z,vx,vy,vz,mass,charge,type,tag,image = 79 B,而 LAMMPS 的 `default_border` 是 `{"id","type","mask","x"}` + per-style 附加,速度要 `comm_modify vel` 才上线,per-type 的质量根本不上线。多出的 32 B/ghost 不是因为有人要读,是因为**没有地方表达"没人读"**。

反过来声明之后,这件事不再需要一个旋钮(原 `comm-maint-07-wire` 的 border mask 任务),它是默认行为:没有 component 声明读 ghost 上的 `mass`,`mass` 就不在 border schema 里。

## Design

**名字与归属(操作者 2026-08-27 二次修正)。** `Schema` **不是 comm 的东西**,是系统级的——这个化学模型/体系由什么字段构成,是 Driver 的成员。所以它落在 `core/schema.h`(`ATV::Schema`),不是 `cpu/communication/`。`PackPlan` / `Wire` / `comm::Schema` 都被它取代。

**Schema 与 Frame 的关系是校验,不是构造。** Frame 是一个 **mapping**——它带着来源给它的东西到达(数据文件、`Block::insert`、另一个 rank),没有任何东西"从 schema 构造 Frame"。Schema 是挂载的 component 声明的并集,`check_schema(required, frame, who)` 检查 Frame 是否齐备,缺了就抛带 how-to 的异常。**这正好把 CLAUDE.md 第 4 条铁律(必需字段缺失就抛)从各组件各自为政变成系统性的一条。**

**API 形状照 pydantic(操作者 2026-08-27 三次修正)。** 借的是三件具体的事,每一件都换来一个原来表达不了的检查:

  * **字段带类型声明**,不是裸名字 —— `require(kAtom_charge, "CoulLong")` 或
    `require(name, dtype, cols, by)`。于是"字段在但类型/宽度不对"能被发现;裸名
    单看不见它,而它正是会静默污染线上的那一类。
  * **`model_validate` 形状:报告全部问题,不是第一个** ——
    `validate_schema(required, frame)` 返回 `vector<SchemaError>`。一次 build
    修一个错是 schema 变得讨厌的原因。
  * **每个错误带 loc** —— 这里的 loc 是 `required_by`,哪个 component 要的。
    于是消息从"缺字段"变成"PairDpd 要这个,而 Frame 里没有"。

外加一条 pydantic 式的性质:**两个 component 对同一字段声明了不同类型**是模型内部
矛盾,在声明期由 `conflicts()` 记录,不静默合并。

**协议撤销。** 原设计里的 `FieldUser` 协议 + `as_field_user()` 访问器已删除:这里没有"user",而且 `Component::default_bindings()` 已经是"组件声明、driver 收集"的现成先例。改为 `Component::declare_fields(Schema&)` / `declare_reverse_fields(Schema&)` 两个普通虚函数,与 `default_bindings()` 并列。

**协议。** 与 `NeighborConsumer` 同形,由引擎 probe 的 opt-in 协议,`Component` 上一个访问器:

```cpp
class FieldUser {
  public:
    virtual ~FieldUser() = default;
    /// Columns this component READS on ghost rows every step.
    virtual void ghost_reads(comm::Schema& out) const {}
    /// Columns this component ACCUMULATES onto ghost rows, to be folded
    /// back onto the owners.
    virtual void ghost_writes(comm::Schema& out) const {}
};
```

三个动词里只留两个:forward(读)与 reverse(写回)。**exchange 不由 component 声明**——迁移必须携带重建一个原子所需的一切,那是 Frame 的属性不是消费者的属性,继续由 `kColExchange` / `kColState` 驱动。这条线是本设计的分界:**exchange 关乎原子的身份,comm/border/reverse 关乎消费者读什么。**

**基础集。** 有些列不属于任何 component,是 halo 自己的需要:forward 必须带 `x,y,z`(ghost 的位置就是 halo 的产品),reverse 必须带 `fx,fy,fz`,border 必须带 `tag`(身份)与 `type`。这与 LAMMPS 的 `default_comm = {"x"}` / `default_border = {"id","type","mask","x"}`(`atom_vec.cpp:36,39`)是同一个概念,必须显式写出而不是隐含。

**收集点。** `build()`——CLAUDE.md 的 "one-time alloc" 阶段,`Runtime` 已经在那里遍历 component。

**列名而不是位。** `Schema` 持列名(`std::string_view`),不是 `AtomFieldId` 的位集:插件用 `Block::insert` 加的运行时列必须能被声明(`tests/mpi/test_halo_exchange.cpp` 就有一个 2 分量的 `halo_aux`),位集表达不了。这也与 LAMMPS 的 `fields_comm = {"q"}` 一致。

**每个 rank 得到同一个 schema。** 并集来自同一个 Plan 上挂载的同一批 component,顺序是 Plan 顺序(确定的),所以按构造一致——与 `derive_uniform_rounds` 的论证同形。既有的集合指纹校验(`require_collective_plan`)继续作为兜底。

**指针解析与 schema 分离。** 这是 `PackPlan` 现在把两件事缠在一起的地方:*哪些列上线*(build 时决定,此后不变)与 *它们此刻在哪*(`grow()` 一动就失效)。分开之后,`Schema` 是不可变的、可以放心传递的;指针解析退化成 pack 调用内部的一步,`frame_owner` / `owner` / 两个 generation / `valid_for` / `writable` 以及 8 处调用方重建**全部消失**。

**`kColComm` / `kColBorder` / `kColReverse` 的去向。** 三个标志位被推导取代,从字段表删除。`kColExchange` / `kColMigrate` / `kColState` 保留(迁移仍是 Frame 的事)。`kColWriteBack` 是另一件事(molrs 写回),不动。

## 已测到的证据(先于实现)

**ghost 速度只有一个真实消费者,而且不在 bench 里。** `pair_dpd.cpp:147-149` 读 `vx[j] - vx[i]`,`j` 是邻居索引——在分解运行里就是 ghost,所以 DPD 确实需要 ghost 速度。除它之外没有 pair style 读 `v[j]`。MIL0PEGM18 bench 挂的是 LJ + CoulLong + bonded + PPPM + NoseHoover,**没有 DPD**,所以这次运行根本不需要 ghost 速度——却每步每 ghost 带着 24 B。

**ghost 质量找不到消费者。** `virial_stress.cpp` 读 `mass` 与 `v` 是为动能项,那是 owner-only 的求和,不碰 ghost 行。没找到任何 pair/bonded 内核索引 `mass[j]`。

所以对 bench 这个 component 集,border 应从 79 B/ghost 降到 **47 B**(去掉 vx,vy,vz,mass),约 −40%;而对一个 DPD 运行它自己就会把速度加回来。这正是"派生而非配置"要买的东西——**上线的字段成为这次运行的推导结果,不是全局猜测**。实现时以实测复核,不以此为准。

## 已知的开放问题

1. **谁声明 `charge`?** `CoulLong` 在 ghost 上读电荷,应由它声明。但 `ChargePacker` 走的是自发起的 `forward_comm(Packer&)` 旁路,不经 border——两条路的关系要在实现时确定,不要让电荷既在 border schema 里又被 packer 重发一遍。
2. **bonded 字段。** 操作者点名"atom 的、bond 的等等"。bonded 元组的解析走 `tag_to_local`,ghost 上需要的是 `tag`;`num_bond`/`bond_atom` 只在 exchange 线上。实现时确认 bonded component 在 ghost 上到底读什么,不要凭直觉给它们加列。
3. **`pair_dpd` 是本设计的判例。** 它必须声明 `ghost_reads = {vx,vy,vz}`,而且要有一个测试证明:挂上 DPD 速度回到 border schema,不挂就不在。没有这个测试,这次改动等于把 24 B 悄悄删掉并赌没人用。

## Files

- `include/atomiverse/cpu/communication/schema.h`(新)— `comm::Schema`
- `include/atomiverse/cpu/component/field_user.h`(新)— 协议
- `include/atomiverse/cpu/component/component.h` — 访问器
- `src/cpu/driver/runtime.cpp` — build 时收集,发布到 `ctx.md`
- `include/atomiverse/core/frame_schema.h` — 删三个标志位
- `atom_pack.{h,cpp}`、`comm_brick.cpp` — 改吃 `Schema`
- 每个在 ghost 上读数据的 component — 声明

## Tasks

- [x] Add ATV::Schema in core/ (field names, idempotent add, deterministic order)
- [x] Add Component::declare_fields / declare_reverse_fields, shaped like default_bindings; no FieldUser protocol, no accessor
- [x] Add check_schema(required, frame, who) — the Schema validates the Frame, throwing with a how-to
- [x] Collect the union in build(), beside the existing halo_cutoff fold; publish it
- [x] State the base set explicitly and document it against LAMMPS default_comm / default_border
- [~] Declare requirements — DONE for the three measured cases (`PairDpd` velocities, `CoulLong` and `PairLJCutCoulLong` charge). REMAINING: sweep every other component that indexes a neighbour row, and resolve open question 1 (ChargePacker vs border) and 2 (bonded)
- [ ] Rewire atom_pack and comm_brick onto Schema; delete PackPlan's staleness surface
- [ ] Delete kColComm / kColBorder / kColReverse; keep kColExchange / kColMigrate / kColState
- [ ] Report the border wire width before and after

## Testing

行为判据是**逐位不变**:`ctest` 全绿,np=16 MIL0PEGM18 单点 `E=1725.258348 pair=-200.208398 eV`,`widebox_parity.sh` 8,8,4 与 16,4,4 dev=0.0。

新增覆盖:一个 component 声明一个列 → 该列出现在 schema 里;没人声明 → 不出现;两个 component 声明同一列 → 只出现一次;每个 rank 得到相同 schema(多 rank 测试)。

**并且必须报告 border 线宽的前后对比**——如果并集后仍是 79 B,说明确实有人在读那些列,那这次改动买到的是"知道为什么"而不是省字节,要如实说。

## Out of scope

- 不引入 atom style。ATV 没有这个概念,Frame 的列集是**实例**属性(`Block::insert` 运行时插列),声明无处安放。
- 不动 exchange 线的字段来源。
- 不动 `ColumnRef` 与 `ATV_ATOM_FIELDS` 表本身(只删三个标志位)。

## Progress 2026-08-27 — foundation landed, rewire outstanding

Landed: `comm::Schema` (`communication/schema.h`), the `FieldUser` protocol
and its `Component::as_field_user()` accessor, the base sets, the fold in
`Runtime::cache_verlet_comm_` beside the existing `halo_cutoff()` fold, and
publication as `ctx.md.ghost_reads` / `ghost_writes` (non-owning pointers, the
`ctx.md.pair_list` precedent — a `Schema` holds a vector so it cannot sit by
value in the trivially-copyable slice).

Declared so far, each from a read line rather than a guess:
`PairDpd` → `vx,vy,vz` (`pair_dpd.cpp:147`, `v[j] - v[i]`);
`CoulLong` and `PairLJCutCoulLong` → `charge` (`pair_lj_cut_coul_long.cpp:493`,
`q_ptr[j]`). `mass` has no declarer because no kernel indexes `mass[j]`.

`tests/cpu/communication/test_comm_schema.cpp`, 8 tests, pins that the wire
follows the mounted set: Coulomb-only carries charge and no velocity, adding
DPD adds exactly three columns, neither carries mass.

**Not yet done, and the layer still runs on the old flags:** `atom_pack` and
`comm_brick` have not been rewired onto `Schema`, so `kColComm`/`kColBorder`/
`kColReverse` are still what actually drives the wire and the border payload
is still 79 B/ghost. The collected schema is published but not yet consumed.
That rewire, plus deleting the three flags, is the rest of this spec. 242/242
green at this checkpoint.

## 2026-08-27 收尾 — 已落地

`ATV::Schema`(`core/schema.h`),pydantic 形状:`Field{name, dtype, cols, required_by}`、
`SchemaError{field, required_by, reason}`、`require()` / `require_all()` /
`fields()` / `conflicts()`;`cpu::validate_schema(schema, frame)` 收集全部问题,
`cpu::check_schema(...)` 把它们汇成一条带 how-to 的异常。校验已接在
`Runtime::build()` 的 schema 折叠之后。

`Component::declare_fields` / `declare_reverse_fields` 两个普通虚函数,与
`default_bindings()` 并列;`FieldUser` 协议与访问器已删除。

声明者(各来自一行读代码,不是猜测):`PairDpd` → `vx,vy,vz`
(`pair_dpd.cpp:147`);`CoulLong` / `PairLJCutCoulLong` → `charge`
(`pair_lj_cut_coul_long.cpp:493`)。`mass` 无人声明。

`tests/cpu/frame/test_schema.cpp` 15 个测试,含 pydantic 特有的四条:错类型被
发现、错宽度被发现、一次报告全部问题、两个 component 声明冲突被记录。

242/242 绿;np=16 MIL0PEGM18 单点 `E=1725.258348 pair=-200.208398 eV` 全程逐位不变。

## 2026-08-27 完成 — wire 已改吃 schema

`Block::resolve_columns(const Schema&)` 按声明顺序解析,`make_wire_plan(frame,
kind, schema)` 据此编译;`CommBrick` 在 `setup()` 缓存三条线并据此重建。
`make_comm_plan`/`make_border_plan`/`make_reverse_plan` 删除,
`kColComm`/`kColBorder`/`kColReverse` 一并删除。`kColExchange`/`kColMigrate`
保留——迁移携带的是"这个原子是什么",是 Frame 的属性,不是挂载集的属性;
这条线正是另外三个能删掉的理由。

**三条线,不是两条。** 实现中发现 `declare_fields` 表达不了"每步都变、需要刷新"
这件事:`test_mpi_halo_exchange` 在两次 `forward_comm` 之间修改插件列并期望被刷新,
把 forward 线钉死在坐标上之后它正确地失败了。补上 `declare_refresh_fields`——
LAMMPS 把同一区分保留为 `fields_comm` vs `fields_border`(`atom_vec.cpp:36-39`)。

**实测**:border 载荷 **79 → 44 B/ghost**(x,y,z 24 + tag 8 + type 4 + charge 8)。
去掉的是 ghost 速度(全树唯一消费者 `pair_dpd.cpp:147` 未挂载)、ghost 质量
(无消费者)、`image`(只在 `owned_count` 下被读,且仍走 exchange 线,迁移需要它)。
能量与力逐位不变(`E=1725.258348 pair=-200.208398 eV`),这就是那 35 字节确实
无人承重的证据。

**闸门(job 1753125)**:np=256 widebox parity 三个几何全部 dev=0.0 ——
8,8,4 @32³、8,8,4 @24³、16,4,4(hops 3,1,1)。242/242 单测绿。

**性能:未下结论。** 单次 np=256 得 1405 steps/s,而 wt10 三次是 1444/1448/1447。
分项里只有 `comm_borders` 朝预期方向明确移动(**0.134 → 0.104,−23%**,正是窄了的
border 线);`comm_forward`/`comm_reverse`/`comm_exchange`/`reduce` 都上浮 7–27%,
而这四项**本次一个字节都没改**——`reduce` 是纯 Allreduce,+27% 只可能是 skew。
n=1 不足以判 3% 的 loop 级移动,已提 3 次重复(job 1753630)。在那之前不宣称净
性能变化。
