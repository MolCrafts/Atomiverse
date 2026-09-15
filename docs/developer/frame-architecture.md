# The Frame: Blocks, Typed Meta, and Box

`cpu::Frame` is the only public token in Atomiverse: every method takes one and
writes one back. This page describes what it physically is, and why reading a
column costs the same as reading a raw array.

The representation is final. There is no parallel storage, no proxy view, no
alias for an older member name, no dual read/write path and no fallback
decoder.

```cpp
class cpu::Frame {
  public:
    BlockMap blocks;
    MetaMap  meta;
    cpu::Box box;
};
```

Those three members are the whole representation. The CUDA frame is switched
separately; that work does not change the CPU API and cannot ask the CPU side
to keep a transitional surface alive.

## 1. Two levels of mapping

`blocks` maps `string -> Block`, and each `Block` maps `string -> Column` in a
deterministic order. All columns of one Block share a row count; different
Blocks need not share an axis-0 length. That is what lets atoms, grids and
method outputs each carry the row semantics that suit them.

Standard atomic state lives in the `"atoms"` Block:

- positions, velocities, masses and charges use `StorageReal`;
- forces use `AccReal`, which is always `double`;
- types and topology counts use native `int`;
- global identity uses the configurable `TagInt`;
- images use three native `int8_t` columns;
- bonds, angles, dihedrals and special lists use per-row variable-width
  topology columns;
- runtime columns are inserted straight into the Block, carrying their own
  dtype, width, group and communication semantics.

A CPU `Column` uniformly owns a two-dimensional `[rows, cols]` xtensor. Scalar
columns have `cols == 1`; fixed vectors and topology strides use the same
representation. The storage variant keeps the real types — `StorageReal`,
`AccReal`, `TagInt`, `int`, `int8_t` — and never funnels a generic get/set
through `double`.

`core/frame_schema.h` is the single semantic manifest for the standard atomic
columns. It fixes the explicit wire ID, dtype, stride, topology group and the
grow / migrate / exchange / halo / write-back flags. The Block owns its columns
and its capacity; the Frame keeps no second list of named fields that could
drift out of sync with it.

## 2. Typed access and the resolution boundary

Calling code resolves a contiguous span through a compile-time typed key:

```cpp
auto x       = frame.span(kAtom_x);
auto force_x = frame.span(kAtom_fx);
auto tags    = frame.span(kAtom_tag);

StorageReal* px  = x.data();
AccReal*     pfx = force_x.data();
for (int i = 0; i < frame.nlocal(); ++i)
    pfx[i] += static_cast<AccReal>(px[i]);
```

Resolution happens at an explicit boundary — a component `launch`, a
communication plan build, or a similar seam. The inner particle, pair and
topology loops only ever receive already-resolved spans and pointers; they
never look up a string key. The physical storage stays a mapping while the
generated hot loop stays ordinary SoA pointer arithmetic.

A Block's *structure generation* increments when a column is inserted or
removed, when capacity changes, and when topology is re-strided. `Span` and
`ColumnRef` capture that generation, so using a stale handle after a structural
change throws `stale_handle_error` instead of touching freed memory. A
long-lived object may keep a typed key or a plan's logical description; it must
not cache a raw pointer across a structural change.

The Frame's *resolution window* counts Block and meta resolutions, so a test
can assert that the resolution count grows with the number of columns rather
than with the number of atoms. It is a diagnostic channel and never enters a
per-particle loop.

## 3. One typed meta, and only one

Every piece of non-column state is owned by `meta`. Standard keys are:

- backend-local: `nlocal`, `nghost`, `nmax`, topology strides, `has_bonded`,
  `next_tag`, and the tag-to-row index;
- serializable: step, the energy components, kinetic, conserved, stress and
  kinetic stress;
- runtime serializable typed keys.

Access is typed the same way:

```cpp
const double energy = frame.meta.get(kEnergy);
frame.meta.set(kStep, std::int64_t{42});
auto stress = frame.meta.ref(kStress);
```

The serializable schema uses the same 19 explicit dtype IDs as molrs: scalar
Bool / I32 / I64 / U32 / U64 / F32 / F64 / String, plus the prescribed array
types of length 3, 6 and 9. The tag-to-row index is a CPU-local 255th kind: it
can only be rebuilt, and never crosses an FFI, an MPI Frame broadcast, or a
file boundary.

`MetaRef` also captures the meta generation. Inserting a runtime key
invalidates older references; modifying an existing value does not change the
schema. Boundary code works from a key-sorted serializable plan and its
fingerprint — it never stringifies a typed value and never accepts the old
string-meta format.

## 4. Box owns the cell

`box` is the Frame's only cell. It owns the cell matrix together with the
`inv_cell`, `inv_L`, `half_L` and orthogonality caches, which are refreshed in
the same call:

```cpp
frame.box = cpu::Box::cubic(40.0);
frame.box.set_cell(cell);
const double lx     = frame.box.length(0);
const double volume = frame.box.volume();
```

Every cell update goes through `Box`, which is what keeps those caches
coherent. The Frame offers no second cell member, and `Box` offers no
element-wise or assignment operators that would let a caller treat a `Mat3` as
a legacy aggregate.

## 5. Structural operations walk the schema

Capacity, row movement and topology operations live on Block and Frame:

- `grow` / `set_capacity` resize every atoms column;
- `copy_row` moves the standard and runtime columns declared as migrating;
- `reserve_topology` / `grow_group` re-stride counted topology uniformly;
- `clone_structure(rows)` copies the full Block/Column schema, the runtime
  columns, the serializable meta schema and values, and the box — while
  rebuilding the backend-local counts and index;
- `memory_bytes` is computed from actual Block / Column / meta ownership.

A new column is registered through
`frame.block("atoms").insert(key, ColumnSpec{...})`. The flags on `ColumnSpec`
state explicitly whether it participates in migrate, exchange, border, comm or
write-back. There is no second, hand-copied list of fields.

## 6. MPI moves the native schema

The communication path first compiles a Block into a narrow `PackPlan`, then
performs native byte copies row by row:

```cpp
auto plan = cpu::comm::make_exchange_plan(frame);
cpu::comm::require_collective_schema(plan, communicator, "migration");
const std::size_t written = cpu::comm::pack_exchange(
    frame, plan, row, bytes, capacity);
```

The wire fingerprint covers the current wire version, the sorted key and wire
IDs, dtypes, native element widths, strides and groups, and the semantic flags.
The insertion order of runtime columns does not affect wire order; a genuine
schema difference fails collectively *before* any payload is exchanged.

The `comm`, `border` and `exchange` plans all preserve the configured native
width. In particular `StorageReal = float`, `smallbig TagInt = int32`, and
64-bit tags above `2^53` never pass through a `double`. An exchange row carries
a count-driven variable-length topology payload; unpacking scans the counts
first, re-strides at most once when it has to, and refreshes the
generation-bound plan.

Whole-Frame broadcast and gather use that same exchange schema plus an explicit
typed-meta and cell protocol. Sending a C++ Frame, Block, Box or variant object
as raw memory is forbidden.

## 7. The molrs boundary

`from_molrs()` is the one entry point from `molrs::Frame` to `cpu::Frame`, and
`to_molrs()` is the one exit. Both read and write Block columns and exact typed
meta directly: there is no intermediate SystemScalars Block, no string
key/value metadata, and no legacy decoder.

The division of labour:

- molrs flat bonds / angles / dihedrals become CPU per-row global-tag topology
  during `from_molrs`;
- element identity has exactly one public owner, `molrs::Element`. Atomiverse
  declares no alias, no second enum and no implicit conversion. The ExtXYZ
  entry point consumes the `species` column that format prescribes, produces
  `atoms.type` through the canonical `molrs::Element`, drops the format column,
  and only then enters the strict `from_molrs`. The general boundary does no
  `type` / `element` / `species` fallback;
- write-back writes the result columns and serializable meta named in the
  current boundary contract, with an explicit range check on tag narrowing;
- CPU-local population, capacity, stride and the tag index are never
  serialized;
- the Box crosses as explicit cell data, never as copied C++ object memory.

molpy uses `molrs.Frame` / `molrs.Block` directly. Atomiverse, molrs and molpy
share one current format version; anything else is rejected rather than
implicitly migrated.

## 8. Performance discipline

Blocking the Frame does not change the computational layout a kernel sees: the
CPU side is still contiguous SoA, and the xsimd / OpenMP inner loops still
receive raw pointers. The only performance risk is resolving the mapping
repeatedly *inside* an inner loop, so acceptance watches:

- the number of resolutions per `launch` is O(1) in N and in pair count;
- steady-state MD, neighbor and communication introduce no allocation
  attributable to the Frame representation;
- LJ neighbor reuse and forced re-neighboring are each compared against the
  cutover baseline;
- `from_molrs`, write-back, pack/unpack, broadcast, halo, migration and gather
  are measured separately;
- the `StorageReal` / `WorkReal` / `TagInt` configuration matrix round-trips
  bit-exactly.

An early 1M-row microbenchmark measured "resolve once in `launch`, then use
pointers" at the same speed as direct SoA pointers. A single string resolution
costs on the order of a nanosecond; repeating that map lookup per row degrades
by orders of magnitude. Resolving at a typed boundary is therefore an
architectural constraint, not a style preference.

## 9. What the CPU surface guarantees

The source and compile-negative gates hold at least the following:

- `cpu::Frame` has exactly three physical members: `blocks`, `meta`, `box`;
- there is no separate aggregate atom storage and no SystemScalars mirror;
- there are no legacy Frame members, Box operators, forwarding inserts or proxy
  macros;
- CPU production code, tests, regressions, examples, tools, the app, the Python
  layer and the documentation teach only `span(ColKey)`, typed meta and `box`;
- no MPI payload funnels through `double`, and runtime columns and topology
  take part in the schema plan;
- molrs typed-meta IDs and versions match the CXX boundary exactly;
- CUDA remains explicitly deferred, and the CPU keeps no compatibility surface
  on its behalf.
