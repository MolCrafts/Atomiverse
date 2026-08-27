---
title: One wire layout per direction, and a symmetric predicate
status: code-complete
created: 2026-08-27
revised: 2026-08-27
chain: comm-maint
---

# One wire layout per direction, and a symmetric predicate

## Summary

**This spec was rewritten after implementation contradicted its premise.** It
originally said the six `if (plan_is_*) { fast } else { generic }` branches had
six dead generic arms to delete. Two of the three wires turned out to be the
opposite: the generic arm is the live, general implementation and the fast arm
is a shape specialization. Only the **border** fast arm was genuinely dead. The
real defect was never the dead code — it was (a) a comment asserting the two
layouts were byte-identical when they are field-major vs row-major, and (b) a
**pack/unpack predicate asymmetry** that could route the two ends down
different layouts on the same message.

## Domain basis

不声明新物理。判据是既有数值逐位不变，加上"pack 与 unpack 不可能选到不同布局"。

## What the tree actually does

Measured, not inferred — each of these was found by deleting the arm and
watching what broke:

| wire | predicate | which arm runs in production | evidence |
| --- | --- | --- | --- |
| comm | `plan_is_xyz_comm` | **both** — fast for x/y/z, generic for plugin runtime columns | `tests/mpi/test_halo_exchange.cpp:163-167` inserts a **2-component** `kColComm` runtime column; deleting the generic arm timed the test out |
| border | `plan_is_scalar_border` | **generic only** | `image` is `Fixed3` (`frame_schema.h:124`) and carries `kColBorder`, so the predicate is false on every production Frame; deleting the generic arm broke EAM/SW/deform multirank |
| reverse | `plan_is_fxyz_reverse` | **both** — fast for f, generic for plugin runtime columns | `tests/mpi/test_halo_exchange.cpp:267` inserts a `kColReverse` runtime column |

The plugin-declared runtime column is a supported capability, not an accident:
`Block::insert` with the right flag bits puts a column on a wire with no
comm-layer edit at all. The generic arms are what carry it.

## The actual defect, and the fix

**(1) The layouts differ and the comment said they did not.** `atom_pack.cpp`
documented the fast path as "byte-identical to the generic path (row-major,
fields in plan order)". It is field-major. Both arms produce the same *total*
size, so the length check passes either way — which is what made the
divergence survivable and invisible.

**(2) The predicate was asymmetric.** pack tested `plan_is_*(plan)`; unpack
tested `plan_is_*(plan) && plan.writable`. A conforming but non-writable plan
therefore packed field-major and unpacked row-major: same byte count, different
meaning, no throw. Production only avoided it by accident — `comm_brick.cpp`
rebuilds the plan when `!writable`.

**Fix, as landed:**

- Writability is hoisted to a **precondition of both arms**, checked before
  either is chosen, with a how-to message. The two sides now evaluate the
  identical shape predicate, so they cannot disagree.
- The **border** fast arm — the one genuinely dead — is deleted, along with
  `plan_is_scalar_border` and `gather_column`. Border has one layout, row-major,
  and a comment saying why.
- The false byte-identity claims are replaced with an explicit statement of
  which arm is field-major, which is row-major, and why both exist.

## Files

- `src/cpu/communication/atom_pack.cpp`
- `include/atomiverse/cpu/communication/atom_pack.h`

## Tasks

- [x] Determine empirically which arm each wire actually takes
- [x] Hoist writability to a precondition of both arms on every unpack
- [x] Make pack and unpack evaluate the identical shape predicate
- [x] Delete the border fast arm, `plan_is_scalar_border`, `gather_column`
- [x] Replace the false byte-identical comments with the real layout contract
- [x] Add unit coverage for the generic arms —
      `tests/cpu/communication/test_atom_pack_generic_wire.cpp`, verified to
      bite: an off-by-one in the comm generic unpack fails it directly and
      serially (previously the same defect surfaced only as an MPI timeout)

## Testing

239/239 `ctest` green; np=16 MIL0PEGM18 single point bit-identical
(`E=1725.258348 pair=-200.208398 eV`) before and after. The remaining task is
the unit coverage: a runtime multi-component column on each wire, packed and
unpacked in-process, so the generic arms are pinned without needing MPI.

## Out of scope

- 不把快路径推广到多分量列——那是有需求时的功能 spec。
- 不改 `pack_exchange` / `unpack_exchange`（变长拓扑行，本来就单路径）。

## Retrospective

The premise came from two independent deep scans plus my own reading, and all
three agreed the generic arms were dead. They were wrong in the same way: each
reasoned from the *canonical* Frame schema, where the predicates do hold for
comm and reverse, and none checked whether a Frame can carry extra columns —
which is exactly the capability the generic arms exist for. The thing that
caught it was deleting the code and running the suite. Worth remembering the
next time a spec's central claim is "this is never executed": the cheap proof
is to remove it and see what screams, not to grep harder.
