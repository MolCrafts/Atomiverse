---
spec: comm-maint-08-wireplan
created: 2026-08-27
criteria:
  - id: ac-001
    summary: the wire plan owns its verbs
    type: code
    pass_when: |
      atom_pack.h declares pack, unpack, bytes, row_bytes, max_row_bytes,
      wire_fingerprint, pack_rows, accumulate, pack_row and unpack_row as
      members of the plan type, and declares no free function taking a plan
      as a parameter.
    status: verified
  - id: ac-002
    summary: no caller selects a kind-specific entry point
    type: code
    pass_when: |
      No call site outside atom_pack picks a pack/unpack function by wire;
      every one calls the plan and the plan dispatches. PackKind may still
      be named by the OWNER of a cached plan validating its own cache
      (comm_brick's three `plan.kind != PackKind::X` refresh guards) — that
      is the cache checking itself, not a caller choosing a function, and
      narrowing the criterion to forbid it would only push the same check
      behind a wrapper.
    status: verified
  - id: ac-003
    summary: the shift is one flat array, not three scalars and not a struct
    type: code
    pass_when: |
      the pack signature takes const double shift[3]; no Shift/Displacement
      aggregate was introduced.
    status: verified
  - id: ac-004
    summary: WITHDRAWN — the four switches are four different operations
    type: code
    pass_when: |
      Withdrawn during implementation. The criterion was written from a grep
      count; reading the sites shows a size table, a type predicate, a
      variant constructor and one typed dispatch. The only twin of the typed
      dispatch was deleted in comm-maint-03, so there is no duplication left
      to remove and a unifying visitor would add indirection.
    status: withdrawn
  - id: ac-005
    summary: the declarative field table is untouched
    type: code
    pass_when: |
      core/frame_schema.h ATV_ATOM_FIELDS and the kCol* bits are unchanged,
      and ColumnRef keeps its members; adding a communicated field is still
      a one-line edit to the table.
    status: verified
  - id: ac-006
    summary: numbers unchanged
    type: code
    pass_when: |
      ctest green; the np=16 MIL0PEGM18 single point reports
      E=1725.258348 pair=-200.208398 eV; widebox_parity.sh reports dev=0.0
      at NP=256 for DIMS=8,8,4 and DIMS=16,4,4.
    status: pending
