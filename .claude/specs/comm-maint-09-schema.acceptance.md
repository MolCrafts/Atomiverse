---
spec: comm-maint-09-schema
created: 2026-08-27
criteria:
  - id: ac-001
    summary: components declare, the engine collects
    type: code
    pass_when: |
      A FieldUser protocol exists with ghost_reads / ghost_writes, probed
      through a Component accessor in the same shape as
      as_neighbor_consumer; Runtime::build folds every mounted component's
      declaration into one comm::Schema per boundary and publishes it.
    status: pending
  - id: ac-002
    summary: an undeclared column does not cross the boundary
    type: code
    pass_when: |
      A test mounts a component set that declares no ghost read of a given
      column and asserts that column is absent from the collected schema
      and absent from the packed payload; adding one component that
      declares it makes it appear exactly once.
    status: pending
  - id: ac-003
    summary: the base set is explicit
    type: code
    pass_when: |
      The columns the halo needs regardless of any component (x/y/z
      forward, f reverse, tag and type on border) are named in one place
      with a comment citing LAMMPS default_comm / default_border, not
      scattered as special cases in the packers.
    status: pending
  - id: ac-004
    summary: every rank collects the same schema
    type: code
    pass_when: |
      A multi-rank test asserts the collected schema and its collective
      fingerprint are identical on every rank.
    status: pending
  - id: ac-005
    summary: the plan staleness surface is gone
    type: code
    pass_when: |
      No type in the communication layer exposes frame_owner, owner,
      frame_generation, block_generation, valid_for or a writable flag as
      caller-visible state; no call site rebuilds a plan on a staleness
      test. Nothing is named PackPlan or Wire.
    status: pending
  - id: ac-006
    summary: the derived flags are deleted
    type: code
    pass_when: |
      kColComm, kColBorder and kColReverse no longer exist in
      core/frame_schema.h; kColExchange, kColMigrate and kColState remain
      and still drive migration.
    status: pending
  - id: ac-007
    summary: numbers unchanged, and the wire width is reported
    type: code
    pass_when: |
      ctest green; np=16 MIL0PEGM18 single point reports
      E=1725.258348 pair=-200.208398 eV; widebox_parity.sh dev=0.0 at
      NP=256 for DIMS=8,8,4 and DIMS=16,4,4. The report states the border
      payload in bytes per ghost before and after, and if it did not
      shrink, says which component declares each surviving column.
    status: pending
