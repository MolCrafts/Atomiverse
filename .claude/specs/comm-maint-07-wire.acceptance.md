---
spec: comm-maint-07-wire
created: 2026-08-27
criteria:
  - id: ac-001
    summary: coordinate columns are marked, not inferred from wire_id
    type: code
    pass_when: |
      core/frame_schema.h carries an explicit per-field flag for columns
      that take a PBC shift; atom_pack.cpp contains no comparison of
      wire_id against AtomFieldId::z; all three former sites consult the
      flag.
    status: pending
  - id: ac-002
    summary: a table reorder fails at compile time
    type: code
    pass_when: |
      a static_assert pins AtomFieldId::x, y and z to 0, 1 and 2, and
      moving any row ahead of x in ATV_ATOM_FIELDS fails the build
      rather than changing runtime behaviour.
    status: pending
  - id: ac-003
    summary: a new column cannot silently acquire a box shift
    type: code
    pass_when: |
      a unit test inserts a Storage runtime column carrying kColComm and
      asserts its ghost values equal the owner values exactly, with no
      box-length offset, under a swap whose pbc flag is non-zero.
    status: pending
  - id: ac-004
    summary: the wire_id documentation matches the code
    type: code
    pass_when: |
      the frame_schema.h comment no longer claims wire_id is never
      derived from a container index, or the code no longer derives it
      that way.
    status: pending
  - id: ac-005
    summary: the border wire is either trimmed or explained
    type: code
    pass_when: |
      either Comm exposes a flat per-run switch that omits ghost
      velocities, with a test showing energy and forces bit-identical
      and the measured wire narrower; or the report states the measured
      cost of shipping vx/vy/vz and mass and the header documents why
      the wire is wider than LAMMPS default_border. Not neither.
    status: pending
  - id: ac-005b
    summary: if a mask lands, every rank derives the same one
    type: code
    pass_when: |
      when the border narrowing is implemented, the mask is computed
      before Comm::setup anchors its fingerprints, from inputs that are
      identical on every rank; a multi-rank test asserts the anchored
      border fingerprint matches across ranks with the mask active. No
      CaptureMask-shaped struct was introduced; the selection rides the
      existing unsigned flag word.
    status: pending
  - id: ac-006
    summary: the Comm base states its real job
    type: code
    pass_when: |
      comm.h documents that the interface exists as a contract and a
      test-double seam (naming SpyComm), and says where shared logic
      would live if a second production implementation is added; it does
      not claim a comm-style selection mechanism that does not exist.
    status: pending
  - id: ac-007
    summary: no two types share the DomainDecomposition name
    type: code
    pass_when: |
      the CUDA single-process spatial split is no longer named
      DomainDecomposition; the CUDA build compiles on GH200; and
      .claude/notes/architecture.md names both types correctly.
    status: pending
