---
spec: comm-maint-03-genericpath
created: 2026-08-27
revised: 2026-08-27
criteria:
  - id: ac-001
    summary: pack and unpack evaluate the identical shape predicate
    type: code
    pass_when: |
      For each of the comm and reverse wires, the pack side and the unpack
      side branch on the same expression; neither carries a term the other
      lacks. Writability is checked before the branch, not inside it.
    status: verified
  - id: ac-002
    summary: a non-writable plan is refused, not silently rerouted
    type: code
    pass_when: |
      unpack_comm, unpack_border and accumulate_reverse_rows each throw a
      column_error naming the make_*_plan(Frame&) fix when handed a plan
      compiled from a const Frame, rather than taking a different layout.
    status: verified
  - id: ac-003
    summary: the border wire has exactly one layout
    type: code
    pass_when: |
      pack_border and unpack_border contain one loop each, row-major;
      plan_is_scalar_border and gather_column no longer exist; a comment
      records that the deleted field-major arm was unreachable because
      `image` is a three-component kColBorder column.
    status: verified
  - id: ac-004
    summary: no comment claims the two layouts are byte-identical
    type: code
    pass_when: |
      grep for "byte-identical" in atom_pack.cpp returns nothing; the
      surviving comments state which arm is field-major, which is row-major,
      and that the generic arm carries plugin-declared runtime columns.
    status: verified
  - id: ac-005
    summary: numbers unchanged
    type: code
    pass_when: |
      ctest green at 239/239 and the np=16 MIL0PEGM18 single point reports
      E=1725.258348 pair=-200.208398 eV, identical to before the change.
    status: verified
  - id: ac-006
    summary: the generic arms are unit-tested, not only MPI-tested
    type: code
    pass_when: |
      A serial unit test inserts a multi-component runtime column carrying
      kColComm and another carrying kColReverse, then packs and unpacks
      in-process and recovers the input values. Deleting either generic arm
      must fail this test directly, not as an MPI timeout.
    status: pending
