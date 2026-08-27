---
spec: comm-maint-01-deadcode
created: 2026-08-27
criteria:
  - id: ac-001
    summary: atom_migration is gone and nothing references it
    type: code
    pass_when: |
      include/atomiverse/cpu/communication/atom_migration.h and
      src/cpu/communication/atom_migration.cpp no longer exist; the
      CMakeLists.txt source list no longer names atom_migration.cpp;
      grep -rn "scatter_atoms_from_root" over src include tests
      regressions app python tools returns no hits; the project builds
      and links.
    status: pending
  - id: ac-002
    summary: the two dead exchange helpers are gone
    type: code
    pass_when: |
      grep -n "pack_atom\|unpack_atom" src/cpu/communication/comm_brick.cpp
      returns no hits, and the stale "atom_migration's MPI_Alltoallv"
      comment is gone.
    status: pending
  - id: ac-003
    summary: the two zero-implementer protocols are deleted
    type: code
    pass_when: |
      include/atomiverse/cpu/component/halo_participant.h and
      atom_exchange.h no longer exist; component.h declares neither
      halo_participant() nor atom_exchange(); no file in the tree
      names HaloParticipant or AtomExchange.
    status: pending
  - id: ac-004
    summary: ForwardPacker leaves the Component base with no replacement accessor
    type: code
    pass_when: |
      component.h declares class Component with no ForwardPacker base
      and adds no as_forward_packer() accessor; FixRattle and FixSettle
      each name ForwardPacker as a base; ChargePacker is unchanged; the
      Component vtable is three virtuals smaller and none were added;
      the rattle, settle and charge-packer behaviours are unchanged.
    status: pending
  - id: ac-004b
    summary: the CUDA twin protocol goes with it
    type: code
    pass_when: |
      cuda/component/halo_participant.cuh no longer exists,
      bind_participants is gone, and cuda Component declares no
      halo_participant(); the CUDA build compiles on GH200.
    status: pending
  - id: ac-005
    summary: no comment or note names a type that does not exist
    type: code
    pass_when: |
      grep -rn "CommBrick\|CommBrick\|CommBrick::forward_comm\|Halo\|Comm::exchange "
      over src include docs tests regressions .claude returns no hits;
      .claude/notes/architecture.md describes cpu/communication with the
      types that are actually there; CLAUDE.md lists the ten md::Phase
      values from core/stage.h and no longer names HaloParticipant or
      AsyncComponent.
    status: pending
  - id: ac-006
    summary: unused-symbol warnings are on and clean
    type: code
    pass_when: |
      the atomiverse target compiles with at least
      -Wunused-function -Wunused-variable, and a full clean rebuild
      emits zero warnings from those flags. If -Wall was also enabled,
      the rebuild is likewise clean; if it was not, the report states
      the measured -Wall warning count and it is recorded as a ticket.
    status: pending
  - id: ac-007
    summary: behaviour is unchanged
    type: code
    pass_when: |
      ctest over the configured build is green with the same test count
      as before the change minus the one deleted halo_participant
      assertion, and no test tolerance or pin was touched.
    status: pending
