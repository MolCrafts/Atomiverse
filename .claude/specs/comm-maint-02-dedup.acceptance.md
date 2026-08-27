---
spec: comm-maint-02-dedup
created: 2026-08-27
criteria:
  - id: ac-001
    summary: Halo and Swap own their own pure helpers
    type: code
    pass_when: |
      stage_end, round_end, round_begin, mirror_swap, the PBC shift and
      the silent-swap predicate are reachable from halo.h (as Halo/Swap
      members or free functions in the same header) and no longer exist
      in the anonymous namespace of comm_brick.cpp; each takes flat
      parameters and introduces no new aggregate type.
    status: pending
  - id: ac-002
    summary: all four wire codecs live in atom_pack
    type: code
    pass_when: |
      pack_reverse_rows, accumulate_reverse_rows, accumulate_column and
      plan_is_fxyz_reverse are declared in atom_pack.h alongside
      pack_comm / pack_border / pack_exchange, and comm_brick.cpp no
      longer defines them.
    status: pending
  - id: ac-003
    summary: the triplicated ownership helpers collapse to one
    type: code
    pass_when: |
      rank_along_dim, slab_along_dim_from_cuts and the owner-rank lookup
      each have exactly one definition in the tree, on
      DomainDecomposition; grep finds no copy in comm_brick.cpp and none
      remaining at domain.cpp:175.
    status: pending
  - id: ac-004
    summary: one overflow-checked multiply and one hash helper
    type: code
    pass_when: |
      the communication directory contains exactly one
      overflow-checked-multiply definition and one FNV hash helper pair;
      gather.cpp, bcast_frame.cpp, comm_brick.cpp and atom_pack.cpp all
      call the shared ones.
    status: pending
  - id: ac-005
    summary: comm_brick.cpp keeps only genuinely file-scoped helpers
    type: code
    pass_when: |
      its anonymous namespace is a single block containing only
      facebufs_bytes, prealloc_bufs, pack_forward_for_swap, pack_to,
      forward_round, forward_cascade, exchange_along_dim and
      zero_ghost_forces, with no duplicate using-declaration.
    status: pending
  - id: ac-006
    summary: the move is bit-for-bit behaviour preserving
    type: code
    pass_when: |
      ctest is green at the pre-change count plus the new helper tests,
      and widebox_parity.sh reports dev=0.0 on every window at NP=256
      for both DIMS=8,8,4 and DIMS=16,4,4.
    status: pending
  - id: ac-007
    summary: newly public helpers are unit tested
    type: code
    pass_when: |
      every helper made reachable by this spec has at least one unit
      test under tests/cpu/communication/, including a pack/accumulate
      round trip for the reverse codec that recovers the input values.
    status: pending
