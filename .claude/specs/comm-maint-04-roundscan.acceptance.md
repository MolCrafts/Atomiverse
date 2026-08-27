---
spec: comm-maint-04-roundscan
created: 2026-08-27
criteria:
  - id: ac-001
    summary: the scan window is a pure Halo query
    type: code
    pass_when: |
      Halo exposes a scan_window query that takes flat parameters,
      performs no MPI and needs no Context, and forward_cascade in
      comm_brick.cpp no longer derives win_first/win_last inline.
    status: verified
  - id: ac-002
    summary: round>1 sendlist contents are pinned
    type: code
    pass_when: |
      tests/cpu/communication/test_halo_rounds.cpp asserts the exact row
      indices a hop-2 low-face swap ships at a 63.0016 A box with an
      8,8,4 grid and an 11 A shell, not merely their count.
    status: verified
  - id: ac-003
    summary: the forwarding band bounds are pinned
    type: code
    pass_when: |
      a test asserts slablo/slabhi for round>1 equal
      (0.5*(sublo+subhi), sublo+cutoff) on the low face and
      (subhi-cutoff, 0.5*(sublo+subhi)) on the high face, and that a row
      exactly on either edge is shipped.
    status: verified
  - id: ac-004
    summary: the 3-hop chain is proven complete and non-duplicating
    type: code
    pass_when: |
      a test at DIMS=16,4,4 (hops 3,1,1) asserts the union of the three
      x rounds covers [sublo-cutoff, subhi+cutoff] and that no row is
      shipped twice at the same periodic image.
    status: verified
  - id: ac-005
    summary: the window takes the opposite face
    type: code
    pass_when: |
      a test asserts that round r>1 of a face scans the slot range the
      OPPOSITE face received at round r-1, and fails if given the same
      face's range.
    status: verified
  - id: ac-006
    summary: the new tests actually bite
    type: code
    pass_when: |
      with the round>1 band temporarily reopened to the sentinel range,
      test_cpu_comm_halo_rounds fails while test_cpu_comm_halo_maxneed and
      test_cpu_comm_halo both still pass; with the band restored, all pass.
      Measured 2026-08-27.
    status: verified
  - id: ac-007
    summary: end-to-end parity is unchanged
    type: code
    pass_when: |
      ctest is green and widebox_parity.sh reports dev=0.0 on every
      window at NP=256 for DIMS=8,8,4 and DIMS=16,4,4.
    status: verified
