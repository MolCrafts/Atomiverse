---
spec: comm-maint-05-dd-split
created: 2026-08-27
criteria:
  - id: ac-001
    summary: the sort schedule no longer lives on DomainDecomposition
    type: code
    pass_when: |
      domain.h declares none of atom_sort_enabled, atom_sort_nfreq,
      atom_sort_binsize, atom_sort_neigh_cut, atom_sort_step or
      atom_sort_next; neighbor_list.h no longer publishes them; the sort
      decision is reached through a query with flat parameters and no
      new aggregate type was introduced.
    status: pending
  - id: ac-002
    summary: the sort clock is the real step index
    type: code
    pass_when: |
      grep finds no ++dd->atom_sort_step anywhere; comm_brick.cpp
      derives the cadence from the LaunchContext step_idx.
    status: pending
  - id: ac-003
    summary: a line-search force re-evaluation does not advance the clock
    type: code
    pass_when: |
      a test drives a Minimize line-search trial through the force slot
      and asserts the sort cadence advanced by zero, while a plain MD
      step advances it by one.
    status: pending
  - id: ac-004
    summary: sorting still fires on cadence
    type: code
    pass_when: |
      a multi-rank test with a known nfreq observes the sort at the
      expected steps and not at others.
    status: pending
  - id: ac-005
    summary: periods is resolved, not left dangling
    type: code
    pass_when: |
      either init_topology assigns periods from a real input and a
      multi-rank test exercises a non-periodic axis, or periods and every
      non-periodic branch in Halo are deleted and the all-periodic
      limitation is stated in domain.h and CLAUDE.md. Not both, not
      neither.
    status: pending
  - id: ac-006
    summary: legal writers are documented
    type: code
    pass_when: |
      the DomainDecomposition class comment names every production
      writer outside src/cpu/communication/ and what each is allowed to
      write.
    status: pending
  - id: ac-007
    summary: MD trajectories unchanged, Minimize cadence deliberately changed
    type: code
    pass_when: |
      ctest is green, widebox_parity.sh reports dev=0.0 for DIMS=8,8,4
      and DIMS=16,4,4 at NP=256, and a fixed-seed MD run reproduces its
      pre-change trajectory bit for bit. For Minimize the report states
      the sort count before and after and shows the new count equals the
      step count rather than the force-evaluation count.
    status: pending
  - id: ac-008
    summary: the existing AtomSortState is reused, not duplicated
    type: code
    pass_when: |
      md::AtomSortState is published on DynamicsServices as a non-owning
      pointer owned by NeighborList; comm_brick.cpp calls its due /
      resolved_binsize / mark_sorted rather than reimplementing them;
      AtomSortParams is gone and configure takes flat arguments;
      DynamicsServices is still trivially copyable.
    status: pending
