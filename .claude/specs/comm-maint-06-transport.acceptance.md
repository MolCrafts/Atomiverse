---
spec: comm-maint-06-transport
created: 2026-08-27
criteria:
  - id: ac-001
    summary: one primitive backs every halo transport
    type: code
    pass_when: |
      comm_brick.cpp contains one round-exchange primitive with flat
      parameters and no configuration struct, and the forward cascade,
      forward replay, reverse replay, owner exchange and packer path all
      route through it rather than posting their own Isend/Irecv/Waitall
      sequences.
    status: pending
  - id: ac-002
    summary: one tag scheme and one buffer pool
    type: code
    pass_when: |
      no MPI call in comm_brick.cpp passes a literal 0 tag; every tag
      comes from halo_tag; forward_comm(Packer&) allocates no
      std::vector per call and uses the shared FaceBufs; its unreachable
      self branch is gone.
    status: pending
  - id: ac-003
    summary: packer buffers are sized once per reneighbour
    type: code
    pass_when: |
      the maximum packer width is pulled from mounted components once
      per run setup, the shared buffers are sized at the end of
      borders(), and a steady-state MD interval performs no buffer
      reallocation (assertable via a capacity probe or an allocation
      counter in a test).
    status: pending
  - id: ac-004
    summary: Comm::setup has an honest contract
    type: code
    pass_when: |
      either an orchestrator calls Comm::setup on box change and a test
      observes it, or setup is no longer part of the public Comm
      interface; in both cases no error message instructs the caller to
      do something the interface cannot express.
    status: pending
  - id: ac-005
    summary: the comm schedule is written once
    type: code
    pass_when: |
      the reneighbor-or-forward decision exists in exactly one helper in
      runtime.cpp, called by the linear path, the verlet setup and the
      verlet step; each caller still walks only the phases its own enum
      defines.
    status: pending
  - id: ac-006
    summary: SpyComm pins the sequence
    type: code
    pass_when: |
      tests/cpu/driver/test_verlet_comm.cpp asserts, for each of the
      three paths, the ordered sequence of Comm methods invoked on a
      reneighbor step and on a non-reneighbor step.
    status: pending
  - id: ac-007
    summary: numbers unchanged and performance not regressed
    type: code
    pass_when: |
      ctest green; widebox_parity.sh dev=0.0 at NP=256 for DIMS=8,8,4
      and DIMS=16,4,4; the MIL0PEGM18 np=256 benchmark Comm section is
      no more than 3% slower than the pre-change 0.842 s / 5200
      launches, and steps/s is no lower than 1400.
    status: pending
