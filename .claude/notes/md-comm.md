<!-- mol:note:topic:md-comm -->
# CPU MD communication

Binding rules: docs/developer/project-contract.md › Ghost data is a lifetime contract. This note keeps
the mechanism and the reasons.

## Wires

- Step halo rides the contract-compiled `PackPlan`: refresh x/y/z, reverse
  fx/fy/fz, border x/y/z/tag/type, each field-major (all scalar). A column
  travels because a mounted component declared it, not because of a flag.
- `make_wire_plan` caches its compile in `CommBrick::setup`; every refold sets
  the communicator stale and `setup` re-runs at the next rebuild.
- A ghost born on a rebuild step is filled by `borders()` and read by the force
  with no refresh in between, so a `step` read is necessarily on both passes —
  presence cannot be spelled separately.

## Why undeclared reads are dangerous

- An undeclared ghost read returns whatever the row holds. Bogus `q_i q_j` is
  symmetric, so `Σ F = 0` and every momentum check still passes while the energy
  is wrong. Guards: `TestGhostContract.EveryGhostChargeReaderDeclaresIt`,
  `regressions/md/domain-decomposition`.
- Charge is `read_stable` on all Coulomb styles. Fluctuating-charge owners
  (QEq, constant-potential electrodes) refresh ghosts themselves through
  `ChargePacker` on the `ForwardPacker` path, after the solve.
- DPD needs ghost velocities fresh every step (`read_each_step`); stale ones
  are invisible to energy checks because DPD does not conserve energy.

## Swap table and transport (LAMMPS `CommBrick`)

- `Halo::expand` gives swap `s` its own face's neighbour to send to and the
  **opposite** face's neighbour to hear from (`procneigh[dim][0/1]`). The table
  is rank-uniform (`derive_uniform_rounds`), so this rank's swap `s` is answered
  by the partner's swap `s` — one message per pair per swap, tag 0, also at
  `dims[d] == 2`.
- A round's two swaps share one scan window (everything the previous round of
  that axis delivered, `CommBrick::borders` `nfirst/nlast`); the `round > 1`
  mid-plane band sorts that window between faces. Keeping round `r > 1` off the
  owners is the window's job (`forward_cascade`).
- Transport: post receive, pack, blocking `MPI_Send` into a listening partner,
  `Wait`, unpack. No send-side `Isend`/`Waitall`.
- `bcast_frame` sends the root's per-column stride in the header: a topology
  column's reserve width is per rank and outside the schema digest.
  `comm::column_schema_fingerprint` is the same digest `PackPlan::fingerprint`
  folds under its `PackKind`.

## Leads

- **CUDA `MultiGpu`** is a second MD stack (partition + halo + force + NVE,
  public `PairFactory`). Split when CUDA MD grows a Runtime communicator; the
  OpenMM cluster/tile engine stays the force path.
- **Four namespaces** in one pillar: `cpu::Comm`, `cpu::comm`,
  `cpu::communication`, `cpu::md::mpi` (deliberate follow-up).
- `tools/debug_amorphous/gather_frame.h` drives `pack_row` / `unpack_row` for a
  variable-length all-gather — migration payload in a debug tool; leave it
  unless the tool is reworked.
- `SpatialGrid` slot map is x-fastest; CPU brick is x-slowest. Do not silently
  unify.
- k-space remap `Alltoallv` is the other wall vs LAMMPS at high rank counts;
  OpenMP over sendlists / MPI datatypes into SoA were not the Comm wall — do
  not add them without a new large-rank measurement.
