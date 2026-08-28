<!-- mol:note:topic:md-comm -->
# CPU MD communication

**Rule**: the per-step halo (`forward_comm` / `reverse_comm`) rides the
Schema-compiled `PackPlan`. Refresh is x/y/z, reverse is fx/fy/fz, border is
x/y/z/tag/type — and each of those is field-major, because every column on
them is scalar. A column travels because a mounted component declared it, not
because it carries a flag.

**Components declare a data-lifetime contract, not a wire.** A style hands
back a `static constexpr std::array<GhostField>` from `declare_ghosts()` —
compile-time data, no builder, no allocation at fold time. Two orthogonal
axes: `read_stable` / `read_each_step` is the owner→ghost **freshness**
guarantee, `reduce_to_owner` is the ghost→owner **data flow**. The Driver
folds every table onto `base_fields::kEngine` into ONE `GhostContract`
(`ctx.md.ghost`); each pass compiles its own `PackPlan` straight off it.

The contract→pass mapping lives once, in `make_wire_plan`: Border = every
read, Comm = reads at `Freshness::step`, Reverse = every reduction. Nothing
above it names a wire; nothing below it names a freshness. There is no
per-wire `Schema` and no intermediate bag — that layer was deleted, not
renamed.

Presence is implied by the read, which is why the old
`every_step ⇒ at_creation` rule is gone rather than enforced: there are not
two sets. A ghost born on a rebuild step is filled by `borders()` and read by
the force with no refresh pass in between, so a `step` read is on both passes
and there is no way to spell one without the other. Getting that wrong is how
`PairDpd` read ghost velocities up to `every` steps old — which no
energy-conservation check catches, because DPD does not conserve energy by
construction.

**The swap table pairs mirrored faces, and that is what fixes the tag.**
`Halo::setup_uniform` gives both faces of a round the *same* partner rank
(`Swap::sendproc == Swap::recvproc == face_neighbour`), so a message this rank
sends on swap `s` arrives at the partner as its `Halo::mirror_swap(s)`. Every
MPI post therefore keys on the swap index: send with tag `s`, receive with tag
`mirror_swap(begin, end, s)`, and the **round** — not the swap — is the unit
that gets drained. LAMMPS can run one swap at a time on tag 0 only because its
`sendproc != recvproc` (send low, hear back from high), which makes both ranks'
swap `i` each other's partner.

Draining ATV's table one swap at a time on tag 0 pairs the wrong faces, and at
`dims[d] == 2` — where both faces of a round address the same rank — deadlocks
outright. That was tried and reverted: it silently shipped the opposite face's
positions (equal byte count, so no MPI error) and cost 23% of the LJ energy at
8 ranks.

Within a round the transport is LAMMPS-shaped: post every receive, then pack
and **blocking `MPI_Send`** into partners that are already listening, then
`Waitall` the receives and unpack. No send-side `Isend`/`Waitall` — that
Waitall was the measured Comm wall at 256 ranks.

`PackPlan` is a compiled Schema projection, not a config bag, and it already
fast-paths the all-scalar wires down to three shifted block copies. It is not
on the critical path for removal.

## Remaining leads

- **Convert the swap table to LAMMPS's `sendproc != recvproc` convention.**
  Send low / hear from high per swap, the way `comm_brick.cpp` sets
  `procneigh`. That is the only thing that makes the literal per-swap tag-0
  shape legal here, and it would delete `mirror_swap` and the round-as-unit
  rule with it. Touches `borders` (count handshake, send-list bands, pbc
  flags), `reverse_comm`, and the multi-hop `round_scan_window`. A spec, not a
  drive-by.
- **`PackPlan` off exchange / gather / bcast / MC.** The migration wire still
  selects by flag (`kColExchange`) rather than by Schema, which is defensible —
  migration carries what an atom IS, not what a consumer reads. Gather / bcast /
  MC `Transaction` have no such excuse. Tests under
  `tests/cpu/communication/test_atom_pack_*.cpp` and
  `include/atomiverse/cpu/montecarlo/transaction.h`.
- **CUDA `MultiGpu` is a second MD stack** in `cuda/communication/`
  (partition + halo + force + NVE, public `PairFactory`). CPU communication
  is DomainRuntime + CommBrick behind `ctx.md.comm`. Split when CUDA MD
  grows a Runtime communicator — do not dual-architect Verlet on CUDA
  (OpenMM cluster/tile stays the force engine).
- **Four namespaces** in one pillar: `cpu::Comm`, `cpu::comm`,
  `cpu::communication`, `cpu::md::mpi`. CLAUDE: namespace alignment is a
  deliberate follow-up, not a drive-by.
- **`SpatialGrid` slot map is x-fastest; CPU brick is x-slowest.** Same
  role, not the same rank linearization. Do not silently unify.
- **kspace remap Alltoallv** is the other measured wall vs LAMMPS at
  np=256 (wt10: ATV 95% overall; Comm 1.92× was Waitall on the old Isend
  path). Next measurement: same MIL0PEGM18 protocol now that the send side is
  a blocking `Send` behind standing receives.
- **OpenMP over sendlists / MPI datatype into SoA x,y,z** were not the
  measured Comm wall (unpack 0.008 s). Do not add either without a new
  256-rank number.
