<!-- mol:note:topic:md-comm -->
# CPU MD communication

**Rule**: the per-step halo (`forward_comm` / `reverse_comm`) rides the
contract-compiled `PackPlan`. Refresh is x/y/z, reverse is fx/fy/fz, border is
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

**A style that does not declare gets nothing, and nothing says so.** When the
wire moved off the global `kColBorder` flag (`82ad7867`), `charge` stopped
riding the border pass unless a mounted component asked for it — and four
styles that read `q[j]` were never given a `declare_ghosts()`: `PairCoulCut`,
`PairCoulSoft`, `PairGaussianCoul`, `PairLJCutTip4pCut`. At one rank nothing
changes (there are no ghosts). At 8 ranks `coul/cut` read whatever those ghost
rows happened to hold and pumped **44% onto the total energy**, while
`Σ_i F_i` stayed exactly 0 — a bogus `q_i q_j` is still symmetric, so every
force-balance and momentum check passes. `regressions/md/multihop-widebox-parity`
caught it; `TestGhostContract.EveryGhostChargeReaderDeclaresIt` is the
unit-level guard. Charge is `read_stable` on all six Coulomb styles: a fixed
charge does not change while a ghost lives, and a fluctuating-charge owner
(QEq, constant-potential electrodes) refreshes ghosts itself through
`ChargePacker` on the self-initiated `ForwardPacker` path. **Adding a style
that reads any ghost column means adding its declaration in the same change**
— the missing case is silent by construction.

**The swap table is LAMMPS's, `sendproc != recvproc` included.**
`Halo::expand` gives swap `s` its own face's neighbour to send to and the
**opposite** face's neighbour to hear back from — LAMMPS `CommBrick::setup`'s
`procneigh[dim][0]` / `procneigh[dim][1]`. The table is rank-uniform
(`derive_uniform_rounds`) and every rank walks it in the same order, so this
rank's swap `s` is answered by the partner's swap `s`. Exactly one message
per pair per swap, therefore: **one swap at a time, tag 0**, at
`dims[d] == 2` (both faces the same rank) as much as anywhere else. There is
no mirrored index and no round-as-a-transport-unit rule.

The mirrored convention that came before (`sendproc == recvproc`: my band
out, the partner's band across the same plane back) moved the identical atom
set but paired swap `s` with the partner's *opposite* swap. That forced a
per-swap tag and a whole round as the drain unit; draining it one swap at a
time on tag 0 shipped the opposite face's positions (equal byte count, so no
MPI error) and cost 23% of the LJ energy at 8 ranks. Under the current
convention the same shape is correct, which is the whole point of converting
the table rather than deleting the tags.

**The round survives as the forwarding unit.** Its two swaps share one scan
window — everything the previous round of that axis delivered, LAMMPS
`CommBrick::borders`' `nfirst = nlast; nlast = nlocal + nghost` — and the
`round > 1` mid-plane band edge is what sorts that shared window between the
faces: a ghost that arrived across one face sits past the opposite edge of
the subdomain, so it fails its own face's band and passes the other's. The
band cannot keep round `r > 1` off the OWNERS (an owner is inside the
subdomain and can be on either side of the mid-plane); that is the window's
job, three lines in `forward_cascade`.

Transport is LAMMPS-shaped throughout: post the receive, pack, **blocking
`MPI_Send`** into a partner that is already listening, `Wait`, unpack. No
send-side `Isend`/`Waitall` — that Waitall was the measured Comm wall at 256
ranks.

**`PackPlan` is a communication artifact and nothing else.** Two makers:
`make_wire_plan` (contract-driven halo passes) and `make_exchange_plan`
(migration, flag-driven on `kColExchange` — migration carries what an atom
IS, which is a property of the Frame, not of whoever is mounted). Everything
else asks the Block directly:

- `gather_to_root` enumerates with `Block::resolve_columns(kColExchange)` and
  `Gatherv`s column-major.
- `bcast_frame` does the same and `Bcast`s **one call per column**; the
  root's per-column stride rides the header, because a topology column's
  reserve width is per rank and deliberately outside the schema digest.
- `mc::Transaction` snapshots and restores whole rows with
  `Frame::copy_row_from` onto a `clone_structure` patch — zero column
  resolutions per trial, where the old two-plan byte round trip cost one.

Rank agreement for the two collectives is `comm::column_schema_fingerprint`,
the same field digest `PackPlan::fingerprint` folds under its `PackKind`.

## Remaining leads

- **CUDA `MultiGpu` is a second MD stack** in `cuda/communication/`
  (partition + halo + force + NVE, public `PairFactory`). CPU communication
  is DomainRuntime + CommBrick behind `ctx.md.comm`. Split when CUDA MD
  grows a Runtime communicator — do not dual-architect Verlet on CUDA
  (OpenMM cluster/tile stays the force engine).
- **Four namespaces** in one pillar: `cpu::Comm`, `cpu::comm`,
  `cpu::communication`, `cpu::md::mpi`. CLAUDE: namespace alignment is a
  deliberate follow-up, not a drive-by.
- **`tools/debug_amorphous/gather_frame.h`** still drives `pack_row` /
  `unpack_row` for an all-gather of variable-length rows. That is the
  migration payload doing migration's job in a debug harness, not the
  borrowed-wire pattern the library paths shed — leave it unless the tool
  itself is being reworked.
- **`SpatialGrid` slot map is x-fastest; CPU brick is x-slowest.** Same
  role, not the same rank linearization. Do not silently unify.
- **kspace remap Alltoallv** is the other measured wall vs LAMMPS at
  np=256 (wt10: ATV 95% overall; Comm 1.92× was Waitall on the old Isend
  path). Next measurement: same MIL0PEGM18 protocol now that the send side is
  a blocking `Send` behind standing receives — and now that the halo posts
  per swap rather than per round, which is a second variable in that number.
- **OpenMP over sendlists / MPI datatype into SoA x,y,z** were not the
  measured Comm wall (unpack 0.008 s). Do not add either without a new
  256-rank number.
