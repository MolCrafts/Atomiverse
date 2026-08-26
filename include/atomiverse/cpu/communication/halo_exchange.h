/// @file cpu/communication/halo_exchange.h
/// @brief Ghost-atom halo exchange components for MPI domain decomposition.
///
/// Structural reimplementation of the LAMMPS CommBrick communication
/// discipline in `src/comm_brick.cpp` (`CommBrick::forward_comm` /
/// `reverse_comm` / `borders`) — not a verbatim dump. Driven by an ordered
/// **swap table** (`comm::SwapList`) instead of a per-step per-face geometry
/// re-derivation:
///   - `HaloSend::launch` re-derives the table at every full rebuild
///     (`setup` + borders: refill each swap's send list from current
///     positions, exchange counts, pack rows with the swap's PBC image
///     shift, `MPI_Isend/Irecv` both swaps of one round concurrently,
///     unpack received ghosts into `atoms` Block rows
///     `[n_local, n_local+n_ghost)` and commit each swap's ghost slots).
///     In between rebuilds it replays the same table, refreshing ghost
///     rows only.
///   - `HaloRecv::launch` walks the same table backwards, reducing
///     ghost-row `fx/fy/fz` contributions back onto the owners.
///
/// **A ghost shell deeper than one subdomain is not a refusal.** When the
/// shell is wider than this rank's slab (narrow box, many ranks on one axis,
/// an NPT / deform shrink, or a bonded span wider than the pair cutoff) the
/// table carries `SwapList::max_hops(d) > 1` **forwarding rounds** per axis,
/// and round `r > 1` of a face ships exactly what round `r - 1` delivered on
/// the OPPOSITE face of the same axis — so rank `c - r`'s atoms arrive
/// through the ranks in between, never as a direct message to a rank two
/// hops away. Rounds are serialized; the two faces of a round are not. The
/// only geometry still refused is the minimum-image bound
/// (`shell <= L_d - w_self`), which no number of hops can satisfy.
///
/// Each endpoint owns its per-swap MPI buffers (`detail::FaceBufs`,
/// per-instance PIMPL). The swap table itself is **owned by `HaloSend`**
/// and published non-owning as `ctx.md.swap_list` (mirroring
/// `NeighborList → ctx.md.pair_list`); `HaloRecv` and `HaloRefresh` read
/// it, and a null pointer means "no table yet" — a rank-uniform no-op.
///
/// The default-constructed `HaloSend{}` (`cutoff = 0`) is a
/// no-op sentinel. Under `ATV_BUILD_MPI`, `cpu::md::MD` auto-injects the
/// `HaloSend{max(1.1 * pair_cutoff, neighbor_reach)}` + `HaloRecv{}` pair when
/// `set_evaluator` is called — the shell has to cover the distance the neighbor
/// list enumerates to (`cutoff + skin`), not just the interaction cutoff, or a
/// cross-rank pair inside the skin band gets no ghost and goes missing from the
/// CSR until the next full rebuild. A manually registered pair suppresses the
/// injection (detected
/// through the `halo_participant()` protocol probe — reverse exchange
/// must run exactly once).
///
/// **Bonded skin auto-extension** (spec A.5): `build()` scans the
/// per-atom bonded topology (`kAtom_num_bond`, `kAtom_bond_atom`, …;
/// tag-aware: multi-rank scans resolve
/// atom tags through `dd->tag_to_local` and `MPI_Allreduce(MAX)` across
/// ranks) to find the widest intra-tuple span. `launch()` uses
/// `max(cutoff, bonded_extension)` as the effective halo width -- the
/// bonded extension only *raises* the halo when the user-supplied pair
/// cutoff is smaller than a bonded tuple's reach. Callers no longer have
/// to hand-roll the bonded skin width on top of the pair cutoff.
///
/// Copyright (2003) Sandia Corporation.  Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software.  This software is distributed under
/// the GNU General Public License.
///
/// References:
///   * S. Plimpton, J. Comput. Phys. **117**, 1 (1995),
///     DOI 10.1006/jcph.1995.1039 — the original swap-list forward/reverse
///     comm (`maxneed`, `sendlist`, `firstrecv`, `sendself`).
///   * A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
///     DOI 10.1016/j.cpc.2021.108171 — modern LAMMPS borders/forward/reverse.
///
/// @see .claude/specs/parallel-md-engine.md  § A.4, A.5
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/communication/swap_list.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/halo_participant.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace comm {

namespace detail {
/// @brief Per-swap MPI buffer storage (PIMPL). Definition lives in
/// `halo_exchange.cpp` so the header stays free of MPI-adjacent state.
/// Held per-instance by `HaloSend` / `HaloRecv` so
/// multiple components in the same process do not clobber each other's
/// send/recv buffers. Sized to the swap count, which is derived
/// (`2 * sum_d maxneed_d`) rather than a fixed six.
struct FaceBufs;
} // namespace detail

/// @brief Forward pass: owned positions → ghost rows on face neighbors.
///
/// Owns the halo's `comm::SwapList` and publishes it as `ctx.md.swap_list`.
/// The table is re-derived on every full rebuild — so an NPT / deform box
/// rescale is picked up, including a shrink that raises the per-axis round
/// count — and each swap's send list is refilled from the current positions.
///
/// Steady-state (refresh / non-full-rebuild) `launch()` path: **zero MPI
/// collectives**, **zero realloc** of the swap table / `FaceBufs` (buffers
/// may `resize` up to pre-reserved capacity), **zero memset** of recv
/// buffers. Schema-fingerprint collectives live in `build()` and the
/// full-rebuild branch only. `HaloRecv`'s layout-guard Allreduce is gated
/// on `dd->halo_generation`. `mpi::Reduce` is a separate NVT temperature
/// reduction, not part of halo.
///
/// Single-rank (`runtime.mpi_size == 1`) and `cutoff == 0` both early-return,
/// leaving the published pointer null.
///
/// Atoms within `cutoff` of a face plane are flagged as ghosts to the
/// corresponding face neighbor, and a deeper shell is filled by forwarding
/// each round's ghosts one rank further. PBC: rows sent across the periodic
/// edge of the global box are shifted by `±L_d` on the way out so the
/// receiver sees them in its own coordinate frame; a forwarded row already
/// carries the shifts of the hops it has taken, and each link adds its own.
///
/// It also publishes the shell it is about to build as
/// `DomainDecomposition::ghost_shell_width` [Angstrom] — at `build()` and at
/// every full rebuild, before any consumer reads it. `CellList` pads the
/// decomposed axes of its bin grid by that width; a multi-hop ghost outside
/// the window would be clamped into the edge cell silently.
///
/// The published width is the shell that is *asked for*,
/// `max(cutoff, bonded_extension)`. A forwarding round carries the open band
/// of `Swap::slablo` (the LAMMPS `ineed >= 2` convention), so the last round
/// of an axis can also deliver rows up to one partner slab **past** that
/// width — surplus, never a missing ghost: those rows are farther from every
/// owner than the shell itself, so no pair kernel and no neighbour list ever
/// reaches them, and the grid's edge clamp on them changes nothing. Trimming
/// them would mean giving rounds `> 1` the round-1 band, which is a
/// `SwapList` geometry decision, not a transport one.
class HaloSend : public cpu::Component, public cpu::HaloParticipant {
  public:
    /// @brief No-op default constructor (cutoff = 0).
    HaloSend();

    /// @brief MPI constructor with the pair cutoff.
    /// @param cutoff Pair-interaction cutoff [Angstrom]. Typically
    ///        `r_cut + skin` where `r_cut` matches the neighbor list /
    ///        pair potential cutoff. The effective halo width used at
    ///        `launch()` is `max(cutoff, bonded_extension)`, where
    ///        `bonded_extension` is the widest intra-tuple span in
    ///        canonical per-atom topology columns computed once at `build()`.
    ///        When `cutoff >= bonded_extension` the bonded scan has no
    ///        effect (pair cutoff already dominates).
    /// @param every Interval (in steps) between **full** rebuilds of
    ///        per-face send lists and ghost-slot assignments. On steps
    ///        between rebuilds the halo only refreshes ghost positions
    ///        for the existing send lists -- slot indices stay stable
    ///        so the downstream neighbor list remains reusable. Matches
    ///        LAMMPS's reneighboring cadence. Default 10 lines up with
    ///        `NeighborList`'s default `every=10`.
    explicit HaloSend(double cutoff, int every = 10);

    // unique_ptr<incomplete type> needs user-defined special members,
    // defined in the .cpp where detail::FaceBufs is complete.
    ~HaloSend() override;
    HaloSend(HaloSend&&) noexcept;
    HaloSend& operator=(HaloSend&&) noexcept;
    HaloSend(const HaloSend&) = delete;
    HaloSend& operator=(const HaloSend&) = delete;

    std::string name() const override { return "halo_send"; }
    /// @brief Send half of the DD halo pair — the driver infrastructure
    /// scan probes this typed accessor, not `name()`.
    HaloParticipant* halo_participant() override { return this; }
    /// @brief Forward exchange endpoint.
    Endpoint halo_endpoint() const override { return Endpoint::send; }
    /// @brief Receive the field-participant list from the driver's
    /// build-time bind pass; field descriptors are flattened once at
    /// first launch and cached.
    void bind_participants(
        const std::vector<const HaloParticipant*>& participants) override {
        participants_ = participants;
    }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::pre_neighbor, 0)};
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Retract the published swap table.
    ///
    /// The table dies with this component, so the pointer on the context
    /// must not outlive it — a runtime unfix would otherwise leave
    /// `HaloRecv` / `HaloRefresh` reading freed memory instead of taking
    /// their null branch.
    /// @param ctx Destroy context carrying the publishing runtime.
    void destroy(const cpu::DestroyContext& ctx) override;

    /// @brief Pair cutoff this exchange was constructed with (no bonded skin).
    double cutoff() const { return cutoff_; }

    /// @brief Bonded-skin extension computed by `build()` from the frame's
    /// bonds / angles / dihedrals. Zero before `build()` or when the frame
    /// has no topology.
    double bonded_extension() const { return bonded_extension_; }

    /// @brief Effective halo cutoff: the larger of the pair cutoff and the
    /// bonded skin extension. An atom near the boundary needs its farthest
    /// pair OR bonded partner ghosted, not both summed. `launch()` uses
    /// this value to decide which owned atoms to ship as ghosts. Returns 0
    /// before `build()`.
    double halo_cutoff() const {
        return (cutoff_ > bonded_extension_) ? cutoff_ : bonded_extension_;
    }

    /// @brief Compute the bonded-skin extension from a frame's topology.
    ///
    /// Scans the per-atom bonded topology columns and returns
    /// the largest pairwise atom-to-atom distance found across any tuple.
    /// For an empty topology the result is `0.0`.
    ///
    /// Intended for unit testing and for callers that want the same number
    /// the component will use internally. The component invokes this
    /// helper once, inside `build()`, and caches the result.
    ///
    /// @param frame Frame whose positions and topology to scan.
    /// @return Maximum intra-tuple pairwise distance in Angstroms.
    static double compute_bonded_extension(const cpu::Frame& frame);

    /// @brief Test-only: per-swap native-byte send-buffer capacity after
    /// `build()` has preallocated the instance's own `FaceBufs`. Returns 0
    /// before `build()` or when MPI is not compiled in. Two instances
    /// built from different `dd->capacity` values are expected to report
    /// different capacities; used as a regression check that the PIMPL
    /// buffers are per-instance rather than a process-global static.
    std::size_t send_buf_capacity_for_test() const;

    /// @brief Resident heap bytes of the per-swap send/recv ghost buffers.
    /// Classified under the "Comm" category by the driver memory summary.
    std::size_t memory_bytes() const override;

  private:
    double cutoff_ = 0.0;
    double bonded_extension_ = 0.0;
    /// The halo's swap table — owned here, published non-owning as
    /// `ctx.md.swap_list`. Empty until `build()`.
    comm::SwapList swaps_;
    std::unique_ptr<detail::FaceBufs> bufs_;
    /// Field participants bound by the driver (non-owning); their
    /// `comm_forward_fields()` ride the per-face forward messages.
    std::vector<const HaloParticipant*> participants_;
    /// Flattened field descriptors, cached on first launch (participants'
    /// arrays are allocated in their build(), which may run after ours).
    std::vector<HaloParticipant::CommField> fwd_fields_;
    /// Reverse fields cached for ghost-row zeroing (forces semantics).
    std::vector<HaloParticipant::CommField> rev_fields_;
    bool fields_cached_ = false;

    // Reneighboring cadence. On `ago_ >= every_` or when migration bumps
    // `dd->neighbor_generation`, the next `launch()` re-derives the swap
    // table and rebuilds its send lists and ghost-slot assignments.
    // Otherwise the launch replays the table and only refreshes ghost
    // x/y/z.
    int every_ = 10;
    int ago_ = 0;
    std::uint64_t last_rebuild_generation_ = 0;
};

/// @brief Reverse pass: ghost-row forces → owner's force arrays.
///
/// Pairs with the most recent `HaloSend` exchange through the published
/// swap table (`ctx.md.swap_list`), walked in reverse so the forward
/// cascade plays backwards: last axis first, and inside an axis the deepest
/// forwarding round first, so a hop's ghost forces land on the intermediate
/// rank's own ghost rows before that rank passes them further. Ghost forces
/// in `kAtom_fx/fy/fz` are packed per swap from
/// `[firstrecv, firstrecv + nrecv)`, `MPI_Isend/Irecv` to the originating
/// ranks, and added into the owners' (or the next hop's) force array rows at
/// the indices in that swap's send list.
///
/// @see HaloSend for the matching producer of the table; a null
///      `ctx.md.swap_list` (no halo in this run) makes the whole pass a
///      rank-uniform no-op.
///
/// ## Reverse-comm payload audit (2026-05-06)
///
/// The reverse exchange transports every atoms-Block column marked
/// `kColReverse`, in its native dtype. The canonical schema marks
/// `fx/fy/fz`; runtime columns opt in with the same flag and require no
/// endpoint payload edit.
///
///   * Pair forces (`LJ`, future Coulomb) write to `fx/fy/fz`. ✓ shipped.
///   * Bonded forces (`BondHarmonic`, `AngleHarmonic`,
///     `DihedralMultiHarmonic`) write to `fx/fy/fz`. ✓ shipped.
///   * Per-atom virial / stress contributions are *accumulated globally*
///     via `frame.stress[6]` (a per-rank scalar tally) and reduced by
///     `Reduce` at `end_of_step @ -50`. They are not stored per
///     ghost row, so there is nothing for the halo reverse to ship.
///   * Per-atom kinetic energy is computed from owner velocities only
///     (loop bound `n_local`). Ghost velocities are never read.
///   * Per-atom potential energy similarly accumulates into the global
///     `frame.energy` scalar.
///
/// A future kernel that writes another ghost-row accumulator declares a
/// runtime column with `kColReverse`; the schema fingerprint and generic
/// pack/accumulate path carry it automatically.
/// @brief Re-send owner positions or velocities into the EXISTING ghost slots, at a
///        phase of the caller's choosing.
///
/// **The capability the step model was missing.** The engine performs exactly one
/// forward halo (`pre_neighbor`) and one reverse halo (`post_force`) per step, and the
/// forward payload carries POSITIONS ONLY. That is enough for every force style — a
/// pair kernel reads ghost positions and writes ghost forces, and both directions are
/// served. It is NOT enough for a **constraint**: RATTLE's velocity stage runs at
/// `final_integrate`, after the second half-kick, and needs the *current* velocities of
/// every atom in a cluster — including its ghosts. No communication happens anywhere
/// near that point, and ghost velocities are never shipped at all.
///
/// `HaloRefresh` closes that hole without disturbing the two exchanges that already
/// exist. It rebuilds nothing: it replays the swap table `HaloSend` published at the
/// last reneighbor — send lists, per-swap receive counts and ghost-slot layout — and
/// simply re-packs one field triple into the same slots. With no table published
/// (`ctx.md.swap_list == nullptr`, e.g. at step 0 before the first `HaloSend` launch,
/// or single-rank) it is a rank-uniform no-op. Bind it wherever the data is needed:
///
/// @code
///   md.on(md::Phase::post_integrate).add(comm::HaloRefresh{Field::positions}, -100);
///   md.on(md::Phase::final_integrate).add(comm::HaloRefresh{Field::velocities}, +150);
/// @endcode
///
/// It is a pure refresh, so it is idempotent and safe to run more than once per step —
/// unlike `HaloRecv`, whose reverse accumulation must run exactly once or it
/// double-applies ghost forces to their owners.
///
/// **Positions take the PBC shift; velocities and charges do not.** A ghost across a
/// periodic boundary sits at `x - L` in the receiver's frame, but its velocity and its
/// charge are unchanged. Shifting velocities would be an exquisitely silent bug:
/// energies stay finite, the geometry stays rigid, and only the constrained dynamics
/// is wrong.
///
/// **Charges** are the constant-potential case: every electrode charge solver
/// (`ConstPotBO`, `ConstPotMaZe`, `ConstPotCG`, `ConstPotentialBO`,
/// `DipoleElectrodeJoint`) rewrites owned charges at `pre_force`, i.e. AFTER the
/// step's forward halo — so the force stages would read ghost charges one solve
/// stale (and zero at step 0). `cpu::md::MD::install_defaults()` auto-injects
/// `HaloRefresh{Field::charges}` at `pre_force @ +900` whenever a registered
/// component declares `updates_owned_charges()` (see
/// Thompson et al., Comput. Phys. Commun. 271, 108171 (2022): LAMMPS
/// `fix electrode` forward-communicates q after every conp solve).
class HaloRefresh : public cpu::Component {
  public:
    /// @brief Which per-atom field set to refresh.
    enum class Field {
        positions,  ///< x, y, z — carries the periodic shift.
        velocities, ///< vx, vy, vz — no shift; a boost is not a translation.
        charges,    ///< charge — no shift; scalar, translation-invariant.
    };

    /// @param field Field set to re-send to the ghosts.
    explicit HaloRefresh(Field field) : field_(field) {}

    std::string name() const override {
        switch (field_) {
        case Field::positions:
            return "halo_refresh_x";
        case Field::velocities:
            return "halo_refresh_v";
        case Field::charges:
            return "halo_refresh_q";
        }
        return "halo_refresh";
    }

    /// @brief No default binding — the caller decides the phase, because only the
    /// caller knows when its data is stale.
    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    Field field_;
};

class HaloRecv : public cpu::Component, public cpu::HaloParticipant {
  public:
    HaloRecv();
    ~HaloRecv() override;
    HaloRecv(HaloRecv&&) noexcept;
    HaloRecv& operator=(HaloRecv&&) noexcept;
    HaloRecv(const HaloRecv&) = delete;
    HaloRecv& operator=(const HaloRecv&) = delete;

    std::string name() const override { return "halo_recv"; }
    /// @brief Receive half of the DD halo pair — the driver infrastructure
    /// scan probes this typed accessor, not `name()`.
    HaloParticipant* halo_participant() override { return this; }
    /// @brief Reverse exchange endpoint.
    Endpoint halo_endpoint() const override { return Endpoint::recv; }
    /// @brief Receive the field-participant list from the driver's
    /// build-time bind pass; reverse-field descriptors are flattened once
    /// at first launch and cached.
    void bind_participants(
        const std::vector<const HaloParticipant*>& participants) override {
        participants_ = participants;
    }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::post_force, -1000)};
    }

    /// @brief Preallocate per-face buffers from `dd->capacity`.
    /// @param ctx Build context.
    void build(const cpu::BuildContext& ctx) override;

    /// @brief Pack ghost-row forces, `MPI_Isend/Irecv` back to owners,
    /// accumulate into owner force arrays.
    /// @param ctx Launch context.
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Resident heap bytes of the per-face reverse-comm buffers.
    /// Classified under the "Comm" category by the driver memory summary.
    std::size_t memory_bytes() const override;

  private:
    std::unique_ptr<detail::FaceBufs> bufs_;
    /// Field participants bound by the driver (non-owning); their
    /// `comm_reverse_fields()` ride the per-face reverse messages.
    std::vector<const HaloParticipant*> participants_;
    /// Flattened reverse-field descriptors, cached on first launch.
    std::vector<HaloParticipant::CommField> rev_fields_;
    bool fields_cached_ = false;
    /// Schema + layout-guard Allreduce run only when `dd->halo_generation`
    /// differs; HaloSend bumps it on every rank at full rebuild.
    std::uint64_t last_checked_halo_generation_{};
};

} // namespace comm
} // namespace cpu
} // namespace ATV
