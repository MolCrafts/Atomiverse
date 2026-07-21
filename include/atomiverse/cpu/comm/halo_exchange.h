/// @file cpu/comm/halo_exchange.h
/// @brief Ghost-atom halo exchange components for MPI domain decomposition.
///
/// LAMMPS-style forward/reverse comm:
///   - `HaloSend::launch` rebuilds per-face send lists from
///     current positions, packs owned positions (with PBC shift across
///     periodic-edge faces), issues 6-face `MPI_Isend/Irecv`, and unpacks
///     received ghosts into `atoms` Block rows `[n_local, n_local+n_ghost)`.
///   - `HaloRecv::launch` symmetrically reduces ghost-row
///     `fx/fy/fz` contributions back onto the owners.
///
/// Each endpoint owns its per-face MPI buffers (`detail::FaceBufs`,
/// per-instance PIMPL); the shared per-face send lists and receive counts
/// live on `md::mpi::DomainDecomposition` (`dd->send_lists`,
/// `dd->recv_counts`). `HaloSend` is the producer (writes the counts and
/// the ghost rows); `HaloRecv` is the consumer (uses the same counts to
/// know how many ghost forces to pack and where they go on the owner's
/// side).
///
/// The default-constructed `HaloSend{}` (`cutoff = 0`) is a
/// no-op sentinel. Under `ATV_BUILD_MPI`, `cpu::md::MD` auto-injects the
/// `HaloSend{1.1 * pair_cutoff}` + `HaloRecv{}` pair when `add_potential` is
/// called; a manually registered pair suppresses the injection (detected
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
/// @see .claude/specs/parallel-md-engine.md  § A.4, A.5
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/halo_participant.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace comm {

namespace detail {
/// @brief Per-face MPI buffer storage (PIMPL). Definition lives in
/// `halo_exchange.cpp` so the header stays free of MPI-adjacent state.
/// Held per-instance by `HaloSend` / `HaloRecv` so
/// multiple components in the same process do not clobber each other's
/// send/recv buffers.
struct FaceBufs;
} // namespace detail

/// @brief Forward pass: owned positions → ghost rows on face neighbors.
///
/// Per-face send lists are rebuilt every step from current owned positions;
/// packed buffers are allocated once in `build()` based on a generous initial
/// estimate (no `launch()` realloc on the steady-state path). Single-rank
/// (`runtime.mpi_size == 1`) and `cutoff == 0` both early-return.
///
/// Atoms within `cutoff` of a face plane are flagged as ghosts to the
/// corresponding face neighbor. PBC: atoms sent across the periodic edge of
/// the global box are shifted by `±L_d` on the way out so the receiver sees
/// them in its own coordinate frame.
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

    /// @brief Test-only: per-face native-byte send-buffer capacity after
    /// `build()` has preallocated the instance's own `FaceBufs`. Returns 0
    /// before `build()` or when MPI is not compiled in. Two instances
    /// built from different `dd->capacity` values are expected to report
    /// different capacities; used as a regression check that the PIMPL
    /// buffers are per-instance rather than a process-global static.
    std::size_t send_buf_capacity_for_test() const;

    /// @brief Resident heap bytes of the per-face send/recv ghost buffers.
    /// Classified under the "Comm" category by the driver memory summary.
    std::size_t memory_bytes() const override;

  private:
    double cutoff_ = 0.0;
    double bonded_extension_ = 0.0;
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
    // `dd->neighbor_generation`, the next `launch()` does a full rebuild
    // of per-face send lists and ghost-slot assignments. Otherwise the
    // launch reuses existing send lists and only refreshes ghost x/y/z.
    int every_ = 10;
    int ago_ = 0;
    std::uint64_t last_rebuild_generation_ = 0;
};

/// @brief Reverse pass: ghost-row forces → owner's force arrays.
///
/// Pairs with the most recent `HaloSend` exchange via
/// the shared per-face state on `md::mpi::DomainDecomposition`. Ghost forces in
/// `kAtom_fx/fy/fz` ghost rows are packed by face,
/// `MPI_Isend/Irecv` to the originating ranks, and added into the owners'
/// local force arrays at the indices in their own `send_lists`.
///
/// @see HaloSend for the matching producer of `dd->send_lists` /
///      `dd->recv_counts`; Reverse consumes those counts to decide how
///      many ghost forces to pack and where to add them on the owner side.
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
/// exist. It rebuilds nothing: it reuses the send lists, per-face receive counts and
/// ghost-slot layout that `HaloSend` established at the last reneighbor, and simply
/// re-packs one field triple into the same slots. Bind it wherever the data is needed:
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
/// **Positions take the PBC shift; velocities do not.** A ghost across a periodic
/// boundary sits at `x - L` in the receiver's frame, but its velocity is unchanged.
/// Shifting velocities would be an exquisitely silent bug: energies stay finite, the
/// geometry stays rigid, and only the constrained dynamics is wrong.
class HaloRefresh : public cpu::Component {
  public:
    /// @brief Which per-atom field triple to refresh.
    enum class Field {
        positions,  ///< x, y, z — carries the periodic shift.
        velocities, ///< vx, vy, vz — no shift; a boost is not a translation.
    };

    /// @param field Field triple to re-send to the ghosts.
    explicit HaloRefresh(Field field) : field_(field) {}

    std::string name() const override {
        return field_ == Field::positions ? "halo_refresh_x" : "halo_refresh_v";
    }

    /// @brief Write-set: it writes ghost rows of the chosen field.

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
};

} // namespace comm
} // namespace cpu
} // namespace ATV
