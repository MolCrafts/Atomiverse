/// @file cpu/neighbor/neighbor_list.h
/// @brief NeighborList -- component wrapper for spatial index strategies.
///
/// @tparam Strategy Spatial indexing strategy (CellList, NSQ, etc.).
///
/// ## LAMMPS correspondence (`neighbor` + `neigh_modify`)
///
/// `NeighborList` mirrors the LAMMPS `neighbor` command (Verlet skin) and
/// the `neigh_modify` keywords. Audited against `lammps/src/neighbor.cpp`
/// (`stable` branch) — the rebuild gate replicates `Neighbor::decide()`
/// (neighbor.cpp:2360) and the displacement test replicates
/// `Neighbor::check_distance()` (neighbor.cpp:2390).
///
/// | LAMMPS keyword            | `NeighborList`            | Default        | Verdict |
/// |---------------------------|---------------------------|----------------|---------|
/// | `neighbor skin style`     | `skin` ctor arg [Å]       | `0.3·cutoff`   | aligned (semantic); LAMMPS has no auto-skin — the auto default is a documented divergence (convenience, opt-out by passing skin explicitly) |
/// | `neigh_modify every N`    | `every` ctor arg          | `1`            | aligned |
/// | `neigh_modify delay M`    | `delay` ctor arg          | `10`           | aligned |
/// | `neigh_modify check yes/no` | `check` ctor arg (bool) | `true` (= `yes`) | aligned — see below |
/// | `neigh_modify one N`      | (not exposed)             | —              | **documented divergence**: Atomiverse's neighbor storage is a growable CSR arena (`PairList::neighbors` is a `std::vector<int>`, the xtensor CSR rows grow via `ensure_at_least`). `one` is purely a LAMMPS allocation bound; with a growable arena it has no semantic effect, so no false parameter is introduced. |
/// | `neigh_modify binsize`    | (not exposed)             | —              | **documented divergence**: the `CellList` strategy derives its cell size internally from `cutoff + skin`. LAMMPS exposes `binsize` to tune the binned-neighboring bin; Atomiverse's internal sizing makes it unnecessary, so the `bin` style is not exposed. |
///
/// **Rebuild gate** — `Neighbor::decide()` is, verbatim:
/// ```
///   ago++;
///   if (ago >= delay && ago % every == 0) {
///     if (build_once) return 0;
///     if (dist_check == 0) return 1;   // check no  → unconditional rebuild
///     return check_distance();          // check yes → displacement-gated
///   } else return 0;
/// ```
/// `NeighborList::launch()` reproduces this exactly: `ago_` is incremented,
/// then `delay_ok = ago_ >= delay_` AND `every_ok = ago_ % every_ == 0`
/// gate the rebuild; when both hold, `check no` (`check_ == false`) rebuilds
/// unconditionally and `check yes` consults the skin displacement test
/// (`max_disp·2 > skin`, the half-skin trigger of `check_distance()`).
/// Atomiverse's `size_stale` short-circuit (atom-count change from MPI
/// migration / halo resize) is an Atomiverse-only addition with no LAMMPS
/// keyword — it is always-on and correct (a changed atom set always
/// invalidates the stored CSR).
#pragma once

#include <string>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/dynamics/atom_sort.h"
#include "atomiverse/cpu/neighbor/cell_list.h"
#include "atomiverse/cpu/neighbor/pair_list.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Neighbor list component -- builds and maintains spatial index.
/// @tparam Strategy Spatial indexing strategy (CellList by default).
///
/// Rebuild cadence mirrors LAMMPS `neigh_modify every N delay M check yes/no`:
/// with `check yes` (default) the Verlet list is re-enumerated only when
/// `ago >= delay` *and* `ago % every == 0` *and* the max-displacement skin
/// check fires; with `check no` the displacement test is skipped and the
/// list rebuilds on *every* `every`/`delay`-eligible step. Between rebuilds
/// the stored CSR is reused verbatim — force kernels continue reading the
/// same `pair_list`. Cost of `launch()` on non-rebuild steps collapses to a
/// single O(n) skin scan (or zero when `ago < every` or `ago < delay`).
/// Defaults match LAMMPS: `every=1`, `delay=10`, `check=true` (`check yes`).
template <typename Strategy = CellList>
class NeighborList : public cpu::Component, public cpu::NeighborProvider {
  public:
    Role role() const override { return role_bit(RoleFlag::FrameImmutable); }

    /// @param cutoff Interaction cutoff [Angstrom].
    /// @param skin Verlet skin width [Angstrom]. Negative = auto (0.3 * cutoff).
    /// @param every Attempt rebuild at most every N steps (LAMMPS `neigh_modify
    ///        every`; default 1 reproduces LAMMPS's built-in default).
    /// @param delay Skip displacement check for this many steps after a rebuild
    ///        (LAMMPS `neigh_modify delay`; default 10 reproduces LAMMPS).
    /// @param check Displacement-gated rebuild (LAMMPS `neigh_modify check`;
    ///        default `true` = `check yes`). `false` (= `check no`) skips the
    ///        skin displacement test, so the list rebuilds on every
    ///        `every`/`delay`-eligible step — matching `dist_check == 0` in
    ///        `Neighbor::decide()` (neighbor.cpp:2373, `return 1`).
    explicit NeighborList(double cutoff, double skin = -1.0, int every = 1,
                          int delay = 10, bool check = true)
        : strategy_(cutoff, skin), cutoff_(cutoff),
          every_(every < 1 ? 1 : every), delay_(delay < 0 ? 0 : delay),
          check_(check) {}

    /// @brief LAMMPS `atom_modify sort Nfreq binsize` — spatial reorder of
    ///        owned atoms. Multi-rank: applied in HaloSend before borders;
    ///        serial: at NeighborList reneighbor. Default: enabled, nfreq=1000.
    NeighborList& set_atom_sort(AtomSortParams p) & {
        sort_state_.configure(p);
        return *this;
    }
    NeighborList set_atom_sort(AtomSortParams p) && {
        set_atom_sort(p);
        return std::move(*this);
    }

    std::string name() const override { return "neighbor_list"; }

    /// @brief Supply side of the neighbor contract — the driver
    /// infrastructure scan probes this typed accessor, not `name()`.
    const cpu::NeighborProvider* as_neighbor_provider() const override {
        return this;
    }

    /// @brief Mutable supply side, for a driver that configures the list.
    cpu::NeighborProvider* neighbor_provider() override { return this; }

    /// @brief Drop `atom_modify sort` permanently (see `NeighborProvider`).
    void forbid_row_reordering() override {
        AtomSortParams p = sort_state_.params;
        p.enabled = false;
        sort_state_.configure(p);
    }

    void build(const cpu::BuildContext& ctx) override {
        reinit_strategy(ctx.runtime, ctx.frame);
        const int n_active = active_count(ctx.runtime, ctx.frame);
        strategy_.build(ctx.frame, n_active);
        // One-time arena sizing. The xtensor CSR arrays grow (never shrink)
        // through `refresh_half`'s `ensure_at_least`; prime them here to
        // the expected upper bound so steady-state rebuilds don't enter
        // the allocator. `pairs_.neighbors` is a `std::vector<int>` and
        // amortises growth itself -- leave it empty, the first refresh
        // reaches the correct size within one resize.
        refresh_(ctx.runtime, ctx.frame, n_active);
        ++pairs_.version;
        trim_inum_to_local(ctx.runtime, pairs_);
        ctx.runtime.md.pair_list = &pairs_;
        ago_ = 0;
        n_rebuilds_ = 1; // initial enumeration counts as rebuild #1
        // Snapshot the halo generation so the first `launch()` doesn't
        // re-rebuild against a freshly built CSR. (Pre-`launch()` the
        // first halo full rebuild may or may not have run yet — under MD
        // it has, via `pre_force`; under `Evaluate` there is no halo
        // launch before the first force eval, hence the
        // `is_multi_rank` guard.)
        if (is_multi_rank(ctx.runtime)) {
            last_halo_generation_ = ctx.runtime.md.dd->halo_generation;
            // Publish sort cadence for HaloSend (exchange → sort → borders).
            auto* dd = ctx.runtime.md.dd;
            dd->atom_sort_enabled = sort_state_.params.enabled;
            dd->atom_sort_nfreq = sort_state_.params.nfreq;
            dd->atom_sort_binsize = sort_state_.params.binsize;
            dd->atom_sort_neigh_cut = cutoff_ + strategy_.skin();
            dd->atom_sort_step = sort_state_.step;
            dd->atom_sort_next = sort_state_.next_sort;
        }
    }

    void launch(const cpu::LaunchContext& ctx) override {
        const int n_active = active_count(ctx.runtime, ctx.frame);
        ++ago_;
        ++sort_state_.step; // LAMMPS ntimestep proxy for atom_modify sort

        // Rebuild decision mirrors LAMMPS `Neighbor::decide()` (neighbor.cpp:2360):
        //     ago++;
        //     if (ago >= delay && ago % every == 0) {
        //       if (dist_check == 0) return 1;   // check no
        //       return check_distance();          // check yes
        //     } else return 0;
        //   size_stale: atom count changed (migration / halo resize) → force
        //               rebuild. Atomiverse-only short-circuit, no LAMMPS keyword.
        //   halo_stale: HaloSend did a full rebuild that may have
        //               reshuffled ghost slot indices (different atoms in the
        //               same slot, or atoms relocated to a different slot
        //               because an earlier-axis swap's received count
        //               changed). The pair
        //               list CSR stores `j` as raw slot indices, so a
        //               reshuffle without a rebuild leaves the LJ + bonded
        //               kernels reading stale (wrong-atom) positions —
        //               observed as a +1-3 eV per-halo-cycle drift of the
        //               MTK extended Hamiltonian under MSU stage-2 NPT,
        //               accumulating to a step-1000 catastrophic ε̇ runaway.
        //   delay gate:  ago < delay → no rebuild attempt (LAMMPS `delay`)
        //   every gate:  ago % every != 0 → no rebuild attempt (LAMMPS `every`)
        //   check yes:   max_disp * 2 > skin → rebuild (half-skin trigger)
        //   check no:    rebuild unconditionally once the every/delay gate opens
        //   mode_stale:  a consumer asked for a FULL list (ctx.md.full_list_requested)
        //               after we had already built a half list — or vice versa.
        //               Component build() runs in Plan order and the neighbor
        //               phase precedes force, so the requester's build() lands
        //               after ours; resolving the mode lazily here is the same
        //               deal the CUDA flat-pairs export strikes.
        const bool mode_stale =
            ctx.runtime.md.full_list_requested != built_full_;
        const bool size_stale = n_active != strategy_.n_at_last_build();
        const bool halo_stale =
            is_multi_rank(ctx.runtime)
            && ctx.runtime.md.dd->halo_generation != last_halo_generation_;
        const bool delay_ok = ago_ >= delay_;
        const bool every_ok = (ago_ % every_) == 0;
        // When the gate is open: check no → always rebuild; check yes →
        // consult the displacement skin test. `needs_rebuild` is only
        // evaluated for `check yes` so `check no` runs pay no O(n) skin scan.
        const bool disp_rebuild =
            delay_ok && every_ok
            && (!check_ || strategy_.needs_rebuild(ctx.frame, n_active));
        if (mode_stale || size_stale || halo_stale || disp_rebuild) {
            // Serial only: multi-rank sort runs in HaloSend before borders
            // (LAMMPS exchange → sort → borders). Sorting here after halo
            // would invalidate the swap table's reverse send lists.
            //
            // Must notify_sorted so BondedTupleCache rebuilds: on serial MD
            // `dd` still exists, and cache staleness is keyed on
            // neighbor_generation. Skipping the bump leaves bonded forces
            // on pre-sort local indices → GAFF trajectory explosion at
            // nfreq=1.
            if (!is_multi_rank(ctx.runtime) && sort_state_.due()) {
                auto& f = ctx.frame;
                const double bs =
                    sort_state_.resolved_binsize(cutoff_ + strategy_.skin());
                // No swap table on this path by construction: it is the
                // single-rank branch, where `HaloSend` early-returns and
                // publishes nothing.
                if (sort_owned_atoms(f, f.nlocal(), bs))
                    notify_sorted(ctx.runtime.md.dd, f, /*swaps=*/nullptr,
                                  /*clear_ghosts=*/true);
                sort_state_.mark_sorted();
            }
            strategy_.build(ctx.frame, n_active);
            refresh_(ctx.runtime, ctx.frame, n_active);
            ++pairs_.version;
            ago_ = 0;
            ++n_rebuilds_;
            if (is_multi_rank(ctx.runtime))
                last_halo_generation_ = ctx.runtime.md.dd->halo_generation;
        }

        trim_inum_to_local(ctx.runtime, pairs_);
        // Re-publish defensively: the component that owns pairs_ is stable
        // across launches, but a subsequent Backend::build_components() cycle
        // (e.g. after a frame-resize) would reset the runtime pointer.
        ctx.runtime.md.pair_list = &pairs_;
    }

    /// @brief Total rebuild count since construction (includes initial build()).
    int n_rebuilds() const { return n_rebuilds_; }

    /// @brief LAMMPS `neigh_modify check` flag (`true` = `check yes`).
    bool check() const { return check_; }

    /// @brief Verlet skin width [Angstrom] (negative input resolved to the
    /// `0.3*cutoff` auto value during construction). Surfaced for the
    /// MD/Evaluate auto-injection path that extends the halo cutoff by
    /// the skin to keep ghosts stable across full-rebuild boundaries.
    double skin() const { return strategy_.skin(); }

    /// @brief Access the current pair list (read-only).
    const PairList& pairs() const { return pairs_; }

    /// @brief Access the underlying strategy.
    const Strategy& strategy() const { return strategy_; }

    /// @brief Resident heap bytes: the CSR neighbor list + the spatial index.
    /// Classified under the "Neighbor" category by the driver memory summary.
    std::size_t memory_bytes() const override {
        return pairs_.bytes() + strategy_.bytes();
    }

  private:
    /// @brief True when running with multiple MPI ranks (and the DD struct
    /// is available). Single-rank or null-`dd` returns false.
    static bool is_multi_rank(const cpu::Context& rt) {
        return rt.md.dd != nullptr && rt.mpi_size > 1;
    }

    /// @brief Active atom count: owners + current ghosts under multi-rank,
    /// otherwise the full atom array (single-rank legacy).
    static int active_count(const cpu::Context& rt, const cpu::Frame& frame) {
        if (is_multi_rank(rt))
            return rt.md.dd->n_local + rt.md.dd->n_ghost;
        return frame.nlocal();
    }

    /// @brief Multi-rank trim: `inum = dd->n_local` so force kernels only
    /// iterate owned atoms as `i`, while `neighbors[]` may still point to
    /// ghost rows. Single-rank: no-op.
    static void trim_inum_to_local(const cpu::Context& rt, PairList& pl) {
        if (is_multi_rank(rt)) {
            pl.inum = rt.md.dd->n_local;
        }
    }

    /// @brief Rebuild the CSR in the mode the consumers asked for.
    ///
    /// Half list (the default) or full list (`ctx.md.full_list_requested` — set by
    /// an EAM/3-body style that needs every neighbour of `i`, both directions).
    /// The full path takes no owner count: the half list needs the `tag[j] < tag[i]`
    /// dedup so an owner-ghost pair is counted on exactly one rank, but a full list
    /// carries both directions by construction and each rank keeps only the legs
    /// incident on its own owners, so there is nothing to deduplicate.
    ///
    /// Two call sites (`build()` and `launch()`) — which is why this is a function
    /// at all.
    void refresh_(const cpu::Context& rt, cpu::Frame& f, int n_active) {
        const bool want_full = rt.md.full_list_requested;
        if (want_full) {
            strategy_.refresh_full(f, n_active, pairs_);
        } else {
            strategy_.refresh_half(f, n_active, pairs_,
                                   is_multi_rank(rt) ? rt.md.dd->n_local : -1);
        }
        built_full_ = want_full;
        pairs_.n_rows =
            n_active; // owners + ghosts; `inum` is trimmed to owners later
    }

    /// @brief (Re-)initialise the underlying spatial-index strategy from
    /// the current frame.box.  Called from `build()` (one-time setup)
    /// and from `launch()` whenever `frame.box` has drifted enough that
    /// the cell-grid sizing is stale.
    void reinit_strategy(const cpu::Context& rt, cpu::Frame& f) {
        if (is_multi_rank(rt)) {
            strategy_.init_extended(f, cutoff_, rt.md.dd, rt.md.dd->capacity);
        } else {
            strategy_.init(f, cutoff_);
        }
    }

    Strategy strategy_;
    PairList pairs_;
    AtomSortState sort_state_{}; ///< LAMMPS atom_modify sort (at reneighbor)
    /// @brief Mode the currently-published CSR was built in. A mismatch against
    /// `ctx.md.full_list_requested` forces a rebuild (see `launch()`).
    bool built_full_ = false;
    double cutoff_;
    int every_ = 1;
    int delay_ = 10;
    /// @brief LAMMPS `neigh_modify check` flag. `true` (= `check yes`):
    /// displacement-gated rebuild. `false` (= `check no`): rebuild on every
    /// `every`/`delay`-eligible step regardless of displacement.
    bool check_ = true;
    /// @brief Steps since the last actual rebuild. Starts at 0 after `build()`
    /// (the initial enumeration), incremented once per `launch()`.
    int ago_ = 0;
    /// @brief Total rebuild count (includes the initial `build()` enumeration).
    int n_rebuilds_ = 0;
    /// @brief Last observed value of `dd->halo_generation` (multi-rank only).
    /// Used to force a CSR rebuild when `HaloSend`'s full rebuild
    /// reshuffles ghost slot indices — see the `halo_stale` branch in
    /// `launch()`. Initialised to 0 so the very first launch (which
    /// initialises the CSR via `build()` already) doesn't double-rebuild.
    std::uint64_t last_halo_generation_ = 0;
};

} // namespace md
} // namespace cpu
} // namespace ATV
