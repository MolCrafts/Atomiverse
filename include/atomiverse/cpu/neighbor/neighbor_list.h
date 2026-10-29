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
/// `NeighborList::decide_reneighbor()` is that function (LAMMPS `nflag`),
/// plus one gate LAMMPS expresses as `box_change` on fixes that edit the
/// cell without moving atoms: a lattice-vector change larger than the
/// skin, a halo-pad change, or a subdomain change rebuilds immediately.
/// A Berendsen rescaling smaller than the skin does not — the affine
/// coordinate change is visible to `check_distance`, and
/// `CellList::build` still resizes bins from the live box when a rebuild
/// runs. MD Verlet calls this from `Runtime::step` and then either
/// `forward_comm` or `exchange+sort+borders+launch()` (build).
/// `launch()` always rebuilds; Runtime owns the decision to invoke it.
// Per-consumer half/full requests and derived half lists follow LAMMPS
// src/neigh_request.h and src/npair_halffull.cpp,
// a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b. Copyright (2003) Sandia
// Corporation. Under Contract DE-AC04-94AL85000, the U.S. Government retains
// certain rights. Distributed under the GNU General Public License.
#pragma once

#include <string>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/component/owned_range.h"
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
/// the stored CSR is reused verbatim. Runtime calls `decide_reneighbor`
/// and invokes `launch()` only when rebuilding; the decision scans owned
/// positions only when cadence permits.
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
    ///        owned atoms. Runtime applies it after exchange and before borders.
    ///        Default: enabled, nfreq=1000.
    /// @param nfreq Sort at the first reneighbor after this many MD steps.
    /// @param binsize Bin edge [Å]; `≤ 0` → `0.5 * (cutoff + skin)`.
    /// @param enabled Master enable.
    NeighborList& set_atom_sort(int nfreq, double binsize = 0.0,
                                bool enabled = true) & {
        sort_state_.configure(nfreq, binsize, enabled);
        return *this;
    }
    NeighborList set_atom_sort(int nfreq, double binsize = 0.0,
                               bool enabled = true) && {
        set_atom_sort(nfreq, binsize, enabled);
        return std::move(*this);
    }

    std::string name() const override { return "neighbor_list"; }
    bool verlet_engine() const override { return true; }

    /// @brief Supply side of the neighbor contract — the driver
    /// infrastructure scan probes this typed accessor, not `name()`.
    const cpu::NeighborProvider* as_neighbor_provider() const override {
        return this;
    }

    /// @brief Mutable supply side, for a driver that configures the list.
    cpu::NeighborProvider* neighbor_provider() override { return this; }

    /// @brief Distance this list enumerates pairs out to [Angstrom]:
    /// `cutoff + skin`, exactly the radius `build()` hands the spatial index.
    ///
    /// The auto-injected MPI halo is sized from this, so the ghost shell covers
    /// the whole skin band rather than the interaction cutoff alone (see
    /// `NeighborProvider::neighbor_reach`).
    /// @return Enumeration radius [Angstrom].
    AccReal supported_cutoff() const override { return cutoff_; }

    AccReal neighbor_reach() const override {
        return static_cast<AccReal>(cutoff_ + strategy_.skin());
    }

    void activate(cpu::Context& rt) override {
        rt.md.atom_sort = &sort_state_;
        rt.md.pair_list = want_half_ ? &pairs_ : nullptr;
        rt.md.full_pair_list = want_full_ ? &full_pairs_ : nullptr;
    }

    void configure_lists(bool half, bool full,
                         bool ghost_rows = false) override {
        if (ghost_rows && !full)
            throw std::invalid_argument(
                "ghost neighbour rows require a full list");
        if (half != want_half_ || full != want_full_
            || ghost_rows != want_ghost_rows_) {
            want_half_ = half;
            want_full_ = full;
            want_ghost_rows_ = ghost_rows;
            lists_stale_ = true;
        }
    }

    /// @brief LAMMPS `Neighbor::decide()` (`nflag`). Increments `ago_`.
    ///        Sort cadence is stamped from `LaunchContext::step` at the
    ///        sort site, not from this call.
    bool decide_reneighbor(cpu::Context& rt, cpu::Frame& frame,
                           bool md_cadence) override {
        ++ago_;
        // `grid_stale()` is true on every NPT step (the cell matrix is
        // compared exactly). That must not bypass delay/check: a sub-skin
        // rescaling is absorbed by the Verlet list, and bin sizes are
        // refreshed inside `launch()` when a rebuild does happen.
        if (lists_stale_ || frame.nlocal() != saved_owned_
            || frame.n_total() != saved_active_
            || strategy_.geometry_forces_rebuild(frame)
            || (rt.md.dd
                && (rt.md.dd->partition_dirty
                    || rt.md.dd->halo_generation != last_halo_generation_)))
            return true;
        // LAMMPS Min::init: every=1, delay=0, dist_check=1 while minimizing.
        if (md_cadence && (ago_ < delay_ || ago_ % every_ != 0))
            return false;
        if (md_cadence && !check_)
            return true;
        const StorageReal* x = frame.span(kAtom_x);
        const StorageReal* y = frame.span(kAtom_y);
        const StorageReal* z = frame.span(kAtom_z);
        const double limit = 0.25 * strategy_.skin() * strategy_.skin();
        // LAMMPS Neighbor::check_distance: owners move continuously between
        // rebuilds. Runtime wraps only before exchange, never between checks.
        for (int i = 0; i < saved_owned_; ++i) {
            const double dx = x[i] - saved_positions_[i][0];
            const double dy = y[i] - saved_positions_[i][1];
            const double dz = z[i] - saved_positions_[i][2];
            if (dx * dx + dy * dy + dz * dz > limit)
                return true;
        }
        return false;
    }

    /// @brief Drop `atom_modify sort` permanently (see `NeighborProvider`).
    void forbid_row_reordering() override { sort_state_.enabled = false; }

    void build(const cpu::BuildContext& ctx) override {
        reinit_strategy(ctx.runtime, ctx.frame);
        saved_positions_.reserve(ctx.frame.capacity());
        saved_owned_ = saved_active_ = -1;
        lists_stale_ = true;
        ago_ = 0;
        n_rebuilds_ = 0;
        activate(ctx.runtime);
    }

    // Runtime is the sole caller that decides whether this rebuild operation
    // runs. Initial build happens after exchange/sort/borders, just like every
    // later rebuild; lifecycle allocation never publishes a pre-halo CSR.
    void launch(const cpu::LaunchContext& ctx) override {
        if (strategy_dd_ != ctx.runtime.md.dd)
            reinit_strategy(ctx.runtime, ctx.frame);
        const int n_active = ctx.frame.n_total();
        strategy_.build(ctx.frame, n_active);
        refresh_(ctx.runtime, ctx.frame, n_active);
        saved_owned_ = ctx.frame.nlocal();
        saved_active_ = n_active;
        saved_positions_.resize(saved_owned_);
        const StorageReal* x = ctx.frame.span(kAtom_x);
        const StorageReal* y = ctx.frame.span(kAtom_y);
        const StorageReal* z = ctx.frame.span(kAtom_z);
        for (int i = 0; i < saved_owned_; ++i)
            saved_positions_[i] = Vec3{x[i], y[i], z[i]};
        ago_ = 0;
        ++n_rebuilds_;
        last_halo_generation_ =
            ctx.runtime.md.dd ? ctx.runtime.md.dd->halo_generation : 0;
        activate(ctx.runtime);
    }

    void destroy(const cpu::DestroyContext& ctx) override {
        if (ctx.runtime.md.pair_list == &pairs_)
            ctx.runtime.md.pair_list = nullptr;
        if (ctx.runtime.md.full_pair_list == &full_pairs_)
            ctx.runtime.md.full_pair_list = nullptr;
        if (ctx.runtime.md.atom_sort == &sort_state_)
            ctx.runtime.md.atom_sort = nullptr;
    }

    /// @brief Completed enumerations since lifecycle build (including setup launch).
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
        return pairs_.bytes() + full_pairs_.bytes() + strategy_.bytes()
               + saved_positions_.capacity() * sizeof(Vec3);
    }

  private:
    /// Reuse one spatial index for the independently requested counting modes.
    void refresh_(const cpu::Context&, cpu::Frame& f, int n_active) {
        const int n_local = f.nlocal();
        if (want_full_) {
            const int n_rows =
                n_local >= 0 && !want_ghost_rows_ ? n_local : n_active;
            strategy_.refresh_full(f, n_active, full_pairs_, n_rows);
            ++full_pairs_.version;
            full_pairs_.inum = n_local;
        }
        if (want_half_) {
            if (want_full_)
                strategy_.derive_half(f, n_active, full_pairs_, pairs_,
                                      n_local);
            else
                strategy_.refresh_half(f, n_active, pairs_, n_local);
            pairs_.n_rows = n_active;
            ++pairs_.version;
            pairs_.inum = n_local;
        }
        lists_stale_ = false;
    }

    /// @brief (Re-)initialise the underlying spatial-index strategy from
    /// the current frame.box.  Called from `build()` (one-time setup)
    /// and from `launch()` whenever `frame.box` has drifted enough that
    /// the cell-grid sizing is stale.
    void reinit_strategy(const cpu::Context& rt, cpu::Frame& f) {
        // The strategy is sized and filtered by the VERLET reach, not the
        // bare cutoff: a list built to `cutoff` alone drops every pair in
        // the skin band and stops being reusable between rebuilds.
        // (`skin()` is the constructor's negative-means-auto value already
        // resolved, so this is exactly what the strategy derives itself.)
        const double cutoff_skin = cutoff_ + strategy_.skin();
        strategy_dd_ = rt.md.dd;
        if (strategy_dd_) {
            strategy_.init_extended(f, cutoff_skin, rt.md.dd, f.capacity());
        } else {
            strategy_.init(f, cutoff_skin);
        }
    }

    Strategy strategy_;
    const mpi::DomainDecomposition* strategy_dd_ = nullptr;
    std::vector<Vec3> saved_positions_;
    int saved_owned_ = -1;
    int saved_active_ = -1;
    PairList pairs_;
    PairList full_pairs_;
    AtomSortState sort_state_{}; ///< LAMMPS atom_modify sort (at reneighbor)
    bool want_half_ = true;
    bool want_full_ = false;
    bool want_ghost_rows_ = false;
    bool lists_stale_ = false;
    double cutoff_;
    int every_ = 1;
    int delay_ = 10;
    /// @brief LAMMPS `neigh_modify check` flag. `true` (= `check yes`):
    /// displacement-gated rebuild. `false` (= `check no`): rebuild on every
    /// `every`/`delay`-eligible step regardless of displacement.
    bool check_ = true;
    /// @brief Steps since the last enumeration, incremented by each decision.
    int ago_ = 0;
    /// @brief Completed enumerations, including the initial setup launch.
    int n_rebuilds_ = 0;
    /// @brief Halo generation at the last enumeration, including self images.
    std::uint64_t last_halo_generation_ = 0;
};

} // namespace md
} // namespace cpu
} // namespace ATV
