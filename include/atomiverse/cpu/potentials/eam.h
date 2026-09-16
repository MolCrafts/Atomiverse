/// @file cpu/potentials/eam.h
/// @brief cpu::EAM — analytic embedded-atom many-body force style.
#pragma once

#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/potentials/eam_potential.h"

namespace ATV {
namespace cpu {

/// @brief Embedded-atom-method force style (single element, analytic Sutton-Chen).
///
/// The first genuinely many-body potential in the engine, and deliberately so:
/// it is the probe that establishes what a non-pairwise style needs from the
/// Component / Neighbor / Comm layers.
///
/// **Why it is not a `Pair`.** `cpu::Pair` models `fpair(r^2)` — one scalar per
/// separation — and marks `launch()`/`role()` `final`; `set_evaluator` is a singular
/// slot. EAM's force on `i` depends on the *density* at `i` **and at every
/// neighbour `j`**, so it is not expressible in that shape. It therefore follows
/// the `CoulLong` precedent instead: a plain `Component + NeighborConsumer`,
/// registered explicitly with `.on(md::Phase::force).add_ref(eam)`.
///
/// **Two passes, one launch.** Pass 1 accumulates `rho_i = sum_j f(r_ij)` for
/// every atom; pass 2 uses `F'(rho_i)` and `F'(rho_j)` to build the forces. Both
/// live in one `launch()` — the `force` phase's order mechanism could host them as
/// two components, but they share a per-atom array and nothing else consumes
/// `rho`, so splitting them would buy nothing and cost a second Plan slot.
///
/// **It needs the FULL neighbor list.** `rho_i` is a sum over *all* neighbours of
/// `i`; a half list gives each atom only the half of its neighbours that landed in
/// its own CSR row. The component sets `ctx.md.full_list_requested` in `build()`,
/// and `NeighborList` switches to `CellList::refresh_full`.
///
/// **Multi-rank works, via a wider halo rather than an extra message.** Pass 2 needs
/// `F'(rho_j)` where `j` may be a ghost, and a ghost's own `rho` is a sum over *its*
/// neighbours — so it can only be completed locally if the ghost shell reaches
/// `2 * cutoff`. The component declares exactly that through
/// `NeighborConsumer::halo_cutoff()`. The alternative — exchanging `rho` in the middle
/// of the force phase — is not available: the engine runs exactly one forward halo
/// (`pre_neighbor`) and one reverse halo (`post_force`, payload audited to fx/fy/fz),
/// with no communication point inside `force`. The wide halo sidesteps that hole
/// instead of punching a new one through it; it costs memory and a fatter ghost
/// exchange, and it costs no new invariant.
class EAM : public cpu::Potential, public cpu::NeighborConsumer {
  public:
    /// @param potential Analytic EAM parameterization (constructed by lookup).
    /// @param cutoff Interaction cutoff [Angstrom]. The Sutton-Chen form is a
    ///        power law with no intrinsic range, so the cutoff is the user's
    ///        convergence knob; the energy is shifted to zero at `cutoff` so the
    ///        pair term is continuous there.
    EAM(EAMPotential potential, double cutoff);

    std::string name() const override { return "eam"; }
    /* role via Potential/kind base */

    /// @brief Cutoff the neighbor list must cover [Angstrom].
    AccReal neighbor_cutoff() const override { return cutoff_; }

    /// @brief The halo must reach 2 * cutoff — a GHOST-OF-GHOST shell.
    ///
    /// Pass 2 needs `F'(rho_j)` for a ghost `j`, and `rho_j` sums over *j's*
    /// neighbours. With a ghost shell of only `r_c`, a ghost's own neighbour list is
    /// truncated and its density is silently too small. Widening the shell to `2 r_c`
    /// makes every ghost's rho locally computable, and buys multi-rank EAM without a
    /// single extra message — which matters, because the engine has exactly one
    /// forward halo (pre_neighbor) and one reverse halo (post_force) per step, and no
    /// communication point at all inside the force phase.
    AccReal halo_cutoff() const override { return 2.0 * cutoff_; }
    const cpu::NeighborConsumer* as_neighbor_consumer() const override {
        return this;
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Resident heap bytes (the per-atom density + embedding-derivative
    /// arrays). Classified under "Pair" by the driver memory summary.
    std::size_t memory_bytes() const override {
        return (rho_.capacity() + dF_.capacity()) * sizeof(double);
    }

  private:
    EAMPotential pot_;
    double cutoff_;
    double phi_shift_ =
        0.0; ///< phi(cutoff), subtracted so the pair term is continuous.

    /// Per-atom local density and embedding derivative. Sized to the atom count
    /// at build and regrown in launch if the frame grew.
    std::vector<double> rho_;
    std::vector<double> dF_;
};

} // namespace cpu
} // namespace ATV
