/// @file cpu/potentials/eam.h
/// @brief cpu::EAM — analytic embedded-atom many-body force style.
///
/// Density/embedding communication algorithm ported from LAMMPS
/// `src/MANYBODY/pair_eam.h` and `src/OPENMP/pair_eam_omp.cpp`, checkout `a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b`.
/// Copyright (2003) Sandia Corporation. Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software. Distributed under the GNU General
/// Public License (GPL-2.0-only).
#pragma once

#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/potentials/eam_potential.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

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
/// **Two half-list passes, one launch.** Density contributions accumulate onto
/// both pair endpoints, then reverse communication sums ghost contributions onto
/// their owners. Each owner computes its embedding derivative; forward
/// communication supplies it to ghosts before the force pass. The coordinate
/// halo therefore needs only one cutoff, and no ghost-centred neighbour rows.
class EAM : public cpu::Potential,
            public cpu::NeighborConsumer,
            private cpu::ForwardPacker,
            private cpu::ReversePacker {
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

    NeighborListMode neighbor_list_mode() const override {
        return NeighborListMode::Half;
    }
    const cpu::NeighborConsumer* as_neighbor_consumer() const override {
        return this;
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;
    void destroy(const cpu::DestroyContext& ctx) override;

    /// @brief Resident heap bytes (the per-atom density + embedding-derivative
    /// arrays). Classified under "Pair" by the driver memory summary.
    std::size_t memory_bytes() const override {
        std::size_t bytes = (rho_.capacity() + dF_.capacity()) * sizeof(double);
        for (const auto& density : rho_threads_)
            bytes += density.capacity() * sizeof(double);
        return bytes;
    }

  private:
    int comm_forward() const override { return 1; }
    int pack_forward_comm(int n, const int* list, double* buf, int pbc_flag,
                          const int* pbc) override;
    void unpack_forward_comm(int n, int first, const double* buf) override;
    int comm_reverse() const override { return 1; }
    int pack_reverse_comm(int n, int first, double* buf) override;
    void unpack_reverse_comm(int n, const int* list,
                             const double* buf) override;

    EAMPotential pot_;
    double cutoff_;
    double phi_shift_ =
        0.0; ///< phi(cutoff), subtracted so the pair term is continuous.

    /// Per-atom local density and embedding derivative. Sized to the atom count
    /// at build and regrown in launch if the frame grew.
    std::vector<double> rho_;
    std::vector<double> dF_;
    std::vector<std::vector<double>> rho_threads_;
};

} // namespace cpu
} // namespace ATV
