/// @file cpu/communication/rebalance.h
/// @brief `Rebalance` -- non-uniform brick load balancer.
///
/// Recomputes per-axis cut planes from a globally-reduced
/// atom-density histogram so each rank ends up with approximately the same
/// number of owned atoms. The decomposition stays a 3D Cartesian brick
/// (the existing halo / migration topology); only the slab widths change.
///
/// Hard prerequisite: a multi-rank `DomainDecomposition` initialised by
/// `DomainRuntime`. Single-rank: structural no-op (`launch()` early-returns).
///
/// Effect of one balance pass:
///   1. Build local atom-position histograms for x, y, z.
///   2. `MPI_Allreduce(SUM)` to a global histogram.
///   3. Bisect each axis's histogram into `dims[d]` equal-count buckets;
///      store the resulting `dims[d] + 1` cut planes (as box-length fractions)
///      in `dd->cuts[d]`.
///   4. Refresh this rank's `sublo[]/subhi[]` via `compute_sublo_subhi(box)`.
///   5. Bump `dd->neighbor_generation` (bonded-cache invalidation) and set
///      `dd->partition_dirty` so `NeighborList::decide_reneighbor` returns
///      true and Verlet runs `exchange` + `borders` on the next step.
///
/// The component itself does not move atoms -- it only updates the partition.
/// Migration is `Comm::exchange`, which consults `dd->cuts[]` when populated.
///
/// **Topology constraint.** This is a non-uniform *brick* balance, not full
/// recursive coordinate bisection (which would produce irregular subdomains
/// incompatible with the existing 6-face halo). The simpler axis-aligned
/// scheme matches LAMMPS's `balance dynamic xyz`.
///
/// Algorithm and communication structure transplanted from LAMMPS
/// `src/balance.cpp` (`balance dynamic xyz`, checkout `78a7ddf`).
///
/// Copyright (2003) Sandia Corporation.  Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software.  This software is distributed under
/// the GNU General Public License (GPL-2.0-only).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {
namespace md {
namespace mpi {

/// @brief Non-uniform brick load balancer based on a global atom histogram.
class Rebalance : public cpu::Component {
  public:
    /// @param every Step interval between balance passes. Default 1000.
    /// @param n_bins Bins per axis for the global histogram. Default 256;
    ///        higher resolution gives finer cut placement at higher comm
    ///        cost. The full reduction is `O(3 * n_bins * dims_max)` doubles.
    explicit Rebalance(int every = 1000, int n_bins = 256);

    std::string name() const override { return "rebalance"; }

    /// @brief Run every `every_` steps.
    /// @return Step stride.
    int frequency() const override { return every_; }

    /// @brief Default binding: `end_of_step @ +1000`.
    /// @return Binding tuple list.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, 1000)};
    }

    /// @brief Compute new cut planes and refresh this rank's slab.
    /// @param ctx Launch context.
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Step interval between balance passes (read-only accessor).
    int every() const noexcept { return every_; }

    /// @brief Bins per axis used for histogram bisection (read-only accessor).
    int n_bins() const noexcept { return n_bins_; }

  private:
    int every_;
    int n_bins_;
};

} // namespace mpi
} // namespace md
} // namespace cpu
} // namespace ATV
