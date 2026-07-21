/// @file cpu/comm/atom_migration.h
/// @brief Atom migration component: moves owned atoms across rank boundaries.
///
/// `MPI_Alltoallv` of atoms + their per-atom bonded topology to new owners
/// whenever this rank's ownership set has drifted beyond the halo skin; bumps
/// `md::mpi::DomainDecomposition::neighbor_generation` on each migration so
/// downstream consumers (neighbor list, halo send lists, bonded index
/// remap) rebuild from tag→local. Bound at `pre_force @ -30`.
///
/// `launch()` never allocates. Capacity growth is handled out of band via
/// the driver-level `rebuild_frame()` hook — see spec § A.4.
///
/// @see .claude/specs/parallel-md-engine.md  § A.4, § A.5
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {

class Frame;

namespace comm {

/// @brief One-shot initial distribution: a frame whose atoms are all held by
///        rank 0 (others empty) is scattered so each rank ends up holding only
///        the atoms in its spatial subdomain. Avoids every rank loading the
///        full data file (per-rank memory ∝ total system size). Box broadcast
///        from rank 0 first; ownership matches `partition_atoms`' uniform
///        Cartesian rule. Non-MPI / single rank: no-op. Uses `MPI_COMM_WORLD`.
void scatter_atoms_from_root(cpu::Frame& frame);

/// @brief Atom migration across rank boundaries. Registers `pre_force @ -30`.
class Migrate : public cpu::Component {
  public:
    /// @param every Attempt owner migration every N steps. The component also
    ///        runs immediately when an external partition update bumps
    ///        `md::mpi::DomainDecomposition::neighbor_generation`.
    explicit Migrate(int every = 1) : every_(every < 1 ? 1 : every) {}

    std::string name() const override { return "migrate"; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::pre_exchange, 0)};
    }
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    int every_ = 1;
    std::uint64_t last_observed_generation_ = ~std::uint64_t{0};
};

} // namespace comm
} // namespace cpu
} // namespace ATV
