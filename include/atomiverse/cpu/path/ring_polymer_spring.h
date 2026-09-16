/// @file cpu/path/ring_polymer_spring.h
/// @brief Ring-polymer harmonic springs between adjacent PIMD beads.
///
/// For bead index s with P beads and spring constant k [eV/Å²]:
///   F_i ← F_i − k (2 r_i^{(s)} − r_i^{(s−1)} − r_i^{(s+1)})
/// with periodic bead indices. Physical potential forces are unchanged.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/replica/replica_set.h"

namespace ATV {
namespace cpu {
namespace path {

/// @brief Closed-chain spring Potential for one PIMD bead (force + bond energy).
///
/// Counts as an MD energy/force source so free-particle PIMD needs no external
/// pair style. Energy contribution is ½ k |r^{(s)} − r^{(s+1)}|² per atom
/// (each polymer bond counted once around the ring).
class RingPolymerSpring : public Potential {
  public:
    /// @param band Shared replica band (must outlive this component).
    /// @param bead_index This bead's index in [0, P).
    /// @param k_spring Spring constant [eV/Å²] (= m ω_P² in energy units).
    RingPolymerSpring(ReplicaSet* band, int bead_index, double k_spring);

    std::string name() const override { return "ring_polymer_spring"; }

    std::vector<DefaultBinding> default_bindings() const override {
        // After pair potentials (order 0); before post_force.
        return {ATV::make_binding(ATV::md::Phase::force, +20)};
    }

    void launch(const LaunchContext& ctx) override;

  private:
    ReplicaSet* band_ = nullptr;
    int bead_ = 0;
    double k_ = 0;
};

} // namespace path
} // namespace cpu
} // namespace ATV
