/// @file cuda/dynamics/kinetic_energy.cuh
/// @brief md::KineticEnergy -- computes system kinetic energy from velocities.
///
/// KE [eV] = 0.5 * ATV::units::mass_v2_to_energy * sum_i( mass_i * (vx_i^2 + vy_i^2 + vz_i^2) ).
///
/// Auto-binds to end_of_step (order -100, before user observers).
/// Accumulates into `frame.scalars->kinetic` AND
/// `frame.scalars->stress_kinetic[6]` (the Σ m·vα·vβ·ATV::units::mass_v2_to_energy kinetic
/// stress tensor — `cuda::Virial` zeroes the buffer at
/// post_force then this kernel atomic-adds into it). Reduction is a
/// packed warp-shuffle (`cuda::block_reduce_sum_n<7>`) — replaces the
/// pre-link-05 two-kernel pattern (separate `kinetic_energy` +
/// `kinetic_stress_k`) and the per-component shared-memory tree
/// reduction.
///
/// Requires ZeroKinetic (order -1000 at end_of_step) and Virial
/// (auto-injected at post_force @ -50) to clear the accumulators
/// before this kernel atomic-adds.
///
/// @see ZeroKinetic for the kinetic-scalar zeroing infrastructure
/// @see Virial for the stress_kinetic[] zeroing
/// @see Thermo for the component that reads frame.scalars->kinetic
#pragma once
#include "atomiverse/core/role.h"

#include <string>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {
namespace md {

/// @brief Computes kinetic energy: KE = 0.5 * ATV::units::mass_v2_to_energy * sum(m_i * v_i^2) [eV].
///
/// Auto-binds to end_of_step at order -100 (after ZeroKinetic at -1000,
/// before user observers at order >= 0).
///
/// `provides_energy()` / `provides_force()` stay false: kinetic energy
/// is a structural diagnostic, not a plugin energy/force source, so
/// `cuda::md::MD` validation does not count it toward `require_force()`.
class KineticEnergy : public Component {
  public:
    /// @brief Component name.
    /// @return "kinetic_energy".
    std::string name() const override { return "kinetic_energy"; }
    Role role() const override {
        return RoleFlag::FrameImmutable | RoleFlag::Deferred;
    }

    /// @brief Default bindings: end_of_step at order -100.
    /// @return Single binding to end_of_step.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, -100)};
    }

    /// @brief Launch KE reduction kernel.
    /// @param ctx Launch context.
    void launch(const LaunchContext& ctx) override;
};

} // namespace md
} // namespace cuda
} // namespace ATV
