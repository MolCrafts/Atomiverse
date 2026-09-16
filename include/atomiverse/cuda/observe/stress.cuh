/// @file cuda/observe/stress.cuh
/// @brief Virial -- ZEROING of the kinetic-stress accumulator.
///
/// The full virial stress tensor is
///   σ_αβ = (1/V) [Σ_i m_i v_iα v_iβ + Σ_{pairs} r_ij_α F_ij_β]
///
/// The device buffers hold the **extensive** bracketed terms [eV] — *not* σ —
/// matching the `kStress` / `kStressKinetic` contract on the CPU side
/// (`core/frame_schema.h`): extensive is what makes `+=` across evaluators and
/// the MPI sum of rank-local partials well-defined. The 1/V is applied by the
/// consumer, as the producer split below states.
///
/// Producer split:
///   - Kinetic term `Σ m·vα·vβ · ATV::units::mass_v2_to_energy` → `cuda::md::KineticEnergy`
///     atomic-adds it into `frame.scalars->stress_kinetic[6]` via the
///     packed warp-shuffle reduction it does for KE (one fused kernel
///     since cuda-md-review-cleanup-05).
///   - Pair / bonded virial → `cuda::PairLJCut`, `cuda::BondHarmonic`,
///     etc. atomic-add into `frame.scalars->stress[6]` from their own
///     force kernels.
///   - Volume division → the consumer (`cuda::md::NoseHooverNPT`,
///     pressure observers).
///
/// This component's residual responsibilities are:
///   - **`provides_system_stress()`** so `cuda::md::MD::install_defaults()`
///     can skip a second Virial when one is already mounted.
///   - **Zeroing** the `stress_kinetic[6]` accumulator on its
///     `post_force @ -50` slot so `KineticEnergy` at
///     `end_of_step @ -100` can atomic-add into a known-zero buffer.
///
/// Cross-driver component: empty default_bindings(), bound by the
/// install_infra_ injector at post_force @ -50. CUDA-free header.
#pragma once
#include "atomiverse/core/role.h"

#include <memory>
#include <string>

#include "atomiverse/cuda/component/observer.cuh"

namespace ATV {
namespace cuda {

/// @brief Capability-declaring zeroing shim for the kinetic-stress
///        accumulator. See file-level docstring for the producer split.
class Virial : public Observer {
  public:
    Virial();
    ~Virial() override;
    Virial(Virial&&) noexcept;
    Virial& operator=(Virial&&) noexcept;

    std::string name() const override { return "virial"; }
    bool provides_system_stress() const override { return true; }
    /* role via Observer */
    /// @brief Zero the kinetic-stress accumulator. Pair/bonded virial
    ///        writes stay on the force kernels; this Observer only
    ///        declares `provides_system_stress()`.
    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace cuda
} // namespace ATV
