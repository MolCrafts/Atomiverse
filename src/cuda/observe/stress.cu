/// @file cuda/observe/stress.cu
/// @brief Virial -- ZEROING of `stress_kinetic[6]`.
///
/// After cuda-md-review-cleanup-05-reductions-and-atomics the kinetic
/// stress tensor `Σ m·vα·vβ · ATV::units::mass_v2_to_energy` (eV units, no volume division)
/// is computed by `cuda::md::KineticEnergy` via a packed warp-shuffle
/// reduction fused with the kinetic-energy scalar. That kernel
/// atomic-adds into `frame.scalars->stress_kinetic[6]`, so this
/// component's role narrows to (a) auto-injection capability
/// declaration (`role()==Observer`, `capabilities()==Stress`,
/// triggered by any `reads()==Stress` consumer per
/// cuda::md::MD::install_infra_), and (b) clearing the 6-component
/// accumulator on its post_force phase slot so KineticEnergy can
/// atomic-add into a known-zero buffer at end_of_step.

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/observe/stress.cuh"
#include <cuda_runtime.h>

namespace ATV {
namespace cuda {

using cuda::prec_t;

struct Virial::State {};

Virial::Virial() : state_(std::make_unique<State>()) {}
Virial::~Virial() = default;
Virial::Virial(Virial&&) noexcept = default;
Virial& Virial::operator=(Virial&&) noexcept = default;

void Virial::build(const BuildContext& /*ctx*/) {
    // No per-component scratch — the compute kernel moved to
    // cuda::md::KineticEnergy in cuda-md-review-cleanup-05.
}

void Virial::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;

    // Overwrite-only: this component clears `stress_kinetic[6]` so that
    // `cuda::md::KineticEnergy` (end_of_step @ -100, runs AFTER this
    // component's post_force @ -50 slot per cuda::md::MD::install_infra_)
    // can atomic-add the freshly-computed Σ m·vα·vβ · ATV::units::mass_v2_to_energy into a
    // known-zero buffer. The pair / bonded virial (`stress[6]`) is
    // owned by `pair_lj_cut` / `bond_harmonic` / `angle_harmonic` /
    // `dihedral_multi_harmonic` and cleared by `ZeroEnergy` at
    // pre_force @ -999 — NOT touched here.
    cudaMemsetAsync(f.scalars->stress_kinetic, 0, 6 * sizeof(prec_t), stream);
}

void Virial::destroy(const DestroyContext& /*ctx*/) {}

} // namespace cuda
} // namespace ATV
