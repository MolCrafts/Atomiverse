/// @file cuda/dynamics/barostat.cu
/// @brief Barostat -- Berendsen isotropic pressure coupling.

#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/barostat.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace ATV {

namespace cuda {
namespace md {

struct Barostat::State {
    mixed_t* d_mu = nullptr; // scale factor [device, 1 element]
    int n_atoms = 0;
};

Barostat::Barostat(double P_target, double tau_p, double dt,
                   double bulk_modulus)
    : P_target_(P_target), tau_p_(tau_p), dt_(dt), bulk_modulus_(bulk_modulus),
      state_(std::make_unique<State>()) {}

Barostat::~Barostat() = default;
Barostat::Barostat(Barostat&&) noexcept = default;
Barostat& Barostat::operator=(Barostat&&) noexcept = default;

std::vector<DefaultBinding> Barostat::default_bindings() const {
    // end_of_step @ -45, matching cpu::md::Barostat and LAMMPS
    // `fix press/berendsen` (an end-of-step fix). Runs after KineticEnergy
    // (end_of_step @ -100) so the instantaneous pressure the barostat reads
    // (virial + current kinetic) is complete — at final_integrate the kinetic
    // for this step is not yet accumulated, giving a stale pressure.
    return {ATV::make_binding(ATV::md::Phase::end_of_step, -45)};
}

namespace {

// Determine the dilation before any block modifies the cell. Computing it
// independently in the position blocks races the box update across blocks.
__global__ void berendsen_dilation_k(const geom_t* box,
                                     const SystemScalars* scalars,
                                     double target, double dt, double damping,
                                     double modulus, mixed_t* mu) {
    const double volume = fabs(box[0] * (box[4] * box[8] - box[5] * box[7])
                               - box[1] * (box[3] * box[8] - box[5] * box[6])
                               + box[2] * (box[3] * box[7] - box[4] * box[6]));
    *mu = volume > 0.0
              ? mixed_t(ATV::md::berendsen_dilation(
                    ATV::md::isotropic_pressure(
                        scalars->stress, scalars->stress_kinetic, volume),
                    target, dt, damping, modulus))
              : mixed_t(1);
}

__global__ void berendsen_scale_k(pos_t* x, pos_t* y, pos_t* z, pos_t* xc,
                                  pos_t* yc, pos_t* zc, geom_t* box,
                                  const mixed_t* dilation, int n) {
    const mixed_t mu = *dilation;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        for (int d = 0; d < 9; ++d)
            box[d] *= geom_t(mu);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        store_position(x, xc, i, load_position(x, xc, i) * mu);
        store_position(y, yc, i, load_position(y, yc, i) * mu);
        store_position(z, zc, i, load_position(z, zc, i) * mu);
    }
}

} // anonymous namespace

void Barostat::build(const BuildContext& ctx) {
    state_->n_atoms = ctx.frame.n_atoms();
    ATV_CUDA_CHECK(cudaMalloc(&state_->d_mu, sizeof(mixed_t)));
}

void Barostat::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int n = f.n_atoms();
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));

    berendsen_dilation_k<<<1, 1, 0, stream>>>(
        f.box, f.scalars, P_target_, dt_, tau_p_, bulk_modulus_, state_->d_mu);
    berendsen_scale_k<<<std::max(1, grid), block, 0, stream>>>(
        f.atoms.x, f.atoms.y, f.atoms.z, f.atoms.x_corr, f.atoms.y_corr,
        f.atoms.z_corr, f.box, state_->d_mu, n);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void Barostat::destroy(const DestroyContext&) {
    ATV_CUDA_CHECK(cudaFree(state_->d_mu));
    *state_ = {};
}

} // namespace md
} // namespace cuda
} // namespace ATV
