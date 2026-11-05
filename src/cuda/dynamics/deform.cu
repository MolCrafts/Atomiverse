/// @file cuda/dynamics/deform.cu
/// @brief Deform -- uniaxial constant-strain-rate deformation, the device
///        twin of `cpu::md::Deform` (LAMMPS `fix deform` erate / trate,
///        `remap x`; `src/fix_deform.cpp` FixDeform::init / apply_strain).

#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/deform.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

namespace ATV {

namespace cuda {
namespace md {

namespace {

struct DeviceState {
    double L0 = 0;        ///< Deformed edge at the first launch [Angstrom].
    double scale = 1;     ///< This step's L_target / L_current.
    long long origin = 0; ///< Step of the first launch after build.
    int armed = 0;
};

// One thread: snapshot L0 on the first launch (FixDeform::init lo_start /
// hi_start), set this step's scale from the live cell and scale the cell's
// `axis` row (r = H s keeps s when row `axis` of H and r scale together).
// The elapsed time comes from the device step: after the launch at step s
// the box reflects s - origin + 1 steps.
__global__ void deform_box(geom_t* __restrict__ box, DeviceState* s, int axis,
                           int trate, double rate, double dt,
                           const long long* step_ptr, long long host_step,
                           const ForceFault* fault) {
    if (state_frozen(fault))
        return;
    const long long step = current_step(step_ptr, host_step);
    const double L_prev = static_cast<double>(box[4 * axis]);
    if (!s->armed) {
        s->L0 = L_prev;
        s->origin = step;
        s->armed = 1;
    }
    const double t_elapsed = static_cast<double>(step - s->origin + 1) * dt;
    const double L_target = trate ? s->L0 * exp(rate * t_elapsed)
                                  : s->L0 * (1.0 + rate * t_elapsed);
    const double scale = L_prev > 0.0 ? L_target / L_prev : 1.0;
    s->scale = scale;
    if (scale == 1.0)
        return;
    for (int j = 0; j < 3; ++j)
        box[3 * axis + j] = static_cast<geom_t>(box[3 * axis + j] * scale);
}

__global__ void deform_rows(pos_t* __restrict__ pos, pos_t* __restrict__ corr,
                            const DeviceState* __restrict__ s, int n,
                            const ForceFault* fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // A faulted run keeps its cell and positions at the faulting step.
    if (i >= n || state_frozen(fault))
        return;
    const double scale = s->scale;
    if (scale == 1.0)
        return;
    store_position(
        pos, corr, i,
        static_cast<mixed_t>(static_cast<double>(load_position(pos, corr, i))
                             * scale));
}

} // anonymous namespace

struct Deform::State {
    DeviceState* device = nullptr;
    void release() {
        if (device)
            cudaFree(device);
        device = nullptr;
    }
    ~State() { release(); }
};

Deform::Deform(Mode mode, double rate, double dt, int axis)
    : mode_(mode), rate_(rate), dt_(dt), axis_(axis),
      state_(std::make_unique<State>()) {
    if (!std::isfinite(rate_))
        throw std::invalid_argument("Deform: rate must be finite");
    if (!(dt_ > 0.0) || !std::isfinite(dt_))
        throw std::invalid_argument("Deform: dt must be finite and positive");
    if (axis_ < 0 || axis_ > 2)
        throw std::invalid_argument("Deform: axis must be 0, 1 or 2");
}

Deform::~Deform() = default;
Deform::Deform(Deform&&) noexcept = default;
Deform& Deform::operator=(Deform&&) noexcept = default;

std::vector<DefaultBinding> Deform::default_bindings() const {
    // initial_integrate @ +200, matching cpu::md::Deform: the affine box
    // remap runs AFTER the integrator's kick/drift (order 0).
    return {ATV::make_binding(ATV::md::Phase::initial_integrate, 200)};
}

void Deform::build(const BuildContext&) {
    // Re-arm the activation snapshot: L0 is taken on the next launch, as
    // LAMMPS re-issuing `fix deform` re-snapshots lo_start / hi_start.
    state_->release();
    ATV_CUDA_CHECK(cudaMalloc(&state_->device, sizeof(DeviceState)));
    const DeviceState zero{};
    ATV_CUDA_CHECK(cudaMemcpy(state_->device, &zero, sizeof(DeviceState),
                              cudaMemcpyHostToDevice));
}

void Deform::destroy(const DestroyContext&) {
    if (state_)
        state_->release();
}

void Deform::launch(const LaunchContext& ctx) {
    if (!state_ || !state_->device)
        throw std::logic_error("Deform: launch before build");
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    deform_box<<<1, 1, 0, stream>>>(
        f.box, state_->device, axis_, mode_ == Mode::trate ? 1 : 0, rate_, dt_,
        ctx.runtime.device_step, static_cast<long long>(ctx.step),
        f.force_fault);
    pos_t* pos = axis_ == 0 ? f.atoms.x : axis_ == 1 ? f.atoms.y : f.atoms.z;
    pos_t* corr = axis_ == 0   ? f.atoms.x_corr
                  : axis_ == 1 ? f.atoms.y_corr
                               : f.atoms.z_corr;
    const int n = f.n_all();
    deform_rows<<<cuda::grid_size(n), cuda::kDefaultBlock, 0, stream>>>(
        pos, corr, state_->device, n, f.force_fault);
    ATV_CUDA_CHECK(cudaGetLastError());
}

} // namespace md
} // namespace cuda
} // namespace ATV
