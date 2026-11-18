/// @file cuda/dynamics/fix_csvr.cu
/// @brief CUDA CSVR (Bussi) thermostat (LAMMPS `fix temp/csvr`, port
///        reference e8c4c278a8; Sandia notice in the header).

#include "atomiverse/core/csvr.h"
#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/system_scalars.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/fix_csvr.cuh"
#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace ATV {
namespace cuda {
namespace md {

namespace {

/// Device-resident per-instance state. `sums` is cleared before every
/// reduction; the rest is written by the last block of the reduction.
struct DeviceState {
    double sums[6];           ///< Σ m, Σ m v (3), Σ m v² (group), Σ m v² (all).
    double alpha;             ///< Velocity scale of this step.
    double shift[3];          ///< (1 − α) v_cm (zero without bias).
    double bath;              ///< This instance's −Σ ΔK [eV].
    unsigned int block_count; ///< Last-block detection counter.
};

struct Params {
    std::uint32_t seed;
    double T_start, T_stop;
    BigInt nfirst, nlast;
    double n_dof;
    double dt, tau;
    double boltz, mvv2e;
    ATV::md::TemperatureBias bias;
};

__device__ __forceinline__ bool
in_group(const int* type, const unsigned char* member, int n_member, int i) {
    if (type == nullptr)
        return true;
    const int t = type[i];
    return t >= 0 && t < n_member && member[t] != 0;
}

/// @brief Group reduction; the last block draws the randoms and forms α.
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock) void reduce_and_resample(
    const mixed_t* __restrict__ vx, const mixed_t* __restrict__ vy,
    const mixed_t* __restrict__ vz, const mixed_t* __restrict__ mass,
    const int* __restrict__ type, const unsigned char* __restrict__ member,
    int n_member, int n, DeviceState* __restrict__ st,
    SystemScalars* __restrict__ scalars, const long long* step_ptr,
    long long host_step, Params p, const ForceFault* fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double v[6] = {};
    if (i < n) {
        const double m = mass[i];
        const double x = vx[i], y = vy[i], z = vz[i];
        const double mvv = m * (x * x + y * y + z * z);
        v[5] = mvv;
        if (in_group(type, member, n_member, i)) {
            v[0] = m;
            v[1] = m * x;
            v[2] = m * y;
            v[3] = m * z;
            v[4] = mvv;
        }
    }
    ::cuda::std::array<double, 6> vals = {v[0], v[1], v[2], v[3], v[4], v[5]};
    using Reduce =
        cub::BlockReduce<::cuda::std::array<double, 6>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto out = Reduce(reduction).Reduce(vals, SumComponents<double, 6>{});
    if (threadIdx.x == 0)
#pragma unroll
        for (int c = 0; c < 6; ++c)
            atomicAdd(&st->sums[c], out[c]);

    __threadfence();
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        const unsigned int old = atomicInc(&st->block_count, gridDim.x - 1);
        is_last = (old == gridDim.x - 1);
    }
    __syncthreads();
    if (!is_last || threadIdx.x != 0)
        return;

    // Default: leave the velocities alone.
    st->alpha = 1.0;
    st->shift[0] = st->shift[1] = st->shift[2] = 0.0;
    // A faulted run keeps its thermostat state at the faulting step.
    if (state_frozen(fault))
        return;

    ATV::md::CsvrSums s;
    volatile const double* sums = st->sums;
    s.mass = sums[0];
    s.p[0] = sums[1];
    s.p[1] = sums[2];
    s.p[2] = sums[3];
    s.mvv = sums[4];
    s.mvv_all = sums[5];
    const double ke = 0.5 * p.mvv2e * ATV::md::csvr_group_mvv(s, p.bias);
    if (!(ke > 0.0))
        return;

    const long long step = current_step(step_ptr, host_step);
    const double T_live = ATV::md::ramp_target(p.T_start, p.T_stop,
                                               BigInt(step), p.nfirst, p.nlast);
    const double ke_bar = 0.5 * p.n_dof * p.boltz * T_live;
    const ATV::md::CsvrNoise noise =
        ATV::md::csvr_noise(p.seed, static_cast<std::int64_t>(step),
                            static_cast<std::int64_t>(p.n_dof));
    const double alpha2 =
        ATV::md::csvr_alpha2(ke, ke_bar, p.n_dof, p.dt, p.tau, noise);
    const double alpha = sqrt(alpha2);
    st->alpha = alpha;
    if (p.bias == ATV::md::TemperatureBias::Com && s.mass > 0.0) {
        const double shift = 1.0 - alpha;
        st->shift[0] = shift * s.p[0] / s.mass;
        st->shift[1] = shift * s.p[1] / s.mass;
        st->shift[2] = shift * s.p[2] / s.mass;
    }
    // Bussi Eq. 15 / LAMMPS `energy += ekin_old (1 − λ²)`.
    const double dk = (alpha2 - 1.0) * ke;
    st->bath -= dk;
    scalars->thermostat_energy -= dk;
    scalars->conserved = scalars->energy + 0.5 * p.mvv2e * s.mvv_all + dk
                         + scalars->thermostat_energy;
}

__global__ void rescale(mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
                        mixed_t* __restrict__ vz, const int* __restrict__ type,
                        const unsigned char* __restrict__ member, int n_member,
                        int n, const DeviceState* __restrict__ st) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !in_group(type, member, n_member, i))
        return;
    const double a = st->alpha;
    vx[i] = mixed_t(a * double(vx[i]) + st->shift[0]);
    vy[i] = mixed_t(a * double(vy[i]) + st->shift[1]);
    vz[i] = mixed_t(a * double(vz[i]) + st->shift[2]);
}

void require_positive(double value, const char* what) {
    if (!std::isfinite(value) || !(value > 0.0))
        throw std::invalid_argument(std::string("cuda::md::FixCSVR: ") + what
                                    + " must be finite and > 0");
}

} // anonymous namespace

struct FixCSVR::State {
    DeviceState* d_state = nullptr;
    unsigned char* d_member = nullptr;
    int n_member = 0;
};

FixCSVR::FixCSVR(double T_start, double tau, double dt, std::uint32_t seed)
    : FixCSVR(T_start, tau, dt, seed, {}, ATV::md::TemperatureBias::None) {}

FixCSVR::FixCSVR(double T_start, double tau, double dt, std::uint32_t seed,
                 std::vector<int> types, ATV::md::TemperatureBias bias)
    : T_start_(T_start), T_stop_(T_start), tau_(tau), dt_(dt), seed_(seed),
      types_(std::move(types)), bias_(bias), T_live_(T_start),
      state_(std::make_unique<State>()) {
    require_positive(T_start, "T_start");
    require_positive(tau, "tau");
    require_positive(dt, "dt");
    std::sort(types_.begin(), types_.end());
    types_.erase(std::unique(types_.begin(), types_.end()), types_.end());
    for (int t : types_) {
        if (t < 0)
            throw std::invalid_argument("cuda::md::FixCSVR: negative atom type "
                                        + std::to_string(t) + " in the group");
        if (member_.size() <= static_cast<std::size_t>(t))
            member_.resize(static_cast<std::size_t>(t) + 1, 0);
        member_[static_cast<std::size_t>(t)] = 1;
    }
}

FixCSVR::~FixCSVR() = default;
FixCSVR::FixCSVR(FixCSVR&&) noexcept = default;
FixCSVR& FixCSVR::operator=(FixCSVR&&) noexcept = default;

void FixCSVR::set_timestep(double dt_fs) {
    if (!(dt_fs > 0.0) || !std::isfinite(dt_fs))
        throw std::invalid_argument(
            "cuda::md::FixCSVR: timestep must be finite and > 0 fs, got "
            + std::to_string(dt_fs));
    dt_ = dt_fs;
}

void FixCSVR::build(const BuildContext& ctx) {
    const cuda::Context& rt = ctx.runtime;
    const cuda::Frame& f = ctx.frame;
    const int n = f.n_atoms();
    if (types_.empty()) {
        n_dof_ = rt.md.n_dof_global;
    } else {
        if (f.atoms.type == nullptr)
            throw std::runtime_error(
                "cuda::md::FixCSVR: a type group needs the per-atom "
                "force-field type column (kAtom_type); provide molpy "
                "atoms.type_id");
        if (!rt.md.constraints_by_type_known)
            throw std::runtime_error(
                "cuda::md::FixCSVR: the backend could not attribute "
                "constraints to atom types, so the group's degrees of "
                "freedom are unknown");
        // Setup only: the group size is read once from the device types.
        std::vector<int> type(static_cast<std::size_t>(n));
        if (n > 0)
            ATV_CUDA_CHECK(cudaMemcpy(type.data(), f.atoms.type,
                                      sizeof(int) * static_cast<std::size_t>(n),
                                      cudaMemcpyDeviceToHost));
        BigInt n_group = 0;
        for (int t : type)
            n_group += t >= 0 && static_cast<std::size_t>(t) < member_.size()
                       && member_[static_cast<std::size_t>(t)] != 0;
        if (n_group == 0)
            throw std::runtime_error(
                "cuda::md::FixCSVR: the type group selects no atoms of the "
                "frame");
        BigInt n_con = 0;
        for (int t : types_)
            if (t < rt.md.n_constraint_types)
                n_con += rt.md.n_constraints_by_type[t];
        n_dof_ = ATV::md::csvr_group_dof(n_group, n_con);
    }

    ATV_CUDA_CHECK(cudaMalloc(&state_->d_state, sizeof(DeviceState)));
    ATV_CUDA_CHECK(cudaMemset(state_->d_state, 0, sizeof(DeviceState)));
    state_->n_member = static_cast<int>(member_.size());
    if (!member_.empty()) {
        ATV_CUDA_CHECK(cudaMalloc(&state_->d_member, member_.size()));
        ATV_CUDA_CHECK(cudaMemcpy(state_->d_member, member_.data(),
                                  member_.size(), cudaMemcpyHostToDevice));
    }
}

void FixCSVR::before_launch(const LaunchContext& ctx) {
    // Host mirror of the device setpoint for T_target(); no device work.
    T_live_ =
        ATV::md::ramp_target(T_start_, T_stop_, ctx.step, nfirst_, nlast_);
}

void FixCSVR::launch(const LaunchContext& ctx) {
    // LAMMPS: "nothing to do, if there are no degrees of freedom". N_f is a
    // build-time host value, never device state.
    if (n_dof_ < 1)
        return;
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    const int n = f.n_atoms();
    if (n == 0)
        return;
    const int block = cuda::kDefaultBlock;
    const int grid = cuda::grid_size(n);
    const int* type = types_.empty() ? nullptr : f.atoms.type;

    Params p{};
    p.seed = seed_;
    p.T_start = T_start_;
    p.T_stop = T_stop_;
    p.nfirst = nfirst_;
    p.nlast = nlast_;
    p.n_dof = static_cast<double>(n_dof_);
    p.dt = dt_;
    p.tau = tau_;
    p.boltz = ctx.runtime.units.boltz;
    p.mvv2e = ctx.runtime.units.mvv2e;
    p.bias = bias_;

    // `sums` is the leading member: clear it without touching the rest.
    static_assert(offsetof(DeviceState, sums) == 0);
    ATV_CUDA_CHECK(
        cudaMemsetAsync(state_->d_state, 0, sizeof(DeviceState::sums), stream));
    reduce_and_resample<<<grid, block, 0, stream>>>(
        f.atoms.vx, f.atoms.vy, f.atoms.vz, f.atoms.mass, type,
        state_->d_member, state_->n_member, n, state_->d_state, f.scalars,
        ctx.runtime.device_step, static_cast<long long>(ctx.step), p,
        f.force_fault);
    rescale<<<grid, block, 0, stream>>>(f.atoms.vx, f.atoms.vy, f.atoms.vz,
                                        type, state_->d_member,
                                        state_->n_member, n, state_->d_state);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void FixCSVR::destroy(const DestroyContext&) {
    cudaFree(state_->d_state);
    cudaFree(state_->d_member);
    *state_ = {};
}

} // namespace md
} // namespace cuda
} // namespace ATV
