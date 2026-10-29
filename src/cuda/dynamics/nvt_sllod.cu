/// @file cuda/dynamics/nvt_sllod.cu
/// @brief Device NVT SLLOD. Scalar laws are `core/sllod_flow.h`.

#include "atomiverse/cuda/dynamics/nvt_sllod.cuh"

#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cuda {
namespace md {
namespace {

struct Thermo {
    double eta = 0;
    double eta_dot = 0;
    double scale = 1;
    double ke = 0;
    unsigned int blocks = 0;
};

int box_slot(int dir) { return ATV::md::kFlowBoxSlot[dir]; }

void read_upper(const geom_t* box, double h[6]) {
    const geom_t lower[3] = {box[3], box[6], box[7]};
    for (geom_t v : lower) {
        if (v != 0)
            throw std::invalid_argument(
                "NvtSllod: cell lower triangle must be zero");
    }
    h[0] = static_cast<double>(box[0]);
    h[1] = static_cast<double>(box[4]);
    h[2] = static_cast<double>(box[8]);
    h[3] = static_cast<double>(box[5]);
    h[4] = static_cast<double>(box[2]);
    h[5] = static_cast<double>(box[1]);
}

__global__ void
kick_bias(mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
          mixed_t* __restrict__ vz, const pos_t* __restrict__ x,
          const pos_t* __restrict__ y, const pos_t* __restrict__ z,
          const pos_t* __restrict__ xc, const pos_t* __restrict__ yc,
          const pos_t* __restrict__ zc, ATV::md::SllodFlow flow, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double ux = vx[i], uy = vy[i], uz = vz[i];
    ATV::md::sllod_apply_bias(ux, uy, uz, load_position(x, xc, i),
                              load_position(y, yc, i), load_position(z, zc, i),
                              flow.h, flow.grad_u, 0.0);
    vx[i] = static_cast<mixed_t>(ux);
    vy[i] = static_cast<mixed_t>(uy);
    vz[i] = static_cast<mixed_t>(uz);
}

__global__ void reduce_peculiar_ke(
    const mixed_t* __restrict__ vx, const mixed_t* __restrict__ vy,
    const mixed_t* __restrict__ vz, const pos_t* __restrict__ x,
    const pos_t* __restrict__ y, const pos_t* __restrict__ z,
    const pos_t* __restrict__ xc, const pos_t* __restrict__ yc,
    const pos_t* __restrict__ zc, const mixed_t* __restrict__ mass,
    Thermo* thermo, ATV::md::SllodFlow flow, int n, int ndof, double target,
    double kT, double Q, double dt) {
    using Reduce = cub::BlockReduce<double, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0;
    if (i < n) {
        double bx, by, bz;
        ATV::md::sllod_deform_bias(load_position(x, xc, i),
                                   load_position(y, yc, i),
                                   load_position(z, zc, i), flow.h_inv,
                                   flow.h_rate, flow.h_ratelo, bx, by, bz);
        const double px = static_cast<double>(vx[i]) - bx;
        const double py = static_cast<double>(vy[i]) - by;
        const double pz = static_cast<double>(vz[i]) - bz;
        val = 0.5 * ATV::units::mass_v2_to_energy * static_cast<double>(mass[i])
              * (px * px + py * py + pz * pz);
    }
    const double sum = Reduce(reduction).Sum(val);
    if (threadIdx.x == 0)
        atomicAdd(&thermo->ke, sum);
    __threadfence();
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        const unsigned int old = atomicInc(&thermo->blocks, gridDim.x - 1);
        is_last = (old == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        ATV::md::SllodChain chain{thermo->eta, thermo->eta_dot, Q};
        double scale = 1;
        if (ndof > 0 && Q > 0)
            scale = ATV::md::sllod_nh_half(chain, thermo->ke, target, kT, dt);
        thermo->eta = chain.eta;
        thermo->eta_dot = chain.eta_dot;
        thermo->scale = scale;
    }
}

__global__ void
scale_peculiar(mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
               mixed_t* __restrict__ vz, const pos_t* __restrict__ x,
               const pos_t* __restrict__ y, const pos_t* __restrict__ z,
               const pos_t* __restrict__ xc, const pos_t* __restrict__ yc,
               const pos_t* __restrict__ zc, const Thermo* thermo,
               ATV::md::SllodFlow flow, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double bx, by, bz;
    const double px = load_position(x, xc, i);
    const double py = load_position(y, yc, i);
    const double pz = load_position(z, zc, i);
    ATV::md::sllod_deform_bias(px, py, pz, flow.h_inv, flow.h_rate,
                               flow.h_ratelo, bx, by, bz);
    const double scale = thermo->scale;
    vx[i] =
        static_cast<mixed_t>(bx + scale * (static_cast<double>(vx[i]) - bx));
    vy[i] =
        static_cast<mixed_t>(by + scale * (static_cast<double>(vy[i]) - by));
    vz[i] =
        static_cast<mixed_t>(bz + scale * (static_cast<double>(vz[i]) - bz));
}

__global__ void nve_v(mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
                      mixed_t* __restrict__ vz, const fixed_t* __restrict__ fx,
                      const fixed_t* __restrict__ fy,
                      const fixed_t* __restrict__ fz,
                      const mixed_t* __restrict__ mass, ForceFault* fault,
                      const std::int64_t* __restrict__ tag, long long step,
                      double dt, int n) {
    if (!forces_consumable(fault, tag, step))
        return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const double scale =
        0.5 * dt * ATV::units::force_to_accel / static_cast<double>(mass[i]);
    vx[i] = static_cast<mixed_t>(static_cast<double>(vx[i])
                                 + scale * decode_force<double>(fx[i]));
    vy[i] = static_cast<mixed_t>(static_cast<double>(vy[i])
                                 + scale * decode_force<double>(fy[i]));
    vz[i] = static_cast<mixed_t>(static_cast<double>(vz[i])
                                 + scale * decode_force<double>(fz[i]));
}

__global__ void nve_x(pos_t* __restrict__ x, pos_t* __restrict__ y,
                      pos_t* __restrict__ z, pos_t* __restrict__ xc,
                      pos_t* __restrict__ yc, pos_t* __restrict__ zc,
                      mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
                      mixed_t* __restrict__ vz, const fixed_t* __restrict__ fx,
                      const fixed_t* __restrict__ fy,
                      const fixed_t* __restrict__ fz,
                      const mixed_t* __restrict__ mass, ForceFault* fault,
                      const std::int64_t* __restrict__ tag, long long step,
                      ATV::md::SllodFlow flow, double dt, int n) {
    if (!forces_consumable(fault, tag, step))
        return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const double kick =
        0.5 * dt * ATV::units::force_to_accel / static_cast<double>(mass[i]);
    double ux = static_cast<double>(vx[i]) + kick * decode_force<double>(fx[i]);
    double uy = static_cast<double>(vy[i]) + kick * decode_force<double>(fy[i]);
    double uz = static_cast<double>(vz[i]) + kick * decode_force<double>(fz[i]);
    double px = load_position(x, xc, i);
    double py = load_position(y, yc, i);
    double pz = load_position(z, zc, i);
    double bx, by, bz;
    ATV::md::sllod_deform_bias(px, py, pz, flow.h_inv, flow.h_rate,
                               flow.h_ratelo, bx, by, bz);
    ux -= bx;
    uy -= by;
    uz -= bz;
    ATV::md::sllod_nve_x(px, py, pz, ux, uy, uz, flow.h, flow.grad_u, dt);
    ATV::md::sllod_apply_bias(ux, uy, uz, px, py, pz, flow.h, flow.grad_u, dt);
    ATV::md::sllod_pbc(px, py, pz, ux, uy, uz, flow.h, flow.h_inv, flow.h_rate);
    store_position(x, xc, i, static_cast<mixed_t>(px));
    store_position(y, yc, i, static_cast<mixed_t>(py));
    store_position(z, zc, i, static_cast<mixed_t>(pz));
    vx[i] = static_cast<mixed_t>(ux);
    vy[i] = static_cast<mixed_t>(uy);
    vz[i] = static_cast<mixed_t>(uz);
}

__global__ void end_of_step(pos_t* __restrict__ x, pos_t* __restrict__ y,
                            pos_t* __restrict__ z, pos_t* __restrict__ xc,
                            pos_t* __restrict__ yc, pos_t* __restrict__ zc,
                            mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
                            mixed_t* __restrict__ vz, geom_t* __restrict__ box,
                            ATV::md::SllodFlow flow, double shift, int slot,
                            double new_value, int axis, int n_local,
                            int n_all) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0)
        box[slot] = static_cast<geom_t>(new_value);
    if (i >= n_all)
        return;
    if (axis >= 0 && shift != 0.0) {
        pos_t* pos = axis == 0 ? x : axis == 1 ? y : z;
        pos_t* corr = axis == 0 ? xc : axis == 1 ? yc : zc;
        store_position(pos, corr, i,
                       load_position(pos, corr, i)
                           + static_cast<mixed_t>(shift));
    }
    if (i < n_local && shift != 0.0) {
        double px = load_position(x, xc, i);
        double py = load_position(y, yc, i);
        double pz = load_position(z, zc, i);
        double ux = vx[i], uy = vy[i], uz = vz[i];
        ATV::md::sllod_pbc(px, py, pz, ux, uy, uz, flow.h, flow.h_inv,
                           flow.h_rate);
        store_position(x, xc, i, static_cast<mixed_t>(px));
        store_position(y, yc, i, static_cast<mixed_t>(py));
        store_position(z, zc, i, static_cast<mixed_t>(pz));
        vx[i] = static_cast<mixed_t>(ux);
        vy[i] = static_cast<mixed_t>(uy);
        vz[i] = static_cast<mixed_t>(uz);
    }
}

} // namespace

struct NvtSllod::State {
    Thermo* thermo = nullptr;
    int ndof = 0;
    double Q = 0;
    ATV::md::SllodFlow flow{};
    ~State() {
        if (thermo)
            cudaFree(thermo);
    }
};

NvtSllod::NvtSllod(double temperature_K, double tau_fs, double dt_fs,
                   double rate_per_fs, int direction)
    : T_(temperature_K), tau_(tau_fs), dt_(dt_fs), rate_(rate_per_fs),
      dir_(direction), state_(std::make_unique<State>()) {
    if (!(T_ > 0.0) || !std::isfinite(T_))
        throw std::invalid_argument("NvtSllod: temperature must be positive");
    if (!(tau_ > 0.0) || !std::isfinite(tau_))
        throw std::invalid_argument("NvtSllod: tau must be positive");
    if (!(dt_ > 0.0) || !std::isfinite(dt_))
        throw std::invalid_argument("NvtSllod: dt must be positive");
    if (!std::isfinite(rate_))
        throw std::invalid_argument("NvtSllod: rate must be finite");
    if (dir_ < 0 || dir_ > 5)
        throw std::invalid_argument(
            "NvtSllod: direction must be 0..5 (x y z yz xz xy)");
}

NvtSllod::~NvtSllod() = default;
NvtSllod::NvtSllod(NvtSllod&&) noexcept = default;
NvtSllod& NvtSllod::operator=(NvtSllod&&) noexcept = default;

void NvtSllod::build(const BuildContext& ctx) {
    if (state_->thermo) {
        cudaFree(state_->thermo);
        state_->thermo = nullptr;
    }
    state_->ndof = static_cast<int>(ctx.runtime.md.n_dof_global);
    state_->Q = state_->ndof > 0 ? static_cast<double>(state_->ndof)
                                       * units::kB_ev_per_k * T_ * tau_ * tau_
                                 : 0;
    armed_ = false;
    kicked_ = false;
    step_ = 0;
    state_->flow = {};
    ATV_CUDA_CHECK(cudaMalloc(&state_->thermo, sizeof(Thermo)));
    Thermo zero;
    ATV_CUDA_CHECK(cudaMemcpy(state_->thermo, &zero, sizeof(Thermo),
                              cudaMemcpyHostToDevice));
}

void NvtSllod::destroy(const DestroyContext&) {
    if (state_ && state_->thermo) {
        cudaFree(state_->thermo);
        state_->thermo = nullptr;
    }
    armed_ = false;
    kicked_ = false;
}

void NvtSllod::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const auto& a = frame.atoms;
    const int n = frame.n_local();
    const int n_all = frame.n_all();
    const int block = cuda::kDefaultBlock;
    auto& flow = state_->flow;

    if (!armed_) {
        geom_t host_box[9];
        ATV_CUDA_CHECK(cudaMemcpy(host_box, frame.box, 9 * sizeof(geom_t),
                                  cudaMemcpyDeviceToHost));
        read_upper(host_box, flow.h0);
        for (int i = 0; i < 6; ++i)
            flow.h[i] = flow.h0[i];
        flow.rate = rate_;
        flow.dir = dir_;
        ATV::md::sllod_erate_rates(flow.h0, rate_, dir_, flow.h_rate,
                                   flow.h_ratelo);
        if (!ATV::md::sllod_refresh(flow))
            throw std::invalid_argument("NvtSllod: box edge must be positive");
        armed_ = true;
    }

    const double kT = units::kB_ev_per_k * T_;
    const double target = static_cast<double>(state_->ndof) * kT;
    auto thermostate = [&]() {
        ATV_CUDA_CHECK(
            cudaMemsetAsync(&state_->thermo->ke, 0, sizeof(double), stream));
        const int grid = cuda::grid_size(n);
        reduce_peculiar_ke<<<grid, block, 0, stream>>>(
            a.vx, a.vy, a.vz, a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr,
            a.mass, state_->thermo, flow, n, state_->ndof, target, kT,
            state_->Q, dt_);
        scale_peculiar<<<grid, block, 0, stream>>>(
            a.vx, a.vy, a.vz, a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr,
            state_->thermo, flow, n);
    };

    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        const int grid = cuda::grid_size(n);
        if (!kicked_) {
            kick_bias<<<grid, block, 0, stream>>>(a.vx, a.vy, a.vz, a.x, a.y,
                                                  a.z, a.x_corr, a.y_corr,
                                                  a.z_corr, flow, n);
            kicked_ = true;
        }
        thermostate();
        nve_x<<<grid, block, 0, stream>>>(a.x, a.y, a.z, a.x_corr, a.y_corr,
                                          a.z_corr, a.vx, a.vy, a.vz, a.fx,
                                          a.fy, a.fz, a.mass, frame.force_fault,
                                          a.tag, ctx.step, flow, dt_, n);
    } else if (ctx.at(ATV::md::Phase::final_integrate)) {
        const int grid = cuda::grid_size(n);
        nve_v<<<grid, block, 0, stream>>>(a.vx, a.vy, a.vz, a.fx, a.fy, a.fz,
                                          a.mass, frame.force_fault, a.tag,
                                          ctx.step, dt_, n);
        thermostate();
    } else if (ctx.at(ATV::md::Phase::end_of_step)) {
        const long long next = step_ + 1;
        double h_new[6];
        ATV::md::sllod_box_at(flow.h0, rate_, dt_, dir_, next, h_new);
        if (dir_ < 3 && !(h_new[dir_] > 0.0))
            throw std::invalid_argument(
                "NvtSllod: deformed edge is not positive");
        const double shift =
            ATV::md::sllod_edge_shift(flow.h[dir_], h_new[dir_], dir_);
        for (int i = 0; i < 6; ++i)
            flow.h[i] = h_new[i];
        if (!ATV::md::sllod_refresh(flow))
            throw std::invalid_argument("NvtSllod: box edge must be positive");
        const int grid = cuda::grid_size(n_all);
        end_of_step<<<grid, block, 0, stream>>>(
            a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, a.vx, a.vy, a.vz,
            frame.box, flow, shift, box_slot(dir_), h_new[dir_],
            dir_ < 3 ? dir_ : -1, n, n_all);
        step_ = next;
    }
}

} // namespace md
} // namespace cuda
} // namespace ATV
