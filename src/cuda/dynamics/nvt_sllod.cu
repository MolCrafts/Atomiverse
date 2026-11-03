/// @file cuda/dynamics/nvt_sllod.cu
/// @brief Device NVT SLLOD. Scalar laws are `core/sllod_flow.h`.
///
/// Step structure ported from LAMMPS `src/fix_nh.cpp` (FixNH
/// initial/final_integrate, nhc_temp_integrate, nh_v_temp),
/// `src/fix_nvt_sllod.cpp` and `src/fix_deform.cpp` (end_of_step,
/// pre_exchange flip), SHA e8c4c278a8599165a0321ebfcb4e4046ef79ad79.
/// Copyright (2003) Sandia Corporation. Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software. GPL-2.0-only.
///
/// Every control decision of a step (thermostat scale, deformed cell, box
/// flip) is made on the device from device-resident state; the host only
/// enqueues kernels (iron law 10). The flow struct is read through a device
/// pointer, so its cell is the one the device last wrote.

#include "atomiverse/cuda/dynamics/nvt_sllod.cuh"

#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ATV {
namespace cuda {
namespace md {
namespace {

using ATV::md::SllodFlow;

/// Device-resident run state. The chain arrays live in a separate
/// allocation of 3 * tchain doubles: eta | eta_dot | mass.
struct Device {
    SllodFlow flow;
    double ke = 0;      ///< Peculiar KE carried between half-steps [eV].
    double ke_sum = 0;  ///< Block-reduction accumulator, re-armed after use.
    double scale = 1;   ///< Peculiar-velocity scale of the last half-step.
    double shift = 0;   ///< Edge origin shift of the last end_of_step [A].
    long long step = 0; ///< Strain clock.
    unsigned int blocks = 0;      ///< KE reduction ticket.
    unsigned int flip_blocks = 0; ///< Flip remap ticket.
};

struct ChainArgs {
    double* chain;
    int length;
    double target; ///< N_f kT [eV].
    double kT;     ///< [eV].
    double dt;
    bool active; ///< N_f > 0 and Q_0 > 0.
};

// LAMMPS Domain::pbc / image_flip image bookkeeping for one row.
__device__ __forceinline__ void add_image_shift(ImageInt* image, int i,
                                                const int shift[3]) {
    if (image == nullptr)
        return;
    for (int d = 0; d < 3; ++d)
        image[3 * i + d] = static_cast<ImageInt>(image[3 * i + d] + shift[d]);
}

__device__ __forceinline__ void chain_half(Device& d, const ChainArgs& c) {
    double scale = 1;
    if (c.active) {
        double* eta = c.chain;
        double* eta_dot = c.chain + c.length;
        const double* mass = c.chain + 2 * c.length;
        scale = ATV::md::sllod_nhc_half(eta, eta_dot, mass, c.length, d.ke,
                                        c.target, c.kT, c.dt);
    }
    d.scale = scale;
}

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

// Row-major cell slots of the Voigt upper triangle (ATV::md::kFlowBoxSlot).
__device__ __forceinline__ void write_upper(geom_t* box, const double* h) {
    box[0] = static_cast<geom_t>(h[0]);
    box[4] = static_cast<geom_t>(h[1]);
    box[8] = static_cast<geom_t>(h[2]);
    box[5] = static_cast<geom_t>(h[3]);
    box[2] = static_cast<geom_t>(h[4]);
    box[1] = static_cast<geom_t>(h[5]);
}

__global__ void
kick_bias(mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
          mixed_t* __restrict__ vz, const pos_t* __restrict__ x,
          const pos_t* __restrict__ y, const pos_t* __restrict__ z,
          const pos_t* __restrict__ xc, const pos_t* __restrict__ yc,
          const pos_t* __restrict__ zc, const Device* __restrict__ d, int n,
          const ForceFault* fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || state_frozen(fault))
        return;
    const SllodFlow& flow = d->flow;
    double ux = vx[i], uy = vy[i], uz = vz[i];
    ATV::md::sllod_apply_bias(ux, uy, uz, load_position(x, xc, i),
                              load_position(y, yc, i), load_position(z, zc, i),
                              flow.h, flow.grad_u, 0.0);
    vx[i] = static_cast<mixed_t>(ux);
    vy[i] = static_cast<mixed_t>(uy);
    vz[i] = static_cast<mixed_t>(uz);
}

// Peculiar KE (LAMMPS ComputeTempDeform::compute_scalar), optionally after
// the final half kick (FixNH::final_integrate nve_v fused in), and the last
// block's chain half-step on the fresh value. A frozen run (or one whose
// forces are not consumable) returns from every block, so no ticket moves.
template <bool Kick>
__global__ void measure_thermostat(
    mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
    mixed_t* __restrict__ vz, const fixed_t* __restrict__ fx,
    const fixed_t* __restrict__ fy, const fixed_t* __restrict__ fz,
    const pos_t* __restrict__ x, const pos_t* __restrict__ y,
    const pos_t* __restrict__ z, const pos_t* __restrict__ xc,
    const pos_t* __restrict__ yc, const pos_t* __restrict__ zc,
    const mixed_t* __restrict__ mass, Device* d, ChainArgs chain, int n,
    ForceFault* fault, const std::int64_t* __restrict__ tag, long long step) {
    if constexpr (Kick) {
        if (!forces_consumable(fault, tag, step))
            return;
    } else {
        if (state_frozen(fault))
            return;
    }
    using Reduce = cub::BlockReduce<double, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0;
    if (i < n) {
        double ux = static_cast<double>(vx[i]);
        double uy = static_cast<double>(vy[i]);
        double uz = static_cast<double>(vz[i]);
        if constexpr (Kick) {
            const double kick = 0.5 * chain.dt * ATV::units::md_ftm2v
                                / static_cast<double>(mass[i]);
            ux += kick * decode_force<double>(fx[i]);
            uy += kick * decode_force<double>(fy[i]);
            uz += kick * decode_force<double>(fz[i]);
            vx[i] = static_cast<mixed_t>(ux);
            vy[i] = static_cast<mixed_t>(uy);
            vz[i] = static_cast<mixed_t>(uz);
            ux = static_cast<double>(vx[i]);
            uy = static_cast<double>(vy[i]);
            uz = static_cast<double>(vz[i]);
        }
        const SllodFlow& flow = d->flow;
        double bx, by, bz;
        ATV::md::sllod_deform_bias(load_position(x, xc, i),
                                   load_position(y, yc, i),
                                   load_position(z, zc, i), flow.h_inv,
                                   flow.h_rate, flow.h_ratelo, bx, by, bz);
        const double px = ux - bx;
        const double py = uy - by;
        const double pz = uz - bz;
        val = 0.5 * ATV::units::md_mvv2e * static_cast<double>(mass[i])
              * (px * px + py * py + pz * pz);
    }
    const double sum = Reduce(reduction).Sum(val);
    if (threadIdx.x == 0)
        atomicAdd(&d->ke_sum, sum);
    __threadfence();
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        const unsigned int old = atomicInc(&d->blocks, gridDim.x - 1);
        is_last = (old == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        __threadfence();
        d->ke = *reinterpret_cast<volatile double*>(&d->ke_sum);
        d->ke_sum = 0.0;
        chain_half(*d, chain);
    }
}

// initial_integrate after the first step: FixNH reuses the t_current the
// previous final_integrate left scaled, so no reduction runs.
__global__ void carried_thermostat(Device* d, ChainArgs chain,
                                   const ForceFault* fault) {
    if (state_frozen(fault))
        return;
    chain_half(*d, chain);
}

__global__ void
scale_peculiar(mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
               mixed_t* __restrict__ vz, const pos_t* __restrict__ x,
               const pos_t* __restrict__ y, const pos_t* __restrict__ z,
               const pos_t* __restrict__ xc, const pos_t* __restrict__ yc,
               const pos_t* __restrict__ zc, const Device* __restrict__ d,
               int n, const ForceFault* fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || state_frozen(fault))
        return;
    const SllodFlow& flow = d->flow;
    double bx, by, bz;
    const double px = load_position(x, xc, i);
    const double py = load_position(y, yc, i);
    const double pz = load_position(z, zc, i);
    ATV::md::sllod_deform_bias(px, py, pz, flow.h_inv, flow.h_rate,
                               flow.h_ratelo, bx, by, bz);
    const double scale = d->scale;
    vx[i] =
        static_cast<mixed_t>(bx + scale * (static_cast<double>(vx[i]) - bx));
    vy[i] =
        static_cast<mixed_t>(by + scale * (static_cast<double>(vy[i]) - by));
    vz[i] =
        static_cast<mixed_t>(bz + scale * (static_cast<double>(vz[i]) - bz));
}

__global__ void
nve_x(pos_t* __restrict__ x, pos_t* __restrict__ y, pos_t* __restrict__ z,
      pos_t* __restrict__ xc, pos_t* __restrict__ yc, pos_t* __restrict__ zc,
      mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
      mixed_t* __restrict__ vz, const fixed_t* __restrict__ fx,
      const fixed_t* __restrict__ fy, const fixed_t* __restrict__ fz,
      const mixed_t* __restrict__ mass, ImageInt* __restrict__ image,
      ForceFault* fault, const std::int64_t* __restrict__ tag, long long step,
      const Device* __restrict__ d, double dt, int n) {
    if (!forces_consumable(fault, tag, step))
        return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const SllodFlow& flow = d->flow;
    const double kick =
        0.5 * dt * ATV::units::md_ftm2v / static_cast<double>(mass[i]);
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
    // CUDA integrators keep rows in the primary cell every step (the
    // neighbor gate and pair kernels take minimum images).
    int shift[3];
    ATV::md::sllod_pbc(px, py, pz, ux, uy, uz, flow.h, flow.h_inv, flow.h_rate,
                       shift);
    add_image_shift(image, i, shift);
    store_position(x, xc, i, static_cast<mixed_t>(px));
    store_position(y, yc, i, static_cast<mixed_t>(py));
    store_position(z, zc, i, static_cast<mixed_t>(pz));
    vx[i] = static_cast<mixed_t>(ux);
    vy[i] = static_cast<mixed_t>(uy);
    vz[i] = static_cast<mixed_t>(uz);
}

// FixDeform::pre_exchange on the device: when end_of_step decided a flip,
// every owned row gets Domain::image_flip and the remap into the flipped
// cell (lambda wrap with deform_vremap); the last block then installs the
// flipped tilts in the flow and the cell and clears the decision. Without
// a pending flip every block returns before the ticket.
__global__ void flip_box(pos_t* __restrict__ x, pos_t* __restrict__ y,
                         pos_t* __restrict__ z, pos_t* __restrict__ xc,
                         pos_t* __restrict__ yc, pos_t* __restrict__ zc,
                         mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
                         mixed_t* __restrict__ vz, ImageInt* __restrict__ image,
                         geom_t* __restrict__ box, Device* d, int n,
                         const ForceFault* fault) {
    if (state_frozen(fault) || !ATV::md::sllod_flip_pending(d->flow))
        return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        SllodFlow flipped = d->flow;
        ATV::md::sllod_apply_flip(flipped);
        ATV::md::sllod_refresh(flipped);
        if (image != nullptr) {
            int b[3] = {image[3 * i], image[3 * i + 1], image[3 * i + 2]};
            ATV::md::sllod_image_flip(b[0], b[1], b[2], flipped.flip);
            for (int k = 0; k < 3; ++k)
                image[3 * i + k] = static_cast<ImageInt>(b[k]);
        }
        double px = load_position(x, xc, i);
        double py = load_position(y, yc, i);
        double pz = load_position(z, zc, i);
        double ux = vx[i], uy = vy[i], uz = vz[i];
        int shift[3];
        ATV::md::sllod_pbc(px, py, pz, ux, uy, uz, flipped.h, flipped.h_inv,
                           flipped.h_rate, shift);
        add_image_shift(image, i, shift);
        store_position(x, xc, i, static_cast<mixed_t>(px));
        store_position(y, yc, i, static_cast<mixed_t>(py));
        store_position(z, zc, i, static_cast<mixed_t>(pz));
        vx[i] = static_cast<mixed_t>(ux);
        vy[i] = static_cast<mixed_t>(uy);
        vz[i] = static_cast<mixed_t>(uz);
    }
    __syncthreads();
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned int old = atomicInc(&d->flip_blocks, gridDim.x - 1);
        is_last = (old == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        SllodFlow& flow = d->flow;
        ATV::md::sllod_apply_flip(flow);
        ATV::md::sllod_refresh(flow);
        flow.flip[0] = flow.flip[1] = flow.flip[2] = 0;
        write_upper(box, flow.h);
    }
}

// FixDeform::end_of_step on the device: advance the strain clock, write the
// deformed cell and decide the flip (sllod_advance_box). The host checked
// the deterministic edge positivity before enqueueing.
__global__ void advance_box(geom_t* __restrict__ box, Device* d,
                            const ForceFault* fault) {
    if (state_frozen(fault))
        return;
    d->step += 1;
    SllodFlow& flow = d->flow;
    d->shift = ATV::md::sllod_advance_box(flow, d->step);
    ATV::md::sllod_refresh(flow);
    write_upper(box, flow.h);
}

// Edge direction: symmetric origin shift of every row, then the wrap of the
// owned rows in the new cell.
__global__ void shift_edge(pos_t* __restrict__ x, pos_t* __restrict__ y,
                           pos_t* __restrict__ z, pos_t* __restrict__ xc,
                           pos_t* __restrict__ yc, pos_t* __restrict__ zc,
                           mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
                           mixed_t* __restrict__ vz,
                           ImageInt* __restrict__ image,
                           const Device* __restrict__ d, int axis, int n_local,
                           int n_all, const ForceFault* fault) {
    if (state_frozen(fault))
        return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_all)
        return;
    const double shift = d->shift;
    if (shift == 0.0)
        return;
    pos_t* pos = axis == 0 ? x : axis == 1 ? y : z;
    pos_t* corr = axis == 0 ? xc : axis == 1 ? yc : zc;
    store_position(pos, corr, i,
                   load_position(pos, corr, i) + static_cast<mixed_t>(shift));
    if (i >= n_local)
        return;
    const SllodFlow& flow = d->flow;
    double px = load_position(x, xc, i);
    double py = load_position(y, yc, i);
    double pz = load_position(z, zc, i);
    double ux = vx[i], uy = vy[i], uz = vz[i];
    int lattice[3];
    ATV::md::sllod_pbc(px, py, pz, ux, uy, uz, flow.h, flow.h_inv, flow.h_rate,
                       lattice);
    add_image_shift(image, i, lattice);
    store_position(x, xc, i, static_cast<mixed_t>(px));
    store_position(y, yc, i, static_cast<mixed_t>(py));
    store_position(z, zc, i, static_cast<mixed_t>(pz));
    vx[i] = static_cast<mixed_t>(ux);
    vy[i] = static_cast<mixed_t>(uy);
    vz[i] = static_cast<mixed_t>(uz);
}

} // namespace

struct NvtSllod::State {
    Device* device = nullptr;
    double* chain = nullptr;
    int ndof = 0;
    double mass0 = 0;
    void release() {
        if (device)
            cudaFree(device);
        if (chain)
            cudaFree(chain);
        device = nullptr;
        chain = nullptr;
    }
    ~State() { release(); }
};

NvtSllod::NvtSllod(double temperature_K, double tau_fs, double dt_fs,
                   double rate_per_fs, int direction, int tchain)
    : T_(temperature_K), tau_(tau_fs), dt_(dt_fs), rate_(rate_per_fs),
      dir_(direction), tchain_(tchain), state_(std::make_unique<State>()) {
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
    if (tchain_ < 1)
        throw std::invalid_argument("NvtSllod: tchain must be at least 1");
}

NvtSllod::~NvtSllod() = default;
NvtSllod::NvtSllod(NvtSllod&&) noexcept = default;
NvtSllod& NvtSllod::operator=(NvtSllod&&) noexcept = default;

void NvtSllod::build(const BuildContext& ctx) {
    state_->release();
    state_->ndof = static_cast<int>(ctx.runtime.md.n_dof_global);
    armed_ = false;
    kicked_ = false;
    measured_ = false;
    step_ = 0;

    // The strain origin h0 is the built cell. Read it here: launch() never
    // reads the device back (iron law 10).
    Device host{};
    {
        auto& flow = host.flow;
        geom_t host_box[9];
        ATV_CUDA_CHECK(cudaMemcpy(host_box, ctx.frame.box, 9 * sizeof(geom_t),
                                  cudaMemcpyDeviceToHost));
        read_upper(host_box, flow.h0);
        for (int i = 0; i < 6; ++i)
            flow.h[i] = flow.h0[i];
        flow.rate = rate_;
        flow.dt = dt_;
        flow.dir = dir_;
        ATV::md::sllod_erate_rates(flow.h0, rate_, dir_, flow.h_rate,
                                   flow.h_ratelo);
        if (!ATV::md::sllod_refresh(flow))
            throw std::invalid_argument("NvtSllod: box edge must be positive");
        // A tilt exists or will: only then can a flip ever be decided.
        can_flip_ = dir_ >= 3 || flow.h0[3] != 0.0 || flow.h0[4] != 0.0
                    || flow.h0[5] != 0.0;
    }
    std::vector<double> chain(3 * static_cast<std::size_t>(tchain_), 0.0);
    if (state_->ndof > 0)
        ATV::md::sllod_chain_masses(chain.data() + 2 * tchain_, tchain_,
                                    static_cast<double>(state_->ndof),
                                    units::md_boltz * T_, tau_);
    state_->mass0 = chain[2 * tchain_];
    ATV_CUDA_CHECK(cudaMalloc(&state_->device, sizeof(Device)));
    ATV_CUDA_CHECK(cudaMemcpy(state_->device, &host, sizeof(Device),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMalloc(&state_->chain, chain.size() * sizeof(double)));
    ATV_CUDA_CHECK(cudaMemcpy(state_->chain, chain.data(),
                              chain.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
    armed_ = true;
}

void NvtSllod::destroy(const DestroyContext&) {
    if (state_)
        state_->release();
    armed_ = false;
    kicked_ = false;
    measured_ = false;
}

ATV::md::SllodChainState NvtSllod::thermostat_chain() const {
    if (!state_ || !state_->chain)
        throw std::logic_error("NvtSllod: thermostat_chain before build");
    std::vector<double> chain(3 * static_cast<std::size_t>(tchain_));
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    ATV_CUDA_CHECK(cudaMemcpy(chain.data(), state_->chain,
                              chain.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
    ATV::md::SllodChainState out;
    out.eta.assign(chain.begin(), chain.begin() + tchain_);
    out.eta_dot.assign(chain.begin() + tchain_, chain.begin() + 2 * tchain_);
    return out;
}

void NvtSllod::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const auto& a = frame.atoms;
    const int n = frame.n_local();
    const int n_all = frame.n_all();
    const int block = cuda::kDefaultBlock;
    Device* d = state_->device;

    if (!armed_)
        throw std::logic_error("NvtSllod: launch before build");

    const double kT = units::md_boltz * T_;
    const ChainArgs chain{state_->chain,
                          tchain_,
                          static_cast<double>(state_->ndof) * kT,
                          kT,
                          dt_,
                          state_->ndof > 0 && state_->mass0 > 0};
    const int grid = cuda::grid_size(n);
    auto scale = [&]() {
        scale_peculiar<<<grid, block, 0, stream>>>(
            a.vx, a.vy, a.vz, a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, d, n,
            frame.force_fault);
    };

    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        if (!kicked_) {
            kick_bias<<<grid, block, 0, stream>>>(
                a.vx, a.vy, a.vz, a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr,
                d, n, frame.force_fault);
            kicked_ = true;
        }
        if (!measured_) {
            // FixNH::setup: the first half-step measures the temperature.
            measure_thermostat<false><<<grid, block, 0, stream>>>(
                a.vx, a.vy, a.vz, a.fx, a.fy, a.fz, a.x, a.y, a.z, a.x_corr,
                a.y_corr, a.z_corr, a.mass, d, chain, n, frame.force_fault,
                a.tag, ctx.step);
            measured_ = true;
        } else {
            carried_thermostat<<<1, 1, 0, stream>>>(d, chain,
                                                    frame.force_fault);
        }
        scale();
        nve_x<<<grid, block, 0, stream>>>(
            a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, a.vx, a.vy, a.vz, a.fx,
            a.fy, a.fz, a.mass, a.image, frame.force_fault, a.tag, ctx.step, d,
            dt_, n);
    } else if (ctx.at(ATV::md::Phase::pre_exchange)) {
        if (!can_flip_)
            return;
        flip_box<<<grid, block, 0, stream>>>(
            a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, a.vx, a.vy, a.vz,
            a.image, frame.box, d, n, frame.force_fault);
    } else if (ctx.at(ATV::md::Phase::final_integrate)) {
        measure_thermostat<true><<<grid, block, 0, stream>>>(
            a.vx, a.vy, a.vz, a.fx, a.fy, a.fz, a.x, a.y, a.z, a.x_corr,
            a.y_corr, a.z_corr, a.mass, d, chain, n, frame.force_fault, a.tag,
            ctx.step);
        measured_ = true;
        scale();
    } else if (ctx.at(ATV::md::Phase::end_of_step)) {
        const long long next = step_ + 1;
        if (dir_ < 3) {
            // Deterministic in the host's own step count; no device read.
            double h_new[6];
            double h0[6] = {0, 0, 0, 0, 0, 0};
            h0[dir_] = 1.0;
            ATV::md::sllod_box_at(h0, rate_, dt_, dir_, next, h_new);
            if (!(h_new[dir_] > 0.0))
                throw std::invalid_argument(
                    "NvtSllod: deformed edge is not positive");
        }
        advance_box<<<1, 1, 0, stream>>>(frame.box, d, frame.force_fault);
        if (dir_ < 3)
            shift_edge<<<cuda::grid_size(n_all), block, 0, stream>>>(
                a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, a.vx, a.vy, a.vz,
                a.image, d, dir_, n, n_all, frame.force_fault);
        step_ = next;
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

} // namespace md
} // namespace cuda
} // namespace ATV
