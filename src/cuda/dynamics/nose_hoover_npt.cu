/// @file cuda/dynamics/nose_hoover_npt.cu
/// @brief Device-resident MTK NPT. Scalar NHC laws are shared with CPU.
/// Martyna et al., JCP 101, 4177 (1994), doi:10.1063/1.467468.
#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover_npt.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thrust/device_vector.h>

namespace ATV::cuda::md {
namespace {
struct NptParameters {
    double temperature, stop_temperature, tau_t, tau_p, pressure, dt, drag;
    int atom_chain, barostat_chain, pressure_dimensions;
    BigInt first, last, dof, atoms;
    bool coupled[3];
    // A faulted run keeps every chain, cell and atom at the faulting step.
    const ForceFault* fault;
};

struct NptHistory {
    double *qt, *qb, *eta_t, *eta_b, *dot_t, *dot_b;
    double kt, mass, kinetic;
    double eps[3], scale[3], drift[3];
    Box next_box;
    bool anisotropic;
};

__global__ void initialize_history(NptHistory* history, NptParameters p) {
    auto& h = *history;
    h.kt = units::kB_ev_per_k * p.temperature;
    h.mass = (double(p.dof) + 3.0) * h.kt * p.tau_p * p.tau_p;
    for (int k = 0; k < p.atom_chain; ++k)
        h.qt[k] = (k == 0 ? double(p.dof) : 1.0) * h.kt * p.tau_t * p.tau_t;
    for (int k = 0; k < p.barostat_chain; ++k)
        h.qb[k] = h.kt * p.tau_p * p.tau_p;
}

__device__ void barostat_chain(NptHistory& h, const NptParameters& p) {
    double ke = 0;
    for (int a = 0; a < 3; ++a)
        if (p.coupled[a])
            ke += (h.mass / 3.0) * h.eps[a] * h.eps[a];
    ke *= 0.5;
    const double scale = ATV::md::nhc_half_step(
        h.eta_b, h.dot_b, h.qb, p.barostat_chain, ke, h.kt, h.kt, 0.5 * p.dt,
        1.0 - p.drag * p.dt / p.tau_p);
    for (int a = 0; a < 3; ++a)
        if (p.coupled[a])
            h.eps[a] *= scale;
}

__global__ void atom_chain(NptHistory* history, NptParameters p, BigInt step,
                           bool initial) {
    if (state_frozen(p.fault))
        return;
    auto& h = *history;
    if (initial) {
        h.kt = units::kB_ev_per_k
               * ATV::md::ramp_target(p.temperature, p.stop_temperature, step,
                                      p.first, p.last);
        for (int k = 0; k < p.atom_chain; ++k)
            h.qt[k] = (k == 0 ? double(p.dof) : 1.0) * h.kt * p.tau_t * p.tau_t;
        barostat_chain(h, p); // P1 precedes atom thermostat P2
    }
    const double scale = ATV::md::nhc_half_step(
        h.eta_t, h.dot_t, h.qt, p.atom_chain, h.kinetic, double(p.dof) * h.kt,
        h.kt, 0.5 * p.dt, 1.0 - p.drag * p.dt / p.tau_t);
    h.scale[0] = h.scale[1] = h.scale[2] = scale;
}

__global__ void scale_velocities(mixed_t* vx, mixed_t* vy, mixed_t* vz,
                                 const NptHistory* history, int n,
                                 const ForceFault* fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !state_frozen(fault)) {
        vx[i] *= mixed_t(history->scale[0]);
        vy[i] *= mixed_t(history->scale[1]);
        vz[i] *= mixed_t(history->scale[2]);
    }
}

__global__ void barostat_scale(NptHistory* history, NptParameters p) {
    if (state_frozen(p.fault))
        return;
    auto& h = *history;
    double sum = 0;
    for (int a = 0; a < 3; ++a)
        if (p.coupled[a])
            sum += h.eps[a];
    const BigInt atoms = p.atoms;
    const double mtk =
        p.pressure_dimensions > 0
            ? sum / (double(p.pressure_dimensions) * double(atoms))
            : 0.0;
    for (int a = 0; a < 3; ++a)
        h.scale[a] = exp(-(h.eps[a] + mtk) * 0.5 * p.dt);
    h.anisotropic = h.scale[0] != h.scale[1] || h.scale[1] != h.scale[2];
    if (!h.anisotropic)
        h.kinetic *= h.scale[0] * h.scale[0];
}

__global__ void kick(mixed_t* vx, mixed_t* vy, mixed_t* vz, const fixed_t* fx,
                     const fixed_t* fy, const fixed_t* fz, const mixed_t* mass,
                     ForceFault* fault, const std::int64_t* tag, long long step,
                     mixed_t half_dt, int n) {
    if (!forces_consumable(fault, tag, step))
        return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const mixed_t factor =
            half_dt * mixed_t(units::force_to_accel) / mass[i];
        vx[i] += factor * decode_force<mixed_t>(fx[i]);
        vy[i] += factor * decode_force<mixed_t>(fy[i]);
        vz[i] += factor * decode_force<mixed_t>(fz[i]);
    }
}
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock, 4) void reduce_ke_axes_k(
    const mixed_t* __restrict__ vx, const mixed_t* __restrict__ vy,
    const mixed_t* __restrict__ vz, const mixed_t* __restrict__ mass,
    acc_t* __restrict__ buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double mx = 0.0, my = 0.0, mz = 0.0;
    if (i < n) {
        double m = static_cast<double>(mass[i]);
        double vxi = static_cast<double>(vx[i]);
        double vyi = static_cast<double>(vy[i]);
        double vzi = static_cast<double>(vz[i]);
        mx = m * vxi * vxi;
        my = m * vyi * vyi;
        mz = m * vzi * vzi;
    }
    // One CUB collective preserves the three independent kinetic components.
    ::cuda::std::array<double, 3> my_vals = {mx, my, mz};

    using Reduce =
        cub::BlockReduce<::cuda::std::array<double, 3>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto out =
        Reduce(reduction).Reduce(my_vals, SumComponents<double, 3>{});
    if (threadIdx.x == 0) {
        double bx = out[0], by = out[1], bz = out[2];
        atomicAdd(&buf[1], bx);
        atomicAdd(&buf[2], by);
        atomicAdd(&buf[3], bz);
        atomicAdd(&buf[0], bx + by + bz);
    }
}

__global__ void seed_kinetic(NptHistory* history,
                             const SystemScalars* scalars) {
    history->kinetic = scalars->kinetic;
}
__global__ void reduced_kinetic(NptHistory* history, const double* ke,
                                const ForceFault* fault) {
    if (state_frozen(fault))
        return;
    history->kinetic = 0.5 * units::mass_v2_to_energy * ke[0];
}

__global__ void pressure_update(NptHistory* history, NptParameters p,
                                const double* ke, const SystemScalars* scalars,
                                const geom_t* cell) {
    if (state_frozen(p.fault))
        return;
    auto& h = *history;
    const double volume = cell[0] * cell[4] * cell[8];
    if (!(volume > 0))
        return;
    double axes[3], diagonal[3], active_ke = 0;
    for (int a = 0; a < 3; ++a) {
        axes[a] = 0.5 * units::mass_v2_to_energy * ke[1 + a];
        diagonal[a] = scalars->stress[a] + 2.0 * axes[a];
        if (p.coupled[a])
            active_ke += axes[a];
    }
    if (h.anisotropic) {
        h.kinetic = 0.5 * units::mass_v2_to_energy * ke[0];
        h.anisotropic = false;
    }
    if (p.pressure_dimensions == 3) {
        const double average = (diagonal[0] + diagonal[1] + diagonal[2]) / 3.0;
        diagonal[0] = diagonal[1] = diagonal[2] = average;
    }
    const BigInt atoms = p.atoms;
    const double mtk =
        p.pressure_dimensions > 0
            ? 6.0 * active_ke / (double(p.pressure_dimensions) * double(atoms))
            : 0.0;
    for (int a = 0; a < 3; ++a)
        if (p.coupled[a])
            h.eps[a] +=
                (3.0 * volume * (diagonal[a] / volume - p.pressure) + mtk)
                / h.mass * (0.5 * p.dt);
}

__global__ void propagate_cell(NptHistory* history, NptParameters p,
                               geom_t* cell, Box seed) {
    if (state_frozen(p.fault))
        return;
    auto& h = *history;
    for (int a = 0; a < 3; ++a) {
        double x = h.eps[a] * p.dt;
        x = x < -0.005 ? -0.005 : (x > 0.005 ? 0.005 : x);
        h.scale[a] = exp(x);
        h.drift[a] = fabs(x) < 1e-4 ? 1.0 + x * (0.5 + x / 6.0) : expm1(x) / x;
        cell[4 * a] *= h.scale[a];
    }
    seed.resident = nullptr;
    seed.assign_cell(cell);
    h.next_box = seed;
}

// Coordinates are reconstructed, propagated and wrapped in integration
// precision against the FP64 next cell, then split again.
__global__ void propagate_positions(pos_t* x, pos_t* y, pos_t* z, pos_t* xc,
                                    pos_t* yc, pos_t* zc, const mixed_t* vx,
                                    const mixed_t* vy, const mixed_t* vz,
                                    const NptHistory* history, mixed_t dt,
                                    int n, const ForceFault* fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || state_frozen(fault))
        return;
    const auto& h = *history;
    mixed_t value[3] = {load_position(x, xc, i) * mixed_t(h.scale[0])
                            + vx[i] * dt * mixed_t(h.drift[0]),
                        load_position(y, yc, i) * mixed_t(h.scale[1])
                            + vy[i] * dt * mixed_t(h.drift[1]),
                        load_position(z, zc, i) * mixed_t(h.scale[2])
                            + vz[i] * dt * mixed_t(h.drift[2])};
    for (int a = 0; a < 3; ++a)
        value[a] -= mixed_t(h.next_box.cell[4 * a])
                    * floor(value[a] * mixed_t(h.next_box.inv_l[a]));
    store_position(x, xc, i, value[0]);
    store_position(y, yc, i, value[1]);
    store_position(z, zc, i, value[2]);
}

__global__ void finish(NptHistory* history, NptParameters p,
                       SystemScalars* scalars, const geom_t* cell) {
    if (state_frozen(p.fault))
        return;
    auto& h = *history;
    barostat_chain(h, p); // P11, after the atom thermostat P10
    double energy = h.kinetic + scalars->energy;
    for (int a = 0; a < 3; ++a)
        if (p.coupled[a])
            energy += 0.5 * (h.mass / 3.0) * h.eps[a] * h.eps[a];
    for (int k = 0; k < p.atom_chain; ++k)
        energy += 0.5 * h.qt[k] * h.dot_t[k] * h.dot_t[k];
    for (int k = 0; k < p.barostat_chain; ++k)
        energy += 0.5 * h.qb[k] * h.dot_b[k] * h.dot_b[k];
    energy += double(p.dof) * h.kt * h.eta_t[0];
    for (int k = 1; k < p.atom_chain; ++k)
        energy += h.kt * h.eta_t[k];
    for (int k = 0; k < p.barostat_chain; ++k)
        energy += h.kt * h.eta_b[k];
    energy += p.pressure * cell[0] * cell[4] * cell[8];
    scalars->kinetic = h.kinetic;
    scalars->conserved = energy;
}
} // namespace

struct NoseHooverNPT::State {
    thrust::device_vector<double> chain;
    thrust::device_vector<double> ke;
    thrust::device_vector<NptHistory> history;
    State() : ke(4), history(1) {}
    NptParameters parameters{};
};

NoseHooverNPT::NoseHooverNPT(double temperature, double tau_t, double pressure,
                             double tau_p, double dt, double drag,
                             int atom_chain, int barostat_chain,
                             std::array<bool, 3> coupled,
                             double stop_temperature)
    : T_target_(temperature),
      T_stop_(stop_temperature < 0 ? temperature : stop_temperature),
      tau_T_(tau_t), P_target_(pressure), tau_P_(tau_p), dt_(dt), drag_(drag),
      mtchain_(atom_chain), mpchain_(barostat_chain), couple_axis_(coupled),
      T_live_(temperature) {
    if (!(temperature > 0) || !(tau_t > 0) || !(tau_p > 0) || !(dt >= 0)
        || atom_chain < 1 || barostat_chain < 1)
        throw std::invalid_argument("NoseHooverNPT requires positive "
                                    "T/tau/chain lengths and nonnegative dt");
    pdim_ = int(coupled[0]) + int(coupled[1]) + int(coupled[2]);
}
NoseHooverNPT::~NoseHooverNPT() = default;
NoseHooverNPT::NoseHooverNPT(NoseHooverNPT&&) noexcept = default;
NoseHooverNPT& NoseHooverNPT::operator=(NoseHooverNPT&&) noexcept = default;

void NoseHooverNPT::build(const BuildContext& ctx) {
    if (!ctx.runtime.host_box.ortho)
        throw std::runtime_error(
            "cuda::md::NoseHooverNPT: triclinic cells are not supported");
    state_ = std::make_unique<State>();
    auto& s = *state_;
    s.parameters = {T_target_,
                    T_stop_,
                    tau_T_,
                    tau_P_,
                    P_target_,
                    dt_,
                    drag_,
                    mtchain_,
                    mpchain_,
                    pdim_,
                    nfirst_,
                    nlast_,
                    ctx.runtime.md.n_dof_global,
                    ctx.frame.n_atoms(),
                    {couple_axis_[0], couple_axis_[1], couple_axis_[2]},
                    nullptr};
    s.chain.resize(3 * (mtchain_ + mpchain_));
    NptHistory h{};
    h.qt = thrust::raw_pointer_cast(s.chain.data());
    h.eta_t = h.qt + mtchain_;
    h.dot_t = h.eta_t + mtchain_;
    h.qb = h.dot_t + mtchain_;
    h.eta_b = h.qb + mpchain_;
    h.dot_b = h.eta_b + mpchain_;
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    ATV_CUDA_CHECK(cudaMemsetAsync(thrust::raw_pointer_cast(s.chain.data()), 0,
                                   s.chain.size() * sizeof(double), stream));
    ATV_CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(s.history.data()), &h,
                              sizeof(h), cudaMemcpyHostToDevice));
    initialize_history<<<1, 1, 0, stream>>>(
        thrust::raw_pointer_cast(s.history.data()), s.parameters);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void NoseHooverNPT::initialize_forces(const LaunchContext& ctx) {
    if (ctx.at(ATV::md::Phase::end_of_step)) {
        seed_kinetic<<<1, 1, 0,
                       static_cast<cudaStream_t>(ctx.runtime.stream)>>>(
            thrust::raw_pointer_cast(state_->history.data()),
            ctx.frame.scalars);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}

void NoseHooverNPT::launch(const LaunchContext& ctx) {
    auto& s = *state_;
    auto& f = ctx.frame;
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    const int n = f.n_atoms(), blocks = std::max(1, grid_size(n));
    auto p = s.parameters;
    p.first = nfirst_;
    p.last = nlast_;
    p.fault = f.force_fault;
    auto scale = [&] {
        scale_velocities<<<blocks, kDefaultBlock, 0, stream>>>(
            f.atoms.vx, f.atoms.vy, f.atoms.vz,
            thrust::raw_pointer_cast(s.history.data()), n, f.force_fault);
    };
    auto reduce = [&] {
        ATV_CUDA_CHECK(cudaMemsetAsync(thrust::raw_pointer_cast(s.ke.data()), 0,
                                       4 * sizeof(double), stream));
        reduce_ke_axes_k<<<blocks, kDefaultBlock, 0, stream>>>(
            f.atoms.vx, f.atoms.vy, f.atoms.vz, f.atoms.mass,
            thrust::raw_pointer_cast(s.ke.data()), n);
    };
    auto force_kick = [&] {
        kick<<<blocks, kDefaultBlock, 0, stream>>>(
            f.atoms.vx, f.atoms.vy, f.atoms.vz, f.atoms.fx, f.atoms.fy,
            f.atoms.fz, f.atoms.mass, f.force_fault, f.atoms.tag, ctx.step,
            mixed_t(0.5 * dt_), n);
    };
    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        T_live_ =
            ATV::md::ramp_target(T_target_, T_stop_, ctx.step, nfirst_, nlast_);
        atom_chain<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(s.history.data()), p, ctx.step,
            true); // P1,P2
        scale();
        reduce();
        pressure_update<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(s.history.data()), p,
            thrust::raw_pointer_cast(s.ke.data()), f.scalars, f.box); // P3
        barostat_scale<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(s.history.data()), p); // P4
        scale();
        force_kick(); // P5
        propagate_cell<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(s.history.data()), p, f.box,
            ctx.runtime.host_box); // P6
        propagate_positions<<<blocks, kDefaultBlock, 0, stream>>>(
            f.atoms.x, f.atoms.y, f.atoms.z, f.atoms.x_corr, f.atoms.y_corr,
            f.atoms.z_corr, f.atoms.vx, f.atoms.vy, f.atoms.vz,
            thrust::raw_pointer_cast(s.history.data()), mixed_t(dt_), n,
            f.force_fault);
    } else if (ctx.at(ATV::md::Phase::final_integrate)) {
        force_kick(); // P7
    } else if (ctx.at(ATV::md::Phase::end_of_step)) {
        reduce();
        reduced_kinetic<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(s.history.data()),
            thrust::raw_pointer_cast(s.ke.data()), f.force_fault);
        barostat_scale<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(s.history.data()), p); // P8
        scale();
        reduce();
        pressure_update<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(s.history.data()), p,
            thrust::raw_pointer_cast(s.ke.data()), f.scalars, f.box); // P9
        atom_chain<<<1, 1, 0, stream>>>(
            thrust::raw_pointer_cast(s.history.data()), p, ctx.step,
            false); // P10
        scale();
        finish<<<1, 1, 0, stream>>>(thrust::raw_pointer_cast(s.history.data()),
                                    p, f.scalars, f.box); // P11
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

void NoseHooverNPT::destroy(const DestroyContext&) { state_.reset(); }
} // namespace ATV::cuda::md
