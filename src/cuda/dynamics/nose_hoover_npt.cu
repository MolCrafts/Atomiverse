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
#include <cstdint>
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
    // The run's LAMMPS `units` factors.
    double boltz, mvv2e, ftm2v;
};

struct NptHistory {
    double *qt, *qb, *eta_t, *eta_b, *dot_t, *dot_b;
    double kt, mass, kinetic;
    double eps[3], scale[3], cell_scale[3], drift[3];
    Box next_box;
    bool anisotropic;
};

__global__ void initialize_history(NptHistory* history, NptParameters p) {
    auto& h = *history;
    h.kt = p.boltz * p.temperature;
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

// ── Scalar stages (one thread; each early-returns on a frozen run) ────────

__device__ void atom_chain(NptHistory& h, const NptParameters& p,
                           long long step, bool initial) {
    if (initial) {
        h.kt = p.boltz
               * ATV::md::ramp_target(p.temperature, p.stop_temperature,
                                      BigInt(step), p.first, p.last);
        for (int k = 0; k < p.atom_chain; ++k)
            h.qt[k] = (k == 0 ? double(p.dof) : 1.0) * h.kt * p.tau_t * p.tau_t;
        barostat_chain(h, p); // P1 precedes atom thermostat P2
    }
    const double scale = ATV::md::nhc_half_step(
        h.eta_t, h.dot_t, h.qt, p.atom_chain, h.kinetic, double(p.dof) * h.kt,
        h.kt, 0.5 * p.dt, 1.0 - p.drag * p.dt / p.tau_t);
    h.scale[0] = h.scale[1] = h.scale[2] = scale;
}

__device__ void barostat_scale(NptHistory& h, const NptParameters& p) {
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

__device__ void pressure_update(NptHistory& h, const NptParameters& p,
                                const double* ke, const SystemScalars* scalars,
                                const geom_t* cell) {
    const double volume = cell[0] * cell[4] * cell[8];
    if (!(volume > 0))
        return;
    double axes[3], diagonal[3], active_ke = 0;
    for (int a = 0; a < 3; ++a) {
        axes[a] = 0.5 * p.mvv2e * ke[1 + a];
        diagonal[a] = scalars->stress[a] + 2.0 * axes[a];
        if (p.coupled[a])
            active_ke += axes[a];
    }
    if (h.anisotropic) {
        h.kinetic = 0.5 * p.mvv2e * ke[0];
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

/// The next cell (P6) from the current strain rate. The cell itself is
/// committed by the position propagation, which reads only `next_box`.
__device__ void next_cell(NptHistory& h, const NptParameters& p,
                          const geom_t* cell, Box seed) {
    geom_t next[9];
    for (int k = 0; k < 9; ++k)
        next[k] = cell[k];
    for (int a = 0; a < 3; ++a) {
        double x = h.eps[a] * p.dt;
        x = x < -0.005 ? -0.005 : (x > 0.005 ? 0.005 : x);
        h.cell_scale[a] = exp(x);
        h.drift[a] = fabs(x) < 1e-4 ? 1.0 + x * (0.5 + x / 6.0) : expm1(x) / x;
        next[4 * a] *= h.cell_scale[a];
    }
    seed.resident = nullptr;
    seed.assign_cell(next);
    h.next_box = seed;
}

__device__ void finish(NptHistory& h, const NptParameters& p,
                       SystemScalars* scalars, const geom_t* cell) {
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

__device__ void clear_kinetic(double* ke) {
    ke[0] = ke[1] = ke[2] = ke[3] = 0.0;
}

// ── Fused launches: one single-thread kernel between grid passes ──────────

/// initial_integrate P1-P2: barostat and atom chains at the ramped target.
__global__ void initial_chains(NptHistory* history, NptParameters p,
                               const long long* step_ptr, long long step) {
    if (state_frozen(p.fault))
        return;
    atom_chain(*history, p, current_step(step_ptr, step), true);
}

/// initial_integrate P3-P4 and the P6 cell: strain rate from the scaled
/// kinetic tensor, the barostat velocity scale and the next cell.
__global__ void initial_barostat(NptHistory* history, NptParameters p,
                                 double* ke, const SystemScalars* scalars,
                                 const geom_t* cell, Box seed) {
    if (state_frozen(p.fault))
        return;
    auto& h = *history;
    pressure_update(h, p, ke, scalars, cell);
    barostat_scale(h, p);
    next_cell(h, p, cell, seed);
    clear_kinetic(ke);
}

/// end_of_step P8: measured kinetic energy, then the barostat scale.
__global__ void final_barostat(NptHistory* history, NptParameters p,
                               double* ke) {
    if (state_frozen(p.fault))
        return;
    auto& h = *history;
    h.kinetic = 0.5 * p.mvv2e * ke[0];
    barostat_scale(h, p);
    clear_kinetic(ke);
}

/// end_of_step P9-P11: strain rate, atom chain, barostat chain and the
/// conserved quantity (none of them reads the velocities P10 rescales).
__global__ void final_chains(NptHistory* history, NptParameters p, double* ke,
                             SystemScalars* scalars, const geom_t* cell,
                             const long long* step_ptr, long long step) {
    if (state_frozen(p.fault))
        return;
    auto& h = *history;
    pressure_update(h, p, ke, scalars, cell);
    atom_chain(h, p, current_step(step_ptr, step), false);
    finish(h, p, scalars, cell);
    clear_kinetic(ke);
}

/// Per-axis kinetic tensor of the block's rows into `buf` (total, x, y, z).
__device__ void accumulate_kinetic(double mx, double my, double mz,
                                   acc_t* __restrict__ buf) {
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

/// Optionally rescale the velocities by the history's scale, then reduce
/// the kinetic tensor of the result.
template <bool Scale, bool Reduce>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock, 4) void scale_reduce(
    mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
    mixed_t* __restrict__ vz, const mixed_t* __restrict__ mass,
    const NptHistory* history, acc_t* __restrict__ buf, int n,
    const ForceFault* fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double mx = 0.0, my = 0.0, mz = 0.0;
    if (i < n) {
        mixed_t ux = vx[i], uy = vy[i], uz = vz[i];
        if (Scale && !state_frozen(fault)) {
            ux *= mixed_t(history->scale[0]);
            uy *= mixed_t(history->scale[1]);
            uz *= mixed_t(history->scale[2]);
            vx[i] = ux;
            vy[i] = uy;
            vz[i] = uz;
        }
        if (Reduce) {
            const double m = static_cast<double>(mass[i]);
            const double a = ux, b = uy, c = uz;
            mx = m * a * a;
            my = m * b * b;
            mz = m * c * c;
        }
    }
    if (Reduce)
        accumulate_kinetic(mx, my, mz, buf);
}

/// P4 velocity scale fused with the P5 / P7 half kick.
template <bool Scale>
__global__ void
scale_kick(mixed_t* vx, mixed_t* vy, mixed_t* vz, const fixed_t* fx,
           const fixed_t* fy, const fixed_t* fz, const mixed_t* mass,
           const NptHistory* history, ForceFault* fault,
           const std::int64_t* tag, const long long* step_ptr, long long step,
           mixed_t half_dt, mixed_t ftm2v, int n) {
    if (!forces_consumable(fault, tag, step_ptr, step))
        return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        mixed_t ux = vx[i], uy = vy[i], uz = vz[i];
        if (Scale) {
            ux *= mixed_t(history->scale[0]);
            uy *= mixed_t(history->scale[1]);
            uz *= mixed_t(history->scale[2]);
        }
        const mixed_t factor = half_dt * ftm2v / mass[i];
        vx[i] = ux + factor * decode_force<mixed_t>(fx[i]);
        vy[i] = uy + factor * decode_force<mixed_t>(fy[i]);
        vz[i] = uz + factor * decode_force<mixed_t>(fz[i]);
    }
}

__global__ void seed_kinetic(NptHistory* history,
                             const SystemScalars* scalars) {
    history->kinetic = scalars->kinetic;
}

// Coordinates are reconstructed, propagated and wrapped in integration
// precision against the FP64 next cell, then split again. The first thread
// commits the next cell (no thread reads the old one here).
__global__ void propagate_positions(pos_t* x, pos_t* y, pos_t* z, pos_t* xc,
                                    pos_t* yc, pos_t* zc, const mixed_t* vx,
                                    const mixed_t* vy, const mixed_t* vz,
                                    ImageInt* image, const NptHistory* history,
                                    geom_t* cell, mixed_t dt, int n,
                                    const ForceFault* fault) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (state_frozen(fault))
        return;
    const auto& h = *history;
    if (i == 0)
        for (int a = 0; a < 3; ++a)
            cell[4 * a] = h.next_box.cell[4 * a];
    if (i >= n)
        return;
    mixed_t value[3] = {load_position(x, xc, i) * mixed_t(h.cell_scale[0])
                            + vx[i] * dt * mixed_t(h.drift[0]),
                        load_position(y, yc, i) * mixed_t(h.cell_scale[1])
                            + vy[i] * dt * mixed_t(h.drift[1]),
                        load_position(z, zc, i) * mixed_t(h.cell_scale[2])
                            + vz[i] * dt * mixed_t(h.drift[2])};
    for (int a = 0; a < 3; ++a) {
        const mixed_t shift = floor(value[a] * mixed_t(h.next_box.inv_l[a]));
        value[a] -= mixed_t(h.next_box.cell[4 * a]) * shift;
        // LAMMPS Domain::pbc: the wrapped lattice translation joins the
        // image flags, keeping x + L·image continuous.
        if (image != nullptr)
            image[3 * i + a] = static_cast<ImageInt>(image[3 * i + a]
                                                     + static_cast<int>(shift));
    }
    store_position(x, xc, i, value[0]);
    store_position(y, yc, i, value[1]);
    store_position(z, zc, i, value[2]);
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
            "cuda::md::NoseHooverNPT: triclinic cells are not supported — "
            "the CUDA MTK barostat scales the three diagonal cell edges only. "
            "Use an orthorhombic cell (zero tilts), or run the triclinic NPT "
            "on the CPU backend (NoseHooverNPT / ParrinelloRahmanNPT)");
    state_ = std::make_unique<State>();
    auto& s = *state_;
    s.parameters = {T_target_,
                    T_stop_,
                    tau_T_,
                    tau_P_,
                    // LAMMPS `p_target / nktv2p`: energy/volume.
                    P_target_ / ctx.runtime.units.nktv2p,
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
                    nullptr,
                    ctx.runtime.units.boltz,
                    ctx.runtime.units.mvv2e,
                    ctx.runtime.units.ftm2v};
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

void NoseHooverNPT::before_launch(const LaunchContext& ctx) {
    // Host mirror of the device ramp for T_target(); no device work.
    if (ctx.at(ATV::md::Phase::initial_integrate))
        T_live_ =
            ATV::md::ramp_target(T_target_, T_stop_, ctx.step, nfirst_, nlast_);
}

// Five launches per stage (two single-thread scalar kernels between three
// grid passes), every one reading the step from the device: capturable.
void NoseHooverNPT::launch(const LaunchContext& ctx) {
    auto& s = *state_;
    auto& f = ctx.frame;
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    const int n = f.n_atoms(), blocks = std::max(1, grid_size(n));
    auto p = s.parameters;
    p.first = nfirst_;
    p.last = nlast_;
    p.fault = f.force_fault;
    NptHistory* history = thrust::raw_pointer_cast(s.history.data());
    double* ke = thrust::raw_pointer_cast(s.ke.data());
    const long long* step_ptr = ctx.runtime.device_step;
    const long long step = static_cast<long long>(ctx.step);
    const auto& a = f.atoms;
    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        initial_chains<<<1, 1, 0, stream>>>(history, p, step_ptr, step); // P1-2
        scale_reduce<true, true><<<blocks, kDefaultBlock, 0, stream>>>(
            a.vx, a.vy, a.vz, a.mass, history, ke, n, f.force_fault);
        initial_barostat<<<1, 1, 0, stream>>>(history, p, ke, f.scalars, f.box,
                                              ctx.runtime.host_box); // P3-4,6
        scale_kick<true><<<blocks, kDefaultBlock, 0, stream>>>(
            a.vx, a.vy, a.vz, a.fx, a.fy, a.fz, a.mass, history, f.force_fault,
            a.tag, step_ptr, step, mixed_t(0.5 * dt_), mixed_t(p.ftm2v),
            n); // P4-5
        propagate_positions<<<blocks, kDefaultBlock, 0, stream>>>(
            a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, a.vx, a.vy, a.vz,
            a.image, history, f.box, mixed_t(dt_), n, f.force_fault); // P6
    } else if (ctx.at(ATV::md::Phase::final_integrate)) {
        scale_kick<false><<<blocks, kDefaultBlock, 0, stream>>>(
            a.vx, a.vy, a.vz, a.fx, a.fy, a.fz, a.mass, history, f.force_fault,
            a.tag, step_ptr, step, mixed_t(0.5 * dt_), mixed_t(p.ftm2v),
            n); // P7
    } else if (ctx.at(ATV::md::Phase::end_of_step)) {
        scale_reduce<false, true><<<blocks, kDefaultBlock, 0, stream>>>(
            a.vx, a.vy, a.vz, a.mass, history, ke, n, f.force_fault);
        final_barostat<<<1, 1, 0, stream>>>(history, p, ke); // P8
        scale_reduce<true, true><<<blocks, kDefaultBlock, 0, stream>>>(
            a.vx, a.vy, a.vz, a.mass, history, ke, n, f.force_fault);
        final_chains<<<1, 1, 0, stream>>>(history, p, ke, f.scalars, f.box,
                                          step_ptr, step); // P9-11
        scale_reduce<true, false><<<blocks, kDefaultBlock, 0, stream>>>(
            a.vx, a.vy, a.vz, a.mass, history, ke, n, f.force_fault); // P10
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

void NoseHooverNPT::destroy(const DestroyContext&) { state_.reset(); }
} // namespace ATV::cuda::md
