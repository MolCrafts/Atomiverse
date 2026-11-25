/// @file cuda/dynamics/gear_pc.cu
/// @brief Fifth-order Gear predictor–corrector on the device (see the
///        header). Mirrors `src/cpu/dynamics/gear_pc.cpp` and
///        `src/cpu/dynamics/gauss_constraint.cpp`.
#include "atomiverse/cuda/dynamics/gear_pc.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/gear_pc.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {
namespace md {

namespace {

constexpr int kVars = 6;    // x y z vx vy vz
constexpr int kHistory = 5; // z1..z5 per variable
/// Species a flow-free axis can tell apart (one mean velocity each).
constexpr int kMaxSpecies = 8;
/// Block size of the kernels that reduce the Gaussian sums.
constexpr int kGearBlock = 128;

/// How one Cartesian axis enters the constrained kinetic energy
/// (`cpu::md::internal::GaussAxis`).
enum GaussAxis : int {
    kBarycentric = 0, ///< w = v − (system mean velocity)
    kSpecies = 1,     ///< w = v − (mean velocity of the atom's species)
    kFree = 2,        ///< axis left out of the constraint (w = 0)
};

/// Thermostat faults, reported when the run returns.
enum GearFault : int {
    kZeroKinetic = 1,  ///< Σ m w² ≤ 0
    kBadLaplacian = 2, ///< Σ ∇²U ≤ 0
};

/// Build-time constants of the Gaussian constraint (kernel argument).
struct GaussSetup {
    int mode[3] = {kBarycentric, kBarycentric, kBarycentric};
    int species_axis = -1;                ///< The `kSpecies` axis, or -1.
    int n_species = 0;                    ///< Groups of that axis.
    const int* species_of_type = nullptr; ///< Device table, by frame type.
    double k0_twice = 0.0;                ///< Target Σ m w² [mass·velocity²].
    bool active = false;
};

/// Device scalars of one integrator: the Gaussian sums and what the
/// finalize kernel derives from them, and the ξ Nordsieck vector.
struct GearScalars {
    // Block sums; `gauss_finalize` consumes and zeroes them.
    double mass;         ///< Σ m
    double momentum[3];  ///< Σ m v_d
    double work[3];      ///< Σ m v_d a_d
    double twice_kin[3]; ///< Σ m v_d²
    double push[3];      ///< Σ m a_d
    double species_mass[kMaxSpecies];
    double species_momentum[kMaxSpecies]; ///< On the species axis.
    double species_push[kMaxSpecies];
    // Derived.
    double mean[3];                   ///< Barycentric mean velocity.
    double species_mean[kMaxSpecies]; ///< Per species, on its axis.
    double alpha;                     ///< Gauss multiplier [1/time].
    double scale;                     ///< Projection scale of w.
    double drift;                     ///< |Σ m w² / K0 − 1| before projecting.
    // Configurational thermostat.
    double xi[ATV::md::kGearValues]; ///< Nordsieck vector of ξ.
    double xi_rate;                  ///< ξ̇ of the last evaluation.
    double gain;                     ///< ξ / T0 of the last evaluation.
    double kick;                     ///< Start-up h²/2 · ξ̇/T0.
    double t_c1;                     ///< T_C1 of the last evaluation [K].
    double laplacian;                ///< Σ ∇²U of the last evaluation.
    int status;                      ///< `GearFault` bits.
};

/// Frame columns the integration reads and writes (kernel argument).
struct GearColumns {
    pos_t *x, *y, *z, *xc, *yc, *zc;
    mixed_t *vx, *vy, *vz;
    const fixed_t *fx, *fy, *fz;
    const mixed_t* mass;
    const int* type;
    ImageInt* image;
    ForceFault* fault;
    const std::int64_t* tag;
    const long long* step_ptr;
    long long step;
};

__device__ __forceinline__ double* hist_of(double* hist, int stride, int var,
                                           int k) {
    return hist + static_cast<std::size_t>(kHistory * var + (k - 1)) * stride;
}

__device__ __forceinline__ int species_of(const GaussSetup& g, const int* type,
                                          int i) {
    return g.species_of_type ? g.species_of_type[type[i]] : 0;
}

/// Peculiar velocity of one component.
__device__ __forceinline__ double peculiar(const GaussSetup& g,
                                           const GearScalars* s, int d,
                                           int species, double v) {
    if (g.mode[d] == kFree)
        return 0.0;
    return v - (g.mode[d] == kSpecies ? s->species_mean[species] : s->mean[d]);
}

/// Block sum of N doubles; valid in thread 0. Every thread of the block
/// must call it; the storage is shared by successive calls.
template <int N>
__device__ ::cuda::std::array<double, N>
block_sum(const ::cuda::std::array<double, N>& values) {
    using Reduce = cub::BlockReduce<::cuda::std::array<double, N>, kGearBlock>;
    __shared__ typename Reduce::TempStorage storage;
    const auto out = Reduce(storage).Reduce(values, SumComponents<double, N>{});
    __syncthreads();
    return out;
}

/// Add one atom (zeros for an idle lane) to the block's Gaussian sums and
/// the block's sums to the device accumulators.
__device__ void gauss_tally(const GaussSetup& g, GearScalars* s, double m,
                            const double* v, const double* ma, int species) {
    ::cuda::std::array<double, 13> values{};
    values[0] = m;
    for (int d = 0; d < 3; ++d) {
        values[1 + d] = m * v[d];
        values[4 + d] = v[d] * ma[d];
        values[7 + d] = m * v[d] * v[d];
        values[10 + d] = ma[d];
    }
    const auto sum = block_sum<13>(values);
    if (threadIdx.x == 0) {
        atomicAdd(&s->mass, sum[0]);
        for (int d = 0; d < 3; ++d) {
            atomicAdd(&s->momentum[d], sum[1 + d]);
            atomicAdd(&s->work[d], sum[4 + d]);
            atomicAdd(&s->twice_kin[d], sum[7 + d]);
            atomicAdd(&s->push[d], sum[10 + d]);
        }
    }
    const int axis = g.species_axis;
    if (axis < 0)
        return;
    for (int group = 0; group < g.n_species; ++group) {
        const double mine = species == group ? 1.0 : 0.0;
        const auto part =
            block_sum<3>({mine * m, mine * m * v[axis], mine * ma[axis]});
        if (threadIdx.x == 0) {
            atomicAdd(&s->species_mass[group], part[0]);
            atomicAdd(&s->species_momentum[group], part[1]);
            atomicAdd(&s->species_push[group], part[2]);
        }
    }
}

/// Gaussian sums of the frame velocities, with the unconstrained
/// accelerations a = ftm2v F/m when `with_force`.
__global__ void gauss_measure(GearColumns c, GaussSetup g, GearScalars* s,
                              double ftm2v, bool with_force, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double m = 0.0, v[3] = {}, ma[3] = {};
    int species = 0;
    if (i < n) {
        m = static_cast<double>(c.mass[i]);
        v[0] = static_cast<double>(c.vx[i]);
        v[1] = static_cast<double>(c.vy[i]);
        v[2] = static_cast<double>(c.vz[i]);
        if (with_force) {
            ma[0] = ftm2v * decode_force<double>(c.fx[i]);
            ma[1] = ftm2v * decode_force<double>(c.fy[i]);
            ma[2] = ftm2v * decode_force<double>(c.fz[i]);
        }
        species = species_of(g, c.type, i);
    }
    gauss_tally(g, s, m, v, ma, species);
}

/// One thread: the group mean velocities, then either the multiplier
/// α = Σ m w·a / Σ m w² or the projection scale √(K0 / Σ m w²), from
/// Σ m w·a = Σ m v·a − Σ_g V_g·(Σ_g m a) and Σ m w² = Σ m v² − Σ_g V_g·P_g.
__global__ void gauss_finalize(GaussSetup g, GearScalars* s,
                               const ForceFault* fault, bool projection) {
    double num = 0.0, den = 0.0;
    for (int d = 0; d < 3; ++d) {
        if (g.mode[d] == kFree)
            continue;
        num += s->work[d];
        den += s->twice_kin[d];
        if (g.mode[d] == kBarycentric) {
            const double mean = s->momentum[d] / s->mass;
            s->mean[d] = mean;
            num -= mean * s->push[d];
            den -= mean * s->momentum[d];
            continue;
        }
        for (int group = 0; group < g.n_species; ++group) {
            const double mean =
                s->species_momentum[group] / s->species_mass[group];
            s->species_mean[group] = mean;
            num -= mean * s->species_push[group];
            den -= mean * s->species_momentum[group];
        }
    }
    s->mass = 0.0;
    for (int d = 0; d < 3; ++d)
        s->momentum[d] = s->work[d] = s->twice_kin[d] = s->push[d] = 0.0;
    for (int group = 0; group < kMaxSpecies; ++group)
        s->species_mass[group] = s->species_momentum[group] =
            s->species_push[group] = 0.0;
    // A frozen state measured nothing: keep the fault that froze it.
    if (state_frozen(fault) || s->status != 0)
        return;
    if (!(den > 0.0)) {
        s->status |= kZeroKinetic;
        return;
    }
    if (projection) {
        s->drift = fabs(den / g.k0_twice - 1.0);
        s->scale = sqrt(g.k0_twice / den);
    } else {
        s->alpha = num / den;
    }
}

/// Scale the peculiar velocities onto Σ m w² = K0.
__global__ void gauss_project(GearColumns c, GaussSetup g, const GearScalars* s,
                              int n) {
    if (state_frozen(c.fault) || s->status != 0)
        return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const int species = species_of(g, c.type, i);
    const double gain = s->scale - 1.0;
    mixed_t* const vel[3] = {c.vx, c.vy, c.vz};
    for (int d = 0; d < 3; ++d) {
        const double v = static_cast<double>(vel[d][i]);
        vel[d][i] =
            static_cast<mixed_t>(v + gain * peculiar(g, s, d, species, v));
    }
}

/// One thread: T_C1 = N/(k_B D), ξ̇ = −Q (T_C1 − T0)/T0 and the gain ξ/T0
/// of the position rates; at the start, ξ's first Nordsieck value and the
/// position kick h²/2 · ξ̇/T0.
__global__ void configurational_rate(const double* __restrict__ sums,
                                     GearScalars* s, const ForceFault* fault,
                                     double kB, double t0, double q, double dt,
                                     bool start) {
    if (state_frozen(fault) || s->status != 0)
        return;
    const double force_sq = sums[0], laplacian = sums[1];
    s->laplacian = laplacian;
    if (!(laplacian > 0.0)) {
        s->status |= kBadLaplacian;
        return;
    }
    s->t_c1 = force_sq / (kB * laplacian);
    s->xi_rate = -q * (s->t_c1 - t0) / t0;
    if (start) {
        s->xi[1] = dt * s->xi_rate;
        s->gain = 0.0;
        s->kick = 0.5 * dt * s->xi[1] / t0;
    } else {
        s->gain = s->xi[0] / t0;
    }
}

/// ∂T_C1/∂r of one atom: −(2 (H F) D + N ∇D) / (k_B D²).
__device__ __forceinline__ void temperature_gradient(const CurvatureRows& rows,
                                                     const double* sums,
                                                     double kB, int i,
                                                     double* out) {
    const double force_sq = sums[0], laplacian = sums[1];
    const double scale = -1.0 / (kB * laplacian * laplacian);
    out[0] = scale * (2.0 * rows.hx[i] * laplacian + force_sq * rows.gx[i]);
    out[1] = scale * (2.0 * rows.hy[i] * laplacian + force_sq * rows.gy[i]);
    out[2] = scale * (2.0 * rows.hz[i] * laplacian + force_sq * rows.gz[i]);
}

/// The right-hand side of one atom at the frame's state: ṙ = v + gain·∂T,
/// v̇ = ftm2v F/m − α w. `grad` is ∂T_C1/∂r (zeros without the
/// configurational thermostat).
__device__ __forceinline__ void rates_of(const GearColumns& c,
                                         const GaussSetup& g,
                                         const GearScalars* s, double ftm2v,
                                         const double* grad, int i,
                                         const double* v, double* rate) {
    const double inv_m = ftm2v / static_cast<double>(c.mass[i]);
    const double f[3] = {decode_force<double>(c.fx[i]),
                         decode_force<double>(c.fy[i]),
                         decode_force<double>(c.fz[i])};
    const int species = g.active ? species_of(g, c.type, i) : 0;
    for (int d = 0; d < 3; ++d) {
        rate[d] = v[d] + s->gain * grad[d];
        rate[3 + d] = inv_m * f[d];
        if (g.active)
            rate[3 + d] -= s->alpha * peculiar(g, s, d, species, v[d]);
    }
}

__device__ __forceinline__ void load_state(const GearColumns& c, int i,
                                           double* y) {
    y[0] = static_cast<double>(load_position(c.x, c.xc, i));
    y[1] = static_cast<double>(load_position(c.y, c.yc, i));
    y[2] = static_cast<double>(load_position(c.z, c.zc, i));
    y[3] = static_cast<double>(c.vx[i]);
    y[4] = static_cast<double>(c.vy[i]);
    y[5] = static_cast<double>(c.vz[i]);
}

/// Write positions (wrapped into the cell, image flags counted) and
/// velocities back to the frame.
__device__ __forceinline__ void
store_state(const GearColumns& c, const Box& box, int i, const double* y) {
    mixed_t px = static_cast<mixed_t>(y[0]), py = static_cast<mixed_t>(y[1]),
            pz = static_cast<mixed_t>(y[2]);
    box.wrap_position(px, py, pz, c.image ? c.image + 3 * i : nullptr);
    store_position(c.x, c.xc, i, px);
    store_position(c.y, c.yc, i, py);
    store_position(c.z, c.zc, i, pz);
    c.vx[i] = static_cast<mixed_t>(y[3]);
    c.vy[i] = static_cast<mixed_t>(y[4]);
    c.vz[i] = static_cast<mixed_t>(y[5]);
}

/// Nordsieck start: z1 = h ẏ; z2 = h²/2 r̈ for the positions, with
/// r̈ = v̇ + (ξ̇/T0) ∂T_C1/∂r at ξ = 0; the higher ones zero.
__global__ void gear_start(GearColumns c, GaussSetup g, const GearScalars* s,
                           CurvatureRows rows, const double* sums, double kB,
                           double* hist, int stride, double dt, double ftm2v,
                           bool configurational, int n) {
    if (state_frozen(c.fault) || s->status != 0)
        return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double y[kVars], rate[kVars], grad[3] = {};
    load_state(c, i, y);
    if (configurational)
        temperature_gradient(rows, sums, kB, i, grad);
    rates_of(c, g, s, ftm2v, grad, i, y + 3, rate);
    for (int var = 0; var < kVars; ++var) {
        hist_of(hist, stride, var, 1)[i] = dt * rate[var];
        for (int k = 2; k <= kHistory; ++k)
            hist_of(hist, stride, var, k)[i] = 0.0;
    }
    for (int d = 0; d < 3; ++d)
        hist_of(hist, stride, d, 2)[i] =
            0.5 * dt * dt * rate[3 + d] + s->kick * grad[d];
}

/// Predictor: Taylor-advance every variable (and ξ) one step.
__global__ void gear_predict_step(GearColumns c, Box box_arg, GearScalars* s,
                                  double* hist, int stride,
                                  bool configurational, int n) {
    if (!forces_consumable(c.fault, c.tag, c.step_ptr, c.step)
        || s->status != 0)
        return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 && configurational)
        ATV::md::gear_predict(s->xi);
    if (i >= n)
        return;
    const Box box = box_arg.current();
    double y[kVars];
    load_state(c, i, y);
    for (int var = 0; var < kVars; ++var) {
        double z[ATV::md::kGearValues] = {y[var]};
        for (int k = 1; k <= kHistory; ++k)
            z[k] = hist_of(hist, stride, var, k)[i];
        ATV::md::gear_predict(z);
        y[var] = z[0];
        for (int k = 1; k <= kHistory; ++k)
            hist_of(hist, stride, var, k)[i] = z[k];
    }
    store_state(c, box, i, y);
}

/// Corrector: evaluate the rates at the predicted state and the new
/// forces, correct every variable (and ξ), and tally the Gaussian sums of
/// the corrected velocities for the projection.
__global__ void gear_correct_step(GearColumns c, Box box_arg, GaussSetup g,
                                  GearScalars* s, CurvatureRows rows,
                                  const double* sums, double kB, double* hist,
                                  int stride, double dt, double ftm2v,
                                  bool configurational, int n) {
    const bool live =
        forces_consumable(c.fault, c.tag, c.step_ptr, c.step) && s->status == 0;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double m = 0.0, v[3] = {};
    const double no_push[3] = {};
    int species = 0;
    if (live && i < n) {
        const Box box = box_arg.current();
        double y[kVars], rate[kVars], grad[3] = {};
        load_state(c, i, y);
        if (configurational)
            temperature_gradient(rows, sums, kB, i, grad);
        rates_of(c, g, s, ftm2v, grad, i, y + 3, rate);
        for (int var = 0; var < kVars; ++var) {
            double z[ATV::md::kGearValues] = {y[var]};
            for (int k = 1; k <= kHistory; ++k)
                z[k] = hist_of(hist, stride, var, k)[i];
            ATV::md::gear_correct(z, dt * rate[var]);
            y[var] = z[0];
            for (int k = 1; k <= kHistory; ++k)
                hist_of(hist, stride, var, k)[i] = z[k];
        }
        store_state(c, box, i, y);
        m = static_cast<double>(c.mass[i]);
        v[0] = y[3];
        v[1] = y[4];
        v[2] = y[5];
        species = g.active ? species_of(g, c.type, i) : 0;
    }
    // ξ̇ and the gain were fixed by `configurational_rate`; no other thread
    // reads the Nordsieck vector here.
    if (live && i == 0 && configurational)
        ATV::md::gear_correct(s->xi, dt * s->xi_rate);
    if (g.active)
        gauss_tally(g, s, m, v, no_push, species);
}

} // namespace

struct GearPC::State {
    double* hist = nullptr; ///< z1..z5 of x y z vx vy vz, 30 rows.
    int stride = 0;         ///< Rows per history column (frame capacity).
    GearScalars* scalars = nullptr;
    int* species_of_type = nullptr;
    GaussSetup gauss{};

    State() = default;
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    ~State() { release(); }
    void release() {
        cudaFree(hist);
        cudaFree(scalars);
        cudaFree(species_of_type);
        hist = nullptr;
        scalars = nullptr;
        species_of_type = nullptr;
        stride = 0;
        gauss = GaussSetup{};
    }
};

namespace {

GearColumns columns_of(const LaunchContext& ctx) {
    const auto& a = ctx.frame.atoms;
    return {a.x,
            a.y,
            a.z,
            a.x_corr,
            a.y_corr,
            a.z_corr,
            a.vx,
            a.vy,
            a.vz,
            a.fx,
            a.fy,
            a.fz,
            a.mass,
            a.type,
            a.image,
            ctx.frame.force_fault,
            a.tag,
            ctx.runtime.device_step,
            static_cast<long long>(ctx.step)};
}

} // namespace

GearPC::GearPC(double dt) : dt_(dt), state_(std::make_unique<State>()) {
    if (!(dt_ > 0.0) || !std::isfinite(dt_))
        throw std::invalid_argument("GearPC: dt must be finite and > 0");
}

GearPC::GearPC(double dt, cpu::md::GaussThermostat thermostat) : GearPC(dt) {
    if (!(thermostat.temperature > 0.0)
        || !std::isfinite(thermostat.temperature))
        throw std::invalid_argument(
            "GearPC: Gaussian thermostat temperature must be finite and > 0");
    if (thermostat.axis < 0 || thermostat.axis > 2)
        throw std::invalid_argument(
            "GaussThermostat: axis must be 0, 1 or 2, got "
            + std::to_string(thermostat.axis));
    gauss_ = thermostat;
}

GearPC::GearPC(double dt, cpu::md::ConfigurationalThermostat thermostat)
    : GearPC(dt) {
    if (!(thermostat.temperature > 0.0)
        || !std::isfinite(thermostat.temperature))
        throw std::invalid_argument("GearPC: configurational thermostat "
                                    "temperature must be finite and > 0");
    if (!(thermostat.Q > 0.0) || !std::isfinite(thermostat.Q))
        throw std::invalid_argument(
            "GearPC: configurational thermostat Q must be finite and > 0");
    configurational_ = thermostat;
}

GearPC::~GearPC() = default;
GearPC::GearPC(GearPC&&) noexcept = default;
GearPC& GearPC::operator=(GearPC&&) noexcept = default;

std::uint64_t GearPC::graph_epoch() const { return assembly_.epoch(); }

void GearPC::set_curvature_terms(std::vector<CurvatureSource*> sources,
                                 std::vector<const ExternalDrive*> drives) {
    assembly_.set_terms(std::move(sources), std::move(drives));
}

void GearPC::invalidate_inputs(InputChange changes, const BuildContext&) {
    if (has_input_change(changes, InputChange::Geometry)
        || has_input_change(changes, InputChange::Velocity)
        || has_input_change(changes, InputChange::Cell))
        started_ = false;
}

void GearPC::build(const BuildContext& ctx) {
    auto& fr = ctx.frame;
    if (fr.n_ghost() != 0)
        throw std::runtime_error(
            "GearPC: needs a single GPU (the Nordsieck history and the "
            "Gaussian sums need every atom on the device)");
    const BigInt n = fr.n_local();
    if (static_cast<BigInt>(ctx.runtime.md.n_dof_global)
        != ATV::md::degrees_of_freedom(n))
        throw std::runtime_error(
            "GearPC: holonomic constraints are not supported (the run "
            "publishes "
            + std::to_string(ctx.runtime.md.n_dof_global)
            + " degrees of freedom, 3N - 3 = "
            + std::to_string(ATV::md::degrees_of_freedom(n)) + ")");
    State& st = *state_;
    st.release();
    st.stride = fr.capacity();
    const std::size_t rows = static_cast<std::size_t>(kVars * kHistory)
                             * static_cast<std::size_t>(st.stride);
    if (rows > 0) {
        ATV_CUDA_CHECK(cudaMalloc(&st.hist, rows * sizeof(double)));
        ATV_CUDA_CHECK(cudaMemset(st.hist, 0, rows * sizeof(double)));
    }
    ATV_CUDA_CHECK(cudaMalloc(&st.scalars, sizeof(GearScalars)));
    ATV_CUDA_CHECK(cudaMemset(st.scalars, 0, sizeof(GearScalars)));
    if (configurational_)
        assembly_.build(ctx);
    started_ = false;
    drift_ = alpha_ = xi_ = t_c1_ = 0.0;
    if (!gauss_)
        return;

    // Species = the distinct frame types, numbered by first appearance
    // (`cpu::md::internal::GaussConstraint::bind`).
    if (n > 0 && !fr.atoms.type)
        throw std::invalid_argument(
            std::string("GearPC: the Gaussian thermostat needs the "
                        "force-field type column; ")
            + atom_field_how_to("type"));
    std::vector<int> types(static_cast<std::size_t>(n));
    if (n > 0)
        ATV_CUDA_CHECK(cudaMemcpy(types.data(), fr.atoms.type,
                                  types.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
    int t_max = -1;
    for (int t : types) {
        if (t < 0)
            throw std::invalid_argument(
                "GaussThermostat: atom types must be >= 0");
        t_max = std::max(t_max, t);
    }
    std::vector<int> species(static_cast<std::size_t>(t_max + 1), -1);
    int n_species = 0;
    for (int t : types)
        if (species[static_cast<std::size_t>(t)] < 0)
            species[static_cast<std::size_t>(t)] = n_species++;
    GaussSetup& g = st.gauss;
    g.active = true;
    long long dof = 3 * (static_cast<long long>(n) - 1);
    if (gauss_->kind == cpu::md::GaussKind::Flow) {
        if (n_species > kMaxSpecies)
            throw std::invalid_argument(
                "GearPC: the flow-free Gaussian thermostat supports at most "
                + std::to_string(kMaxSpecies) + " species on CUDA, got "
                + std::to_string(n_species));
        g.mode[gauss_->axis] = kSpecies;
        g.species_axis = gauss_->axis;
        g.n_species = n_species;
        dof += 1 - n_species;
        if (!species.empty()) {
            ATV_CUDA_CHECK(
                cudaMalloc(&st.species_of_type, species.size() * sizeof(int)));
            ATV_CUDA_CHECK(cudaMemcpy(st.species_of_type, species.data(),
                                      species.size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
        }
        g.species_of_type = st.species_of_type;
    } else if (gauss_->kind == cpu::md::GaussKind::Transverse) {
        g.mode[gauss_->axis] = kFree;
        dof -= static_cast<long long>(n) - 1;
    }
    if (dof <= 0)
        throw std::runtime_error(
            "GearPC: the Gaussian thermostat constrains no degree of freedom");
    g.k0_twice = static_cast<double>(dof)
                 * ctx.runtime.units.boltzmann_constant() * gauss_->temperature
                 / ctx.runtime.units.kinetic_energy.scale();
}

void GearPC::destroy(const DestroyContext&) {
    state_->release();
    assembly_.destroy();
    started_ = false;
}

// Gaussian sums of the frame velocities, then the multiplier or the
// projection scale.
void GearPC::measure_(const LaunchContext& ctx, bool with_force,
                      bool projection) {
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    const int n = ctx.frame.n_local();
    const State& st = *state_;
    gauss_measure<<<grid_size(n, kGearBlock), kGearBlock, 0, stream>>>(
        columns_of(ctx), st.gauss, st.scalars,
        ctx.runtime.units.velocity_impulse.scale(), with_force, n);
    gauss_finalize<<<1, 1, 0, stream>>>(st.gauss, st.scalars,
                                        ctx.frame.force_fault, projection);
}

void GearPC::project_(const LaunchContext& ctx) {
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    const int n = ctx.frame.n_local();
    const State& st = *state_;
    gauss_finalize<<<1, 1, 0, stream>>>(st.gauss, st.scalars,
                                        ctx.frame.force_fault, true);
    gauss_project<<<grid_size(n), kDefaultBlock, 0, stream>>>(
        columns_of(ctx), st.gauss, st.scalars, n);
}

void GearPC::throw_on_fault_(int status, double laplacian) const {
    if (status & kZeroKinetic)
        throw std::runtime_error(
            "GearPC: the constrained kinetic energy is zero; give the atoms "
            "peculiar velocities before a Gaussian-thermostatted run");
    if (status & kBadLaplacian)
        throw std::runtime_error(
            "GearPC: the configurational thermostat needs a positive "
            "Laplacian of the potential energy, got "
            + std::to_string(laplacian));
}

// Setup, outside the step loop: project onto the constraint, evaluate the
// rates at the initial state and write the history; then one read of the
// fault word.
void GearPC::start_(const LaunchContext& ctx) {
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    const int n = ctx.frame.n_local();
    const State& st = *state_;
    ATV_CUDA_CHECK(cudaMemsetAsync(st.scalars, 0, sizeof(GearScalars), stream));
    if (gauss_) {
        measure_(ctx, /*with_force=*/false, /*projection=*/true);
        gauss_project<<<grid_size(n), kDefaultBlock, 0, stream>>>(
            columns_of(ctx), st.gauss, st.scalars, n);
        measure_(ctx, /*with_force=*/true, /*projection=*/false);
    }
    if (configurational_) {
        assembly_.evaluate(ctx);
        configurational_rate<<<1, 1, 0, stream>>>(
            assembly_.sums(), st.scalars, ctx.frame.force_fault,
            ctx.runtime.units.boltzmann_constant(),
            configurational_->temperature, configurational_->Q, dt_,
            /*start=*/true);
    }
    gear_start<<<grid_size(n), kDefaultBlock, 0, stream>>>(
        columns_of(ctx), st.gauss, st.scalars, assembly_.rows(),
        assembly_.sums(), ctx.runtime.units.boltzmann_constant(), st.hist,
        st.stride, dt_, ctx.runtime.units.velocity_impulse.scale(),
        configurational_.has_value(), n);
    ATV_CUDA_CHECK(cudaGetLastError());
    GearScalars host{};
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    ATV_CUDA_CHECK(cudaMemcpy(&host, st.scalars, sizeof(GearScalars),
                              cudaMemcpyDeviceToHost));
    throw_on_fault_(host.status, host.laplacian);
    started_ = true;
}

void GearPC::initialize_forces(const LaunchContext& ctx) {
    if (!started_ && ctx.at(ATV::md::Phase::post_force))
        start_(ctx);
}

void GearPC::launch(const LaunchContext& ctx) {
    // The post_force binding only carries the start (initialize_forces).
    if (ctx.at(ATV::md::Phase::post_force))
        return;
    if (!started_)
        throw std::logic_error(
            "GearPC: integration before the initial force evaluation");
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    const int n = ctx.frame.n_local();
    const State& st = *state_;
    const bool configurational = configurational_.has_value();
    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        gear_predict_step<<<grid_size(n), kDefaultBlock, 0, stream>>>(
            columns_of(ctx), ctx.runtime.host_box, st.scalars, st.hist,
            st.stride, configurational, n);
        ATV_CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (gauss_)
        measure_(ctx, /*with_force=*/true, /*projection=*/false);
    if (configurational) {
        assembly_.evaluate(ctx);
        configurational_rate<<<1, 1, 0, stream>>>(
            assembly_.sums(), st.scalars, ctx.frame.force_fault,
            ctx.runtime.units.boltzmann_constant(),
            configurational_->temperature, configurational_->Q, dt_,
            /*start=*/false);
    }
    gear_correct_step<<<grid_size(n, kGearBlock), kGearBlock, 0, stream>>>(
        columns_of(ctx), ctx.runtime.host_box, st.gauss, st.scalars,
        assembly_.rows(), assembly_.sums(),
        ctx.runtime.units.boltzmann_constant(), st.hist, st.stride, dt_,
        ctx.runtime.units.velocity_impulse.scale(), configurational, n);
    if (gauss_)
        project_(ctx);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// The run has synchronized: read the diagnostics back (end-of-run only).
void GearPC::on_retrieve(const BuildContext&) {
    if (state_->scalars == nullptr)
        return;
    GearScalars host{};
    ATV_CUDA_CHECK(cudaMemcpy(&host, state_->scalars, sizeof(GearScalars),
                              cudaMemcpyDeviceToHost));
    drift_ = host.drift;
    alpha_ = host.alpha;
    xi_ = host.xi[0];
    t_c1_ = host.t_c1;
    throw_on_fault_(host.status, host.laplacian);
}

} // namespace md
} // namespace cuda
} // namespace ATV
