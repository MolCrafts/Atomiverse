/// @file src/cpu/dynamics/gear_pc.cpp
/// @brief Fifth-order Gear predictor–corrector (see the header).
#include "atomiverse/cpu/dynamics/gear_pc.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/gear_pc.h"
#include "atomiverse/core/timer.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/driver/step_timing.h"

namespace ATV {
namespace cpu {
namespace md {

namespace {

constexpr int kVars = 6;    // x y z vx vy vz
constexpr int kHistory = 5; // z1..z5 per variable

// Rows each thread must get before a team is worth opening: predicting or
// correcting a row is a few dozen flops, so a small system (the paper's 512
// ions) runs on the calling thread instead of paying a fork and join per
// variable (cf. VelocityVerlet).
constexpr int kRowsPerThread = 4096;

int team_for(const cpu::LaunchContext& ctx, int rows) {
    return std::max(1, std::min(ctx.runtime.n_threads, rows / kRowsPerThread));
}

// The thermostats run inside gear_pc's launches (the Gauss multiplier and
// projection; T_C1 and its gradient for the configurational one); scoped
// rows let a timing report split them from the integrator.
constexpr const char* kGaussScope = "gear_pc/gauss";
constexpr const char* kCurvatureScope = "gear_pc/curvature";
constexpr int kTimerCategory = static_cast<int>(StepTiming::Modify);

} // namespace

GearPC::GearPC(double dt_fs) : dt_(dt_fs) {
    if (!(dt_ > 0.0) || !std::isfinite(dt_))
        throw std::invalid_argument("GearPC: dt must be finite and > 0");
}

GearPC::GearPC(double dt_fs, GaussThermostat thermostat) : GearPC(dt_fs) {
    if (!(thermostat.temperature > 0.0)
        || !std::isfinite(thermostat.temperature))
        throw std::invalid_argument(
            "GearPC: Gaussian thermostat temperature must be finite and > 0");
    constraint_ = internal::GaussConstraint(thermostat.kind, thermostat.axis);
    gauss_ = thermostat;
}

GearPC::GearPC(double dt_fs, ConfigurationalThermostat thermostat)
    : GearPC(dt_fs) {
    if (!(thermostat.temperature > 0.0)
        || !std::isfinite(thermostat.temperature))
        throw std::invalid_argument("GearPC: configurational thermostat "
                                    "temperature must be finite and > 0");
    if (!(thermostat.Q > 0.0) || !std::isfinite(thermostat.Q))
        throw std::invalid_argument(
            "GearPC: configurational thermostat Q must be finite and > 0");
    configurational_ = thermostat;
}

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

void GearPC::build(const cpu::BuildContext& ctx) {
    if (multi_rank_dd(ctx.runtime))
        throw std::runtime_error(
            "GearPC: spatial decomposition is not supported (the Nordsieck "
            "history and the Gaussian sums need every atom on the rank)");
    const BigInt n = ctx.frame.nlocal();
    if (static_cast<BigInt>(ctx.runtime.md.n_dof_global)
        != ATV::md::degrees_of_freedom(n))
        throw std::runtime_error(
            "GearPC: holonomic constraints are not supported (the run "
            "publishes "
            + std::to_string(ctx.runtime.md.n_dof_global)
            + " degrees of freedom, 3N - 3 = "
            + std::to_string(ATV::md::degrees_of_freedom(n)) + ")");
    started_ = false;
    drift_ = 0.0;
    alpha_ = 0.0;
}

void GearPC::load_state_(const cpu::LaunchContext& ctx) {
    const Frame& f = ctx.frame;
    const StorageReal* cols[kVars] = {
        f.span(kAtom_x).data(),  f.span(kAtom_y).data(),
        f.span(kAtom_z).data(),  f.span(kAtom_vx).data(),
        f.span(kAtom_vy).data(), f.span(kAtom_vz).data()};
    for (int var = 0; var < kVars; ++var)
        for (int i = 0; i < n_; ++i)
            state_[var * n_ + i] = cols[var][i];
    const StorageReal* mass = f.span(kAtom_mass).data();
    for (int i = 0; i < n_; ++i)
        mass_[i] = mass[i];
}

// Right-hand side at the state in `state_` and the frame's forces:
// ṙ = v + (ξ/T0) ∂T_C1/∂r, v̇ = F/m − α w, ξ̇ = −Q (T_C1 − T0)/T0.
void GearPC::rates_(const cpu::LaunchContext& ctx) {
    const Frame& f = ctx.frame;
    const AccReal* force[3] = {f.span(kAtom_fx).data(), f.span(kAtom_fy).data(),
                               f.span(kAtom_fz).data()};
    const double ftm2v = ctx.runtime.units.ftm2v;
    for (int d = 0; d < 3; ++d)
        for (int i = 0; i < n_; ++i) {
            rate_[d * n_ + i] = state_[(3 + d) * n_ + i];
            rate_[(3 + d) * n_ + i] = ftm2v * force[d][i] / mass_[i];
        }
    if (configurational_) {
        ATV_PROFILE_SCOPE(ctx, kCurvatureScope, ctx.stage, kTimerCategory);
        const ConfigurationalSample sample = assembly_.evaluate(ctx);
        if (!(sample.laplacian > 0.0))
            throw std::runtime_error(
                "GearPC: the configurational thermostat needs a positive "
                "Laplacian of the potential energy, got "
                + std::to_string(sample.laplacian));
        t_c1_ = sample.temperature;
        const double t0 = configurational_->temperature;
        assembly_.temperature_gradient(ctx.runtime.units.boltz, grad_t_);
        const double gain = xi_[0] / t0;
        for (int k = 0; k < 3 * n_; ++k)
            rate_[k] += gain * grad_t_[k];
        xi_rate_ = -configurational_->Q * (t_c1_ - t0) / t0;
    }
    if (!gauss_)
        return;
    ATV_PROFILE_SCOPE(ctx, kGaussScope, ctx.stage, kTimerCategory);
    const internal::ConstRows3 v = {
        state_.data() + 3 * n_, state_.data() + 4 * n_, state_.data() + 5 * n_};
    const internal::Rows3 w = {w_.data(), w_.data() + n_, w_.data() + 2 * n_};
    const internal::ConstRows3 wc = {w[0], w[1], w[2]};
    const internal::ConstRows3 a = {
        rate_.data() + 3 * n_, rate_.data() + 4 * n_, rate_.data() + 5 * n_};
    constraint_.peculiar(n_, mass_.data(), f.span(kAtom_type).data(), v, w);
    alpha_ = internal::GaussConstraint::multiplier(n_, mass_.data(), wc, a);
    for (int d = 0; d < 3; ++d)
        for (int i = 0; i < n_; ++i)
            rate_[(3 + d) * n_ + i] -= alpha_ * w[d][i];
}

// Scale the peculiar velocities onto Σ m w² = k0_twice_ (state_ and frame).
void GearPC::project_(const cpu::LaunchContext& ctx) {
    ATV_PROFILE_SCOPE(ctx, kGaussScope, ctx.stage, kTimerCategory);
    Frame& f = ctx.frame;
    const internal::ConstRows3 v = {
        state_.data() + 3 * n_, state_.data() + 4 * n_, state_.data() + 5 * n_};
    const internal::Rows3 w = {w_.data(), w_.data() + n_, w_.data() + 2 * n_};
    constraint_.peculiar(n_, mass_.data(), f.span(kAtom_type).data(), v, w);
    const double twice_k = internal::GaussConstraint::twice_kinetic(
        n_, mass_.data(), {w[0], w[1], w[2]});
    if (!(twice_k > 0.0))
        throw std::runtime_error(
            "GearPC: the constrained kinetic energy is zero; give the atoms "
            "peculiar velocities before a Gaussian-thermostatted run");
    drift_ = std::fabs(twice_k / k0_twice_ - 1.0);
    const double scale = std::sqrt(k0_twice_ / twice_k);
    StorageReal* vel[3] = {f.span(kAtom_vx).data(), f.span(kAtom_vy).data(),
                           f.span(kAtom_vz).data()};
    for (int d = 0; d < 3; ++d)
        for (int i = 0; i < n_; ++i) {
            double& vi = state_[(3 + d) * n_ + i];
            vi += (scale - 1.0) * w[d][i];
            vel[d][i] = static_cast<StorageReal>(vi);
        }
}

void GearPC::start_(const cpu::LaunchContext& ctx) {
    const Frame& f = ctx.frame;
    n_ = owned_count(ctx);
    const auto tags = f.span(kAtom_tag);
    row_tag_.assign(tags.data(), tags.data() + n_);
    hist_store_.assign(static_cast<std::size_t>(kVars * kHistory) * n_, 0.0);
    state_.assign(static_cast<std::size_t>(kVars) * n_, 0.0);
    rate_.assign(static_cast<std::size_t>(kVars) * n_, 0.0);
    mass_.assign(static_cast<std::size_t>(n_), 0.0);
    w_.assign(static_cast<std::size_t>(3) * n_, 0.0);
    load_state_(ctx);
    if (gauss_) {
        constraint_.bind(n_, f.span(kAtom_type).data());
        const long long dof = constraint_.dof();
        if (dof <= 0)
            throw std::runtime_error(
                "GearPC: the Gaussian thermostat constrains no degree of "
                "freedom");
        k0_twice_ = static_cast<double>(dof) * ctx.runtime.units.boltz
                    * gauss_->temperature / ctx.runtime.units.mvv2e;
        project_(ctx);
    }
    for (double& z : xi_)
        z = 0.0;
    rates_(ctx);
    // z1 = h ẏ; z2 = h²/2 r̈ for positions, r̈ = v̇ + (ξ̇/T0) ∂T_C1/∂r at
    // ξ = 0; higher ones zero.
    for (int var = 0; var < kVars; ++var)
        for (int i = 0; i < n_; ++i)
            hist_(var, 1)[i] = dt_ * rate_[var * n_ + i];
    for (int d = 0; d < 3; ++d)
        for (int i = 0; i < n_; ++i)
            hist_(d, 2)[i] = 0.5 * dt_ * hist_(3 + d, 1)[i];
    if (configurational_) {
        xi_[1] = dt_ * xi_rate_;
        const double kick = 0.5 * dt_ * xi_[1] / configurational_->temperature;
        for (int d = 0; d < 3; ++d)
            for (int i = 0; i < n_; ++i)
                hist_(d, 2)[i] += kick * grad_t_[d * n_ + i];
    }
    started_ = true;
}

// Rows follow the atoms by tag: a spatial sort between two launches permutes
// the frame, and the history must permute with it.
void GearPC::follow_rows_(const cpu::LaunchContext& ctx) {
    const int n = owned_count(ctx);
    if (n != n_)
        throw std::runtime_error("GearPC: the number of atoms changed from "
                                 + std::to_string(n_) + " to "
                                 + std::to_string(n) + " during the run");
    const auto tags = ctx.frame.span(kAtom_tag);
    bool same = true;
    for (int i = 0; i < n_ && same; ++i)
        same = tags[i] == row_tag_[i];
    if (same)
        return;
    std::unordered_map<TagInt, int> old_row;
    old_row.reserve(static_cast<std::size_t>(n_));
    for (int i = 0; i < n_; ++i)
        old_row.emplace(row_tag_[i], i);
    std::vector<double> moved(hist_store_.size());
    for (int i = 0; i < n_; ++i) {
        const auto it = old_row.find(tags[i]);
        if (it == old_row.end())
            throw std::runtime_error("GearPC: atom tag "
                                     + std::to_string(tags[i])
                                     + " has no integration history");
        for (int slot = 0; slot < kVars * kHistory; ++slot)
            moved[slot * n_ + i] = hist_store_[slot * n_ + it->second];
    }
    hist_store_.swap(moved);
    row_tag_.assign(tags.data(), tags.data() + n_);
}

void GearPC::predict_(const cpu::LaunchContext& ctx) {
    load_state_(ctx);
    if (configurational_)
        ATV::md::gear_predict(xi_);
    Frame& f = ctx.frame;
    StorageReal* cols[kVars] = {
        f.span(kAtom_x).data(),  f.span(kAtom_y).data(),
        f.span(kAtom_z).data(),  f.span(kAtom_vx).data(),
        f.span(kAtom_vy).data(), f.span(kAtom_vz).data()};
    const int n = n_;
    [[maybe_unused]] const int team = team_for(ctx, n);
    // One team for all six variables; the rows are independent, so no
    // barrier between them.
#pragma omp parallel num_threads(team) if (team > 1)
    for (int var = 0; var < kVars; ++var) {
        double* h1 = hist_(var, 1);
        double* h2 = hist_(var, 2);
        double* h3 = hist_(var, 3);
        double* h4 = hist_(var, 4);
        double* h5 = hist_(var, 5);
        StorageReal* col = cols[var];
        const double* y = state_.data() + var * n;
#pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i) {
            double z[ATV::md::kGearValues] = {y[i],  h1[i], h2[i],
                                              h3[i], h4[i], h5[i]};
            ATV::md::gear_predict(z);
            col[i] = static_cast<StorageReal>(z[0]);
            h1[i] = z[1];
            h2[i] = z[2];
            h3[i] = z[3];
            h4[i] = z[4];
            h5[i] = z[5];
        }
    }
}

void GearPC::correct_(const cpu::LaunchContext& ctx) {
    load_state_(ctx);
    rates_(ctx);
    Frame& f = ctx.frame;
    StorageReal* cols[kVars] = {
        f.span(kAtom_x).data(),  f.span(kAtom_y).data(),
        f.span(kAtom_z).data(),  f.span(kAtom_vx).data(),
        f.span(kAtom_vy).data(), f.span(kAtom_vz).data()};
    const double h = dt_;
    const int n = n_;
    [[maybe_unused]] const int team = team_for(ctx, n);
#pragma omp parallel num_threads(team) if (team > 1)
    for (int var = 0; var < kVars; ++var) {
        double* h1 = hist_(var, 1);
        double* h2 = hist_(var, 2);
        double* h3 = hist_(var, 3);
        double* h4 = hist_(var, 4);
        double* h5 = hist_(var, 5);
        StorageReal* col = cols[var];
        double* y = state_.data() + var * n;
        const double* rate = rate_.data() + var * n;
#pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i) {
            double z[ATV::md::kGearValues] = {y[i],  h1[i], h2[i],
                                              h3[i], h4[i], h5[i]};
            ATV::md::gear_correct(z, h * rate[i]);
            y[i] = z[0];
            col[i] = static_cast<StorageReal>(z[0]);
            h1[i] = z[1];
            h2[i] = z[2];
            h3[i] = z[3];
            h4[i] = z[4];
            h5[i] = z[5];
        }
    }
    if (configurational_)
        ATV::md::gear_correct(xi_, h * xi_rate_);
    if (gauss_)
        project_(ctx);
}

void GearPC::launch(const cpu::LaunchContext& ctx) {
    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        if (!started_)
            start_(ctx);
        else
            follow_rows_(ctx);
        predict_(ctx);
    } else if (ctx.at(ATV::md::Phase::final_integrate)) {
        if (!started_)
            throw std::logic_error(
                "GearPC: final_integrate before any initial_integrate");
        follow_rows_(ctx);
        correct_(ctx);
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
