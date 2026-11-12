/// @file src/cpu/dynamics/gear_pc.cpp
/// @brief Fifth-order Gear predictor–corrector (see the header).
#include "atomiverse/cpu/dynamics/gear_pc.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/gear_pc.h"
#include "atomiverse/cpu/component/owned_range.h"

namespace ATV {
namespace cpu {
namespace md {

namespace {

constexpr int kVars = 6;    // x y z vx vy vz
constexpr int kHistory = 5; // z1..z5 per variable

} // namespace

GearPC::GearPC(double dt_fs) : dt_(dt_fs) {
    if (!(dt_ > 0.0) || !std::isfinite(dt_))
        throw std::invalid_argument("GearPC: dt must be finite and > 0");
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
// ṙ = v, v̇ = F/m.
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
    load_state_(ctx);
    rates_(ctx);
    // z1 = h ẏ; z2 = h²/2 r̈ = (h/2)(h v̇) for positions; higher ones zero.
    for (int var = 0; var < kVars; ++var)
        for (int i = 0; i < n_; ++i)
            hist_(var, 1)[i] = dt_ * rate_[var * n_ + i];
    for (int d = 0; d < 3; ++d)
        for (int i = 0; i < n_; ++i)
            hist_(d, 2)[i] = 0.5 * dt_ * hist_(3 + d, 1)[i];
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
    Frame& f = ctx.frame;
    StorageReal* cols[kVars] = {
        f.span(kAtom_x).data(),  f.span(kAtom_y).data(),
        f.span(kAtom_z).data(),  f.span(kAtom_vx).data(),
        f.span(kAtom_vy).data(), f.span(kAtom_vz).data()};
    for (int var = 0; var < kVars; ++var) {
        double* h1 = hist_(var, 1);
        double* h2 = hist_(var, 2);
        double* h3 = hist_(var, 3);
        double* h4 = hist_(var, 4);
        double* h5 = hist_(var, 5);
        StorageReal* col = cols[var];
        const double* y = state_.data() + var * n_;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_; ++i) {
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
    for (int var = 0; var < kVars; ++var) {
        double* h1 = hist_(var, 1);
        double* h2 = hist_(var, 2);
        double* h3 = hist_(var, 3);
        double* h4 = hist_(var, 4);
        double* h5 = hist_(var, 5);
        StorageReal* col = cols[var];
        double* y = state_.data() + var * n_;
        const double* rate = rate_.data() + var * n_;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_; ++i) {
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
