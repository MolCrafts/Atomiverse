/// @file cpu/dynamics/piston_electrode.cpp
/// @brief Rigid electrode piston for constant \(P_z\).

#include "atomiverse/cpu/dynamics/piston_electrode.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace md {

PistonElectrode::PistonElectrode(int type_id, double P_target, Side side)
    : P_target(P_target), type_id_(type_id), side_(side) {}

void PistonElectrode::resolve_rows_(const cpu::Frame& f, int n_owned) {
    rows_.clear();
    const auto type = f.span(kAtom_type);
    rows_.reserve(static_cast<std::size_t>(n_owned));
    for (int i = 0; i < n_owned; ++i) {
        if (static_cast<int>(type[i]) == type_id_)
            rows_.push_back(i);
    }
}

void PistonElectrode::snapshot_xy_offsets_(const cpu::Frame& f) {
    xy_off_by_tag_.clear();
    z_off_by_tag_.clear();
    if (rows_.empty())
        return;

    const auto tag = f.span(kAtom_tag);
    const auto x = f.span(kAtom_x);
    const auto y = f.span(kAtom_y);
    const auto z = f.span(kAtom_z);
    const auto mass = f.span(kAtom_mass);

    double M = 0.0;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (int i : rows_) {
        const double m = static_cast<double>(mass[i]);
        M += m;
        cx += m * static_cast<double>(x[i]);
        cy += m * static_cast<double>(y[i]);
        cz += m * static_cast<double>(z[i]);
    }
    if (!(M > 0.0))
        throw std::runtime_error("PistonElectrode: total mass must be > 0");
    cx /= M;
    cy /= M;
    cz /= M;
    last_mass_ = M;
    last_z_com_ = cz;

    for (int i : rows_) {
        const TagInt t = tag[i];
        xy_off_by_tag_[t] = {static_cast<StorageReal>(x[i] - cx),
                             static_cast<StorageReal>(y[i] - cy)};
        z_off_by_tag_[t] = static_cast<StorageReal>(z[i] - cz);
    }
}

void PistonElectrode::apply_forces_(cpu::Frame& f) {
    if (rows_.empty()) {
        last_F_z_atoms_ = 0.0;
        last_F_net_ = 0.0;
        last_P_z_ = 0.0;
        last_area_ = 0.0;
        return;
    }

    auto fx = f.span(kAtom_fx);
    auto fy = f.span(kAtom_fy);
    auto fz = f.span(kAtom_fz);
    const auto mass = f.span(kAtom_mass);
    const auto vz = f.span(kAtom_vz);
    const auto z = f.span(kAtom_z);

    double M = 0.0;
    double F_atoms = 0.0;
    double cz = 0.0;
    double pz = 0.0;
    for (int i : rows_) {
        const double m = static_cast<double>(mass[i]);
        M += m;
        F_atoms += static_cast<double>(fz[i]);
        cz += m * static_cast<double>(z[i]);
        pz += m * static_cast<double>(vz[i]);
    }
    if (!(M > 0.0))
        throw std::runtime_error("PistonElectrode: total mass must be > 0");
    cz /= M;
    const double vz_com = pz / M;
    last_mass_ = M;
    last_z_com_ = cz;
    last_vz_com_ = vz_com;
    last_F_z_atoms_ = F_atoms;

    const double Lx = f.box.length(0);
    const double Ly = f.box.length(1);
    const double A = Lx * Ly;
    last_area_ = A;
    if (!(A > 0.0))
        throw std::runtime_error("PistonElectrode: wall area must be > 0");

    // External force: fluid between walls.
    // Upper: F_ext = -P A (pushes down). Lower: F_ext = +P A (pushes up).
    const double sign = (side_ == Side::Upper) ? -1.0 : +1.0;
    const double F_ext = sign * P_target * A;

    // Optional COM friction: a_damp = -v/tau ⇒
    // F_damp = -M * v_com / (tau * force_to_accel)  [eV/Å].
    double F_damp = 0.0;
    if (tau_damp > 0.0) {
        F_damp = -M * vz_com / (tau_damp * units::force_to_accel);
    }

    last_F_net_ = F_atoms + F_ext + F_damp;

    // Instantaneous pressure from atom forces only (external load is the target).
    last_P_z_ = (side_ == Side::Upper) ? (F_atoms / A) : (-F_atoms / A);

    for (int i : rows_) {
        const double m = static_cast<double>(mass[i]);
        const double fi = last_F_net_ * (m / M);
        fx[i] = AccReal(0);
        fy[i] = AccReal(0);
        fz[i] = AccReal(fi);
    }
}

void PistonElectrode::enforce_rigid_(cpu::Frame& f, bool update_positions) {
    if (rows_.empty())
        return;

    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    auto vx = f.span(kAtom_vx);
    auto vy = f.span(kAtom_vy);
    auto vz = f.span(kAtom_vz);
    const auto mass = f.span(kAtom_mass);
    const auto tag = f.span(kAtom_tag);

    double M = 0.0;
    double cz = 0.0;
    double pz = 0.0;
    double cx = 0.0, cy = 0.0;
    for (int i : rows_) {
        const double m = static_cast<double>(mass[i]);
        M += m;
        cz += m * static_cast<double>(z[i]);
        pz += m * static_cast<double>(vz[i]);
        cx += m * static_cast<double>(x[i]);
        cy += m * static_cast<double>(y[i]);
    }
    if (!(M > 0.0))
        return;
    cz /= M;
    cx /= M;
    cy /= M;
    const double vz_com = pz / M;
    last_mass_ = M;
    last_z_com_ = cz;
    last_vz_com_ = vz_com;

    for (int i : rows_) {
        vx[i] = StorageReal(0);
        vy[i] = StorageReal(0);
        vz[i] = static_cast<StorageReal>(vz_com);
        if (!update_positions)
            continue;
        const TagInt t = tag[i];
        const auto itxy = xy_off_by_tag_.find(t);
        const auto itz = z_off_by_tag_.find(t);
        if (itxy != xy_off_by_tag_.end()) {
            x[i] = static_cast<StorageReal>(cx + itxy->second[0]);
            y[i] = static_cast<StorageReal>(cy + itxy->second[1]);
        }
        if (itz != z_off_by_tag_.end()) {
            // Rigid z-pattern about current COM (COM advanced by VV).
            z[i] = static_cast<StorageReal>(cz + itz->second);
        }
    }
}

void PistonElectrode::build(const cpu::BuildContext& ctx) {
    const int n = owned_count(ctx);
    resolve_rows_(ctx.frame, n);
    if (rows_.empty())
        throw std::runtime_error(
            "PistonElectrode: no owned atoms of the requested type");
    snapshot_xy_offsets_(ctx.frame);
    auto& f = ctx.frame;
    auto vx = f.span(kAtom_vx);
    auto vy = f.span(kAtom_vy);
    for (int i : rows_) {
        vx[i] = StorageReal(0);
        vy[i] = StorageReal(0);
    }
}

void PistonElectrode::launch(const cpu::LaunchContext& ctx) {
    const int n = owned_count(ctx);
    resolve_rows_(ctx.frame, n);

    if (ctx.at(ATV::md::Phase::post_force)) {
        apply_forces_(ctx.frame);
        return;
    }
    if (ctx.at(ATV::md::Phase::post_integrate)) {
        enforce_rigid_(ctx.frame, /*update_positions=*/true);
        return;
    }
    if (ctx.at(ATV::md::Phase::final_integrate)) {
        enforce_rigid_(ctx.frame, /*update_positions=*/false);
        return;
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
