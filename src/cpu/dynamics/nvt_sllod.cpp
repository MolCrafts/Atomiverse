/// @file cpu/dynamics/nvt_sllod.cpp
/// @brief CPU NVT SLLOD. Formulas live in `core/sllod_flow.h`.
///
/// Step structure ported from LAMMPS `src/fix_nh.cpp` (FixNH
/// initial/final_integrate, nhc_temp_integrate, nh_v_temp),
/// `src/fix_nvt_sllod.cpp` and `src/fix_deform.cpp` (end_of_step,
/// pre_exchange flip), SHA e8c4c278a8599165a0321ebfcb4e4046ef79ad79.
/// Copyright (2003) Sandia Corporation. Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software. GPL-2.0-only.

#include "atomiverse/cpu/dynamics/nvt_sllod.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace md {
namespace {

void read_upper(const Mat3& cell, double h[6]) {
    const double lower[3] = {cell(1, 0), cell(2, 0), cell(2, 1)};
    for (double v : lower) {
        if (v != 0.0)
            throw std::invalid_argument(
                "NvtSllod: cell lower triangle must be zero");
    }
    h[0] = cell(0, 0);
    h[1] = cell(1, 1);
    h[2] = cell(2, 2);
    h[3] = cell(1, 2);
    h[4] = cell(0, 2);
    h[5] = cell(0, 1);
}

void write_upper(Box& box, const double h[6]) {
    Mat3 cell = box.cell();
    cell(0, 0) = h[0];
    cell(1, 1) = h[1];
    cell(2, 2) = h[2];
    cell(1, 2) = h[3];
    cell(0, 2) = h[4];
    cell(0, 1) = h[5];
    cell(1, 0) = 0;
    cell(2, 0) = 0;
    cell(2, 1) = 0;
    box.set_cell(cell);
}

} // namespace

NvtSllod::NvtSllod(double temperature_K, double tau_fs, double dt_fs,
                   double rate_per_fs, int direction, int tchain)
    : T_(temperature_K), tau_(tau_fs), dt_(dt_fs), rate_(rate_per_fs),
      dir_(direction), tchain_(tchain) {
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

void NvtSllod::build(const cpu::BuildContext& ctx) {
    n_dof_ = static_cast<int>(ctx.runtime.md.n_dof_global);
    armed_ = false;
    kicked_ = false;
    step_ = 0;
    flow_ = {};
    eta_.assign(tchain_, 0.0);
    eta_dot_.assign(tchain_, 0.0);
    eta_mass_.assign(tchain_, 0.0);
    if (n_dof_ > 0)
        ATV::md::sllod_chain_masses(eta_mass_.data(), tchain_,
                                    static_cast<double>(n_dof_),
                                    units::md_boltz * T_, tau_);
    ke_current_ = 0;
    ke_valid_ = false;
}

void NvtSllod::capture_(const cpu::LaunchContext& ctx) {
    read_upper(ctx.frame.box.cell(), flow_.h0);
    for (int i = 0; i < 6; ++i)
        flow_.h[i] = flow_.h0[i];
    flow_.rate = rate_;
    flow_.dt = dt_;
    flow_.dir = dir_;
    ATV::md::sllod_erate_rates(flow_.h0, rate_, dir_, flow_.h_rate,
                               flow_.h_ratelo);
    if (!ATV::md::sllod_refresh(flow_))
        throw std::invalid_argument("NvtSllod: box edge must be positive");
    armed_ = true;
}

// LAMMPS ComputeTempDeform::compute_scalar: KE of the velocity relative to
// the streaming profile at the current position.
double NvtSllod::peculiar_ke_(const cpu::LaunchContext& ctx) const {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const StorageReal* __restrict mass = f.span(kAtom_mass);
    const StorageReal* __restrict vx = f.span(kAtom_vx);
    const StorageReal* __restrict vy = f.span(kAtom_vy);
    const StorageReal* __restrict vz = f.span(kAtom_vz);
    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);

    double ke_local = 0;
#pragma omp parallel for schedule(static) reduction(+ : ke_local)
    for (int i = 0; i < n; ++i) {
        double bx, by, bz;
        ATV::md::sllod_deform_bias(x[i], y[i], z[i], flow_.h_inv, flow_.h_rate,
                                   flow_.h_ratelo, bx, by, bz);
        const double px = vx[i] - bx;
        const double py = vy[i] - by;
        const double pz = vz[i] - bz;
        ke_local += mass[i] * (px * px + py * py + pz * pz);
    }
    ke_local *= 0.5 * units::md_mvv2e;

    double ke = ke_local;
#ifdef ATV_HAS_MPI
    if (ctx.runtime.mpi_size > 1 && ctx.runtime.md.dd
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        MPI_Allreduce(&ke_local, &ke, 1, MPI_DOUBLE, MPI_SUM,
                      ctx.runtime.md.dd->cart_comm);
    }
#endif
    return ke;
}

// LAMMPS FixNH::nhc_temp_integrate + FixNH::nh_v_temp (temperature bias
// removed and restored around the scaling). `measure` is final_integrate's
// `t_current = temperature->compute_scalar()`; initial_integrate reuses the
// value the previous half-step left scaled (FixNH::setup measures once).
void NvtSllod::thermostat_(const cpu::LaunchContext& ctx, bool measure) {
    if (n_dof_ <= 0 || !(eta_mass_[0] > 0.0))
        return;
    if (measure || !ke_valid_)
        ke_current_ = peculiar_ke_(ctx);
    ke_valid_ = true;

    const double kT = units::md_boltz * T_;
    const double target = static_cast<double>(n_dof_) * kT;
    const double scale =
        ATV::md::sllod_nhc_half(eta_.data(), eta_dot_.data(), eta_mass_.data(),
                                tchain_, ke_current_, target, kT, dt_);

    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    StorageReal* __restrict vx = f.span(kAtom_vx);
    StorageReal* __restrict vy = f.span(kAtom_vy);
    StorageReal* __restrict vz = f.span(kAtom_vz);
    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double bx, by, bz;
        ATV::md::sllod_deform_bias(x[i], y[i], z[i], flow_.h_inv, flow_.h_rate,
                                   flow_.h_ratelo, bx, by, bz);
        vx[i] = bx + scale * (vx[i] - bx);
        vy[i] = by + scale * (vy[i] - by);
        vz[i] = bz + scale * (vz[i] - bz);
    }
}

void NvtSllod::nve_v_(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const StorageReal* __restrict mass = f.span(kAtom_mass);
    StorageReal* __restrict vx = f.span(kAtom_vx);
    StorageReal* __restrict vy = f.span(kAtom_vy);
    StorageReal* __restrict vz = f.span(kAtom_vz);
    const AccReal* __restrict fx = f.span(kAtom_fx);
    const AccReal* __restrict fy = f.span(kAtom_fy);
    const AccReal* __restrict fz = f.span(kAtom_fz);
    const double dtf = 0.5 * dt_ * units::md_ftm2v;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const double dtfm = dtf / mass[i];
        vx[i] += dtfm * fx[i];
        vy[i] += dtfm * fy[i];
        vz[i] += dtfm * fz[i];
    }
}

// FixNVTSllod::nve_x (REVERSIBLE). No wrap here: LAMMPS integrates
// continuous coordinates and wraps in Domain::pbc on reneighbor steps, which
// the Verlet displacement check relies on.
void NvtSllod::nve_x_(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    StorageReal* __restrict x = f.span(kAtom_x);
    StorageReal* __restrict y = f.span(kAtom_y);
    StorageReal* __restrict z = f.span(kAtom_z);
    StorageReal* __restrict vx = f.span(kAtom_vx);
    StorageReal* __restrict vy = f.span(kAtom_vy);
    StorageReal* __restrict vz = f.span(kAtom_vz);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double px = x[i], py = y[i], pz = z[i];
        double ux = vx[i], uy = vy[i], uz = vz[i];
        double bx, by, bz;
        ATV::md::sllod_deform_bias(px, py, pz, flow_.h_inv, flow_.h_rate,
                                   flow_.h_ratelo, bx, by, bz);
        ux -= bx;
        uy -= by;
        uz -= bz;
        ATV::md::sllod_nve_x(px, py, pz, ux, uy, uz, flow_.h, flow_.grad_u,
                             dt_);
        ATV::md::sllod_apply_bias(ux, uy, uz, px, py, pz, flow_.h, flow_.grad_u,
                                  dt_);
        x[i] = px;
        y[i] = py;
        z[i] = pz;
        vx[i] = ux;
        vy[i] = uy;
        vz[i] = uz;
    }
}

// FixDeform::pre_exchange (flip: new tilts, Domain::image_flip, then
// remap_all) followed by Verlet's Domain::pbc with deform_vremap; both wrap
// with the same lambda / velocity-jump law, so one sllod_pbc pass covers
// them. Runs on reneighbor steps only; a decided flip requested this one.
void NvtSllod::pre_exchange_(const cpu::LaunchContext& ctx) {
    const bool flip = ATV::md::sllod_flip_pending(flow_);
    if (flip) {
        ATV::md::sllod_apply_flip(flow_);
        if (!ATV::md::sllod_refresh(flow_))
            throw std::invalid_argument("NvtSllod: box edge must be positive");
        write_upper(ctx.frame.box, flow_.h);
    }
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    StorageReal* x = f.span(kAtom_x);
    StorageReal* y = f.span(kAtom_y);
    StorageReal* z = f.span(kAtom_z);
    StorageReal* vx = f.span(kAtom_vx);
    StorageReal* vy = f.span(kAtom_vy);
    StorageReal* vz = f.span(kAtom_vz);
    auto image = f.span(kAtom_image);
    const int flips[3] = {flow_.flip[0], flow_.flip[1], flow_.flip[2]};
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int box[3] = {static_cast<int>(image(i, 0)),
                      static_cast<int>(image(i, 1)),
                      static_cast<int>(image(i, 2))};
        if (flip)
            ATV::md::sllod_image_flip(box[0], box[1], box[2], flips);
        double px = x[i], py = y[i], pz = z[i];
        double ux = vx[i], uy = vy[i], uz = vz[i];
        int shift[3];
        ATV::md::sllod_pbc(px, py, pz, ux, uy, uz, flow_.h, flow_.h_inv,
                           flow_.h_rate, shift);
        for (int d = 0; d < 3; ++d)
            image(i, d) = static_cast<ImageInt>(box[d] + shift[d]);
        x[i] = px;
        y[i] = py;
        z[i] = pz;
        vx[i] = ux;
        vy[i] = uy;
        vz[i] = uz;
    }
    flow_.flip[0] = flow_.flip[1] = flow_.flip[2] = 0;
}

void NvtSllod::end_of_step_(const cpu::LaunchContext& ctx) {
    const long long next = step_ + 1;
    if (dir_ < 3) {
        double h_new[6];
        ATV::md::sllod_box_at(flow_.h0, rate_, dt_, dir_, next, h_new);
        if (!(h_new[dir_] > 0.0))
            throw std::invalid_argument(
                "NvtSllod: deformed edge is not positive");
    }
    const double shift = ATV::md::sllod_advance_box(flow_, next);
    if (!ATV::md::sllod_refresh(flow_))
        throw std::invalid_argument("NvtSllod: box edge must be positive");
    write_upper(ctx.frame.box, flow_.h);
    // LAMMPS next_reneighbor = ntimestep + 1: the flip runs in the next
    // step's pre_exchange, before that step's neighbor build.
    if (ATV::md::sllod_flip_pending(flow_))
        ctx.runtime.md.reneighbor_requested = true;

    if (shift != 0.0) {
        auto& f = ctx.frame;
        StorageReal* coord = nullptr;
        switch (dir_) {
        case 0:
            coord = f.span(kAtom_x);
            break;
        case 1:
            coord = f.span(kAtom_y);
            break;
        default:
            coord = f.span(kAtom_z);
            break;
        }
        const int n_total = force_count(ctx);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_total; ++i)
            coord[i] += shift;
    }
    step_ = next;
}

void NvtSllod::launch(const cpu::LaunchContext& ctx) {
    if (!armed_)
        capture_(ctx);
    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        if (!kicked_) {
            auto& f = ctx.frame;
            const int n = owned_count(ctx);
            StorageReal* vx = f.span(kAtom_vx);
            StorageReal* vy = f.span(kAtom_vy);
            StorageReal* vz = f.span(kAtom_vz);
            const StorageReal* x = f.span(kAtom_x);
            const StorageReal* y = f.span(kAtom_y);
            const StorageReal* z = f.span(kAtom_z);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                double ux = vx[i], uy = vy[i], uz = vz[i];
                ATV::md::sllod_apply_bias(ux, uy, uz, x[i], y[i], z[i], flow_.h,
                                          flow_.grad_u, 0.0);
                vx[i] = ux;
                vy[i] = uy;
                vz[i] = uz;
            }
            kicked_ = true;
        }
        thermostat_(ctx, /*measure=*/false);
        nve_v_(ctx);
        nve_x_(ctx);
    } else if (ctx.at(ATV::md::Phase::pre_exchange)) {
        pre_exchange_(ctx);
    } else if (ctx.at(ATV::md::Phase::final_integrate)) {
        nve_v_(ctx);
        thermostat_(ctx, /*measure=*/true);
    } else if (ctx.at(ATV::md::Phase::end_of_step)) {
        end_of_step_(ctx);
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
