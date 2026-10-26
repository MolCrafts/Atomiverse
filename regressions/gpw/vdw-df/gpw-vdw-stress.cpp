/// @file regressions/gpw/gpw-vdw-stress/reg_gpw-vdw-stress.cpp
/// @brief END-TO-END: fixed-P strain-difference stress with nonlocal vdW-DF (DRSLL).
///
/// Water dimer Γ GPW PBE + NON_LOCAL DRSLL. Fixed-P vs relaxed-SCF strain FD
/// of E_tot; soft absolute class vs offline CP2K STRESS + NON_LOCAL twin.
/// See REFERENCE.md.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"
#include "harness/registry.h"

namespace atv_reg_gpw_gpw_vdw_stress {

namespace {

const double kAng[6][3] = {
    {0.000000, 0.000000, 0.000000},  // O
    {0.758602, 0.000000, 0.504284},  // H
    {0.758602, 0.000000, -0.504284}, // H
    {3.000000, 0.000000, 0.000000},  // O
    {3.758602, 0.000000, 0.504284},  // H
    {3.758602, 0.000000, -0.504284}, // H
};
const int kZ[6] = {8, 1, 1, 8, 1, 1};

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

ATV::cpu::Frame make_frame() {
    using namespace ATV;
    cpu::Frame frame(6);
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_type)[i] = kZ[i];
        frame.span(kAtom_x)[i] = kAng[i][0];
        frame.span(kAtom_y)[i] = kAng[i][1];
        frame.span(kAtom_z)[i] = kAng[i][2];
    }
    frame.box = cpu::Box::cubic(12.0);
    return frame;
}

std::vector<ATV::cpu::gpw::Kind> water_kinds() {
    using namespace ATV::cpu::gpw;
    return {Kind{molrs::Element::O,
                 GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
                 GthPotential(molrs::Element::O, "GTH-PBE-q6")},
            Kind{molrs::Element::H,
                 GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
                 GthPotential(molrs::Element::H, "GTH-PBE-q1")}};
}

} // namespace

int run() {
    using namespace ATV::cpu::gpw;
    const auto section_start = std::chrono::steady_clock::now();

    // Energy twin class of gpw-vdw-df-ice / vdw-kpoint-forces (Γ DRSLL).
    constexpr double kEDispCp2k = 0.151943904803; // Ha (post-SCF class)
    constexpr double kEDispTol = 5.0e-4;
    constexpr double kETotSoftTol = 5.0e-2; // Ha twin-family residual

    ATV::cpu::Frame frame = make_frame();

    auto make_cfg = [&]() {
        GpwDft m(water_kinds(), 200.0, XcKind::PBE);
        m.n_occ = 8;
        m.max_iter = 40;
        m.tol = 1e-7;
        m.mix = 0.4;
        m.nonperiodic = false;
        m.vdw_df = true;
        m.local_pp_on_grid = true;
        return m;
    };

    const double h = 2e-3;
    // Public stress performs one SCF and publishes that state's energy and
    // result too. Read them here rather than cold-solving the same state
    // first for energy and then again for fixed-P stress.
    GpwDft energy = make_cfg();
    energy.stress_strain_h = h;
    energy.build(frame);
    constexpr double kToHaBohr3 =
        1.0 / ATV::units::kHartreePerBohr3InEvPerAngstrom3;
    const auto product_start = std::chrono::steady_clock::now();
    auto san = energy.stress(frame);
    for (auto& value : san)
        value *= kToHaBohr3;
    const auto product_end = std::chrono::steady_clock::now();
    const double wall_product =
        std::chrono::duration<double>(product_end - product_start).count();
    const auto& r = energy.last_result();
    std::printf("E=%.10f Ha E_disp=%.10f Ha iters=%d wall_energy+stress=%.3f s "
                "conv=%d\n",
                r.e_total, r.e_disp, r.iters, wall_product,
                r.converged ? 1 : 0);
    check(r.converged, "SCF converged");
    check(std::isfinite(r.e_total) && std::isfinite(r.e_disp), "finite E");
    check(std::abs(r.e_disp - kEDispCp2k) <= kEDispTol, "E_disp vs CP2K class");
    check(std::abs(r.e_disp) > 1e-4, "non-vacuous nonlocal E_disp");
    auto stress_ha_bohr3 = [&](GpwDft m) {
        m.stress_strain_h = h;
        m.build(frame);
        auto stress = m.stress(frame);
        for (auto& value : stress)
            value *= kToHaBohr3;
        return stress;
    };
    const auto t3 = std::chrono::steady_clock::now();
    auto sfd = gpw_stress_fd(frame, make_cfg(), h);
    for (auto& v : sfd)
        v *= kToHaBohr3;
    const auto t4 = std::chrono::steady_clock::now();
    const double wall_fd = std::chrono::duration<double>(t4 - t3).count();

    double max_diag_err = 0.0;
    const char* names[3] = {"xx", "yy", "zz"};
    for (int ax = 0; ax < 3; ++ax) {
        const double an = san[static_cast<std::size_t>(ax)];
        const double fd = sfd[static_cast<std::size_t>(ax)];
        const double err = std::abs(an - fd);
        max_diag_err = std::max(max_diag_err, err);
        std::printf("  σ_%s  fixed-P=% .8e  fd=% .8e  |d|=%.3e Ha/Bohr3\n",
                    names[ax], an, fd, err);
        check(std::isfinite(an) && std::isfinite(fd), "finite stress");
    }
    for (int k = 3; k < 6; ++k)
        check(std::abs(san[static_cast<std::size_t>(k)]) <= 1e-8,
              "off-diagonal ~ 0");

    // Fixed-P strain difference vs relaxed-SCF strain FD; residual SCF + O(h²) + nonlocal.
    constexpr double kAnFdTol = 5.0e-4; // Ha/Bohr³
    std::printf("max|σ_fixedP-σ_fd|=%.3e Ha/Bohr3 wall_energy+stress=%.3f s "
                "wall_fd=%.3f s\n",
                max_diag_err, wall_product, wall_fd);
    check(max_diag_err <= kAnFdTol, "fixed-P vs relaxed FD max diag (vdW-DF)");

    // Wiring: stress without nonlocal must differ if E_disp is large enough.
    {
        auto bare = make_cfg();
        bare.vdw_df = false;
        const auto s_bare = stress_ha_bohr3(std::move(bare));
        double max_diff = 0.0;
        for (int ax = 0; ax < 3; ++ax)
            max_diff = std::max(
                max_diff, std::abs(san[static_cast<std::size_t>(ax)]
                                   - s_bare[static_cast<std::size_t>(ax)]));
        std::printf("max|σ_vdw−σ_bare|=%.3e Ha/Bohr3\n", max_diff);
        check(max_diff > 1e-6, "nonlocal stress contribution non-vacuous");
    }

    (void)kETotSoftTol;
    std::printf("T_atv_energy+stress=%.3f s\n", wall_product);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    const double wall = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - section_start)
                            .count();
    std::printf("whole section got=%.6f want<=60 seconds\n", wall);
    if (wall > 60.0)
        return 1;
    std::printf("PASS: gpw-vdw-stress  E=%.10f  E_disp=%.6e  max|dσ|=%.3e  "
                "T_energy+stress=%.3f s\n",
                r.e_total, r.e_disp, max_diag_err, wall_product);
    return 0;
}

} // namespace atv_reg_gpw_gpw_vdw_stress

ATV_REGRESSION("gpw/vdw-df", "gpw-vdw-stress", atv_reg_gpw_gpw_vdw_stress::run);
