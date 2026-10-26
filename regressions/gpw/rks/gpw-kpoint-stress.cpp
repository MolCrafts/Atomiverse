/// @file regressions/gpw/rks/gpw-kpoint-stress.cpp
/// @brief END-TO-END: GPW fixed-P cell-strain stress at Monkhorst-Pack k points.
///
/// Diamond Si bct 4-atom cell (DZVP-MOLOPT-GTH + GTH-PBE-q4, PBE), mesh
/// 1x1x2, single grid. CP2K analytical tensor is the external
/// oracle; relaxed strain finite differences supplement it. See REFERENCE.md.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molpy.hpp"
#include "harness/registry.h"

namespace atv_reg_gpw_gpw_kpoint_stress {

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

/// Diamond Si as a 4-atom BCT cell (a/√2, a/√2, a) of cubic edge `a_ang` [Å].
ATV::cpu::Frame si_bct(double a_ang) {
    using namespace ATV;
    const double at = a_ang / std::sqrt(2.0);
    const double frac[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
    cpu::Frame frame(4);
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_type)[i] = 14;
        frame.span(kAtom_x)[i] = frac[i][0] * at;
        frame.span(kAtom_y)[i] = frac[i][1] * at;
        frame.span(kAtom_z)[i] = frac[i][2] * a_ang;
    }
    frame.box = cpu::Box::orthorhombic(at, at, a_ang);
    return frame;
}

} // namespace

int run() {
    using namespace ATV;
    using namespace ATV::cpu::gpw;
    g_fail = 0;
    const auto start = std::chrono::steady_clock::now();
    const cpu::Frame frame = si_bct(5.43);

    // Actual CP2K 2026.2 CORE_PPL GRID energy and analytical tensor.
    // Raw output/input/hash: tools/cp2k_oracle/inputs/gpw-kpoint-stress/.
    constexpr double energy_pin = -15.348312942859359;
    constexpr std::array<double, 6> stress_pin = {
        0.00088302882243018852, 0.00088302882243018852,
        0.00026101337099615431, -1.2004166081076166e-17,
        4.1518324258410863e-16, 7.3908856940892354e-18};
    constexpr double energy_tol = 1e-3; // Ha
    constexpr double stress_tol = 1e-4; // Ha/Bohr^3
    constexpr double fd_tol = 1e-4; // Ha/Bohr^3, independent consistency gate
    constexpr double h = 2e-3;
    const char* names[] = {"xx", "yy", "zz", "xy", "xz", "yz"};
    auto check_near = [&](double got, double want, double tol,
                          const char* quantity) {
        const bool ok = std::isfinite(got) && std::abs(got - want) <= tol;
        std::printf("  %s got=% .12e want=% .12e tol=%.3e %s\n", quantity, got,
                    want, tol, ok ? "ok" : "FAIL");
        if (!ok)
            ++g_fail;
    };

    auto make_method = []() {
        GpwDft method{
            {Kind{molrs::Element::Si,
                  GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
                  GthPotential(molrs::Element::Si, "GTH-PBE-q4")}},
            100.0,
            XcKind::PBE};
        method.n_occ = 8;
        method.max_iter = 100;
        method.tol = 1e-10;
        method.residual_tol = 1e-8;
        method.eps_default = 1e-12;
        method.local_pp_on_grid = true;
        method.k_mesh[0] = 1;
        method.k_mesh[1] = 1;
        method.k_mesh[2] = 2; // MP kappa_z=+/-1/4, time reversal folded
        return method;
    };
    const auto begin = std::chrono::steady_clock::now();
    GpwDft method = make_method();
    method.stress_strain_h = h;
    cpu::Frame state = frame;
    method.build(state);
    auto stress = method.stress(state);
    const auto& result = method.last_result();
    check(result.converged, "unstrained SCF converged");
    check(result.n_kpts >= 1 && !result.C_occ_k.empty(),
          "Bloch occupied-orbital payload present");
    check_near(result.e_total, energy_pin, energy_tol, "energy [Ha] vs CP2K");
    for (double& value : stress)
        value /= units::kHartreePerBohr3InEvPerAngstrom3;
    const auto product_done = std::chrono::steady_clock::now();

    // Additional internal consistency check: all +/- samples start from
    // the same unstrained density and independently reconverge SCF.
    const auto relaxed_ev = gpw_stress_fd(frame, make_method(), h);
    double largest = 0;
    for (std::size_t component = 0; component < stress.size(); ++component) {
        char quantity[96];
        std::snprintf(quantity, sizeof(quantity), "sigma_%s [Ha/Bohr3] vs CP2K",
                      names[component]);
        check_near(stress[component], stress_pin[component], stress_tol,
                   quantity);
        if (component < 3) {
            const double relaxed =
                relaxed_ev[component] / units::kHartreePerBohr3InEvPerAngstrom3;
            std::snprintf(quantity, sizeof(quantity),
                          "sigma_%s fixed-P vs relaxed FD", names[component]);
            check_near(stress[component], relaxed, fd_tol, quantity);
            largest = std::max(largest, std::abs(stress[component]));
        } else {
            check_near(stress[component], 0.0, 1e-8,
                       "off-diagonal symmetry [Ha/Bohr3]");
        }
    }
    std::printf("  stress magnitude got=%.6e want=>1e-6 Ha/Bohr3\n", largest);
    check(largest > 1e-6, "nonzero stress magnitude floor");
    const auto end = std::chrono::steady_clock::now();
    std::printf("  product %.6fs relaxed-FD %.6fs total %.6fs\n",
                std::chrono::duration<double>(product_done - begin).count(),
                std::chrono::duration<double>(end - product_done).count(),
                std::chrono::duration<double>(end - begin).count());
    const double wall = std::chrono::duration<double>(end - start).count();
    std::printf("  whole section got=%.6f want<=60 seconds\n", wall);
    check(wall <= 60.0, "whole section within 60 seconds");
    if (g_fail) {
        std::fprintf(stderr, "FAILED: gpw-kpoint-stress %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf(
        "PASS: gpw-kpoint-stress CP2K single-grid tensor and relaxed FD\n");
    return 0;
}

} // namespace atv_reg_gpw_gpw_kpoint_stress

ATV_REGRESSION("gpw/rks", "gpw-kpoint-stress",
               atv_reg_gpw_gpw_kpoint_stress::run);
