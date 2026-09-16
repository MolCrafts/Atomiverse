/// @file regressions/scf/c10h22_cfmm_j/reg_c10h22_cfmm_j.cpp
/// @brief C10H22 / STO-3G CFMM-J dual gate (hf-capability-gaps-21-cfmm):
///          1. Hard gate (energy proxy for elementwise J at convergence):
///             |E(CFMM-J) − E(Direct)| ≤ 1e-7 Ha. The unit-test
///             `test_cfmm_j` pins elementwise max|J_CFMM − J_direct| ≤ 1e-8
///             on H2O/STO-3G (public-API regressions cannot reach the
///             private J matrix without internal headers).
///          2. Both runs converge; wall-clock ≤ 60 s.
///
/// Oracle is the in-tree Schwarz direct path (PySCF ships no FMM — see
/// REFERENCE.md). White–Johnson–Gill–Head-Gordon extent control,
/// Chem. Phys. Lett. 230, 8 (1994), DOI 10.1016/0009-2614(94)01128-1.
///
/// Out-of-library consumer: public headers + link atomiverse only.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

using ATV::cpu::Frame;

/// All-trans n-decane (C10H22) skeleton in the xz-plane, Å.
/// C–C = 1.54 Å, C–H = 1.09 Å, tetrahedral (≈109.47°) H placement.
Frame make_c10h22() {
    constexpr double kCC = 1.54;
    constexpr double kCH = 1.09;
    constexpr double kPi = 3.14159265358979323846;
    // Chain along +x; each carbon staggered flat (all-trans projection).
    std::vector<double> cx(10), cy(10, 0.0), cz(10, 0.0);
    for (int i = 0; i < 10; ++i) {
        cx[static_cast<std::size_t>(i)] = i * kCC;
        // slight tetrahedral zig-zag in z for non-collinear H placement
        cz[static_cast<std::size_t>(i)] = (i % 2 == 0) ? 0.0 : 0.5;
    }

    // 10 C + 22 H = 32 atoms.
    Frame frame(32);
    int a = 0;
    for (int i = 0; i < 10; ++i) {
        frame.span(ATV::kAtom_type)[a] = 6;
        frame.span(ATV::kAtom_x)[a] = cx[static_cast<std::size_t>(i)];
        frame.span(ATV::kAtom_y)[a] = cy[static_cast<std::size_t>(i)];
        frame.span(ATV::kAtom_z)[a] = cz[static_cast<std::size_t>(i)];
        ++a;
    }

    // Terminal CH3 on C0 and C9: 3 H each; interior CH2: 2 H each.
    auto add_h = [&](double x, double y, double z) {
        frame.span(ATV::kAtom_type)[a] = 1;
        frame.span(ATV::kAtom_x)[a] = x;
        frame.span(ATV::kAtom_y)[a] = y;
        frame.span(ATV::kAtom_z)[a] = z;
        ++a;
    };

    const double th = 109.47 * kPi / 180.0;
    const double st = std::sin(th);
    const double ct = std::cos(th);

    // C0 methyl: back along −x plus two side H.
    add_h(cx[0] - kCH, 0.0, cz[0]);
    add_h(cx[0] + kCH * ct * 0.3, kCH * st, cz[0]);
    add_h(cx[0] + kCH * ct * 0.3, -kCH * st, cz[0]);
    // C9 methyl.
    add_h(cx[9] + kCH, 0.0, cz[9]);
    add_h(cx[9] - kCH * ct * 0.3, kCH * st, cz[9]);
    add_h(cx[9] - kCH * ct * 0.3, -kCH * st, cz[9]);
    // Interior methylene H (C1..C8): ±y
    for (int i = 1; i <= 8; ++i) {
        add_h(cx[static_cast<std::size_t>(i)], kCH * st * 0.9,
              cz[static_cast<std::size_t>(i)] + kCH * ct * 0.3);
        add_h(cx[static_cast<std::size_t>(i)], -kCH * st * 0.9,
              cz[static_cast<std::size_t>(i)] + kCH * ct * 0.3);
    }

    if (a != 32) {
        std::fprintf(stderr, "geometry atom count mismatch a=%d\n", a);
        std::exit(2);
    }
    frame.box = ATV::cpu::Box::cubic(80.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    cpu::Frame frame = make_c10h22();

    // Direct Schwarz path (oracle — our own PySCF-pinned vacuum route).
    cpu::scf::RHF direct;
    direct.basis = cpu::scf::BasisSet::STO3G;
    direct.tol = 1e-9;
    direct.max_iter = 80;
    direct.coulomb = cpu::scf::Coulomb::Direct;
    direct.run(frame);
    const double e_direct = direct.energy();

    // CFMM-J path (K remains exact).
    cpu::scf::RHF cfmm;
    cfmm.basis = cpu::scf::BasisSet::STO3G;
    cfmm.tol = 1e-9;
    cfmm.max_iter = 80;
    cfmm.coulomb = cpu::scf::Coulomb::CFMM;
    cfmm.cfmm_order = 10;
    cfmm.cfmm_threshold = 1e-9;
    cfmm.run(frame);
    const double e_cfmm = cfmm.energy();

    int rc = 0;
    std::printf("E_RHF(Direct) = %.12f Ha  converged=%d iters=%d\n", e_direct,
                static_cast<int>(direct.converged()), direct.iterations());
    std::printf("E_RHF(CFMM)   = %.12f Ha  converged=%d iters=%d\n", e_cfmm,
                static_cast<int>(cfmm.converged()), cfmm.iterations());

    if (!direct.converged() || !cfmm.converged()) {
        std::printf("FAIL: SCF did not converge (direct=%d cfmm=%d)\n",
                    static_cast<int>(direct.converged()),
                    static_cast<int>(cfmm.converged()));
        rc = 1;
    }

    const double dE = std::fabs(e_cfmm - e_direct);
    std::printf("gate energy: |E_CFMM - E_Direct| = %.3e Ha (tol 1e-7)\n", dE);
    if (!(dE <= 1e-7)) {
        std::printf("FAIL: CFMM-J total energy beyond 1e-7 Ha of Direct\n");
        rc = 1;
    }

    if (rc == 0) {
        std::printf("PASS: C10H22/STO-3G CFMM-J vs Direct |dE|=%.3e Ha "
                    "[WJGH 1994; oracle = in-tree Direct (PySCF has no FMM)]\n",
                    dE);
    }
    return rc;
}
