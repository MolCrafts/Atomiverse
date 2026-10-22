/// @file regressions/scf/uhf/h2_uhf_sto3g.h
/// @brief Shared PySCF pins for the STO-3G UHF energy + gradient case.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"

namespace uhf_case {

struct Out {
    double energy_ha = 0;
    bool converged = false;
    xt::xtensor<double, 2> forces;
};

inline ATV::cpu::Frame h2_frame() {
    using namespace ATV;
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[0] = 0;
    frame.span(kAtom_y)[0] = 0;
    frame.span(kAtom_z)[0] = 0;
    frame.span(kAtom_x)[1] = 0;
    frame.span(kAtom_y)[1] = 0;
    frame.span(kAtom_z)[1] = 0.74;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

inline ATV::cpu::Frame water_frame() {
    using namespace ATV;
    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0;
    frame.span(kAtom_y)[0] = 0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = 0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

inline double grad_ha_bohr(const xt::xtensor<double, 2>& f, int A, int d) {
    using namespace ATV;
    return -units::convert(f(A, d), units::Force::eVPerAngstrom,
                           units::Force::HartreePerBohr);
}

inline int check_grad(const char* tag, const Out& got, double e_want,
                      const double* g_want, int natm) {
    constexpr double kE = 1e-8;
    constexpr double kG = 1e-6;
    if (!got.converged) {
        std::printf("FAIL %s: SCF did not converge\n", tag);
        return 1;
    }
    const double de = std::fabs(got.energy_ha - e_want);
    std::printf("%s E got=%.12f want=%.12f |dE|=%.3e\n", tag, got.energy_ha,
                e_want, de);
    if (!(de < kE)) {
        std::printf("FAIL %s: energy\n", tag);
        return 1;
    }
    double max_err = 0;
    double net[3] = {0, 0, 0};
    for (int A = 0; A < natm; ++A) {
        for (int d = 0; d < 3; ++d) {
            const double g = grad_ha_bohr(got.forces, A, d);
            net[d] += g;
            const double err = std::fabs(g - g_want[A * 3 + d]);
            if (err > max_err)
                max_err = err;
        }
    }
    const double net_max = std::max(
        std::fabs(net[0]), std::max(std::fabs(net[1]), std::fabs(net[2])));
    std::printf("%s max|dg|=%.3e |sum g|=%.3e\n", tag, max_err, net_max);
    if (!(max_err < kG) || !(net_max < 1e-8)) {
        std::printf("FAIL %s: gradient\n", tag);
        return 1;
    }
    std::printf("PASS %s\n", tag);
    return 0;
}

template <class Run>
int run_all(Run run) {
    // PySCF 2.14.0, tools/pyscf_oracle/pin_scf.py uhf --geom h2 --basis sto-3g
    // --charge 1 --spin 1 --grad --conv-tol 1e-12
    constexpr double e_h2p = -0.538205447565;
    const double g_h2p[6] = {0, 0, 0.191355066881, 0, 0, -0.191355066881};
    auto frame = h2_frame();
    Out h2p = run(frame, /*na=*/1, /*nb=*/0, /*charge=*/1);
    int rc = check_grad("H2+ UHF/STO-3G", h2p, e_h2p, g_h2p, 2);

    // uhf --geom h2 --charge 0 --spin 0 --grad
    constexpr double e_h2 = -1.116759307396;
    const double g_h2[6] = {0, 0, -0.027679600707, 0, 0, 0.027679600707};
    frame = h2_frame();
    Out h2 = run(frame, 1, 1, 0);
    rc |= check_grad("H2 UHF/STO-3G", h2, e_h2, g_h2, 2);

    // uhf --geom water --charge 1 --spin 1 --grad
    constexpr double e_w = -74.655898713621;
    const double g_w[9] = {0,
                           0,
                           -0.095196667501,
                           0,
                           -0.084875484950,
                           0.047598333750,
                           0,
                           0.084875484950,
                           0.047598333750};
    frame = water_frame();
    Out w = run(frame, 5, 4, 1);
    rc |= check_grad("H2O+ UHF/STO-3G", w, e_w, g_w, 3);
    return rc;
}

} // namespace uhf_case
