/// @file regressions/scf/uhf/cuda-uhf.h
/// @brief Shared PySCF pins for O2 and OH STO-3G UHF energy + gradient.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"

namespace cuda_uhf_case {

struct Out {
    double energy_ha = 0;
    bool converged = false;
    xt::xtensor<double, 2> forces;
};

inline ATV::cpu::Frame o2_frame() {
    using namespace ATV;
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_type)[1] = 8;
    frame.span(kAtom_x)[0] = 0;
    frame.span(kAtom_y)[0] = 0;
    frame.span(kAtom_z)[0] = 0;
    frame.span(kAtom_x)[1] = 0;
    frame.span(kAtom_y)[1] = 0;
    frame.span(kAtom_z)[1] = 1.2075;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

inline ATV::cpu::Frame oh_frame() {
    using namespace ATV;
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[0] = 0;
    frame.span(kAtom_y)[0] = 0;
    frame.span(kAtom_z)[0] = 0;
    frame.span(kAtom_x)[1] = 0;
    frame.span(kAtom_y)[1] = 0;
    frame.span(kAtom_z)[1] = 0.96966;
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
    // PySCF 2.14.0 pin_scf.py uhf --geom o2 --basis sto-3g --spin 2 --grad
    constexpr double e_o2 = -147.633946785450;
    const double g_o2[6] = {0, 0, 0.024814046022, 0, 0, -0.024814046022};
    auto frame = o2_frame();
    Out o2 = run(frame, /*na=*/9, /*nb=*/7, /*charge=*/0);
    int rc = check_grad("O2 UHF/STO-3G", o2, e_o2, g_o2, 2);

    // uhf --geom oh --spin 1 --grad ; r(OH) = 0.96966 A
    constexpr double e_oh = -74.362633276842;
    const double g_oh[6] = {0, 0, 0.056145481913, 0, 0, -0.056145481913};
    frame = oh_frame();
    Out oh = run(frame, /*na=*/5, /*nb=*/4, /*charge=*/0);
    rc |= check_grad("OH UHF/STO-3G", oh, e_oh, g_oh, 2);
    return rc;
}

} // namespace cuda_uhf_case
