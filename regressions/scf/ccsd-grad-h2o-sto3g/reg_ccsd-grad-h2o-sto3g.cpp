/// @file regressions/scf/ccsd-grad-h2o-sto3g/reg_ccsd-grad-h2o-sto3g.cpp
/// @brief H2O / STO-3G RHF-CCSD analytic nuclear gradient vs offline PySCF pin.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Checks (REFERENCE.md, PySCF 2.14.0, conv_tol=1e-12):
///   1. element-wise |g_analytic − g_PySCF| ≤ 1e-5 Ha/Bohr
///   2. translational invariance |Σ_A g_A| ≤ 1e-8 Ha/Bohr
///
/// Exit non-zero on mismatch. <= 60 s, single CPU.

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    const cpu::scf::GradientResult res = rhf.ccsd_forces(frame);

    auto grad_ha_bohr = [&](int A, int d) {
        return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // PySCF 2.14.0 CCSD analytic gradient [Ha/Bohr] (REFERENCE.md).
    const double want[3][3] = {
        {0.0, 0.0, -1.106969560091e-01},
        {0.0, -4.698911449636e-02, 5.534847800455e-02},
        {0.0, 4.698911449636e-02, 5.534847800455e-02},
    };
    const double tol = 1e-5;

    double max_err = 0.0;
    double net[3] = {0.0, 0.0, 0.0};
    for (int A = 0; A < 3; ++A) {
        for (int d = 0; d < 3; ++d) {
            const double g = grad_ha_bohr(A, d);
            net[d] += g;
            const double err = std::fabs(g - want[A][d]);
            if (err > max_err)
                max_err = err;
            std::printf("g[%d][%d] got=% .12e want=% .12e |err|=%.2e\n", A, d,
                        g, want[A][d], err);
        }
    }
    const double net_max =
        std::max({std::fabs(net[0]), std::fabs(net[1]), std::fabs(net[2])});
    std::printf("max_elementwise_err=%.3e tol=%.0e  |Σ_A g_A|_max=%.3e\n",
                max_err, tol, net_max);
    std::printf("E_CCSD = %.12f eV\n", res.energy);

    if (!(max_err < tol)) {
        std::printf("FAIL: H2O CCSD gradient off PySCF oracle\n");
        return 1;
    }
    if (!(net_max < 1e-8)) {
        std::printf("FAIL: gradient not translationally invariant\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
