/// @file regressions/scf/ump2-grad-o2-sto3g/reg_ump2-grad-o2-sto3g.cpp
/// @brief O2 triplet UMP2/STO-3G analytic nuclear gradient vs offline PySCF pin.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// ROHF density warm-start → UHF (same multi-minima path as ump2-o2-sto3g)
/// then `UHF::mp2_forces`. Checks (REFERENCE.md, PySCF 2.14.0, conv_tol=1e-12):
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

namespace {

ATV::cpu::Frame o2_frame() {
    ATV::cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_type)[i] = 8;
        frame.span(ATV::kAtom_mass)[i] = 15.999;
        frame.span(ATV::kAtom_x)[i] = 0.0;
        frame.span(ATV::kAtom_y)[i] = 0.0;
    }
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 1.2075;
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = o2_frame();

    // Cold UHF often lands on a higher local solution for O2; warm-start from
    // ROHF densities to reach the PySCF UHF minimum used for the UMP2 pin.
    cpu::scf::ROHF rohf(9, 7);
    rohf.basis = cpu::scf::BasisSet::STO3G;
    rohf.tol = 1e-12;
    rohf.max_iter = 200;
    rohf.run(frame);
    if (!rohf.converged()) {
        std::printf("FAIL: ROHF warm-start did not converge\n");
        return 1;
    }

    using Spin = cpu::scf::SlaterDeterminant::Spin;
    cpu::scf::SCFInitialGuess guess;
    guess.P_alpha = rohf.determinant().density(Spin::alpha);
    guess.P_beta = rohf.determinant().density(Spin::beta);
    guess.has_alpha = true;
    guess.has_beta = true;

    cpu::scf::UHF uhf(9, 7);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.tol = 1e-12;
    uhf.max_iter = 200;

    const cpu::scf::GradientResult res = uhf.mp2_forces(frame, &guess);

    auto grad_ha_bohr = [&](int A, int d) {
        return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // PySCF 2.14.0 UMP2 analytic gradient [Ha/Bohr] (REFERENCE.md).
    const double want[2][3] = {
        {0.0, 0.0, 1.665132663664e-01},
        {0.0, 0.0, -1.665132663664e-01},
    };
    const double tol = 1e-5;

    double max_err = 0.0;
    double net[3] = {0.0, 0.0, 0.0};
    for (int A = 0; A < 2; ++A) {
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
    std::printf("E_UMP2 = %.12f eV\n", res.energy);

    if (!uhf.converged()) {
        std::printf("FAIL: UHF reference did not converge\n");
        return 1;
    }
    if (!(max_err < tol)) {
        std::printf("FAIL: O2 UMP2 gradient off PySCF oracle\n");
        return 1;
    }
    if (!(net_max < 1e-8)) {
        std::printf("FAIL: gradient not translationally invariant\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
