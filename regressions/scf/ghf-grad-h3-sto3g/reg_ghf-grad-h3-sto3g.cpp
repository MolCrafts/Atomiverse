/// @file regressions/scf/ghf-grad-h3-sto3g/reg_ghf-grad-h3-sto3g.cpp
/// @brief Equilateral H3/STO-3G GHF analytic nuclear gradient vs pinned
///        PySCF-integral oracle (REFERENCE.md) — noncollinear spin-frustrated
///        surface, not the collinear UHF saddle.
///
/// Out-of-library consumer. Checks (PySCF 2.14.0, cart, 120° Néel guess):
///   1. element-wise |g − g_pin| ≤ 1e-5 Ha/Bohr
///   2. translational invariance |Σ_A g_A| ≤ 1e-8 Ha/Bohr
///   3. energy still matches the energy-sibling pin (1e-8 Ha)

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

// Equilateral triangle, side 1.0 Å, xy-plane — same as h3_ghf_sto3g.
ATV::cpu::Frame h3_frame() {
    ATV::cpu::Frame frame(3);
    const double hx[3] = {0.0, 1.0, 0.5};
    const double hy[3] = {0.0, 0.0, 0.8660254037844386};
    for (int i = 0; i < 3; ++i) {
        frame.span(ATV::kAtom_type)[i] = 1;
        frame.span(ATV::kAtom_mass)[i] = 1.008;
        frame.span(ATV::kAtom_x)[i] = hx[i];
        frame.span(ATV::kAtom_y)[i] = hy[i];
        frame.span(ATV::kAtom_z)[i] = 0.0;
    }
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = h3_frame();
    cpu::scf::GHF ghf(3);
    ghf.basis = cpu::scf::BasisSet::STO3G;
    ghf.tol = 1e-12;
    ghf.max_iter = 200;
    ghf.guess_spin_rotation = 2.0 * M_PI / 3.0;

    const cpu::scf::GradientResult res = ghf.forces(frame);

    const double pin_e = -1.34044034354119; // Ha — energy sibling
    const double e_ha =
        units::convert(res.energy, units::Energy::eV, units::Energy::Hartree);
    std::printf("E_GHF = %.12f Ha (pin %.12f, dE = %.3e)\n", e_ha, pin_e,
                e_ha - pin_e);

    auto grad_ha_bohr = [&](int A, int d) {
        return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // PySCF-integral analytic GHF gradient [Ha/Bohr] (REFERENCE.md).
    const double want[3][3] = {
        {9.473114183e-02, 5.469305020e-02, 0.0},
        {-9.473114183e-02, 5.469305020e-02, 0.0},
        {0.0, -1.093861004e-01, 0.0},
    };
    const double tol = 1e-5;

    double max_err = 0.0, net[3] = {0, 0, 0};
    for (int A = 0; A < 3; ++A)
        for (int d = 0; d < 3; ++d) {
            const double g = grad_ha_bohr(A, d);
            net[d] += g;
            const double err = std::fabs(g - want[A][d]);
            max_err = std::max(max_err, err);
            std::printf("g[%d][%d] got=% .10e want=% .10e |err|=%.2e\n", A, d,
                        g, want[A][d], err);
        }
    const double net_max =
        std::max({std::fabs(net[0]), std::fabs(net[1]), std::fabs(net[2])});
    std::printf("max_elementwise_err=%.3e tol=%.0e  |Σ_A g_A|_max=%.3e\n",
                max_err, tol, net_max);

    if (!(std::fabs(e_ha - pin_e) <= 1e-8)) {
        std::printf("FAIL: GHF energy off energy-sibling pin\n");
        return 1;
    }
    if (!(max_err < tol)) {
        std::printf("FAIL: H3 GHF gradient off PySCF-integral oracle\n");
        return 1;
    }
    if (!(net_max < 1e-8)) {
        std::printf("FAIL: gradient not translationally invariant\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
