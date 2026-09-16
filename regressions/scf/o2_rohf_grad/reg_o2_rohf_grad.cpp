/// @file regressions/scf/o2_rohf_grad/reg_o2_rohf_grad.cpp
/// @brief O2 triplet ROHF/STO-3G nuclear gradient vs pinned PySCF analytic
///        gradient — the ROHF-forces chain link's e2e (hf-parity-02).
///
/// Out-of-library consumer. ROHF reuses the unrestricted J−K derivative
/// contraction (shared-spatial-orbital α/β densities); the energy-weighted
/// density is the correct ROHF Lagrangian W = O_a·F_a·O_a + O_b·F_b·O_b
/// (PySCF grad/rohf.py make_rdm1e). This run checks the full ROHF gradient
/// against a third-party analytic gradient.
///
/// Checks (REFERENCE.md, PySCF 2.14.0, conv_tol=1e-12, cart):
///   1. element-wise |g − g_PySCF| ≤ 1e-5 Ha/Bohr
///   2. translational invariance |Σ_A g_A| ≤ 1e-8 Ha/Bohr

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 8;
        frame.span(kAtom_mass)[i] = 15.999;
    }
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_z)[1] = 1.2075;
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::ROHF rohf(9, 7); // O2 triplet
    rohf.basis = cpu::scf::BasisSet::STO3G;
    rohf.tol = 1e-12;
    rohf.max_iter = 200;
    const cpu::scf::GradientResult res = rohf.forces(frame);

    auto grad_ha_bohr = [&](int A, int d) {
        return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // PySCF 2.14.0 analytic ROHF gradient [Ha/Bohr] (REFERENCE.md).
    const double want[2][3] = {
        {0.0, 0.0, 2.607023240746e-02},
        {0.0, 0.0, -2.607023240746e-02},
    };
    const double tol = 1e-5;

    double max_err = 0.0, net[3] = {0, 0, 0};
    for (int A = 0; A < 2; ++A)
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

    if (!(max_err < tol)) {
        std::printf("FAIL: O2 ROHF gradient off PySCF oracle\n");
        return 1;
    }
    if (!(net_max < 1e-8)) {
        std::printf("FAIL: gradient not translationally invariant\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
