/// @file regressions/scf/ecp_i2_grad/reg_ecp_i2_grad.cpp
/// @brief I2 / def2-SVP + def2-ECP RHF nuclear gradient vs the externally
///        pinned PySCF `nuc_grad_method().kernel()` value — the ECP-forces
///        chain link's e2e (hf-parity-01-ecp-forces).
///
/// Out-of-library consumer: public headers + link atomiverse only. The ECP
/// gradient is semi-numerical: every term is analytic except the ECP one-
/// electron block dU/dR, which is central-differenced (a cheap, smooth 1e
/// matrix). This run checks it against a third-party ANALYTIC gradient.
///
/// Checks (REFERENCE.md, PySCF 2.14.0, conv_tol=1e-12, cart=False):
///   1. element-wise |g_analytic − g_PySCF| ≤ 1e-5 Ha/Bohr
///   2. translational invariance |Σ_A g_A| ≤ 1e-8 Ha/Bohr
///
/// Exit non-zero on mismatch, printing got / want / tol. <= 60 s, single CPU.

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    // Pinned geometry (identical to REFERENCE.md and the energy sibling):
    // I(0,0,0); I(0,0,2.666) Å.
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 53;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_type)[1] = 53;
    frame.span(kAtom_z)[1] = 2.666;
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::Def2SVP;
    rhf.ecp = "def2-ECP";
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    const cpu::scf::GradientResult res = rhf.forces(frame);

    // Atomiverse reports FORCE (−dE/dR) in eV/Å; convert to gradient dE/dR
    // in Ha/Bohr to compare with the PySCF analytic gradient.
    auto grad_ha_bohr = [&](int A, int d) {
        return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // PySCF 2.14.0 analytic gradient [Ha/Bohr] (REFERENCE.md). Homonuclear
    // diatomic along z: equal-and-opposite, x/y numerically zero.
    const double want[2][3] = {
        {0.0, 0.0, 5.752712116941e-03},  // I (atom 0)
        {0.0, 0.0, -5.752712116951e-03}, // I (atom 1)
    };
    const double tol = 1e-5;

    double max_err = 0.0;
    double net[3] = {0.0, 0.0, 0.0};
    for (int A = 0; A < 2; ++A) {
        for (int d = 0; d < 3; ++d) {
            const double g = grad_ha_bohr(A, d);
            net[d] += g;
            const double err = std::fabs(g - want[A][d]);
            max_err = std::max(max_err, err);
            std::printf("g[%d][%d] got=% .10e want=% .10e |err|=%.2e\n", A, d,
                        g, want[A][d], err);
        }
    }
    const double net_max =
        std::max({std::fabs(net[0]), std::fabs(net[1]), std::fabs(net[2])});
    std::printf("max_elementwise_err=%.3e tol=%.0e  |Σ_A g_A|_max=%.3e\n",
                max_err, tol, net_max);

    if (!(max_err < tol)) {
        std::printf("FAIL: I2 def2-SVP+ECP RHF gradient off PySCF oracle\n");
        return 1;
    }
    if (!(net_max < 1e-8)) {
        std::printf("FAIL: gradient not translationally invariant\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
