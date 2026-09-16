/// @file regressions/scf/embed_h2o_grad/reg_embed_h2o_grad.cpp
/// @brief H2O RHF/STO-3G point-charge-embedding (QM/MM) forces vs pinned PySCF
///        `qmmm.mm_charge` analytic gradient — first environment-force member
///        (hf-parity-03-environment-forces).
///
/// Out-of-library consumer. The embed force = the standard SCF gradient (on the
/// embedded density) + Tr[P·dV_embed/dR] (semi-numerical, ECP pattern) +
/// analytic nucleus–MM Coulomb. NOTE: fixed MM charges exert a NET force on the
/// QM system, so Σ_QM ≠ 0 (no translational invariance to assert).
///
/// Checks (REFERENCE.md, PySCF 2.14.0, conv_tol=1e-12, cart):
///   element-wise |g_QM − g_PySCF| ≤ 1e-5 Ha/Bohr.

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
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.point_charges = {{0.0, 0.0, 3.0, 0.5}, {0.0, 0.0, -3.0, -0.5}};
    const cpu::scf::GradientResult res = rhf.forces(frame);

    auto grad_ha_bohr = [&](int A, int d) {
        return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // PySCF 2.14.0 qmmm.mm_charge analytic gradient on QM atoms [Ha/Bohr].
    const double want[3][3] = {
        {0.0, 0.0, -6.3053441384e-02},              // O
        {0.0, -1.7882928801e-02, 3.1195488026e-02}, // H
        {0.0, 1.7882928801e-02, 3.1195488026e-02},  // H
    };
    const double tol = 1e-5;

    double max_err = 0.0;
    for (int A = 0; A < 3; ++A)
        for (int d = 0; d < 3; ++d) {
            const double g = grad_ha_bohr(A, d);
            const double err = std::fabs(g - want[A][d]);
            max_err = std::max(max_err, err);
            std::printf("g[%d][%d] got=% .10e want=% .10e |err|=%.2e\n", A, d,
                        g, want[A][d], err);
        }
    std::printf("max_elementwise_err=%.3e tol=%.0e\n", max_err, tol);

    if (!(max_err < tol)) {
        std::printf("FAIL: embed (QM/MM) gradient off PySCF oracle\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
