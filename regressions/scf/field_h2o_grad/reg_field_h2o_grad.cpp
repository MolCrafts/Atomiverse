/// @file regressions/scf/field_h2o_grad/reg_field_h2o_grad.cpp
/// @brief H2O RHF/STO-3G external-field forces vs finite difference of the
///        field-SCF energy — second environment-force member
///        (hf-parity-03-environment-forces).
///
/// Out-of-library consumer. The field 1e operator V_field = Σ_a F_a⟨μ|r_a|ν⟩ is
/// folded into Hcore (ECP-U-like), and the classical nuclear field energy into
/// V_nuc. The analytic force adds both explicit derivatives semi-numerically.
/// The field ENERGY is already validated externally (the −dE/dF ≡ μ identity vs
/// PySCF in h2o_embedding_field); this run checks the analytic FORCE is the
/// derivative of that validated energy, so FD of the field-SCF energy is the
/// correctness oracle here (no turnkey PySCF field-gradient exists).
///
/// Check: element-wise |g_analytic − g_FD| ≤ 1e-5 Ha/Bohr.

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {
using namespace ATV;

cpu::Frame water() {
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_z)[0] = 0.1173;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_y)[1] = 0.7572;
    f.span(kAtom_z)[1] = -0.4692;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_y)[2] = -0.7572;
    f.span(kAtom_z)[2] = -0.4692;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

const std::array<double, 3> kField{{0.0, 0.0, 0.01}}; // a.u.

double field_energy(const cpu::Frame& frame) {
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.external_field = kField;
    cpu::Frame f = frame;
    rhf.run(f);
    return rhf.energy(); // Ha
}
} // namespace

int main() {
    using namespace ATV;
    const cpu::Frame frame = water();

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.external_field = kField;
    cpu::Frame ff = frame;
    const cpu::scf::GradientResult res = rhf.forces(ff);

    // Analytic gradient dE/dR [Ha/Bohr] = −force(eV/Å) converted.
    auto grad_ha_bohr = [&](int A, int d) {
        return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // Finite-difference dE/dR from the field-SCF energy [Ha/Bohr].
    const double h_bohr = 1e-3;
    const double h_ang = h_bohr * units::kBohrInAngstrom;
    auto displaced = [&](int A, int d, double dx) {
        cpu::Frame f = frame;
        if (d == 0)
            f.span(kAtom_x)[A] += dx;
        else if (d == 1)
            f.span(kAtom_y)[A] += dx;
        else
            f.span(kAtom_z)[A] += dx;
        return f;
    };

    double max_err = 0.0;
    for (int A = 0; A < 3; ++A)
        for (int d = 0; d < 3; ++d) {
            const double fd = (field_energy(displaced(A, d, h_ang))
                               - field_energy(displaced(A, d, -h_ang)))
                              / (2.0 * h_bohr);
            const double an = grad_ha_bohr(A, d);
            const double err = std::fabs(an - fd);
            max_err = std::max(max_err, err);
            std::printf("g[%d][%d] analytic=% .8e fd=% .8e |err|=%.2e\n", A, d,
                        an, fd, err);
        }
    std::printf("max_elementwise_err=%.3e tol=1e-05\n", max_err);
    if (!(max_err < 1e-5)) {
        std::printf("FAIL: field force != FD of field-SCF energy\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
