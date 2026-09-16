/// @file regressions/scf/embed_h2o_mm_forces/reg_embed_h2o_mm_forces.cpp
/// @brief H2O RHF/STO-3G embedding forces on MM sites vs offline PySCF pin
///        (esmm-qmmm-01). Public API only; no live PySCF.
///
/// Geometry and charges match `embed_h2o_grad`. Pins are ∂E/∂R_MM [Ha/Bohr]
/// from PySCF 2.14.0 `qmmm.mm_charge` energy FD (see REFERENCE.md). Also
/// checks Σ_QM F + Σ_MM F ≈ 0 with `embed_self_energy=false`.

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

    // PySCF qmmm.mm_charge e_tot FD does NOT include MM–MM Coulomb; match
    // that surface with embed_self_energy=false (QM↔MM seam only).
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.embed_self_energy = false;
    rhf.point_charges = {{0.0, 0.0, 3.0, 0.5}, {0.0, 0.0, -3.0, -0.5}};
    const cpu::scf::GradientResult res = rhf.forces(frame);

    if (res.embedding_forces.shape(0) != 2u
        || res.embedding_forces.shape(1) != 3u) {
        std::printf("FAIL: embedding_forces shape\n");
        return 1;
    }

    auto g_mm = [&](int c, int d) {
        // F = −∂E/∂R [eV/Å] → ∂E/∂R [Ha/Bohr]
        return -units::convert(res.embedding_forces(c, d),
                               units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // PySCF 2.14.0 offline FD pin [Ha/Bohr] (REFERENCE.md).
    const double want[2][3] = {
        {0.0, 0.0, 4.053776692103e-03},  // charge +0.5 at z=+3
        {0.0, 0.0, -3.391311764744e-03}, // charge −0.5 at z=−3
    };
    const double tol = 1e-5;

    double max_err = 0.0;
    for (int c = 0; c < 2; ++c)
        for (int d = 0; d < 3; ++d) {
            const double g = g_mm(c, d);
            const double err = std::fabs(g - want[c][d]);
            max_err = std::max(max_err, err);
            std::printf("g_mm[%d][%d] got=% .12e want=% .12e |err|=%.2e\n", c,
                        d, g, want[c][d], err);
        }
    std::printf("max_elementwise_err=%.3e tol=%.0e\n", max_err, tol);
    if (!(max_err < tol)) {
        std::printf("FAIL: MM embedding gradient off PySCF oracle\n");
        return 1;
    }

    // Third law on the same run (self-energy already off).
    double sum[3] = {0, 0, 0};
    for (int A = 0; A < 3; ++A)
        for (int d = 0; d < 3; ++d)
            sum[d] += res.forces(A, d);
    for (int c = 0; c < 2; ++c)
        for (int d = 0; d < 3; ++d)
            sum[d] += res.embedding_forces(c, d);
    const double net =
        std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
    std::printf("net_force_eVA=%.3e (tol 1e-9)\n", net);
    if (!(net < 1e-9)) {
        std::printf("FAIL: third-law net force\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
