/// @file regressions/scf/x2c_hcl_grad/reg_x2c_hcl_grad.cpp
/// @brief HCl / 6-31G RHF x2c-1e analytic gradient vs PySCF sfx2c1e_grad.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Oracle: tools/pyscf_oracle/pin_scf.py rhf --geom hcl --basis 6-31g
/// --grad --x2c (PySCF 2.14.0, xuncontract=False). See REFERENCE.md.
///
/// Exit non-zero on mismatch. <= 60 s.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"

namespace atv_reg_scf_x2c_hcl_grad {

int run() {
    using namespace ATV;

    // Cl(0,0,0), H(0,0,1.35) Å. Stretched; force is along z.
    cpu::Frame frame(2);
    frame.span(kAtom_atomic_number)[0] = 17;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_atomic_number)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 1.35;
    frame.box = cpu::Box::cubic(30.0);

    cpu::scf::RHF rhf{cpu::scf::BasisSet::Pople631G, 0, 1e-12, 200};
    rhf.x2c = true;
    const cpu::scf::GradientResult res = rhf.forces(frame);
    if (!rhf.converged()) {
        std::printf("FAIL: x2c RHF forces did not converge\n");
        return 1;
    }

    const double e_ha =
        units::convert(res.energy, units::Energy::eV, units::Energy::Hartree);
    const double e_want = -461.49265438490323;
    std::printf("E got=%.12f want=%.12f |err|=%.2e Ha\n", e_ha, e_want,
                std::fabs(e_ha - e_want));
    if (!(std::fabs(e_ha - e_want) < 1e-6)) {
        std::printf("FAIL: x2c energy off PySCF oracle\n");
        return 1;
    }

    auto grad_ha_bohr = [&](int A, int d) {
        return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // PySCF grad_ha_bohr. Transverse entries are ~1e-12; pinned as 0.
    const double want[2][3] = {
        {0.0, 0.0, -0.028498915583206497},
        {0.0, 0.0, 0.028498915583214046},
    };
    const double tol = 1e-6;
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
    std::printf("max_elementwise_err=%.3e tol=%.0e  |Σ g|_max=%.3e\n", max_err,
                tol, net_max);
    if (!(max_err < tol)) {
        std::printf("FAIL: x2c gradient off PySCF oracle\n");
        return 1;
    }
    if (!(net_max < 1e-8)) {
        std::printf("FAIL: gradient not translationally invariant\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

} // namespace atv_reg_scf_x2c_hcl_grad

ATV_REGRESSION("scf/relativistic", "x2c_hcl_grad",
               atv_reg_scf_x2c_hcl_grad::run);
