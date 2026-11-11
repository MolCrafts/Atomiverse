/// @file regressions/scf/hf-x2c-forces/reg_hf-x2c-forces.cpp
/// @brief H2/STO-3G and H2O/6-31G RHF x2c-1e analytic forces vs PySCF.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Oracle: tools/pyscf_oracle/pin_scf.py rhf --grad --x2c (PySCF 2.14.0,
/// xuncontract=False). See REFERENCE.md. An atom is not a force pin.
///
/// Exit non-zero on mismatch. <= 60 s.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"

namespace atv_reg_scf_hf_x2c_forces {

namespace {

struct Case {
    const char* label;
    ATV::cpu::scf::BasisSet basis;
    int n;
    int Z[3];
    double xyz[3][3];
    double box;
    double e_want;
    double g_want[3][3];
};

int check(const Case& c) {
    using namespace ATV;
    cpu::Frame frame(static_cast<std::size_t>(c.n));
    for (int A = 0; A < c.n; ++A) {
        frame.span(kAtom_atomic_number)[A] = c.Z[A];
        frame.span(kAtom_x)[A] = c.xyz[A][0];
        frame.span(kAtom_y)[A] = c.xyz[A][1];
        frame.span(kAtom_z)[A] = c.xyz[A][2];
    }
    frame.box = cpu::Box::cubic(c.box);

    cpu::scf::RHF rhf{c.basis, 0, 1e-12, 200};
    rhf.x2c = true;
    const cpu::scf::GradientResult res = rhf.forces(frame);
    if (!rhf.converged()) {
        std::printf("FAIL: %s x2c RHF forces did not converge\n", c.label);
        return 1;
    }

    const double e_ha =
        units::convert(res.energy, units::Energy::eV, units::Energy::Hartree);
    const double e_err = std::fabs(e_ha - c.e_want);
    std::printf("%s E got=%.12f want=%.12f |err|=%.2e Ha\n", c.label, e_ha,
                c.e_want, e_err);
    int fail = 0;
    if (!(e_err < 1e-6)) {
        std::printf("FAIL: %s x2c energy off PySCF oracle\n", c.label);
        fail = 1;
    }

    auto grad_ha_bohr = [&](int A, int d) {
        return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    const double tol = 1e-6;
    double max_err = 0.0;
    double max_abs = 0.0;
    double net[3] = {0.0, 0.0, 0.0};
    for (int A = 0; A < c.n; ++A) {
        for (int d = 0; d < 3; ++d) {
            const double g = grad_ha_bohr(A, d);
            net[d] += g;
            const double err = std::fabs(g - c.g_want[A][d]);
            if (err > max_err)
                max_err = err;
            if (std::fabs(g) > max_abs)
                max_abs = std::fabs(g);
            std::printf("%s g[%d][%d] got=% .12e want=% .12e |err|=%.2e\n",
                        c.label, A, d, g, c.g_want[A][d], err);
        }
    }
    const double net_max =
        std::max({std::fabs(net[0]), std::fabs(net[1]), std::fabs(net[2])});
    std::printf("%s max_elementwise_err=%.3e tol=%.0e  |g|_max=%.3e  "
                "|Σ g|_max=%.3e\n",
                c.label, max_err, tol, max_abs, net_max);
    if (!(max_abs > 1e-3)) {
        std::printf("FAIL: %s nuclear force is ~0 (not a force pin)\n",
                    c.label);
        fail = 1;
    }
    if (!(max_err < tol)) {
        std::printf("FAIL: %s x2c gradient off PySCF oracle\n", c.label);
        fail = 1;
    }
    if (!(net_max < 1e-8)) {
        std::printf("FAIL: %s gradient not translationally invariant\n",
                    c.label);
        fail = 1;
    }
    return fail;
}

} // namespace

int run() {
    const Case h2{
        "H2/STO-3G",
        ATV::cpu::scf::BasisSet::STO3G,
        2,
        {1, 1, 0},
        {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}, {0.0, 0.0, 0.0}},
        30.0,
        -1.1167713747002186,
        {{0.0, 0.0, -0.02767894667735915},
         {0.0, 0.0, 0.027678946677359373},
         {0.0, 0.0, 0.0}},
    };
    const Case h2o{
        "H2O/6-31G",
        ATV::cpu::scf::BasisSet::Pople631G,
        3,
        {8, 1, 1},
        {{0.0, 0.0, 0.1173}, {0.0, 0.7572, -0.4692}, {0.0, -0.7572, -0.4692}},
        20.0,
        -76.03879282908161,
        {{0.0, 0.0, 0.02372121511333436},
         {0.0, -0.004279885030643804, -0.011860607556359426},
         {0.0, 0.004279885031505337, -0.011860607556974934}},
    };
    const int fail = check(h2) | check(h2o);
    if (fail) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

} // namespace atv_reg_scf_hf_x2c_forces

ATV_REGRESSION("scf/relativistic", "hf-x2c-forces",
               atv_reg_scf_hf_x2c_forces::run);
