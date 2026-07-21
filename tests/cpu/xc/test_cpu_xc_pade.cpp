/// @file tests/cpu/xc/test_cpu_xc_pade.cpp
/// @brief Validation of the native GTH Padé LDA functional (spec X2).
///
/// PADE is CP2K-native (not in libxc), so there is no libxc oracle. We validate
/// internally: (1) the potential is the analytic derivative of the energy
/// (finite difference), (2) unpolarized symmetry, (3) the energy matches the
/// Padé [3/4] formula evaluated independently. Full bit-parity against CP2K is
/// checked later at the GPW M1 gate (Ar.inp / H2.inp).

#include <cmath>

#include <gtest/gtest.h>

#include "atomiverse/cpu/xc/pade.h"

using ATV::cpu::xc::pade_lsd;
using ATV::cpu::xc::PadeXC;

namespace {

// E_xc density [Hartree/Bohr^3] = rho_total * exc.
double energy_density(double ra, double rb) {
    return (ra + rb) * pade_lsd(ra, rb).exc;
}

struct Pt {
    double ra, rb;
    const char* label;
};

constexpr Pt kPoints[] = {
    {0.5, 0.5, "unpolarized rho=1"}, {0.8, 0.2, "polarized 0.8/0.2"},
    {1.0, 0.05, "highly polarized"}, {1.0e-3, 5.0e-4, "low density"},
    {50.0, 30.0, "high density"},
};

} // namespace

// (1) Potential == d(rho*exc)/drho_sigma via central finite difference.
TEST(PadeLDA, PotentialIsEnergyDerivative) {
    for (const auto& pt : kPoints) {
        const double scale = pt.ra + pt.rb;
        const double h = 1.0e-6 * scale;

        const double dEa = (energy_density(pt.ra + h, pt.rb)
                            - energy_density(pt.ra - h, pt.rb))
                           / (2.0 * h);
        const double dEb = (energy_density(pt.ra, pt.rb + h)
                            - energy_density(pt.ra, pt.rb - h))
                           / (2.0 * h);

        PadeXC xc = pade_lsd(pt.ra, pt.rb);
        EXPECT_NEAR(xc.vrho_a, dEa, 1e-5 * std::max(1.0, std::abs(dEa)))
            << "vrho_a vs FD at " << pt.label;
        EXPECT_NEAR(xc.vrho_b, dEb, 1e-5 * std::max(1.0, std::abs(dEb)))
            << "vrho_b vs FD at " << pt.label;
    }
}

// (2) Unpolarized: equal spins give equal potentials.
TEST(PadeLDA, UnpolarizedSymmetry) {
    for (double rho : {0.1, 1.0, 10.0}) {
        PadeXC xc = pade_lsd(0.5 * rho, 0.5 * rho);
        EXPECT_DOUBLE_EQ(xc.vrho_a, xc.vrho_b);
    }
    // Spin swap leaves exc invariant and swaps the potentials.
    PadeXC ab = pade_lsd(0.8, 0.2);
    PadeXC ba = pade_lsd(0.2, 0.8);
    EXPECT_DOUBLE_EQ(ab.exc, ba.exc);
    EXPECT_DOUBLE_EQ(ab.vrho_a, ba.vrho_b);
    EXPECT_DOUBLE_EQ(ab.vrho_b, ba.vrho_a);
}

// (3) Energy matches the Padé [3/4] formula evaluated independently.
TEST(PadeLDA, EnergyMatchesFormula) {
    // Unpolarized rho=1 => x=0 => paramagnetic coefficients only.
    const double rs = std::cbrt(3.0 / (4.0 * M_PI)); // rho_total = 1
    const double a0 = 0.4581652932831429, a1 = 0.2217058676663745e+1,
                 a2 = 0.7405551735357053, a3 = 0.1968227878617998e-1;
    const double b1 = 1.0, b2 = 0.4504130959426697e+1,
                 b3 = 0.1110667363742916e+1, b4 = 0.2359291751427506e-1;
    const double p = a0 + (a1 + (a2 + a3 * rs) * rs) * rs;
    const double q = (b1 + (b2 + (b3 + b4 * rs) * rs) * rs) * rs;
    const double expected = -p / q;

    EXPECT_NEAR(pade_lsd(0.5, 0.5).exc, expected, 1e-12);
    // Coarse literal guard so a shared constant error is still caught.
    EXPECT_NEAR(pade_lsd(0.5, 0.5).exc, -0.8097, 2e-3);
}

// Below the density cutoff everything is zero.
TEST(PadeLDA, DensityCutoff) {
    PadeXC xc = pade_lsd(1e-14, 1e-14);
    EXPECT_DOUBLE_EQ(xc.exc, 0.0);
    EXPECT_DOUBLE_EQ(xc.vrho_a, 0.0);
    EXPECT_DOUBLE_EQ(xc.vrho_b, 0.0);
}
