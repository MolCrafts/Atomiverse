/// @file test_ewald.cpp
#include "atomiverse/core/units.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace ATV;
using namespace ATV::cpu::pw::internal;

TEST(Ewald, SiTwoAtoms) {
    cpu::Box box = cpu::Box::cubic(10.26); // Å (Si-scale cell)
    std::vector<Vec3> pos = {{0, 0, 0}, {2.565, 2.565, 2.565}};
    std::vector<AccReal> charges = {4, 4};
    AccReal ecut = AccReal(15); // Hartree

    EwaldSummation e1(pos, charges, box, ecut);
    EwaldSummation e2(pos, charges, box, ecut * AccReal(2));
    AccReal diff = std::abs(e1.total_energy() - e2.total_energy());
    EXPECT_LT(static_cast<double>(diff), 1e-4); // Hartree
}

TEST(Ewald, ZeroCharges) {
    cpu::Box box = cpu::Box::cubic(10);
    std::vector<Vec3> pos = {{0, 0, 0}, {2, 2, 2}};
    std::vector<AccReal> charges = {0, 0};
    EwaldSummation e(pos, charges, box, AccReal(15));
    EXPECT_NEAR(static_cast<double>(e.total_energy()), 0.0, 1e-12);
}

TEST(Ewald, ForceMatchesEnergyFd) {
    // Analytic Ewald force vs central FD of total_energy (same functional).
    cpu::Box box = cpu::Box::cubic(8.0); // Å
    std::vector<Vec3> pos = {{0.0, 0.0, 0.0}, {1.5, 0.2, -0.1}};
    std::vector<AccReal> charges = {1.0, 1.0};
    AccReal ecut = AccReal(10.0);
    const AccReal h = AccReal(1e-4); // Å

    EwaldSummation e0(pos, charges, box, ecut);
    ASSERT_EQ(e0.forces_bohr().size(), 2u);

    const double to_ev_a = units::convert(1.0, units::Force::HartreePerBohr,
                                          units::Force::eVPerAngstrom);
    // FD of energy [Ha] w.r.t. R [Å]: F_Ha/Å = -dE/dR_Å;
    // convert to Ha/Bohr: F_Ha/Bohr = F_Ha/Å / a2b
    const double a2b = 1.8897259886;

    for (int ia = 0; ia < 2; ++ia) {
        for (int k = 0; k < 3; ++k) {
            auto pp = pos;
            auto pm = pos;
            pp[static_cast<std::size_t>(ia)][k] += h;
            pm[static_cast<std::size_t>(ia)][k] -= h;
            const double ep = static_cast<double>(
                EwaldSummation(pp, charges, box, ecut).total_energy());
            const double em = static_cast<double>(
                EwaldSummation(pm, charges, box, ecut).total_energy());
            const double f_fd_ha_per_ang =
                -(ep - em) / (2.0 * static_cast<double>(h));
            const double f_fd_ha_bohr = f_fd_ha_per_ang / a2b;
            const double f_an = static_cast<double>(
                e0.forces_bohr()[static_cast<std::size_t>(ia)][k]);
            EXPECT_NEAR(f_an, f_fd_ha_bohr, 5e-4)
                << " atom=" << ia << " axis=" << k << " analytic=" << f_an
                << " fd=" << f_fd_ha_bohr << " (eV/Å analytic would be "
                << f_an * to_ev_a << ")";
        }
    }
}
