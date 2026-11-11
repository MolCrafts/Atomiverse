/// @file tests/cpu/scf/test_x2c_hcore_grad.cpp
/// @brief Unit: dh^X2C/dR matches a central difference of X2cWorkspace::hcore.
///
/// No SCF. H2/STO-3G, one Cartesian displacement of the second proton.
/// The analytic tensor is `internal::x2c_hcore_gradient`
/// (pyscf.x2c.sfx2c1e_grad.gen_sf_hfw).

#include <cmath>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/internal/integral_derivatives.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/x2c.h"

using namespace ATV;

namespace {

cpu::scf::internal::IntegralWorkspace pack_at(double z_h_ang) {
    cpu::Frame f(2);
    f.span(kAtom_atomic_number)[0] = 1;
    f.span(kAtom_atomic_number)[1] = 1;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = z_h_ang;
    f.box = cpu::Box::cubic(30.0);
    return cpu::scf::internal::IntegralWorkspace{f,
                                                 cpu::scf::BasisSet::STO3G,
                                                 0,
                                                 /*compute_eri=*/false,
                                                 {},
                                                 {},
                                                 /*compute_pvp=*/true};
}

} // namespace

TEST(X2cHcoreGrad, MatchesCentralDifferenceOnH2) {
    constexpr double z0 = 0.74;
    constexpr double delta_bohr = 1e-5;
    const double delta_ang = delta_bohr * units::kBohrInAngstrom;

    const auto p0 = pack_at(z0);
    ASSERT_GT(p0.W.size(), 0u);
    const auto view = p0.view();
    const cpu::scf::IntegralDerivPack dpack{view};
    const auto dW = cpu::scf::pvp_nuclear_deriv(view);
    const auto dh = cpu::scf::internal::x2c_hcore_gradient(
        p0.S, p0.T, p0.V, p0.W, dpack.dS, dpack.dT, dpack.dV, dW);

    const auto pp = pack_at(z0 + delta_ang);
    const auto pm = pack_at(z0 - delta_ang);
    const cpu::scf::internal::X2cWorkspace wp{
        cpu::scf::RealMatrix(pp.S), cpu::scf::RealMatrix(pp.T),
        cpu::scf::RealMatrix(pp.V), cpu::scf::RealMatrix(pp.W)};
    const cpu::scf::internal::X2cWorkspace wm{
        cpu::scf::RealMatrix(pm.S), cpu::scf::RealMatrix(pm.T),
        cpu::scf::RealMatrix(pm.V), cpu::scf::RealMatrix(pm.W)};

    const int n = p0.n_basis;
    ASSERT_EQ(static_cast<int>(dh.shape(0)), 2);
    ASSERT_EQ(static_cast<int>(dh.shape(2)), n);
    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const double fd =
                (wp.hcore()(i, j) - wm.hcore()(i, j)) / (2.0 * delta_bohr);
            const double err = std::abs(fd - dh(1, 2, i, j));
            if (err > max_err)
                max_err = err;
            // Translational invariance of the core operator along z.
            const double net = dh(0, 2, i, j) + dh(1, 2, i, j);
            EXPECT_LT(std::abs(net), 1e-8) << "i=" << i << " j=" << j;
        }
    }
    EXPECT_LT(max_err, 5e-6) << "max |analytic − FD| on dh^X2C/dZ_H";
}
