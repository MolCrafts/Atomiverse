/// @file tests/cpu/scf/test_periodic_exchange_lattice.cpp
/// @brief Unit tests for Γ-point truncated lattice-summed exchange.

#if defined(ATV_HAS_HF)

#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/periodic_hf.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/native_integrals.h"

using namespace ATV::cpu::scf;

namespace {

native::ShellBlock s_shell(const double* alpha, const double* coeff,
                           const double* xyz, int ao) {
    native::ShellBlock s{};
    s.n_prim = 1;
    s.alpha = alpha;
    s.coeff = coeff;
    s.center[0] = xyz[0];
    s.center[1] = xyz[1];
    s.center[2] = xyz[2];
    s.l = 0;
    s.n_cart = 1;
    s.ao_start = ao;
    return s;
}

} // namespace

TEST(TestPeriodicExchangeLattice, ExtraShellDoesNotChangeK) {
    // Compact α = 2.0 s primitives. At L = 6 Bohr the |T| = 18 Bohr
    // (m = 3) images are already ~exp(−2·18²) and must not move K.
    const double alpha = 2.0;
    const double coeff = native::phys_norm(2.0);
    const double xyz0[3] = {0.0, 0.0, 0.0};
    const double xyz1[3] = {1.4, 0.0, 0.0};
    const auto A = s_shell(&alpha, &coeff, xyz0, 0);
    const auto B = s_shell(&alpha, &coeff, xyz1, 1);
    const std::vector<native::ShellBlock> blocks{A, B};

    RealMatrix P({2, 2});
    P.fill(0.0);
    P(0, 0) = 1.0;
    P(1, 1) = 1.0;
    P(0, 1) = 0.2;
    P(1, 0) = 0.2;

    internal::LatticeSpec s2;
    s2.L[0] = s2.L[1] = s2.L[2] = 6.0;
    s2.m[0] = s2.m[1] = s2.m[2] = 2;
    internal::LatticeSpec s3 = s2;
    s3.m[0] = s3.m[1] = s3.m[2] = 3;

    RealMatrix K2, K3;
    // erfc(ωr)/r decays; bare 1/r does not (needs Ewald).
    constexpr double kOmega = 1.0;
    internal::exchange_lattice(blocks, P, s2, K2, kOmega);
    internal::exchange_lattice(blocks, P, s3, K3, kOmega);

    double worst = 0.0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            worst = std::max(worst, std::abs(K2(i, j) - K3(i, j)));
    EXPECT_LT(worst, 1e-8) << "K2(0,0)=" << K2(0, 0) << " K3(0,0)=" << K3(0, 0)
                           << " K2(0,1)=" << K2(0, 1)
                           << " K3(0,1)=" << K3(0, 1);
}

TEST(TestPeriodicExchangeLattice, MolecularWindowIsTZeroOnly) {
    const double L[3] = {40.0, 40.0, 40.0};
    const auto spec = internal::lattice_spec(L, /*rcut=*/12.0);
    EXPECT_FALSE(spec.has_extra());
    EXPECT_EQ(spec.m[0], 0);
}

TEST(TestPeriodicExchangeLattice, KIsHermitian) {
    const double alpha = 1.0;
    const double coeff = native::phys_norm(1.0);
    const double xyz0[3] = {0.0, 0.0, 0.0};
    const double xyz1[3] = {1.4, 0.0, 0.0};
    const std::vector<native::ShellBlock> blocks{
        s_shell(&alpha, &coeff, xyz0, 0), s_shell(&alpha, &coeff, xyz1, 1)};
    RealMatrix P({2, 2});
    P.fill(0.0);
    P(0, 0) = P(1, 1) = 1.0;
    internal::LatticeSpec spec;
    spec.L[0] = spec.L[1] = spec.L[2] = 4.0;
    spec.m[0] = spec.m[1] = spec.m[2] = 2;
    RealMatrix K;
    internal::exchange_lattice(blocks, P, spec, K);
    EXPECT_NEAR(K(0, 1), K(1, 0), 1e-12);
}

TEST(TestPeriodicExchangeLattice, OverlapImagesAreHermitianAndSpd) {
    using namespace ATV;
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.74;
    f.box = cpu::Box::cubic(2.0);

    internal::IntegralWorkspace ws{f, BasisSet::STO3G};
    const double L[3] = {
        units::convert(2.0, units::Length::Angstrom, units::Length::Bohr),
        units::convert(2.0, units::Length::Angstrom, units::Length::Bohr),
        units::convert(2.0, units::Length::Angstrom, units::Length::Bohr),
    };
    auto spec = internal::lattice_spec(L, 8.0);
    ASSERT_TRUE(spec.has_extra());
    spec.m[0] = spec.m[1] = spec.m[2] = 1;
    internal::apply_gamma_images(ws, spec);

    const int n = ws.n_basis;
    double smin = 1e300;
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(ws.S(i, i), ws.S(i, i), 0.0);
        smin = std::min(smin, ws.S(i, i));
        for (int j = 0; j < i; ++j)
            EXPECT_NEAR(ws.S(i, j), ws.S(j, i), 1e-12);
    }
    EXPECT_GT(smin, 0.0);
}

#endif // ATV_HAS_HF
