/// @file test_cpu_gpw_ksv.cpp
/// @brief Unit tests for GPW Berry-phase string geometry, link-det phase
///        math and quantum wrapping. No SCF e2e (that lives in
///        regressions/gpw/periodic-berry-gpw).

#include <array>
#include <cmath>
#include <complex>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/berry_polarization.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using namespace ATV::cpu::gpw;

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kA2B = 1.0 / 0.52917720859;
} // namespace

TEST(GpwBerryStrings, LayoutStringMajorAndSpacing) {
    BerryPolarization berry;
    berry.k_perp = 2;
    berry.k_string = 4;

    const auto kf = berry.string_kpoints(/*dir=*/2);
    ASSERT_EQ(kf.size(), 2u * 2u * 4u);

    for (int s = 0; s < 4; ++s) {
        for (int j = 0; j < 4; ++j) {
            const auto& f = kf[static_cast<std::size_t>(s * 4 + j)];
            EXPECT_NEAR(f[2], static_cast<double>(j) / 4.0, 1e-12);
            EXPECT_NEAR(f[0], kf[static_cast<std::size_t>(s * 4)][0], 1e-12);
            EXPECT_NEAR(f[1], kf[static_cast<std::size_t>(s * 4)][1], 1e-12);
            for (int d = 0; d < 3; ++d) {
                EXPECT_GE(f[static_cast<std::size_t>(d)], 0.0);
                EXPECT_LT(f[static_cast<std::size_t>(d)], 1.0);
            }
        }
    }

    std::set<std::pair<double, double>> perp;
    for (int s = 0; s < 4; ++s)
        perp.insert({kf[static_cast<std::size_t>(s * 4)][0],
                     kf[static_cast<std::size_t>(s * 4)][1]});
    EXPECT_EQ(perp.size(), 4u);
    EXPECT_TRUE(perp.count({0.0, 0.0}) == 1);
    EXPECT_TRUE(perp.count({0.5, 0.5}) == 1);
}

TEST(GpwBerryStrings, DirectionAxisSelects) {
    BerryPolarization berry;
    berry.k_perp = 1;
    berry.k_string = 3;
    for (int dir = 0; dir < 3; ++dir) {
        const auto kf = berry.string_kpoints(dir);
        ASSERT_EQ(kf.size(), 3u);
        for (int j = 0; j < 3; ++j) {
            for (int d = 0; d < 3; ++d) {
                const double want =
                    (d == dir) ? static_cast<double>(j) / 3.0 : 0.0;
                EXPECT_NEAR(kf[static_cast<std::size_t>(j)]
                              [static_cast<std::size_t>(d)],
                            want, 1e-12);
            }
        }
    }
}

TEST(GpwBerryStrings, InvalidArgumentsThrow) {
    BerryPolarization berry;
    EXPECT_THROW(berry.string_kpoints(-1), std::invalid_argument);
    EXPECT_THROW(berry.string_kpoints(3), std::invalid_argument);
    berry.k_perp = 0;
    EXPECT_THROW(berry.string_kpoints(0), std::invalid_argument);
    berry.k_perp = 2;
    berry.k_string = 1;
    EXPECT_THROW(berry.string_kpoints(0), std::invalid_argument);
}

TEST(GpwBerryPhaseMath, ProductOfUnitPhases) {
    const std::complex<double> w = std::polar(1.0, kPi / 6.0);
    const double phi = phase_from_link_dets({w, w, w});
    EXPECT_NEAR(phi, kPi / 2.0, 1e-12);
}

TEST(GpwBerryPhaseMath, ModulusIsNormalizedAway) {
    const std::vector<std::complex<double>> dets = {
        std::polar(0.3, 0.4), std::polar(7.0, -1.1), std::polar(0.05, 0.9)};
    EXPECT_NEAR(phase_from_link_dets(dets), 0.4 - 1.1 + 0.9, 1e-12);
}

TEST(GpwBerryPhaseMath, WrapsIntoMinusPiPi) {
    const std::complex<double> w = std::polar(1.0, 2.0 * kPi / 3.0);
    EXPECT_NEAR(phase_from_link_dets({w, w, w}), 0.0, 1e-12);
    const std::complex<double> v = std::polar(1.0, 3.0 * kPi / 4.0);
    EXPECT_NEAR(std::abs(phase_from_link_dets({v, v, v, v})), kPi, 1e-12);
}

TEST(GpwBerryPhaseMath, ThrowsOnDeadLinkOrEmpty) {
    EXPECT_THROW(phase_from_link_dets({}), std::invalid_argument);
    EXPECT_THROW(phase_from_link_dets({{1.0, 0.0}, {0.0, 0.0}}),
                 std::invalid_argument);
    EXPECT_THROW(phase_from_link_dets({{5e-13, 0.0}}), std::invalid_argument);
}

TEST(GpwBerryWrap, CenteredInterval) {
    EXPECT_NEAR(wrap_centered(0.3, 1.0), 0.3, 1e-12);
    EXPECT_NEAR(wrap_centered(0.7, 1.0), -0.3, 1e-12);
    EXPECT_NEAR(wrap_centered(-0.7, 1.0), 0.3, 1e-12);
    EXPECT_NEAR(wrap_centered(1.0, 2.0), 1.0, 1e-12);
    EXPECT_NEAR(wrap_centered(1.1, 2.0), -0.9, 1e-12);
    EXPECT_THROW(wrap_centered(0.1, 0.0), std::invalid_argument);
}

TEST(GpwBerryComputeGates, MetallicAndEmptyThrow) {
    const double a = 5.43 * kA2B;
    const GaussianBasis bSi(molrs::Element::Si, "DZVP-MOLOPT-GTH");
    const GthPotential pSi(molrs::Element::Si, "GTH-PBE-q4");
    const double pos[3] = {0, 0, 0};
    std::vector<GpwAtom> atoms;
    atoms.emplace_back(pos, bSi, pSi, 14);

    GpwDft method;
    method.cell[0] = method.cell[1] = a / std::sqrt(2.0);
    method.cell[2] = a;
    method.cutoff_ry = 50.0;
    method.n_occ = 2;
    method.xc = XcKind::PADE;
    method.build(atoms);

    BerryPolarization berry;
    berry.k_perp = 1;
    berry.k_string = 2;
    berry.directions = {true, false, false};

    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.02;
    EXPECT_THROW(berry.compute(method, atoms), std::invalid_argument);

    method.smearing = Smearing::none;
    method.n_occ = 0;
    // rebuild validation for empty occ is on compute, not build if already built
    EXPECT_THROW(berry.compute(method, atoms), std::invalid_argument);
}
