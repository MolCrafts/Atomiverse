/// @file cpu/semiempirical/test_nddo_geometry.cpp
/// @brief NDDO physical-contact refusal; no self-consistency loop.
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_solver.h"
#include "atomiverse/semiempirical/nddo_geometry.h"

namespace {
using namespace ATV;
using ATV::semiempirical::require_nddo_contacts;
using ATV::semiempirical::SemiEmpiricalMethod;

struct Atoms {
    std::vector<int> z;
    std::vector<StorageReal> x, y, w;
    void add(int atomic_number, double px, double py = 0.0, double pz = 0.0) {
        z.push_back(atomic_number);
        x.push_back(static_cast<StorageReal>(px));
        y.push_back(static_cast<StorageReal>(py));
        w.push_back(static_cast<StorageReal>(pz));
    }
    void check(SemiEmpiricalMethod method = SemiEmpiricalMethod::AM1) const {
        require_nddo_contacts(method, z.size(), z.data(), x.data(), y.data(),
                              w.data());
    }
};

std::string refusal(const Atoms& atoms) {
    try {
        atoms.check();
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    ADD_FAILURE() << "geometry was not refused";
    return {};
}

TEST(NddoGeometry, CovalentRadiiAreCorderoAndHaveNoFallback) {
    EXPECT_DOUBLE_EQ(ATV::semiempirical::covalent_radius_angstrom(1), 0.31);
    EXPECT_DOUBLE_EQ(ATV::semiempirical::covalent_radius_angstrom(6), 0.76);
    EXPECT_DOUBLE_EQ(ATV::semiempirical::covalent_radius_angstrom(35), 1.20);
    EXPECT_DOUBLE_EQ(ATV::semiempirical::covalent_radius_angstrom(86), 1.50);
    EXPECT_THROW(ATV::semiempirical::covalent_radius_angstrom(0),
                 std::invalid_argument);
    EXPECT_THROW(ATV::semiempirical::covalent_radius_angstrom(87),
                 std::invalid_argument);
}

TEST(NddoGeometry, ShortestRealBondsAreLegal) {
    for (const auto& [za, zb, r] :
         {std::tuple{1, 1, 0.74}, std::tuple{7, 7, 1.098},
          std::tuple{6, 8, 1.128}, std::tuple{6, 6, 1.20},
          std::tuple{1, 9, 0.917}}) {
        Atoms atoms;
        atoms.add(za, 0.0);
        atoms.add(zb, r);
        EXPECT_NO_THROW(atoms.check()) << za << "-" << zb << " at " << r;
    }
}

TEST(NddoGeometry, LimitIsSixTenthsOfTheCovalentRadiusSum) {
    // H-H limit = 0.6 x (0.31 + 0.31) = 0.372 A.
    Atoms legal;
    legal.add(1, 0.0);
    legal.add(1, 0.373);
    EXPECT_NO_THROW(legal.check());
    Atoms refused;
    refused.add(1, 0.0);
    refused.add(1, 0.371);
    EXPECT_THROW(refused.check(), std::invalid_argument);
    Atoms coincident;
    coincident.add(6, 1.0, 2.0, 3.0);
    coincident.add(6, 1.0, 2.0, 3.0);
    EXPECT_THROW(coincident.check(), std::invalid_argument);
}

TEST(NddoGeometry, RefusalNamesCountIndicesElementsDistanceAndLimit) {
    // Methane plus a stray carbon 0.64 A from the central carbon: the
    // assembly fault of the original 205-atom report (C13-C101, 0.643 A).
    Atoms atoms;
    atoms.add(6, 0.0);
    for (const double s : {0.629, -0.629}) {
        atoms.add(1, s, s, 0.629);
        atoms.add(1, -s, s, -0.629);
    }
    atoms.add(6, 0.0, 0.0, 0.64);
    const std::string message = refusal(atoms);
    EXPECT_NE(message.find("AM1"), std::string::npos) << message;
    EXPECT_NE(message.find("before the SCF"), std::string::npos) << message;
    EXPECT_NE(message.find("1 atom pair"), std::string::npos) << message;
    EXPECT_NE(message.find("0.6 x"), std::string::npos) << message;
    EXPECT_NE(message.find("C0-C5 0.640 A (limit 0.912 A)"), std::string::npos)
        << message;
}

TEST(NddoGeometry, RefusalListsClosestPairsFirstAndCountsTheRest) {
    // Seven H2 "molecules" at shrinking separations, 10 A apart.
    Atoms atoms;
    for (int k = 0; k < 7; ++k) {
        atoms.add(1, 10.0 * k);
        atoms.add(1, 10.0 * k + 0.10 + 0.02 * k);
    }
    const std::string message = refusal(atoms);
    EXPECT_NE(message.find("7 atom pair"), std::string::npos) << message;
    EXPECT_NE(message.find("H0-H1 0.100 A"), std::string::npos) << message;
    EXPECT_LT(message.find("H0-H1"), message.find("H2-H3")) << message;
    EXPECT_NE(message.find("H8-H9"), std::string::npos) << message;
    EXPECT_EQ(message.find("H10-H11"), std::string::npos) << message;
    EXPECT_NE(message.find("(2 more)"), std::string::npos) << message;
}

TEST(NddoGeometry, SolverRefusesOverlapBeforeScfAndCanRetry) {
    cpu::Frame frame(2);
    for (int a = 0; a < 2; ++a) {
        frame.span(kAtom_type)[a] = 1;
        frame.span(kAtom_x)[a] = 0.74 * a;
        frame.span(kAtom_y)[a] = frame.span(kAtom_z)[a] = 0;
    }
    for (const auto method :
         {SemiEmpiricalMethod::AM1, SemiEmpiricalMethod::RM1,
          SemiEmpiricalMethod::MNDO, SemiEmpiricalMethod::PM3}) {
        cpu::semiempirical::NDDOConfig config;
        config.method = method;
        cpu::semiempirical::NDDOSolver solver{config};
        solver.build(frame);
        frame.span(kAtom_x)[1] = 0.30;
        EXPECT_THROW(solver.scf_prepare(frame), std::invalid_argument);
        EXPECT_FALSE(solver.result().converged);
        cpu::semiempirical::NDDOSolver cold{config};
        EXPECT_THROW(cold.build(frame), std::invalid_argument);
        frame.span(kAtom_x)[1] = 0.74;
        EXPECT_NO_THROW(solver.scf_prepare(frame));
    }
}

} // namespace
