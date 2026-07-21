/// @file tests/cpu/scf/test_scf_solver.cpp
/// @brief Concrete methods `RHF` / `UKS` converge on H2O and are stable
///        across independent instances and repeated `run()` calls.

#ifdef ATV_HAS_HF

#include <gtest/gtest.h>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

cpu::Frame make_h2o() {
    cpu::Frame f(3);
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = 0.7572;
    f.span(ATV::kAtom_x)[2] = -0.7572;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_y)[2] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_z)[1] = -0.5860;
    f.span(ATV::kAtom_z)[2] = -0.5860;
    f.span(ATV::kAtom_type)[0] = 8;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_mass)[0] = 15.999;
    f.span(ATV::kAtom_mass)[1] = 1.008;
    f.span(ATV::kAtom_mass)[2] = 1.008;
    f.simbox = ATV::cpu::Box::cubic(50.0);
    return f;
}

} // namespace

TEST(RHF, H2OConvergesAcrossIndependentInstances) {
    auto frame = make_h2o();
    cpu::scf::RHF a;
    a.basis = cpu::scf::BasisSet::STO3G;
    a.run(frame);

    EXPECT_TRUE(a.converged()) << "H2O / STO-3G RHF should converge";
    EXPECT_GT(a.iterations(), 0);
    EXPECT_LT(a.iterations(), a.max_iter);

    cpu::scf::RHF b;
    b.basis = cpu::scf::BasisSet::STO3G;
    b.run(frame);
    EXPECT_NEAR(a.energy(), b.energy(), 1e-10);
    // The old restricted result's P_alpha was the factor-2 total density.
    const auto P_first = a.determinant().density_total();
    const auto P_second = b.determinant().density_total();
    ASSERT_EQ(P_first.shape(0), P_second.shape(0));
    for (std::size_t i = 0; i < P_first.shape(0); ++i) {
        for (std::size_t j = 0; j < P_first.shape(1); ++j) {
            EXPECT_NEAR(P_first(i, j), P_second(i, j), 1e-10);
        }
    }
}

TEST(RHF, RunIsIdempotentAcrossInvocations) {
    auto frame = make_h2o();
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;

    rhf.run(frame);
    const double e1 = rhf.energy();
    // The old restricted result's P_alpha was the factor-2 total density.
    const auto P1 = rhf.determinant().density_total();

    rhf.run(frame);
    EXPECT_NEAR(e1, rhf.energy(), 1e-10);
    const auto P2 = rhf.determinant().density_total();
    for (std::size_t i = 0; i < P1.shape(0); ++i)
        for (std::size_t j = 0; j < P1.shape(1); ++j)
            EXPECT_NEAR(P1(i, j), P2(i, j), 1e-10);
}

TEST(UKS, H2OConverges) {
    auto frame = make_h2o();
    // Closed-shell water: 5 alpha + 5 beta
    cpu::scf::UKS uks(/*n_alpha=*/5, /*n_beta=*/5, cpu::scf::XCType::LSDA);
    uks.basis = cpu::scf::BasisSet::STO3G;
    uks.max_iter = 80;
    uks.run(frame);
    EXPECT_TRUE(uks.converged());
    EXPECT_LT(uks.energy(), 0.0);
}

#endif // ATV_HAS_HF
