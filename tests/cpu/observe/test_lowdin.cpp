/// @file tests/cpu/observer/test_lowdin.cpp
/// @brief Unit: Löwdin populations on hand-built shells + SlaterDeterminant —
///        no SCF run anywhere in this file.
///
/// Single-s analytic case: with one normalized s AO the overlap is exactly
/// S = 1, so S^{1/2} = 1 and q = Z − (S^{1/2} P S^{1/2})_{00} = Z − P_00.
/// Two-atom symmetric case: a bonding-MO density on two identical centers
/// must split evenly and conserve Σq.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/lowdin.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

using namespace ATV;
using cpu::scf::MeanFieldOutputs;
using cpu::scf::SlaterDeterminant;
using cpu::scf::native::Shell;

namespace {

constexpr double kPi = 3.14159265358979323846;

/// One normalized single-primitive s AO per given center [Bohr].
struct SyntheticSystem {
    std::vector<double> storage; // [alpha, coeff] shared by every shell
    std::shared_ptr<std::vector<Shell>> shells;
    std::shared_ptr<SlaterDeterminant> det;
    std::shared_ptr<bool> ready;

    explicit SyntheticSystem(
        const std::vector<std::array<double, 3>>& centers) {
        const double alpha = 1.0;
        const double norm = std::pow(2.0 * alpha / kPi, 0.75);
        storage = {alpha, norm};
        shells = std::make_shared<std::vector<Shell>>();
        for (const auto& c : centers)
            shells->push_back(
                Shell{1, &storage[0], &storage[1], {c[0], c[1], c[2]}, 0, 0});
        ready = std::make_shared<bool>(true);
    }

    MeanFieldOutputs handle() const {
        return {det, shells, ready,
                std::make_shared<const xt::xtensor<double, 2>>()};
    }
};

cpu::Frame frame_at(const std::vector<std::array<double, 3>>& pos_ang,
                    const std::vector<int>& z) {
    cpu::Frame f(static_cast<int>(pos_ang.size()));
    for (std::size_t i = 0; i < pos_ang.size(); ++i) {
        f.span(kAtom_x)[static_cast<int>(i)] = pos_ang[i][0];
        f.span(kAtom_y)[static_cast<int>(i)] = pos_ang[i][1];
        f.span(kAtom_z)[static_cast<int>(i)] = pos_ang[i][2];
        f.span(kAtom_type)[static_cast<int>(i)] = z[i];
    }
    f.box = cpu::Box::cubic(50.0);
    return f;
}

} // namespace

/// Single s AO: S = 1 exactly, so q = Z − P_00 to 1e-12. Restricted
/// determinant with one doubly occupied orbital ⇒ P_00 = 2.
TEST(Lowdin, SingleSAnalytic) {
    SyntheticSystem sys{{{0.0, 0.0, 0.0}}};
    cpu::linalg::RealMatrix C =
        xt::ones<double>({std::size_t(1), std::size_t(1)});
    cpu::linalg::RealVector eps = xt::zeros<double>({std::size_t(1)});
    cpu::linalg::RealVector occ = xt::ones<double>({std::size_t(1)});
    sys.det = std::make_shared<SlaterDeterminant>(C, eps, occ, 1);

    auto frame = frame_at({{0.0, 0.0, 0.0}}, {2}); // He
    const auto q = cpu::lowdin_charge(sys.handle(), frame);
    ASSERT_EQ(q.size(), 1u);
    EXPECT_NEAR(q[0], 2.0 - 2.0, 1e-12); // q = Z − P_00
}

/// Two identical atoms with a symmetric bonding-MO density: equal charges,
/// and Σq = ΣZ − N_electrons (charge conservation).
TEST(Lowdin, TwoAtomSymmetricDensity) {
    const double d_bohr = 1.4;
    SyntheticSystem sys{{{0.0, 0.0, 0.0}, {0.0, 0.0, d_bohr}}};

    // Bonding MO c = (φ1 + φ2)/sqrt(2(1+S12)); S12 of two unit-exponent
    // normalized s Gaussians at distance R: exp(−α R²/2) = exp(−R²/2)... the
    // exact value is irrelevant — any symmetric normalized column works
    // because the assertions are symmetry + conservation, and Tr(PS) = N is
    // only exact when c is S-normalized, so build it from the analytic S12.
    const double s12 = std::exp(-0.5 * d_bohr * d_bohr);
    const double n = 1.0 / std::sqrt(2.0 * (1.0 + s12));
    cpu::linalg::RealMatrix C =
        xt::zeros<double>({std::size_t(2), std::size_t(1)});
    C(0, 0) = n;
    C(1, 0) = n;
    cpu::linalg::RealVector eps = xt::zeros<double>({std::size_t(1)});
    cpu::linalg::RealVector occ = xt::ones<double>({std::size_t(1)});
    sys.det = std::make_shared<SlaterDeterminant>(C, eps, occ, 1); // 2 e⁻ total

    const double d_ang = d_bohr * units::kBohrInAngstrom;
    auto frame = frame_at({{0.0, 0.0, 0.0}, {0.0, 0.0, d_ang}}, {1, 1});
    const auto q = cpu::lowdin_charge(sys.handle(), frame);
    ASSERT_EQ(q.size(), 2u);
    EXPECT_NEAR(q[0], q[1], 1e-12);       // symmetric split
    EXPECT_NEAR(q[0] + q[1], 0.0, 1e-10); // ΣZ − N = 2 − 2 = 0
}

/// Unrestricted determinant: the α-only electron gives σ_A = 0.5 per atom
/// (symmetric), and the restricted form gives exact zeros.
TEST(Lowdin, SpinDensity) {
    const double d_bohr = 1.4;
    SyntheticSystem sys{{{0.0, 0.0, 0.0}, {0.0, 0.0, d_bohr}}};
    const double s12 = std::exp(-0.5 * d_bohr * d_bohr);
    const double n = 1.0 / std::sqrt(2.0 * (1.0 + s12));
    cpu::linalg::RealMatrix C =
        xt::zeros<double>({std::size_t(2), std::size_t(1)});
    C(0, 0) = n;
    C(1, 0) = n;
    cpu::linalg::RealVector eps = xt::zeros<double>({std::size_t(1)});
    cpu::linalg::RealVector f_a = xt::ones<double>({std::size_t(1)});
    cpu::linalg::RealVector f_b = xt::zeros<double>({std::size_t(1)});
    sys.det = std::make_shared<SlaterDeterminant>(C, eps, f_a, 1, C, eps, f_b,
                                                  0); // H2⁺-like: 1 α, 0 β

    const double d_ang = d_bohr * units::kBohrInAngstrom;
    auto frame = frame_at({{0.0, 0.0, 0.0}, {0.0, 0.0, d_ang}}, {1, 1});
    const auto sigma = cpu::lowdin_spin(sys.handle(), frame);
    ASSERT_EQ(sigma.size(), 2u);
    EXPECT_NEAR(sigma[0], 0.5, 1e-10);
    EXPECT_NEAR(sigma[1], 0.5, 1e-10);

    // Restricted form: P^α = P^β ⇒ exact zeros.
    cpu::linalg::RealVector occ = xt::ones<double>({std::size_t(1)});
    sys.det = std::make_shared<SlaterDeterminant>(C, eps, occ, 1);
    const auto zero = cpu::lowdin_spin(sys.handle(), frame);
    EXPECT_NEAR(zero[0], 0.0, 1e-14);
    EXPECT_NEAR(zero[1], 0.0, 1e-14);
}

/// A shell centered away from every atom position must throw (no silent
/// mis-assignment), and a not-ready handle throws invalid_argument.
TEST(Lowdin, UnmatchedShellAndNotReadyThrow) {
    SyntheticSystem sys{{{0.0, 0.0, 5.0}}}; // center matches no atom below
    cpu::linalg::RealMatrix C =
        xt::ones<double>({std::size_t(1), std::size_t(1)});
    cpu::linalg::RealVector eps = xt::zeros<double>({std::size_t(1)});
    cpu::linalg::RealVector occ = xt::ones<double>({std::size_t(1)});
    sys.det = std::make_shared<SlaterDeterminant>(C, eps, occ, 1);

    auto frame = frame_at({{0.0, 0.0, 0.0}}, {2});
    EXPECT_THROW(cpu::lowdin_charge(sys.handle(), frame), std::runtime_error);

    *sys.ready = false;
    EXPECT_THROW(cpu::lowdin_charge(sys.handle(), frame),
                 std::invalid_argument);
}
