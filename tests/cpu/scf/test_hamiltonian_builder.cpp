/// @file tests/cpu/scf/test_hamiltonian_builder.cpp
/// @brief Fock assembly of every HF/KS branch through the one exact J/K
///        engine, against literal four-index contractions (water/def2-SVP:
///        spherical d shells). No SCF loop: fixed densities, two successive
///        builds so the streamed incremental path is exercised.

#ifdef ATV_HAS_HF

#include <cmath>
#include <span>

#include <gtest/gtest.h>
#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/exact_jk.h"
#include "atomiverse/cpu/scf/internal/hamiltonian_builder.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/xc_grid_engine.h"
#include "atomiverse/cpu/xc/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::scf;
using ATV::cpu::xc::XCType;

namespace {

cpu::Frame water() {
    cpu::Frame f(3);
    const int z[] = {8, 1, 1};
    const double xyz[][3] = {
        {0, 0, 0.1173}, {0, 0.7572, -0.4692}, {0, -0.7572, -0.4692}};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_atomic_number)[i] = z[i];
        f.span(kAtom_x)[i] = xyz[i][0];
        f.span(kAtom_y)[i] = xyz[i][1];
        f.span(kAtom_z)[i] = xyz[i][2];
    }
    f.box = cpu::Box::cubic(50);
    return f;
}

/// Symmetric positive-ish spin density with off-diagonal structure.
RealMatrix spin_density(int n, double phase) {
    RealMatrix P = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            P(i, j) =
                0.05 * std::cos(0.3 * (i - j) + phase) + (i == j ? 0.2 : 0.0);
    return P;
}

struct Fixture {
    cpu::Frame frame = water();
    internal::IntegralWorkspace ints{frame, BasisSet::Def2SVP, 0, true};
    xt::xtensor<double, 4> eri = ints.ERI;
    MolecularGrid grid{ints.view().atoms.data(), 3, GridLevel::SG0, 50, 110};
    Fixture() { ints.ERI = {}; } // force the shell-driven engine
};

const Fixture& fixture() {
    static const Fixture f;
    return f;
}

void literal_jk(const xt::xtensor<double, 4>& eri, const RealMatrix& P,
                RealMatrix& J, RealMatrix& K) {
    const int n = static_cast<int>(P.shape()[0]);
    J = xt::zeros<double>({n, n});
    K = J;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l) {
                    J(i, j) += eri(i, j, k, l) * P(l, k);
                    K(i, k) += eri(i, j, k, l) * P(j, l);
                }
}

void expect_close(const RealMatrix& a, const RealMatrix& b, double tol) {
    ASSERT_EQ(a.shape(), b.shape());
    for (std::size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(a.data()[i], b.data()[i], tol) << "element " << i;
}

internal::HamiltonianBuilder prepared(bool is_hf, double memory_mb) {
    const Fixture& fx = fixture();
    internal::HamiltonianBuilder hb;
    hb.prepare(fx.ints.view(), is_hf, GridLevel::SG0, 50, 110, &fx.grid,
               RI::None, AuxBasis::Def2JKFIT, Coulomb::Direct, 10, 1e-9,
               memory_mb);
    return hb;
}

} // namespace

TEST(HamiltonianBuilder, HartreeFockMatchesLiteralJK) {
    const Fixture& fx = fixture();
    const auto view = fx.ints.view();
    const int n = fx.ints.n_basis;
    for (double mb : {4000.0, 0.0}) {
        auto rhf = prepared(true, mb);
        auto uhf = prepared(true, mb);
        for (int it = 0; it < 2; ++it) {
            const RealMatrix Pa = spin_density(n, 0.1 * it);
            const RealMatrix Pb = spin_density(n, 0.5 + 0.1 * it);
            const RealMatrix Pt = Pa + Pb;
            RealMatrix J, K, Ja, Ka, Jb, Kb;
            literal_jk(fx.eri, Pt, J, K);
            literal_jk(fx.eri, Pa, Ja, Ka);
            literal_jk(fx.eri, Pb, Jb, Kb);

            const RealMatrix P2 = 2.0 * Pa;
            RealMatrix J2, K2;
            literal_jk(fx.eri, P2, J2, K2);
            const auto r =
                rhf.step(view, true, true, XCType::LSDA, -1.0, P2, P2);
            expect_close(r.F_alpha, view.Hcore + J2 - 0.5 * K2, 1e-9);
            expect_close(r.F_beta, r.F_alpha, 0.0);

            const auto u =
                uhf.step(view, false, true, XCType::LSDA, -1.0, Pa, Pb);
            expect_close(u.J, J, 1e-9);
            expect_close(u.F_alpha, view.Hcore + J - Ka, 1e-9);
            expect_close(u.F_beta, view.Hcore + J - Kb, 1e-9);
        }
    }
}

TEST(HamiltonianBuilder, PureKsCoulombMatchesLiteralJ) {
    const Fixture& fx = fixture();
    const auto view = fx.ints.view();
    const int n = fx.ints.n_basis;
    const cpu::xc::BuiltinXc pbe(XCType::PBE);
    auto hb = prepared(false, 0.0);
    for (int it = 0; it < 2; ++it) {
        const RealMatrix Pa = spin_density(n, 0.2 * it);
        const RealMatrix P = 2.0 * Pa;
        RealMatrix J, K;
        literal_jk(fx.eri, P, J, K);
        const auto r = hb.step(view, true, false, XCType::PBE, -1.0, P, P);
        const RealMatrix Pc = xt::linalg::dot(
            view.c2s, RealMatrix(xt::linalg::dot(Pa, xt::transpose(view.c2s))));
        auto xc = evaluate_xc_batched(fx.grid, view.shells, view.n_ao_cart(),
                                      Pc, Pc, pbe);
        const RealMatrix V = xt::linalg::dot(
            xt::transpose(view.c2s),
            RealMatrix(xt::linalg::dot(xc.Vxc_alpha, view.c2s)));
        expect_close(r.J, J, 1e-9);
        expect_close(r.F_alpha, view.Hcore + J + V, 1e-8);
        EXPECT_EQ(r.E_exact_exchange, 0.0);
    }
}

TEST(HamiltonianBuilder, HybridExchangeIsScaledExactK) {
    const Fixture& fx = fixture();
    const auto view = fx.ints.view();
    const int n = fx.ints.n_basis;
    for (bool restricted : {true, false}) {
        auto hyb = prepared(false, 0.0);
        auto pure = prepared(false, 0.0); // same functional, alpha forced to 0
        for (int it = 0; it < 2; ++it) {
            const RealMatrix Pa = spin_density(n, 0.3 * it);
            const RealMatrix Pb = spin_density(n, 0.9 + 0.3 * it);
            const RealMatrix P_in = restricted ? RealMatrix(2.0 * Pa) : Pa;
            const RealMatrix Pb_in = restricted ? P_in : Pb;
            const auto h = hyb.step(view, restricted, false, XCType::B3LYP,
                                    -1.0, P_in, Pb_in);
            const auto p = pure.step(view, restricted, false, XCType::B3LYP,
                                     0.0, P_in, Pb_in);
            EXPECT_EQ(p.E_exact_exchange, 0.0);
            if (restricted) {
                RealMatrix J, K;
                literal_jk(fx.eri, P_in, J, K);
                expect_close(h.F_alpha, p.F_alpha - 0.1 * K, 1e-9);
                EXPECT_NEAR(h.E_exact_exchange,
                            -0.25 * 0.2 * xt::sum(P_in * K)(), 1e-9);
            } else {
                RealMatrix Ja, Ka, Jb, Kb;
                literal_jk(fx.eri, Pa, Ja, Ka);
                literal_jk(fx.eri, Pb, Jb, Kb);
                expect_close(h.F_alpha, p.F_alpha - 0.2 * Ka, 1e-9);
                expect_close(h.F_beta, p.F_beta - 0.2 * Kb, 1e-9);
                EXPECT_NEAR(h.E_exact_exchange,
                            -0.5 * 0.2
                                * (xt::sum(Pa * Ka)() + xt::sum(Pb * Kb)()),
                            1e-9);
            }
        }
    }
}

TEST(HamiltonianBuilder, RangeSeparatedAddsErfExchange) {
    const Fixture& fx = fixture();
    const auto view = fx.ints.view();
    const int n = fx.ints.n_basis;
    const auto rsh = cpu::xc::rsh_exchange_params(XCType::CAMB3LYP);
    internal::ExactJK lr;
    lr.prepare(view, 0.0, false, rsh.mu);
    auto hb = prepared(false, 0.0);
    const RealMatrix Pa = spin_density(n, 0.0), Pb = spin_density(n, 1.3);
    // alpha forced to zero isolates the erf-attenuated term.
    const auto u = hb.step(view, false, false, XCType::CAMB3LYP, 0.0, Pa, Pb);
    RealMatrix Ka, Kb;
    const internal::JKRequest requests[] = {{&Pa, nullptr, &Ka, 1.0, 0},
                                            {&Pb, nullptr, &Kb, 1.0, 1}};
    lr.compute(requests);
    EXPECT_NEAR(u.E_exact_exchange,
                -0.5 * rsh.beta * (xt::sum(Pa * Ka)() + xt::sum(Pb * Kb)()),
                1e-9);
    auto ref = prepared(false, 0.0);
    const auto r = ref.step(view, false, false, XCType::CAMB3LYP, -1.0, Pa, Pb);
    RealMatrix Ja, Ka1, Jb, Kb1;
    literal_jk(fx.eri, Pa, Ja, Ka1);
    literal_jk(fx.eri, Pb, Jb, Kb1);
    expect_close(r.F_alpha, u.F_alpha - rsh.alpha * Ka1, 1e-9);
    expect_close(r.F_beta, u.F_beta - rsh.alpha * Kb1, 1e-9);
}

#else
#include <gtest/gtest.h>
TEST(HamiltonianBuilder, SkippedWithoutHF) { GTEST_SKIP() << "ATV_HAS_HF off"; }
#endif
