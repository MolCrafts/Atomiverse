#include "atomiverse/cpu/scf/internal/exact_jk.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/rys_eri.h"
#include <cmath>
#include <gtest/gtest.h>
#include <span>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ATV;
using namespace ATV::cpu::scf;

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

// Independent literal four-index contractions test every symmetry degeneracy
// and, unlike symmetric-only comparisons, detect transposed GHF exchange.
void check(BasisSet basis, bool nonsymmetric) {
    auto f = water();
    internal::IntegralWorkspace ints(f, basis, 0, true);
    const int n = ints.n_basis;
    RealMatrix P = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            P(i, j) = std::sin(nonsymmetric ? i + 2.1 * j : i + j + 0.2);
    RealMatrix Jref = xt::zeros<double>({n, n});
    RealMatrix Kref = Jref;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l) {
                    Jref(i, j) += ints.ERI(i, j, k, l) * P(l, k);
                    Kref(i, k) += ints.ERI(i, j, k, l) * P(j, l);
                }
    // Removing the dense reference forces real shell construction / screening.
    ints.ERI = {};
    internal::ExactJK engine;
    for (double mb : {4000.0, 0.0, 4000.0}) {
        engine.prepare(ints.view(), mb);
        EXPECT_EQ(engine.cached(), mb > 0);
        if (mb > 0) {
            const std::size_t nc = ints.n_basis;
            const auto np = nc * (nc + 1) / 2;
            EXPECT_EQ(engine.cache_bytes(), np * (np + 1) / 2 * sizeof(double));
        } else {
            EXPECT_EQ(engine.cache_bytes(), 0u);
        }
        RealMatrix J, K;
        engine.compute(P, J, K);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                EXPECT_NEAR(J(i, j), Jref(i, j), 2e-10);
                EXPECT_NEAR(K(i, j), Kref(i, j), 2e-10);
            }
    }
}
} // namespace

TEST(ExactJK, SymmetricDensity) { check(BasisSet::STO3G, false); }
TEST(ExactJK, NonsymmetricDensity) { check(BasisSet::STO3G, true); }
TEST(ExactJK, SphericalDShells) { check(BasisSet::Def2SVP, true); }
TEST(ExactJK, MultipleThreads) {
#ifdef _OPENMP
    const int previous = omp_get_max_threads();
    omp_set_num_threads(4);
    check(BasisSet::STO3G, true);
    omp_set_num_threads(previous);
#else
    GTEST_SKIP() << "OpenMP disabled";
#endif
}
TEST(ExactJK, InvalidBudget) {
    auto f = water();
    internal::IntegralWorkspace ints(f, BasisSet::STO3G);
    internal::ExactJK engine;
    EXPECT_THROW(engine.prepare(ints.view(), -1), std::invalid_argument);
    EXPECT_THROW(engine.prepare(ints.view(), NAN), std::invalid_argument);
}

TEST(ExactJK, IncrementalSpinChannelsAndGeometryReset) {
    auto f = water();
    internal::IntegralWorkspace ints(f, BasisSet::STO3G);
    internal::ExactJK engine;
    engine.prepare(ints.view(), 0);
    const int n = ints.n_basis;
    for (int geometry = 0; geometry < 2; ++geometry) {
        if (geometry) {
            f.span(kAtom_x)[1] += .1;
            ints = internal::IntegralWorkspace(f, BasisSet::STO3G);
            engine.prepare(ints.view(), 0);
        }
        for (int iteration = 0; iteration < 3; ++iteration)
            for (int channel = 0; channel < 4; ++channel) {
                RealMatrix P = xt::zeros<double>({n, n});
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j)
                        P(i, j) = std::sin(i + 1.2 * j + channel)
                                  + iteration * 1e-3 * std::cos(i + j);
                RealMatrix J, K, Jref, Kref;
                engine.step(P, J, K, channel);
                engine.compute(P, Jref, Kref);
                for (std::size_t i = 0; i < J.size(); ++i) {
                    EXPECT_NEAR(J.data()[i], Jref.data()[i], 2e-10);
                    EXPECT_NEAR(K.data()[i], Kref.data()[i], 2e-10);
                }
            }
    }
}

namespace {
// Literal J[P] and K[P] from a dense ERI tensor (working basis).
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

RealMatrix test_density(int n, double phase) {
    RealMatrix P = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            P(i, j) = std::sin(i + 2.1 * j + phase);
    return P;
}

void expect_close(const RealMatrix& a, const RealMatrix& b, double tol) {
    ASSERT_EQ(a.shape(), b.shape());
    for (std::size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(a.data()[i], b.data()[i], tol);
}
} // namespace

// One pass over several densities: J-only, scaled K-only and J+K requests
// reproduce the literal contractions (cached and streamed, spherical d).
TEST(ExactJK, MultiDensityJOnlyScaledKOnly) {
    auto f = water();
    internal::IntegralWorkspace ints(f, BasisSet::Def2SVP, 0, true);
    const int n = ints.n_basis;
    const RealMatrix P0 = test_density(n, 0.0), P1 = test_density(n, 0.7),
                     P2 = test_density(n, 1.9);
    RealMatrix J0, K0, J1, K1, J2, K2;
    literal_jk(ints.ERI, P0, J0, K0);
    literal_jk(ints.ERI, P1, J1, K1);
    literal_jk(ints.ERI, P2, J2, K2);
    ints.ERI = {};
    internal::ExactJK engine;
    for (double mb : {4000.0, 0.0}) {
        engine.prepare(ints.view(), mb);
        RealMatrix Ja, Kb, Jc, Kc;
        const internal::JKRequest requests[] = {{&P0, &Ja, nullptr, 1.0, 0},
                                                {&P1, nullptr, &Kb, -0.3, 1},
                                                {&P2, &Jc, &Kc, 0.5, 2}};
        engine.compute(requests);
        expect_close(Ja, J0, 2e-10);
        expect_close(Kb, RealMatrix(-0.3 * K1), 2e-10);
        expect_close(Jc, J2, 2e-10);
        expect_close(Kc, RealMatrix(0.5 * K2), 2e-10);
    }
}

// Per-block-pair density screening must keep every quartet a sparse density
// touches: a density living on one AO pair still gives the literal J/K.
TEST(ExactJK, LocalizedDensityScreeningKeepsTouchedQuartets) {
    auto f = water();
    internal::IntegralWorkspace ints(f, BasisSet::STO3G, 0, true);
    const int n = ints.n_basis;
    RealMatrix P = xt::zeros<double>({n, n});
    P(5, 2) = 1e-3;
    P(2, 5) = 2e-3;
    RealMatrix Jref, Kref;
    literal_jk(ints.ERI, P, Jref, Kref);
    ints.ERI = {};
    internal::ExactJK engine;
    engine.prepare(ints.view(), 0.0);
    RealMatrix J, K;
    engine.compute(P, J, K);
    expect_close(J, Jref, 1e-13);
    expect_close(K, Kref, 1e-13);
}

// The erf(omega r)/r engine matches a literal contraction over the Rys
// erf-attenuated block integrals; omega = 0 is the bare Coulomb operator.
TEST(ExactJK, ErfAttenuatedExchange) {
    auto f = water();
    internal::IntegralWorkspace ints(f, BasisSet::STO3G, 0, true);
    const int n = ints.n_basis;
    const double omega = 0.33;
    const ShellBlockPairList pairs{ints.view(), 0.0};
    xt::xtensor<double, 4> eri = xt::zeros<double>({n, n, n, n});
    std::vector<double> block;
    for (const auto& A : pairs.blocks)
        for (const auto& B : pairs.blocks)
            for (const auto& C : pairs.blocks)
                for (const auto& D : pairs.blocks) {
                    block.assign(static_cast<std::size_t>(A.n_cart) * B.n_cart
                                     * C.n_cart * D.n_cart,
                                 0.0);
                    native::rys_eri_erf_block(A, B, C, D, omega, block.data());
                    std::size_t q = 0;
                    for (int a = 0; a < A.n_cart; ++a)
                        for (int b = 0; b < B.n_cart; ++b)
                            for (int c = 0; c < C.n_cart; ++c)
                                for (int d = 0; d < D.n_cart; ++d)
                                    eri(A.ao_start + a, B.ao_start + b,
                                        C.ao_start + c, D.ao_start + d) =
                                        block[q++];
                }
    const RealMatrix P = test_density(n, 0.4);
    RealMatrix Jref, Kref, Jc, Kc;
    literal_jk(eri, P, Jref, Kref);
    literal_jk(ints.ERI, P, Jc, Kc);
    internal::ExactJK lr, coulomb;
    for (double mb : {4000.0, 0.0}) {
        lr.prepare(ints.view(), mb, false, omega);
        RealMatrix K;
        const internal::JKRequest request{&P, nullptr, &K, 1.0, 0};
        lr.compute(std::span<const internal::JKRequest>(&request, 1));
        expect_close(K, Kref, 1e-11);
        coulomb.prepare(ints.view(), mb, false, 0.0);
        RealMatrix J0, K0;
        coulomb.compute(P, J0, K0);
        expect_close(K0, Kc, 2e-10);
    }
    EXPECT_THROW(lr.prepare(ints.view(), 0.0, false, -1.0),
                 std::invalid_argument);
}

// Incremental multi-request steps track each channel; a changed exchange
// scale on a channel restarts its history instead of mixing scales.
TEST(ExactJK, IncrementalMultiRequestAndScaleChange) {
    auto f = water();
    internal::IntegralWorkspace ints(f, BasisSet::STO3G);
    internal::ExactJK engine;
    engine.prepare(ints.view(), 0);
    const int n = ints.n_basis;
    for (int it = 0; it < 3; ++it) {
        const RealMatrix Pt = test_density(n, 0.1 * it),
                         Pa = test_density(n, 1.0 + 0.1 * it);
        const double scale = it == 2 ? 0.25 : 0.5;
        RealMatrix J, K, Jref, Kref, Jdummy, Kref_a;
        const internal::JKRequest step[] = {{&Pt, &J, nullptr, 1.0, 0},
                                            {&Pa, nullptr, &K, scale, 1}};
        engine.step(step);
        engine.compute(Pt, Jref, Kref);
        engine.compute(Pa, Jdummy, Kref_a);
        expect_close(J, Jref, 2e-10);
        expect_close(K, RealMatrix(scale * Kref_a), 2e-10);
    }
    RealMatrix J, K;
    const RealMatrix P = test_density(n, 0.0);
    const internal::JKRequest twice[] = {{&P, &J, nullptr, 1.0, 0},
                                         {&P, nullptr, &K, 1.0, 0}};
    EXPECT_THROW(engine.step(twice), std::invalid_argument);
}
