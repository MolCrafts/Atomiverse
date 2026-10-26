#include "atomiverse/cpu/scf/internal/exact_jk.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include <cmath>
#include <gtest/gtest.h>
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
        f.span(kAtom_type)[i] = z[i];
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
