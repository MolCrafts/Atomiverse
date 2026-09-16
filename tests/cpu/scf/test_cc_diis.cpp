/// @file tests/cpu/scf/test_cc_diis.cpp
/// @brief Isolated unit tests for the CC amplitude DIIS accelerator
///        (internal/cc_diis.h, ATV::cpu::scf::internal::CcAmplitudeDiis).
///
/// scf-review-08-ccsd-spatial-diis, deliverable A. `CcAmplitudeDiis` mirrors
/// `internal::DIISMixer` (max_vectors ctor, step/reset/size, deque history) but
/// over FLATTENED amplitude vectors T = [t1; t2] with error e = Δt (the Jacobi
/// increment). Pure algebra — no SCF, no CCSD solve. Written RED-first: the
/// header does not exist yet.
///
/// Reference: Scuseria, Lee, Schaefer, Chem. Phys. Lett. 130, 236 (1986);
/// Pulay, Chem. Phys. Lett. 73, 393 (1980).

#ifdef ATV_HAS_HF

#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/scf/internal/cc_diis.h"

using ATV::cpu::scf::internal::CcAmplitudeDiis;

// (a) happy path: a contractive linear fixed-point map g(x) = A x + b, with the
//     DIIS error the Jacobi increment e = g(x) − x, driven below a hard-coded
//     residual threshold and onto the hard-coded exact fixed point. We assert
//     the final residual ≤ threshold and x → x* — NOT any speedup factor.
//
//     A = [[0.5, 0.1], [0.2, 0.4]]  (eigenvalues 0.6, 0.3 → contractive),
//     b = [0.4, 0.4]  ⇒  x* = (I − A)^{-1} b = [1.0, 1.0].
TEST(TestCcAmplitudeDiis, LinearFixedPointConvergesToExactSolution) {
    const double A[2][2] = {{0.5, 0.1}, {0.2, 0.4}};
    const double b[2] = {0.4, 0.4};
    const std::vector<double> x_star = {1.0,
                                        1.0}; // hard-coded exact fixed point

    CcAmplitudeDiis diis(8);
    std::vector<double> x = {0.0, 0.0}; // cold start
    for (int k = 0; k < 12; ++k) {
        const std::vector<double> gx = {A[0][0] * x[0] + A[0][1] * x[1] + b[0],
                                        A[1][0] * x[0] + A[1][1] * x[1] + b[1]};
        const std::vector<double> err = {gx[0] - x[0], gx[1] - x[1]};
        x = diis.step(gx, err); // store (amp, err), return extrapolated amp
    }

    // Final Jacobi increment at the converged x.
    const std::vector<double> gx = {A[0][0] * x[0] + A[0][1] * x[1] + b[0],
                                    A[1][0] * x[0] + A[1][1] * x[1] + b[1]};
    const double final_res = std::sqrt((gx[0] - x[0]) * (gx[0] - x[0])
                                       + (gx[1] - x[1]) * (gx[1] - x[1]));

    EXPECT_LE(final_res, 1e-9)
        << "DIIS must drive the increment below threshold";
    EXPECT_NEAR(x[0], x_star[0], 1e-8);
    EXPECT_NEAR(x[1], x_star[1], 1e-8);
}

// (b) edge: a single stored history vector must extrapolate to the input amp
//     verbatim (c_0 = 1). Mirrors DIISMixer's SingleEntry contract.
TEST(TestCcAmplitudeDiis, SingleHistoryReturnsInputVerbatim) {
    CcAmplitudeDiis diis(8);
    const std::vector<double> amp = {0.3, -1.2, 4.5};
    const std::vector<double> err = {0.1, -0.2, 0.05};

    const std::vector<double> out = diis.step(amp, err);
    ASSERT_EQ(out.size(), amp.size());
    for (std::size_t i = 0; i < amp.size(); ++i)
        EXPECT_DOUBLE_EQ(out[i], amp[i]);
    EXPECT_EQ(diis.size(), 1);
}

// (c) edge: two nearly-parallel error vectors make the Pulay B matrix
//     near-singular; regularization / oldest-drop must keep the extrapolant
//     finite (no NaN / inf).
TEST(TestCcAmplitudeDiis, NearSingularBMatrixReturnsFinite) {
    CcAmplitudeDiis diis(8);
    diis.step({1.0, 2.0}, {1.0, 1.0}); // first pair
    const std::vector<double> out =
        diis.step({1.5, 2.5}, {1.0, 1.0 + 1e-11}); // nearly parallel error

    ASSERT_EQ(out.size(), std::size_t{2});
    for (double v : out)
        EXPECT_TRUE(std::isfinite(v));
}

// (d) state: size() counts stored (amp, err) pairs up to max_vectors (oldest
//     dropped beyond), and reset() empties the history. Four length-4 error
//     vectors are linearly independent, so the bookkeeping is exercised without
//     numerical degeneracy.
TEST(TestCcAmplitudeDiis, SizeCountsHistoryAndResetClears) {
    CcAmplitudeDiis diis(3); // max_vectors = 3
    EXPECT_EQ(diis.size(), 0);

    diis.step({1.0, 2.0, 3.0, 4.0}, {1.0, 0.0, 0.0, 0.0});
    EXPECT_EQ(diis.size(), 1);
    diis.step({2.0, 3.0, 4.0, 5.0}, {0.0, 1.0, 0.0, 0.0});
    EXPECT_EQ(diis.size(), 2);
    diis.step({3.0, 4.0, 5.0, 6.0}, {0.0, 0.0, 1.0, 0.0});
    EXPECT_EQ(diis.size(), 3);
    diis.step({4.0, 5.0, 6.0, 7.0}, {0.0, 0.0, 0.0, 1.0});
    EXPECT_EQ(diis.size(), 3) << "oldest pair dropped beyond max_vectors";

    diis.reset();
    EXPECT_EQ(diis.size(), 0);
}

#endif // ATV_HAS_HF
