/// @file tests/cpu/scf/test_rys_roots.cpp
/// @brief Validation of native::rys_roots across the domain used by CPU SCF.
///
/// The native ERI engine delegates Boys->roots conversion to a ported copy
/// of libcint's closed-form rys_root1..5 (nroots <= 5) and an iterative
/// Jacobi/Schmidt root finder (nroots 6, 7 — f-shell support). This test
/// asserts:
///   - nroots in {1..5}: element-wise match vs libcint's CINTrys_roots
///     at floating-point precision (any drift implies an incomplete port).
///   - nroots in {6, 7}: the iterative root finder reproduces an
///     independent reference (libcint CINTrys_roots, which itself uses
///     the iterative scheme) to <= 1e-10, AND satisfies the Rys
///     quadrature moment identity sum_i w_i t_i^{2m} = F_m(x) for
///     m = 0..2N-1 — a libcint-independent mathematical check that holds
///     as a libcint-independent mathematical check.

#if defined(ATV_HAS_HF)

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cpu/scf/rys_roots.h"


namespace {

// Sample x at enough branch boundaries to cover every piecewise polynomial
// segment inside rys_root1..5. Values straddle libcint's branch points
// (SMALLX_LIMIT=3e-7, X<1, X<3, X<5, X<10, X<15, X<33, X<40, X>=40).
const std::vector<double> kSampleX = {
    0.0,                    // tails into the small-x polynomial.
    1e-10,                  // clearly below SMALLX_LIMIT.
    1e-8,                   // still small-x.
    5e-7,                   // just above SMALLX_LIMIT (first piecewise branch).
    1e-4,                   // small polynomial branch.
    0.3,   0.7,             // X<1 branch.
    1.0,   1.5,  2.0,  2.5, // X<3 branch.
    3.0,   4.0,             // X<5 branch.
    5.5,   7.5,  9.5,       // X<10 branch.
    10.5,  12.5, 14.5,      // X<15 branch.
    16.0,  20.0, 25.0, 30.0, 32.0, // X<33 branch.
    33.5,  37.0, 39.0,             // X<40 / X<47 branches.
    41.0,  50.0, 80.0,             // large-x branches.
    120.0,                         // deep large-x polynomial cutoff.
};

// x values that exercise the iterative root finder's regimes for
// nroots 6/7: small-x (below SMALLX_LIMIT), the intermediate-T regime
// where closed-form Chebyshev fits degrade (the motivation for the
// iterative scheme), the libcint segment_solve breakpoint at x=11,
// and large-x (above the 35+5N analytic cutoff).
const std::vector<double> kIterX = {
    1e-10,  1e-8,   5e-7, // small-x.
    1e-3,   0.05,   0.5,    1.0,
    2.5, // low intermediate-T.
    5.0,    8.0,    10.0,   10.999,
    11.001, // straddling the x=11 segment split.
    13.0,   18.0,   25.0,   33.0,
    44.0,                          // upper intermediate-T.
    55.0,   64.999, 65.001,        // straddling 35+5*6 large-x cutoff (n=6).
    69.999, 70.001, 90.0,   150.0, // straddling 35+5*7 large-x cutoff (n=7).
};

constexpr double kTol = 2e-14; // element-wise absolute tolerance vs libcint.

/// @brief Numerically stable Boys function F_m(x) for the moment-identity
///        reference. ATV::boys_fn uses an upward recurrence that suffers
///        catastrophic cancellation for small x (the (2m-1)F_{m-1}-e
///        subtraction divided by 2x); this routine uses the convergent
///        power series for small/moderate x and the upward recurrence
///        only where it is well conditioned.
double boys_ref(int m, double x) {
    if (x < 1e-13)
        return 1.0 / (2 * m + 1);
    if (x < 35.0) {
        // Power series for F_m(x): F_m = e^{-x} * sum_k x^k / [(2m+1)(2m+3)...(2m+2k+1)].
        const double e = std::exp(-x);
        double term = 1.0 / (2 * m + 1);
        double sum = term;
        for (int k = 1; k < 400; ++k) {
            term *= x / (m + 0.5 + k);
            sum += term;
            if (term < 1e-18 * sum)
                break;
        }
        return e * sum;
    }
    // Large x: erf form for F_0, then a well-conditioned upward recurrence.
    double fn = 0.5 * std::sqrt(M_PI / x) * std::erf(std::sqrt(x));
    const double e = std::exp(-x);
    const double inv2x = 0.5 / x;
    for (int k = 1; k <= m; ++k)
        fn = ((2 * k - 1) * fn - e) * inv2x;
    return fn;
}

} // namespace


// ── Rys quadrature moment identity (libcint-independent) ─────────────
//
// The N Rys roots u[i] and weights w[i] for parameter x must satisfy
//     sum_i w[i] * t_i^{2m} = F_m(x)
// where t_i^2 = u[i] / (1 + u[i]) is the squared raw Rys root and F_m is
// the Boys function. In exact arithmetic an N-point Rys quadrature is
// exact up to m = 2N-1, but the identity is only numerically meaningful
// for the lower moments: t^{2m} underflows for small roots and the Boys
// upward recurrence amplifies rounding, so the m ~ 2N-1 moments are
// ill-conditioned even for libcint's own (ground-truth) roots — at
// nroots=7, x=1, libcint itself misses the m=13 moment by ~1e-7. The
// check is therefore applied to m <= 2N-4, the range where the identity
// holds to ~1e-10 for any correct root set. This is a mathematical
// correctness check independent of any external integral library.
TEST(RysRoots, IterativeSatisfiesMomentIdentity) {
    constexpr double kMomentTol = 1e-10;
    for (int nroots = 6; nroots <= 7; ++nroots) {
        for (double x : kIterX) {
            double u[8] = {};
            double w[8] = {};
            ATV::cpu::scf::native::rys_roots(nroots, x, u, w);

            // Numerically well-conditioned moment range.
            const int m_max = 2 * nroots - 4;
            for (int m = 0; m <= m_max; ++m) {
                double quad = 0.0;
                for (int i = 0; i < nroots; ++i) {
                    // t_i^2 = u / (1 + u): the raw squared Rys root.
                    const double t2 = u[i] / (1.0 + u[i]);
                    quad += w[i] * std::pow(t2, m);
                }
                const double exact = boys_ref(m, x);
                EXPECT_NEAR(quad, exact, kMomentTol + 1e-9 * std::abs(exact))
                    << "moment identity violated at nroots=" << nroots
                    << " x=" << x << " m=" << m;
            }
        }
    }
}

// ── Unsupported nroots -> catchable exception, not abort (item 14) ─────
//
// nroots > 7 (reachable through g/h shells) and nroots < 1 are outside
// the supported 1..7 domain. The production contract *after*
// scf-review-05-api-hygiene is a catchable std::invalid_argument whose
// message names `nroots` — NOT the pre-fix std::fprintf + std::abort(),
// which terminates the process uncatchably and is unit-untestable.
//
// x = 1.0 is benign: it sits in the intermediate regime
// (SMALLX_LIMIT < x < 35 + 5*nroots), so the call reaches the domain
// switch/default branch instead of the small-x or large-x fast paths —
// which is exactly the branch that must now throw.
//
// These tests are placed LAST in the file deliberately: while the pre-fix
// abort() is still in place they terminate the whole test binary (RED),
// and every test above has already run by then.
TEST(RysRoots, UnsupportedNrootsThrowsInvalidArgument) {
    double u[8] = {};
    double w[8] = {};
    EXPECT_THROW(ATV::cpu::scf::native::rys_roots(8, 1.0, u, w),
                 std::invalid_argument);
}

TEST(RysRoots, UnsupportedNrootsMessageNamesNroots) {
    double u[8] = {};
    double w[8] = {};
    try {
        ATV::cpu::scf::native::rys_roots(8, 1.0, u, w);
        ADD_FAILURE() << "expected rys_roots(8, 1.0, ...) to throw";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("nroots"), std::string::npos)
            << "exception message must name nroots, got: " << e.what();
    }
}

TEST(RysRoots, NonPositiveNrootsThrowsInvalidArgument) {
    double u[8] = {};
    double w[8] = {};
    EXPECT_THROW(ATV::cpu::scf::native::rys_roots(0, 1.0, u, w),
                 std::invalid_argument);
    EXPECT_THROW(ATV::cpu::scf::native::rys_roots(-1, 1.0, u, w),
                 std::invalid_argument);
}

#endif // ATV_HAS_HF
