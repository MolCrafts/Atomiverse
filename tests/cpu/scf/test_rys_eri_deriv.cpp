/// @file tests/cpu/scf/test_rys_eri_deriv.cpp
/// @brief Unit test for the Rys-quadrature ERI derivative engine.
///
/// Strategy: finite-difference the plain `rys_eri_block` at shifted
/// shell centres and compare against the analytic output of
/// `rys_eri_block_deriv`. If the two agree to FD tolerance (~1e-7 with
/// h=1e-4) for every l-combination, the analytic derivative is
/// self-consistent with the integral it is differentiating.
///
/// Parametrized over angular momentum combinations reachable from
/// STO-3G: (ssss), (spss), (psps), (ppss), (ppps), (pppp) plus
/// reversed bra (spss with A=s, B=p) to exercise the non-canonical
/// swap path.
///
/// The Stage-1 `test_gradient_vs_libcint.cpp` handles absolute
/// correctness vs libcint at the gradient-contraction level.

#ifdef ATV_HAS_HF

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/rys_eri.h"

namespace {

using namespace ATV::cpu::scf::native;

struct PrimShell {
    int l;
    int cart; ///< For l=1, pick one cart component; 0 otherwise.
    double alpha[3];
    double coeff[3];
    int n_prim;
    double center[3];
};

/// Convert the plain description into a `ShellBlock` for the Rys engine.
/// `ShellBlock::coeff` is pointer-into-buffer; we keep the underlying
/// vectors in the test closure.
ShellBlock make_shell_block(const PrimShell& s) {
    ShellBlock b;
    b.n_prim = s.n_prim;
    b.alpha = s.alpha;
    b.coeff = s.coeff;
    b.center[0] = s.center[0];
    b.center[1] = s.center[1];
    b.center[2] = s.center[2];
    b.l = s.l;
    b.n_cart = (s.l + 1) * (s.l + 2) / 2;
    b.ao_start = 0;
    return b;
}

/// Compute the full integral block `(A B | C D)` via rys_eri_block.
std::vector<double> eri_block(const PrimShell& A, const PrimShell& B,
                              const PrimShell& C, const PrimShell& D) {
    const auto Ab = make_shell_block(A);
    const auto Bb = make_shell_block(B);
    const auto Cb = make_shell_block(C);
    const auto Db = make_shell_block(D);
    std::vector<double> out(static_cast<size_t>(Ab.n_cart) * Bb.n_cart
                                * Cb.n_cart * Db.n_cart,
                            0.0);
    rys_eri_block(Ab, Bb, Cb, Db, out.data());
    return out;
}

/// Compute the 3-axis derivative block via rys_eri_block_deriv.
std::vector<double> deriv_block(const PrimShell& A, const PrimShell& B,
                                const PrimShell& C, const PrimShell& D,
                                int deriv_center) {
    const auto Ab = make_shell_block(A);
    const auto Bb = make_shell_block(B);
    const auto Cb = make_shell_block(C);
    const auto Db = make_shell_block(D);
    const size_t n =
        static_cast<size_t>(Ab.n_cart) * Bb.n_cart * Cb.n_cart * Db.n_cart;
    std::vector<double> out(3 * n, 0.0);
    rys_eri_block_deriv(Ab, Bb, Cb, Db, deriv_center, out.data());
    return out;
}

/// Build a test shell with given angular momentum, 2 primitives and a
/// simple pair of (α, c) coefficients. Numerical values are chosen to
/// resemble real STO-3G primitives without depending on any particular
/// element's basis table.
PrimShell make_shell(int l, const double center[3]) {
    PrimShell s;
    s.l = l;
    s.cart = 0;
    s.n_prim = 3;
    s.alpha[0] = 3.0;
    s.coeff[0] = 0.4;
    s.alpha[1] = 1.0;
    s.coeff[1] = 0.5;
    s.alpha[2] = 0.3;
    s.coeff[2] = 0.3;
    s.center[0] = center[0];
    s.center[1] = center[1];
    s.center[2] = center[2];
    return s;
}

/// Finite-difference derivative of the ERI block w.r.t. one component
/// of one centre. Returns `(ERI(+h) - ERI(-h)) / (2h)` per integral.
std::vector<double> fd_deriv_block(PrimShell A, PrimShell B, PrimShell C,
                                   PrimShell D, int deriv_center, int axis,
                                   double h) {
    PrimShell& target = (deriv_center == 0)   ? A
                        : (deriv_center == 1) ? B
                        : (deriv_center == 2) ? C
                                              : D;
    const double saved = target.center[axis];
    target.center[axis] = saved + h;
    auto plus = eri_block(A, B, C, D);
    target.center[axis] = saved - h;
    auto minus = eri_block(A, B, C, D);
    target.center[axis] = saved;
    std::vector<double> fd(plus.size(), 0.0);
    for (size_t i = 0; i < plus.size(); ++i)
        fd[i] = (plus[i] - minus[i]) / (2.0 * h);
    return fd;
}

/// Compare analytic deriv to finite difference over all (axis, ia, ib,
/// ic, id) combinations. Tolerance is the user-provided FD-noise floor.
::testing::AssertionResult expect_close(const std::vector<double>& analytic,
                                        const std::vector<double>& fd_x,
                                        const std::vector<double>& fd_y,
                                        const std::vector<double>& fd_z,
                                        size_t n_block, double tol) {
    double max_diff = 0.0;
    int worst_axis = -1, worst_idx = -1;
    for (int axis = 0; axis < 3; ++axis) {
        const std::vector<double>& fd = (axis == 0)   ? fd_x
                                        : (axis == 1) ? fd_y
                                                      : fd_z;
        for (size_t i = 0; i < n_block; ++i) {
            const double a = analytic[axis * n_block + i];
            const double f = fd[i];
            const double d = std::abs(a - f);
            if (d > max_diff) {
                max_diff = d;
                worst_axis = axis;
                worst_idx = static_cast<int>(i);
            }
        }
    }
    if (max_diff < tol)
        return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << "max |analytic - FD| = " << max_diff << " at axis=" << worst_axis
           << " idx=" << worst_idx << " (tol=" << tol << ")";
}

/// Run the analytic-vs-FD comparison for a given shell quartet and
/// deriv-centre. Covers all 3 Cartesian axes in one call.
void check_deriv_vs_fd(const PrimShell& A, const PrimShell& B,
                       const PrimShell& C, const PrimShell& D, int deriv_center,
                       double tol = 1e-6) {
    const auto analytic = deriv_block(A, B, C, D, deriv_center);
    const double h = 1e-4;
    const auto fd_x = fd_deriv_block(A, B, C, D, deriv_center, 0, h);
    const auto fd_y = fd_deriv_block(A, B, C, D, deriv_center, 1, h);
    const auto fd_z = fd_deriv_block(A, B, C, D, deriv_center, 2, h);
    const size_t n_block = fd_x.size();
    EXPECT_TRUE(expect_close(analytic, fd_x, fd_y, fd_z, n_block, tol));
}

// ── Shared centres ───────────────────────────────────────────────────

const double RA[3] = {0.0, 0.0, 0.0};
const double RB[3] = {0.9, 0.0, 0.0};
const double RC[3] = {0.3, 0.7, 0.0};
const double RD[3] = {-0.2, 0.1, 0.5};

// ── Tests: l-combinations × deriv-centres ────────────────────────────

TEST(RysEriDeriv, SSSS_All4Centers) {
    const auto A = make_shell(0, RA);
    const auto B = make_shell(0, RB);
    const auto C = make_shell(0, RC);
    const auto D = make_shell(0, RD);
    for (int dc = 0; dc < 4; ++dc)
        check_deriv_vs_fd(A, B, C, D, dc);
}

TEST(RysEriDeriv, PSSS_DerivOnP) {
    const auto A = make_shell(1, RA); // p-shell on the A (i-side) center
    const auto B = make_shell(0, RB);
    const auto C = make_shell(0, RC);
    const auto D = make_shell(0, RD);
    check_deriv_vs_fd(A, B, C, D, 0);
}

TEST(RysEriDeriv, SPSS_DerivOnP_nonCanonicalBra) {
    // A is s, B is p: requires the wrapper's swap_ab path.
    const auto A = make_shell(0, RA);
    const auto B = make_shell(1, RB);
    const auto C = make_shell(0, RC);
    const auto D = make_shell(0, RD);
    check_deriv_vs_fd(A, B, C, D, 0); // ∂/∂R_A (s-side)
    check_deriv_vs_fd(A, B, C, D, 1); // ∂/∂R_B (p-side, originally j)
}

TEST(RysEriDeriv, PPPS_DerivOnAllBra) {
    const auto A = make_shell(1, RA);
    const auto B = make_shell(1, RB);
    const auto C = make_shell(1, RC);
    const auto D = make_shell(0, RD);
    check_deriv_vs_fd(A, B, C, D, 0);
    check_deriv_vs_fd(A, B, C, D, 1);
}

TEST(RysEriDeriv, PPPS_DerivOnKet) {
    const auto A = make_shell(1, RA);
    const auto B = make_shell(1, RB);
    const auto C = make_shell(1, RC);
    const auto D = make_shell(0, RD);
    check_deriv_vs_fd(A, B, C, D, 2); // ∂/∂R_C
    check_deriv_vs_fd(A, B, C, D, 3); // ∂/∂R_D (non-canonical ket via l=1>0)
}

TEST(RysEriDeriv, PPPP_All4Centers) {
    const auto A = make_shell(1, RA);
    const auto B = make_shell(1, RB);
    const auto C = make_shell(1, RC);
    const auto D = make_shell(1, RD);
    for (int dc = 0; dc < 4; ++dc)
        check_deriv_vs_fd(A, B, C, D, dc);
}

TEST(RysEriDeriv, SPPS_BothSwaps) {
    // A=s, B=p, C=s, D=p: both bra and ket non-canonical.
    const auto A = make_shell(0, RA);
    const auto B = make_shell(1, RB);
    const auto C = make_shell(0, RC);
    const auto D = make_shell(1, RD);
    for (int dc = 0; dc < 4; ++dc)
        check_deriv_vs_fd(A, B, C, D, dc);
}

// ── d-shell coverage (Phase 1 Stage A) ───────────────────────────────
// Prerequisite for def2-SVP / 6-31G* basis sets. Rys engine advertises
// d support (kRysMaxL = 2); these tests prove it end-to-end — both the
// plain ERI path (exercised inside the ±h finite difference) and the
// analytic derivative on d-containing quartets.
//
// DDDD has Ltot = 8 for plain ERI → 5 Rys roots; DDDD-deriv raises one
// bra or ket axis to 3, Ltot = 9 → still 5 roots under the canonical
// ceil((L+1)/2) formula. kRysMaxRoots = 5 is therefore exactly at the
// budget ceiling — a DDDD failure here would mean the root table
// accuracy at nroots = 5 is insufficient on derivative grids.

TEST(RysEriDeriv, DSSS_DerivOnD) {
    const auto A = make_shell(2, RA);
    const auto B = make_shell(0, RB);
    const auto C = make_shell(0, RC);
    const auto D = make_shell(0, RD);
    check_deriv_vs_fd(A, B, C, D, 0);
}

TEST(RysEriDeriv, SDSS_DerivOnD_nonCanonicalBra) {
    // A=s, B=d: requires the swap_ab path at Ltot=2 on bra.
    const auto A = make_shell(0, RA);
    const auto B = make_shell(2, RB);
    const auto C = make_shell(0, RC);
    const auto D = make_shell(0, RD);
    check_deriv_vs_fd(A, B, C, D, 0);
    check_deriv_vs_fd(A, B, C, D, 1);
}

TEST(RysEriDeriv, DPSS_DerivOnBra) {
    const auto A = make_shell(2, RA);
    const auto B = make_shell(1, RB);
    const auto C = make_shell(0, RC);
    const auto D = make_shell(0, RD);
    check_deriv_vs_fd(A, B, C, D, 0);
    check_deriv_vs_fd(A, B, C, D, 1);
}

TEST(RysEriDeriv, DDSS_DerivOnBra) {
    const auto A = make_shell(2, RA);
    const auto B = make_shell(2, RB);
    const auto C = make_shell(0, RC);
    const auto D = make_shell(0, RD);
    check_deriv_vs_fd(A, B, C, D, 0);
    check_deriv_vs_fd(A, B, C, D, 1);
}

TEST(RysEriDeriv, DDPP_DerivOnBra) {
    const auto A = make_shell(2, RA);
    const auto B = make_shell(2, RB);
    const auto C = make_shell(1, RC);
    const auto D = make_shell(1, RD);
    check_deriv_vs_fd(A, B, C, D, 0);
    check_deriv_vs_fd(A, B, C, D, 1);
}

TEST(RysEriDeriv, DDPP_DerivOnKet) {
    const auto A = make_shell(2, RA);
    const auto B = make_shell(2, RB);
    const auto C = make_shell(1, RC);
    const auto D = make_shell(1, RD);
    check_deriv_vs_fd(A, B, C, D, 2);
    check_deriv_vs_fd(A, B, C, D, 3);
}

TEST(RysEriDeriv, DDDD_All4Centers) {
    // Maximum-angular-momentum coverage reachable from def2-SVP-sized
    // bases. Plain ERI Ltot=8 (5 roots); derivative enlarges to Ltot=9
    // (still 5 roots under ceil rule). This is the tightest budget
    // check in the suite.
    const auto A = make_shell(2, RA);
    const auto B = make_shell(2, RB);
    const auto C = make_shell(2, RC);
    const auto D = make_shell(2, RD);
    for (int dc = 0; dc < 4; ++dc)
        check_deriv_vs_fd(A, B, C, D, dc);
}

} // namespace

#endif // ATV_HAS_HF
