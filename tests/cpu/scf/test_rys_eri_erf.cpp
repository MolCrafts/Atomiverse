/// @file tests/cpu/scf/test_rys_eri_erf.cpp
/// @brief Unit tests for native::rys_eri_erf_block (range-separated erf Coulomb).

#if defined(ATV_HAS_HF)

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/rys_eri.h"

using namespace ATV::cpu::scf;

namespace {

native::ShellBlock s_shell(const double* alpha, const double* coeff,
                           const double* xyz) {
    native::ShellBlock s{};
    s.n_prim = 1;
    s.alpha = alpha;
    s.coeff = coeff;
    s.center[0] = xyz[0];
    s.center[1] = xyz[1];
    s.center[2] = xyz[2];
    s.l = 0;
    s.n_cart = 1;
    s.ao_start = 0;
    return s;
}

native::ShellBlock p_shell(const double* alpha, const double* coeff,
                           const double* xyz) {
    native::ShellBlock p = s_shell(alpha, coeff, xyz);
    p.l = 1;
    p.n_cart = 3;
    return p;
}

} // namespace

TEST(TestRysEriErf, OmegaZeroMatchesBareSAndP) {
    const double alpha = 1.0;
    const double coeff = native::phys_norm(1.0);
    const double xyz_a[3] = {0.0, 0.0, 0.0};
    const double xyz_b[3] = {0.4, 0.1, -0.2};
    const auto A = s_shell(&alpha, &coeff, xyz_a);
    const auto B = p_shell(&alpha, &coeff, xyz_b);
    const auto C = s_shell(&alpha, &coeff, xyz_a);
    const auto D = p_shell(&alpha, &coeff, xyz_b);

    const int n = A.n_cart * B.n_cart * C.n_cart * D.n_cart;
    std::vector<double> bare(n, 0.0);
    std::vector<double> erf0(n, 0.0);
    native::rys_eri_block(A, B, C, D, bare.data());
    native::rys_eri_erf_block(A, B, C, D, /*omega=*/0.0, erf0.data());

    double worst = 0.0;
    for (int i = 0; i < n; ++i)
        worst = std::max(worst, std::abs(bare[i] - erf0[i]));
    EXPECT_LE(worst, 1e-12);
}

TEST(TestRysEriErf, SsssErfPinVsPyscfLibcint) {
    // One s primitive, α = 1 Bohr^{-2}, libcint-normalized coeff.
    // Offline pin: PySCF 2.14.0 `mol.intor('int2e_cart')` with
    // `mol.omega = 0.33`, atom H at origin, basis [[0, [1.0, 1.0]]],
    // cart=True, unit=Bohr, 2026-08-17.
    const double alpha = 1.0;
    const double coeff = native::phys_norm(1.0);
    const double xyz[3] = {0.0, 0.0, 0.0};
    const auto S = s_shell(&alpha, &coeff, xyz);

    std::vector<double> val(1, 0.0);
    native::rys_eri_erf_block(S, S, S, S, /*omega=*/0.33, val.data());

    constexpr double kPin = 0.353608590761111; // Ha, PySCF 2.14.0 / libcint
    EXPECT_NEAR(val[0], kPin, 1e-10);
}

TEST(TestRysEriErf, NegativeOmegaThrows) {
    const double alpha = 1.0;
    const double coeff = native::phys_norm(1.0);
    const double xyz[3] = {0.0, 0.0, 0.0};
    const auto S = s_shell(&alpha, &coeff, xyz);
    double out = 0.0;
    EXPECT_THROW(native::rys_eri_erf_block(S, S, S, S, -0.1, &out),
                 std::invalid_argument);
}

TEST(TestRysEriErf, OffCenterValueAndDerivativeVsPyscf) {
    // PySCF 2.14.0 / libcint, four normalized s primitives (alpha=1),
    // mol.with_range_coulomb(0.33); negative int2e_ip1_cart is the nuclear
    // derivative on A. Reproduce with
    // tools/diagnostics/molecular-dft/pin_erf.py (full output retained there).
    const double alpha = 1.0, coeff = native::phys_norm(alpha);
    const double xyz[4][3] = {
        {0.1, -0.2, 0.3}, {0.8, 0.1, -0.4}, {-0.2, 0.6, 0.5}, {0.3, -0.5, 0.9}};
    const auto A = s_shell(&alpha, &coeff, xyz[0]);
    const auto B = s_shell(&alpha, &coeff, xyz[1]);
    const auto C = s_shell(&alpha, &coeff, xyz[2]);
    const auto D = s_shell(&alpha, &coeff, xyz[3]);
    double value = 0.0, deriv[3] = {};
    native::rys_eri_erf_block(A, B, C, D, 0.33, &value);
    native::rys_eri_erf_block_deriv(A, B, C, D, 0, 0.33, deriv);
    EXPECT_NEAR(value, 0.08996683143489044, 1e-12);
    EXPECT_NEAR(deriv[0], 0.06182119282155195, 1e-12);
    EXPECT_NEAR(deriv[1], 0.02727894672618497, 1e-12);
    EXPECT_NEAR(deriv[2], -0.06081005228653951, 1e-12);
}

TEST(TestRysEriErf, DerivativeMatchesBareAtZeroAndLargeOmega) {
    const double alpha = 0.8;
    const double coeff = native::phys_norm(alpha);
    const double a[3] = {0.1, -0.3, 0.2}, b[3] = {0.7, 0.2, -0.4};
    const auto A = s_shell(&alpha, &coeff, a);
    const auto B = p_shell(&alpha, &coeff, b);
    const int n = 3 * A.n_cart * B.n_cart * B.n_cart * A.n_cart;
    for (int center = 0; center < 4; ++center) {
        std::vector<double> bare(n), zero(n), large(n);
        native::rys_eri_block_deriv(A, B, B, A, center, bare.data());
        native::rys_eri_erf_block_deriv(A, B, B, A, center, 0.0, zero.data());
        native::rys_eri_erf_block_deriv(A, B, B, A, center, 1e6, large.data());
        for (int i = 0; i < n; ++i) {
            EXPECT_DOUBLE_EQ(zero[i], bare[i]);
            EXPECT_NEAR(large[i], bare[i], 1e-10);
        }
    }
}

TEST(TestRysEriErf, NuclearDerivativesAllCentersAndShellSwaps) {
    // Independent finite differences of the value kernel exercise both
    // canonicalization swaps, angular raising/lowering, all nuclear centers
    // and contracted primitives. Molecular CAM-B3LYP oracle checks live in
    // tools/diagnostics/molecular-dft rather than this integral unit test.
    const double alpha[2] = {0.4, 1.2}, coeff[2] = {0.7, 0.3};
    const double xyz[4][3] = {
        {0.1, -0.2, 0.3}, {0.8, 0.1, -0.4}, {-0.2, 0.6, 0.5}, {0.3, -0.5, 0.9}};
    for (const auto angular : {std::array{0, 1, 1, 2}, std::array{2, 1, 0, 1},
                               std::array{1, 0, 3, 1}}) {
        std::array<native::ShellBlock, 4> shells;
        int n = 1;
        for (int i = 0; i < 4; ++i) {
            shells[i] = s_shell(alpha, coeff, xyz[i]);
            shells[i].n_prim = 2;
            shells[i].l = angular[i];
            shells[i].n_cart = n_cart_of(angular[i]);
            n *= shells[i].n_cart;
        }
        for (double omega : {0.12, 0.33, 1.1}) {
            std::vector<double> sum(3 * n);
            for (int center = 0; center < 4; ++center) {
                std::vector<double> deriv(3 * n);
                native::rys_eri_erf_block_deriv(shells[0], shells[1], shells[2],
                                                shells[3], center, omega,
                                                deriv.data());
                for (int i = 0; i < 3 * n; ++i)
                    sum[i] += deriv[i];
                for (int axis = 0; axis < 3; ++axis) {
                    auto plus = shells, minus = shells;
                    constexpr double h = 1e-5;
                    plus[center].center[axis] += h;
                    minus[center].center[axis] -= h;
                    std::vector<double> vp(n), vm(n);
                    native::rys_eri_erf_block(plus[0], plus[1], plus[2],
                                              plus[3], omega, vp.data());
                    native::rys_eri_erf_block(minus[0], minus[1], minus[2],
                                              minus[3], omega, vm.data());
                    for (int i = 0; i < n; ++i)
                        EXPECT_NEAR(deriv[axis * n + i],
                                    (vp[i] - vm[i]) / (2 * h), 2e-8);
                }
            }
            for (double value : sum)
                EXPECT_NEAR(value, 0.0, 1e-11);
        }
    }
}

TEST(TestRysEriErf, DerivativeRejectsInvalidParameters) {
    const double alpha = 1.0, coeff = 1.0, xyz[3] = {};
    const auto S = s_shell(&alpha, &coeff, xyz);
    double result[3] = {};
    for (double omega : {-0.1, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW(
            native::rys_eri_erf_block_deriv(S, S, S, S, 0, omega, result),
            std::invalid_argument);
    for (int center : {-1, 4})
        EXPECT_THROW(
            native::rys_eri_erf_block_deriv(S, S, S, S, center, 0.33, result),
            std::invalid_argument);
}

#endif // ATV_HAS_HF
