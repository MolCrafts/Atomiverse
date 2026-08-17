/// @file tests/cpu/scf/test_rys_eri_erf.cpp
/// @brief Unit tests for native::rys_eri_erf_block (range-separated erf Coulomb).

#if defined(ATV_HAS_HF)

#include <gtest/gtest.h>

#include <cmath>
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

#endif // ATV_HAS_HF
