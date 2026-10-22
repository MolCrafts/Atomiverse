/// @file tests/cpu/scf/test_kpoint_hf.cpp
/// @brief Unit pins for the FFTDF k-point HF pieces (no SCF loop).
///
/// Ewald and the FFT mesh are the PySCF 2.14.0 numbers for the H₂
/// cell used by regressions/scf/periodic-hf (4 Bohr cube).

#if defined(ATV_HAS_HF)

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/scf/internal/kpoint_hf.h"

using ATV::cpu::scf::internal::ewald_nuclear;
using ATV::cpu::scf::internal::fft3;
using ATV::cpu::scf::internal::fft_mesh_orthorhombic;
using ATV::cpu::scf::internal::monkhorst_kpts;
using ATV::cpu::scf::internal::PgtoSpec;

TEST(KpointHf, EwaldH2MatchesPyscf) {
    const double Z[2] = {1.0, 1.0};
    const double R[2][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
    const double L[3] = {4.0, 4.0, 4.0};
    // pyscf.pbc.gto.cell.ewald on this cell (precision 1e-8).
    const double got = ewald_nuclear(Z, R, 2, L, 1e-8);
    EXPECT_NEAR(got, -0.6281049380287604, 1e-10);
}

TEST(KpointHf, Sto3gMeshMatchesPyscf) {
    // libcint contraction coeffs of H STO-3G after cell.build (l = 0).
    const PgtoSpec prims[] = {
        {0, 3.42525091, 0.98170675},
        {0, 0.62391373, 0.94946401},
        {0, 0.16885540, 0.29590645},
    };
    const double L[3] = {4.0, 4.0, 4.0};
    const auto mesh = fft_mesh_orthorhombic(L, prims, 3, 1e-8);
    EXPECT_EQ(mesh[0], 31);
    EXPECT_EQ(mesh[1], 31);
    EXPECT_EQ(mesh[2], 31);
}

TEST(KpointHf, MonkhorstPackGammaThenHalf) {
    const int n[3] = {1, 1, 2};
    const double L[3] = {4.0, 4.0, 4.0};
    std::vector<std::array<double, 3>> k;
    monkhorst_kpts(n, L, k);
    ASSERT_EQ(k.size(), 2u);
    EXPECT_NEAR(k[0][0], 0.0, 1e-14);
    EXPECT_NEAR(k[0][1], 0.0, 1e-14);
    EXPECT_NEAR(k[0][2], 0.0, 1e-14);
    // s_z = 1/2, b_z = 2π/4, wrap_around is off.
    EXPECT_NEAR(k[1][2], 0.5 * 2.0 * 3.14159265358979323846 / 4.0, 1e-14);
    EXPECT_NEAR(k[1][0], 0.0, 1e-14);
}

TEST(KpointHf, FftRoundTripConstant) {
    const int n = 8;
    std::vector<std::complex<double>> f(static_cast<std::size_t>(n * n * n),
                                        {2.5, 0.0});
    auto g = f;
    fft3(g, n, n, n, false);
    EXPECT_NEAR(g[0].real(), 2.5 * n * n * n, 1e-9);
    EXPECT_NEAR(g[0].imag(), 0.0, 1e-9);
    double rest = 0.0;
    for (std::size_t i = 1; i < g.size(); ++i)
        rest += std::abs(g[i]);
    EXPECT_LT(rest, 1e-8);
    fft3(g, n, n, n, true);
    for (std::size_t i = 0; i < f.size(); ++i) {
        EXPECT_NEAR(g[i].real(), f[i].real(), 1e-9);
        EXPECT_NEAR(g[i].imag(), 0.0, 1e-9);
    }
}

#endif
