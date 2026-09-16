/// @file tests/cpu/scf/test_integrals.cpp
/// @brief Comprehensive validation of native C++ molecular integrals.
///
/// Two test suites:
///   NativeIntegrals     -- always-on (ATV_HAS_HF): symmetry, analytical, multi-geometry

#ifdef ATV_HAS_HF

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/basis_data/accessors.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/native_integrals.h"

using namespace ATV;
using namespace ATV::cpu::scf;

namespace {

/// @brief Build an H2 frame at a given bond length along x.
cpu::Frame make_h2(double bond_length) {
    cpu::Frame frame(2);
    frame.span(ATV::kAtom_x)[0] = double(0);
    frame.span(ATV::kAtom_y)[0] = double(0);
    frame.span(ATV::kAtom_z)[0] = double(0);
    frame.span(ATV::kAtom_x)[1] = bond_length;
    frame.span(ATV::kAtom_y)[1] = double(0);
    frame.span(ATV::kAtom_z)[1] = double(0);
    frame.span(ATV::kAtom_mass)[0] = double(1.008);
    frame.span(ATV::kAtom_mass)[1] = double(1.008);
    frame.span(ATV::kAtom_type)[0] = 1;
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.box = ATV::cpu::Box::cubic(100.0);
    return frame;
}

/// @brief Build Shell structs for native API (physical normalization).
///
/// Exponents/coeffs from `basis_data::pure_s_sto3g_shell(1)` — no retyped table.
struct H2Setup {
    std::vector<double> norm_coeff;
    double centers[2][3];
    native::Shell shells[2];
    native::Atom atoms[2];

    explicit H2Setup(double bond_angstrom = 0.74) {
        const auto& h1s = basis_data::pure_s_sto3g_shell(/*Z=*/1);
        const int n_prim = static_cast<int>(h1s.alpha.size());
        norm_coeff.resize(static_cast<std::size_t>(n_prim));
        for (int p = 0; p < n_prim; ++p)
            norm_coeff[static_cast<std::size_t>(p)] =
                h1s.coeff[static_cast<std::size_t>(p)]
                * native::phys_norm(h1s.alpha[static_cast<std::size_t>(p)]);
        centers[0][0] = 0;
        centers[0][1] = 0;
        centers[0][2] = 0;
        centers[1][0] = bond_angstrom * units::kAngstromInBohr;
        centers[1][1] = 0;
        centers[1][2] = 0;
        for (int s = 0; s < 2; ++s) {
            shells[s].n_prim = n_prim;
            shells[s].alpha = h1s.alpha.data();
            shells[s].coeff = norm_coeff.data();
            shells[s].center[0] = centers[s][0];
            shells[s].center[1] = centers[s][1];
            shells[s].center[2] = centers[s][2];
            shells[s].l = 0;
            shells[s].cart = 0;
        }
        for (int a = 0; a < 2; ++a) {
            atoms[a].Z = 1;
            atoms[a].pos[0] = centers[a][0];
            atoms[a].pos[1] = centers[a][1];
            atoms[a].pos[2] = centers[a][2];
        }
    }
};

// Hardcoded reference values for H2/STO-3G at 0.74 Angstrom.
// Captured from native C++ implementation (physical normalization, double precision).
// These use raw Shell/Atom structs with phys_norm coefficients.
constexpr double kRefS00 = 0.9999999909;
constexpr double kRefS01 = 0.6598731506;
constexpr double kRefT00 = 0.7600318766;
constexpr double kRefT01 = 0.2369602946;
constexpr double kRefV00 = -1.8809913573;
constexpr double kRefV01 = -1.1963361202;
constexpr double kRefVnuc = 0.7151044;
// ERI: (00|00), (00|11), (01|01)
constexpr double kRefERI_0000 = 0.7746059298;
constexpr double kRefERI_0011 = 0.5699948919;
constexpr double kRefERI_0101 = 0.2975905808;

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════
// Suite A: NativeIntegrals (always-on, no libcint needed)
// ═══════════════════════════════════════════════════════════════════════

// ── Boys function ─────────────────────────────────────────────────────

TEST(NativeIntegrals, BoysF0_ZeroArg) {
    EXPECT_NEAR(native::boys_f0(0.0), 1.0, 1e-15);
}

TEST(NativeIntegrals, BoysF0_KnownValues) {
    EXPECT_NEAR(native::boys_f0(1.0), 0.5 * std::sqrt(M_PI) * std::erf(1.0),
                1e-12);
    EXPECT_NEAR(native::boys_f0(10.0),
                0.5 * std::sqrt(M_PI / 10.0) * std::erf(std::sqrt(10.0)),
                1e-12);
}

TEST(NativeIntegrals, BoysF0_LargeArg) {
    // Asymptotic: F_0(x) → 0.5 * sqrt(pi/x) for large x
    EXPECT_NEAR(native::boys_f0(100.0), 0.5 * std::sqrt(M_PI / 100.0), 1e-12);
}

TEST(NativeIntegrals, BoysF0_SmallArg) {
    EXPECT_NEAR(native::boys_f0(1e-15), 1.0, 1e-7);
}

// ── Overlap symmetry and normalization ────────────────────────────────

TEST(NativeIntegrals, Overlap_Symmetry) {
    H2Setup h2;
    double S01 = native::overlap_ss(h2.shells[0], h2.shells[1]);
    double S10 = native::overlap_ss(h2.shells[1], h2.shells[0]);
    EXPECT_NEAR(S01, S10, 1e-15) << "S must be symmetric";
}

TEST(NativeIntegrals, Overlap_SelfNormalization) {
    H2Setup h2;
    double S00 = native::overlap_ss(h2.shells[0], h2.shells[0]);
    EXPECT_NEAR(S00, 1.0, 1e-7) << "Normalized shell self-overlap must be ~1";
}

// ── Kinetic symmetry ──────────────────────────────────────────────────

TEST(NativeIntegrals, Kinetic_Symmetry) {
    H2Setup h2;
    double T01 = native::kinetic_ss(h2.shells[0], h2.shells[1]);
    double T10 = native::kinetic_ss(h2.shells[1], h2.shells[0]);
    EXPECT_NEAR(T01, T10, 1e-15) << "T must be symmetric";
}

TEST(NativeIntegrals, Kinetic_Positive) {
    H2Setup h2;
    double T00 = native::kinetic_ss(h2.shells[0], h2.shells[0]);
    EXPECT_GT(T00, 0.0) << "Diagonal kinetic energy must be positive";
}

// ── Nuclear attraction symmetry ───────────────────────────────────────

TEST(NativeIntegrals, Nuclear_Symmetry) {
    H2Setup h2;
    double V01 = native::nuclear_ss(h2.shells[0], h2.shells[1], h2.atoms, 2);
    double V10 = native::nuclear_ss(h2.shells[1], h2.shells[0], h2.atoms, 2);
    EXPECT_NEAR(V01, V10, 1e-15) << "V must be symmetric";
}

TEST(NativeIntegrals, Nuclear_Negative) {
    H2Setup h2;
    double V00 = native::nuclear_ss(h2.shells[0], h2.shells[0], h2.atoms, 2);
    EXPECT_LT(V00, 0.0) << "Nuclear attraction must be negative";
}

// ── ERI permutation symmetry (8-fold) ─────────────────────────────────

TEST(NativeIntegrals, ERI_PermutationSymmetry) {
    H2Setup h2;
    auto& s = h2.shells;
    // Test all 8-fold permutations of (01|01)
    double base = native::eri_ssss(s[0], s[1], s[0], s[1]); // (01|01)
    EXPECT_NEAR(native::eri_ssss(s[1], s[0], s[0], s[1]), base, 1e-12)
        << "(10|01) != (01|01)";
    EXPECT_NEAR(native::eri_ssss(s[0], s[1], s[1], s[0]), base, 1e-12)
        << "(01|10) != (01|01)";
    EXPECT_NEAR(native::eri_ssss(s[1], s[0], s[1], s[0]), base, 1e-12)
        << "(10|10) != (01|01)";
    EXPECT_NEAR(native::eri_ssss(s[0], s[1], s[0], s[1]), base, 1e-12)
        << "(01|01) self-check";
    // Bra-ket exchange: (ij|kl) == (kl|ij)
    EXPECT_NEAR(native::eri_ssss(s[0], s[1], s[0], s[1]),
                native::eri_ssss(s[0], s[1], s[0], s[1]), 1e-12)
        << "(01|01) != (01|01)";
}

TEST(NativeIntegrals, ERI_Positive) {
    H2Setup h2;
    double eri_0000 = native::eri_ssss(h2.shells[0], h2.shells[0], h2.shells[0],
                                       h2.shells[0]);
    EXPECT_GT(eri_0000, 0.0) << "Self-repulsion ERI must be positive";
}

// ── Hardcoded reference regression (no libcint needed) ────────────────

TEST(NativeIntegrals, RegressionValues_H2) {
    H2Setup h2;
    double tol = 1e-5; // float-safe tolerance

    EXPECT_NEAR(native::overlap_ss(h2.shells[0], h2.shells[0]), kRefS00, tol);
    EXPECT_NEAR(native::overlap_ss(h2.shells[0], h2.shells[1]), kRefS01, tol);
    EXPECT_NEAR(native::kinetic_ss(h2.shells[0], h2.shells[0]), kRefT00, tol);
    EXPECT_NEAR(native::kinetic_ss(h2.shells[0], h2.shells[1]), kRefT01, tol);
    EXPECT_NEAR(native::nuclear_ss(h2.shells[0], h2.shells[0], h2.atoms, 2),
                kRefV00, tol);
    EXPECT_NEAR(native::nuclear_ss(h2.shells[0], h2.shells[1], h2.atoms, 2),
                kRefV01, tol);

    EXPECT_NEAR(native::eri_ssss(h2.shells[0], h2.shells[0], h2.shells[0],
                                 h2.shells[0]),
                kRefERI_0000, tol);
    EXPECT_NEAR(native::eri_ssss(h2.shells[0], h2.shells[0], h2.shells[1],
                                 h2.shells[1]),
                kRefERI_0011, tol);
    EXPECT_NEAR(native::eri_ssss(h2.shells[0], h2.shells[1], h2.shells[0],
                                 h2.shells[1]),
                kRefERI_0101, tol);
}

// ── internal::IntegralWorkspace sanity ───────────────────────────────────────────────

TEST(NativeIntegrals, IntegralWorkspace_Sanity) {
    auto frame = make_h2(double(0.74));
    auto pack = cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G};

    EXPECT_NEAR(static_cast<double>(pack.S(0, 0)), 1.0, 0.01);
    EXPECT_NEAR(static_cast<double>(pack.V_nuc), kRefVnuc, 0.01);
    EXPECT_EQ(pack.n_basis, 2);
    EXPECT_EQ(pack.n_electrons, 2);
    EXPECT_EQ(pack.n_occ(), 1);
}

// ── Multi-geometry tests ──────────────────────────────────────────────

TEST(NativeIntegrals, MultiGeometry_OverlapDecay) {
    // Overlap between shells should decrease with distance
    double bond_lengths[] = {0.5, 0.74, 1.0, 1.5, 2.0};
    double prev_S01 = 1.0;
    for (double bl : bond_lengths) {
        H2Setup h2(bl);
        double S01 = native::overlap_ss(h2.shells[0], h2.shells[1]);
        EXPECT_LT(S01, prev_S01)
            << "Overlap should decrease with distance at " << bl;
        EXPECT_GT(S01, 0.0) << "Overlap should remain positive at " << bl;
        prev_S01 = S01;
    }
}

TEST(NativeIntegrals, MultiGeometry_IntegralWorkspace) {
    // build_integrals should work at various geometries
    double bond_lengths[] = {0.5, 0.74, 1.0, 1.5, 2.0};
    for (double bl : bond_lengths) {
        auto frame = make_h2(double(bl));
        auto pack = cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G};
        // S diagonal must be ~1 regardless of geometry
        EXPECT_NEAR(static_cast<double>(pack.S(0, 0)), 1.0, 0.01)
            << "S(0,0) != 1.0 at bond_length=" << bl;
        // V_nuc should be Z1*Z2/R (in Hartree with Bohr distance)
        double R_bohr = bl * units::kAngstromInBohr;
        double expected_vnuc = 1.0 / R_bohr; // Z=1 for both H atoms
        EXPECT_NEAR(static_cast<double>(pack.V_nuc), expected_vnuc, 1e-5)
            << "V_nuc mismatch at bond_length=" << bl;
    }
}

// ── p-type tests (always-on) ─────────────────────────────────────────

/// @brief Build a water (H2O) frame.
cpu::Frame make_h2o() {
    cpu::Frame f(3);
    // Standard geometry: O at origin, H at ~0.96 A, angle 104.5 deg
    f.span(ATV::kAtom_x)[0] = double(0.0);
    f.span(ATV::kAtom_y)[0] = double(0.0);
    f.span(ATV::kAtom_z)[0] = double(0.0);
    f.span(ATV::kAtom_x)[1] = double(0.757);
    f.span(ATV::kAtom_y)[1] = double(0.587);
    f.span(ATV::kAtom_z)[1] = double(0.0);
    f.span(ATV::kAtom_x)[2] = double(-0.757);
    f.span(ATV::kAtom_y)[2] = double(0.587);
    f.span(ATV::kAtom_z)[2] = double(0.0);
    f.span(ATV::kAtom_mass)[0] = double(15.999);
    f.span(ATV::kAtom_mass)[1] = double(1.008);
    f.span(ATV::kAtom_mass)[2] = double(1.008);
    f.span(ATV::kAtom_type)[0] = 8;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_type)[2] = 1;
    f.box = ATV::cpu::Box::cubic(100.0);
    return f;
}

/// @brief Build a methane (CH4) frame.
cpu::Frame make_ch4() {
    cpu::Frame f(5);
    // Tetrahedral geometry, C at origin, H at ~1.089 A
    f.span(ATV::kAtom_x)[0] = double(0.0);
    f.span(ATV::kAtom_y)[0] = double(0.0);
    f.span(ATV::kAtom_z)[0] = double(0.0);
    f.span(ATV::kAtom_x)[1] = double(0.6276);
    f.span(ATV::kAtom_y)[1] = double(0.6276);
    f.span(ATV::kAtom_z)[1] = double(0.6276);
    f.span(ATV::kAtom_x)[2] = double(-0.6276);
    f.span(ATV::kAtom_y)[2] = double(-0.6276);
    f.span(ATV::kAtom_z)[2] = double(0.6276);
    f.span(ATV::kAtom_x)[3] = double(-0.6276);
    f.span(ATV::kAtom_y)[3] = double(0.6276);
    f.span(ATV::kAtom_z)[3] = double(-0.6276);
    f.span(ATV::kAtom_x)[4] = double(0.6276);
    f.span(ATV::kAtom_y)[4] = double(-0.6276);
    f.span(ATV::kAtom_z)[4] = double(-0.6276);
    for (int i = 0; i < 5; ++i)
        f.span(ATV::kAtom_mass)[i] = (i == 0) ? double(12.011) : double(1.008);
    f.span(ATV::kAtom_type)[0] = 6;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_type)[3] = 1;
    f.span(ATV::kAtom_type)[4] = 1;
    f.box = ATV::cpu::Box::cubic(100.0);
    return f;
}

TEST(NativeIntegrals, H2O_IntegralWorkspace_Sanity) {
    auto frame = make_h2o();
    auto pack = cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G};
    // O: 1s + 2s + 2px + 2py + 2pz = 5, each H: 1s = 1
    EXPECT_EQ(pack.n_basis, 7);
    EXPECT_EQ(pack.n_electrons, 10);
    // S diagonal should be ~1 for normalized basis
    for (int i = 0; i < pack.n_basis; ++i)
        EXPECT_NEAR(static_cast<double>(pack.S(i, i)), 1.0, 0.01)
            << "S(" << i << "," << i << ") != 1.0";
    // S must be symmetric
    for (int i = 0; i < pack.n_basis; ++i)
        for (int j = 0; j < i; ++j)
            EXPECT_NEAR(static_cast<double>(pack.S(i, j)),
                        static_cast<double>(pack.S(j, i)), 1e-12)
                << "S not symmetric at (" << i << "," << j << ")";
}

TEST(NativeIntegrals, CH4_IntegralWorkspace_Sanity) {
    auto frame = make_ch4();
    auto pack = cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G};
    // C: 5 bf, 4 H: 1 bf each = 9 total
    EXPECT_EQ(pack.n_basis, 9);
    EXPECT_EQ(pack.n_electrons, 10);
    for (int i = 0; i < pack.n_basis; ++i)
        EXPECT_NEAR(static_cast<double>(pack.S(i, i)), 1.0, 0.01)
            << "S(" << i << "," << i << ") != 1.0 for CH4";
}

TEST(NativeIntegrals, PType_SelfOverlap) {
    // px should have <px|px> = 1 for a properly normalized shell
    auto frame = make_h2o();
    auto pack = cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G};
    // Basis order for O: 1s(0), 2s(1), 2px(2), 2py(3), 2pz(4), H1(5), H2(6)
    EXPECT_NEAR(static_cast<double>(pack.S(2, 2)), 1.0, 0.01) << "<px|px> != 1";
    EXPECT_NEAR(static_cast<double>(pack.S(3, 3)), 1.0, 0.01) << "<py|py> != 1";
    EXPECT_NEAR(static_cast<double>(pack.S(4, 4)), 1.0, 0.01) << "<pz|pz> != 1";
}

TEST(NativeIntegrals, PType_Orthogonality) {
    // px, py, pz on the SAME center should be orthogonal
    auto frame = make_h2o();
    auto pack = cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G};
    EXPECT_NEAR(static_cast<double>(pack.S(2, 3)), 0.0, 1e-10)
        << "<px|py> != 0";
    EXPECT_NEAR(static_cast<double>(pack.S(2, 4)), 0.0, 1e-10)
        << "<px|pz> != 0";
    EXPECT_NEAR(static_cast<double>(pack.S(3, 4)), 0.0, 1e-10)
        << "<py|pz> != 0";
}

/// @brief Verify block-level g_rhf matches AO-level g_rhf for a given frame.
static void compare_g_ao_vs_block(const cpu::Frame& frame, const char* label) {
    auto pack = cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G};
    int n = pack.n_basis;

    // Dense test density with off-diagonal elements to exercise all contributions.
    RealMatrix P = xt::zeros<double>({(size_t)n, (size_t)n});
    for (int i = 0; i < n; ++i)
        P(i, i) = 0.5;
    for (int i = 0; i < n - 1; ++i) {
        P(i, i + 1) = 0.1;
        P(i + 1, i) = 0.1;
    }

    auto sp_ao = ShellPairList{pack};
    auto sp_blk = ShellBlockPairList{pack};

    RealMatrix G_ao, G_blk;
    sp_ao.g_rhf(P, G_ao);
    sp_blk.g_rhf(P, G_blk);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            EXPECT_NEAR(static_cast<double>(G_blk(i, j)),
                        static_cast<double>(G_ao(i, j)), 1e-9)
                << label << " G[" << i << "," << j << "] mismatch";
}

TEST(NativeIntegrals, BlockG_H2O) { compare_g_ao_vs_block(make_h2o(), "H2O"); }

TEST(NativeIntegrals, BlockG_CH4) { compare_g_ao_vs_block(make_ch4(), "CH4"); }

TEST(NativeIntegrals, BlockG_H2) { compare_g_ao_vs_block(make_h2(0.74), "H2"); }


#else // !ATV_HAS_HF

#include <gtest/gtest.h>
TEST(NativeIntegrals, Disabled) {
    GTEST_SKIP() << "Native integral tests require -DATV_BUILD_HF=ON";
}

#endif
