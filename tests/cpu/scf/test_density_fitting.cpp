/// @file tests/cpu/scf/test_density_fitting.cpp
/// @brief Unit tests for RI-J / RI-JK density fitting (link 12):
///        def2-JKFIT static table rows, l = 4 (g) 2-/3-center integrals
///        against externally pinned PySCF values, metric whitening
///        algebra, and the fitted J/K math on H2/STO-3G — all single
///        calls, NO converged SCF (the H2O/def2-TZVP end-to-end double
///        gate lives in regressions/scf/h2o_ri_jk/).
///
/// Oracle: PySCF 2.14.0 (~/work/.x86_64/bin/python). Every pinned number
/// below records the generating command in the adjacent comment.
///
/// Engine-vs-PySCF normalization note (g shells): the engine's shared
/// Cartesian g normalization `phys_norm_g(α) = (2α/π)^{3/4}(4α)²` differs
/// from PySCF's shared cart norm by the α-INDEPENDENT amplitude ratio
///   s_g = 8.671834327921664
/// (derived from mol.intor('int1e_ovlp', cart=True) on a single-primitive
/// g shell: s_g = phys_norm_g(α) / sqrt(S_xxxx / <x⁴e^{-αr²}|x⁴e^{-αr²}>);
/// s = 1 for l = 0). The pins below are PySCF integrals rescaled once by
/// the product of the per-shell factors — recorded post-scale, so the
/// test compares raw engine output. Pins are checked to 1e-10 relative
/// (engine-convention g self-norms are O(10²), so a fixed absolute
/// 1e-10 would demand sub-ULP agreement on the large entries).

#if defined(ATV_HAS_HF)

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_data/def2_jkfit.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/internal/basis_workspace.h"
#include "atomiverse/cpu/scf/internal/density_builder.h"
#include "atomiverse/cpu/scf/internal/density_fitting.h"
#include "atomiverse/cpu/scf/internal/eigen_solver.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/rys_eri.h"

using namespace ATV;
using namespace ATV::cpu::scf;

namespace {

/// Pin comparison: 1e-10 relative for large-magnitude engine-convention
/// entries, 1e-10 absolute below magnitude 1 (see file header).
void expect_pin(double got, double want, const char* what) {
    const double tol = 1e-10 * std::max(1.0, std::fabs(want));
    EXPECT_NEAR(got, want, tol) << what;
}

/// Single-primitive Cartesian shell block in the ENGINE normalization
/// (coeff = raw_coeff × phys_norm_l(α)), mirroring ShellList's emission.
struct PrimBlock {
    double alpha_store, coeff_store;
    native::ShellBlock blk;
    PrimBlock(int l, double alpha, const double center[3]) {
        alpha_store = alpha;
        const double n = (l == 0)   ? native::phys_norm(alpha)
                         : (l == 3) ? native::phys_norm_f(alpha)
                         : (l == 4) ? native::phys_norm_g(alpha)
                                    : native::gto_norm(l, alpha);
        coeff_store = n;
        blk.n_prim = 1;
        blk.alpha = &alpha_store;
        blk.coeff = &coeff_store;
        blk.center[0] = center[0];
        blk.center[1] = center[1];
        blk.center[2] = center[2];
        blk.l = l;
        blk.n_cart = n_cart_of(l);
        blk.ao_start = 0;
    }
};

/// Zero-exponent unit s dummy (the 3c/2c collapse trick).
struct Dummy {
    double zero = 0.0, one = 1.0;
    native::ShellBlock at(const double c[3]) {
        native::ShellBlock b;
        b.n_prim = 1;
        b.alpha = &zero;
        b.coeff = &one;
        b.center[0] = c[0];
        b.center[1] = c[1];
        b.center[2] = c[2];
        b.l = 0;
        b.n_cart = 1;
        b.ao_start = 0;
        return b;
    }
};

cpu::Frame single_atom_frame(int Z) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = Z;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.box = cpu::Box::cubic(50.0);
    return f;
}

cpu::Frame h2_frame(double bond_ang = 0.74) {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
    }
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = bond_ang;
    f.box = cpu::Box::cubic(50.0);
    return f;
}

} // namespace

// ── def2-JKFIT static table: spot checks vs Weigend 2002 published
//    values (as shipped by PySCF; pinned from
//    `gto.basis.load('def2-universal-jkfit', El)`, PySCF 2.14.0). ──────
TEST(Def2JKFITTable, WeigendSpotChecks) {
    const auto& tab = basis_data::def2_jkfit();

    // H (Z=1): first shell is the 3-primitive s contraction.
    const auto& h = tab[1];
    ASSERT_EQ(h.shells.size(), 6u);
    EXPECT_EQ(h.shells[0].l, 0);
    ASSERT_EQ(h.shells[0].alpha.size(), 3u);
    EXPECT_DOUBLE_EQ(h.shells[0].alpha[0], 22.068343);
    EXPECT_DOUBLE_EQ(h.shells[0].alpha[1], 4.3905712);
    EXPECT_DOUBLE_EQ(h.shells[0].alpha[2], 1.0540787);
    EXPECT_DOUBLE_EQ(h.shells[0].coeff[0], 0.053033986);
    EXPECT_DOUBLE_EQ(h.shells[0].coeff[1], 0.3946522022);
    EXPECT_DOUBLE_EQ(h.shells[0].coeff[2], 0.9172987712);
    // H last shell: uncontracted d, alpha 0.3292649.
    EXPECT_EQ(h.shells[5].l, 2);
    EXPECT_DOUBLE_EQ(h.shells[5].alpha[0], 0.3292649);
    EXPECT_DOUBLE_EQ(h.shells[5].coeff[0], 1.0);

    // C (Z=6): 25 shells; first s contraction head and the g tail.
    const auto& c = tab[6];
    ASSERT_EQ(c.shells.size(), 25u);
    EXPECT_EQ(c.shells[0].l, 0);
    EXPECT_DOUBLE_EQ(c.shells[0].alpha[0], 384.4438241);
    EXPECT_DOUBLE_EQ(c.shells[0].coeff[0], 0.1825786758);
    EXPECT_EQ(c.shells[24].l, 4); // one g shell per first-row element
    EXPECT_DOUBLE_EQ(c.shells[24].alpha[0], 1.1430602);
    EXPECT_DOUBLE_EQ(c.shells[24].coeff[0], 1.0);

    // O (Z=8): first f shell and the g shell.
    const auto& o = tab[8];
    ASSERT_EQ(o.shells.size(), 25u);
    EXPECT_DOUBLE_EQ(o.shells[0].alpha[0], 625.2829811);
    EXPECT_DOUBLE_EQ(o.shells[0].coeff[0], 0.1847924989);
    EXPECT_EQ(o.shells[22].l, 3);
    EXPECT_DOUBLE_EQ(o.shells[22].alpha[0], 3.0293422);
    EXPECT_DOUBLE_EQ(o.shells[22].coeff[0], 0.7615479114);
    EXPECT_EQ(o.shells[24].l, 4);
    EXPECT_DOUBLE_EQ(o.shells[24].alpha[0], 1.6934809);

    // He (Z=2) is deliberately absent (not shipped; no fallback).
    EXPECT_TRUE(tab[2].shells.empty());
}

// ── Missing JKFIT row throws, naming the element and the aux basis. ──
TEST(Def2JKFITTable, MissingElementThrows) {
    AtomList atoms;
    atoms.atoms.resize(1);
    atoms.atoms[0].Z = 2; // He: no def2-universal-jkfit row shipped
    atoms.atoms[0].pos[0] = 0.0;
    atoms.atoms[0].pos[1] = 0.0;
    atoms.atoms[0].pos[2] = 0.0;
    try {
        ShellList aux(atoms, AuxBasis::Def2JKFIT);
        FAIL() << "expected std::runtime_error for missing JKFIT element";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("def2-universal-jkfit"), std::string::npos) << msg;
        EXPECT_NE(msg.find("Z=2"), std::string::npos) << msg;
    }
}

// ── l = 4 (g) 2-center integrals vs pinned PySCF values. ─────────────
//
// Oracle command (PySCF 2.14.0):
//   mol2 = gto.M(atom=[['X1',(0,0,0)],['X2',(0,0,1.2)]],
//                basis={'X1':[[4,[0.8,1.0]]], 'X2':[[4,[1.1,1.0]]]},
//                cart=True, unit='Bohr')
//   V = mol2.intor('int2c2e')
// then rescaled once to the engine convention with s_g² (cross block)
// / s_g² (same-center block), s_g = 8.671834327921664 (file header).
TEST(GShellIntegrals, TwoCenterVsPySCF) {
    const double c0[3] = {0.0, 0.0, 0.0};
    const double c1[3] = {0.0, 0.0, 1.2}; // Bohr
    PrimBlock P(4, 0.8, c0);
    PrimBlock Q(4, 1.1, c1);
    Dummy d;
    const auto Pd = d.at(c0);
    const auto Qd = d.at(c1);

    // Cross-center block (g0.8 @ 0 | g1.1 @ 1.2 z).
    std::vector<double> buf(15 * 15, 0.0);
    native::rys_eri_block(P.blk, Pd, Q.blk, Qd, buf.data());
    expect_pin(buf[0 * 15 + 0], 1206.9512111633148, "(xxxx|xxxx)");
    expect_pin(buf[0 * 15 + 14], 1061.11117210219, "(xxxx|zzzz)");
    expect_pin(buf[2 * 15 + 2], 4.713770050017124, "(xxxz|xxxz)");
    expect_pin(buf[4 * 15 + 4], 0.447490560947105, "(xxyz|xxyz)");
    expect_pin(buf[14 * 15 + 14], 1213.2978801921795, "(zzzz|zzzz)");
    expect_pin(buf[5 * 15 + 9], 0.0, "(xxzz|xzzz) odd-x zero");
    expect_pin(buf[7 * 15 + 11], 0.0, "(xyyz|yyyz) odd-x zero");

    // Same-center block (g0.8 | g0.8).
    std::vector<double> buf0(15 * 15, 0.0);
    native::rys_eri_block(P.blk, Pd, P.blk, Pd, buf0.data());
    expect_pin(buf0[0 * 15 + 0], 1712.7663948071208, "(xxxx|xxxx) 1c");
    expect_pin(buf0[0 * 15 + 3], 492.93084731325547, "(xxxx|xxyy) 1c");
    expect_pin(buf0[0 * 15 + 10], 1286.7065911202765, "(xxxx|yyyy) 1c");
    expect_pin(buf0[4 * 15 + 4], 8.826379360085607, "(xxyz|xxyz) 1c");
}

// ── l = 4 (g) 3-center integrals (μν|P) vs pinned PySCF values. ──────
//
// Oracle command (PySCF 2.14.0):
//   mol    = gto.M(atom=[['X1',(0,0,0)],['X2',(0,0,1.4)]],
//                  basis={'X1':[[0,[0.5,1.0]]], 'X2':[[0,[1.3,1.0]]]},
//                  cart=True, unit='Bohr')
//   auxmol = gto.M(same atoms,
//                  basis={'X1':[[4,[0.9,1.0]]], 'X2':[[0,[0.4,1.0]]]},
//                  cart=True, unit='Bohr')
//   j3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
// rescaled once: aux g columns × s_g, everything else × 1 (s shells).
TEST(GShellIntegrals, ThreeCenterVsPySCF) {
    const double c0[3] = {0.0, 0.0, 0.0};
    const double c1[3] = {0.0, 0.0, 1.4}; // Bohr
    PrimBlock s1(0, 0.5, c0);
    PrimBlock s2(0, 1.3, c1);
    PrimBlock g(4, 0.9, c0);
    PrimBlock saux(0, 0.4, c1);
    Dummy d;

    // (s1 s1 | g), (s1 s2 | g), (s2 s2 | g): 1×1×15 buffers each.
    std::vector<double> b11(15, 0.0), b12(15, 0.0), b22(15, 0.0);
    native::rys_eri_block(s1.blk, s1.blk, g.blk, d.at(c0), b11.data());
    native::rys_eri_block(s1.blk, s2.blk, g.blk, d.at(c0), b12.data());
    native::rys_eri_block(s2.blk, s2.blk, g.blk, d.at(c0), b22.data());
    expect_pin(b11[0], 28.198750186717568, "(s1 s1|g xxxx)");
    expect_pin(b12[0], 10.811013211343417, "(s1 s2|g xxxx)");
    expect_pin(b11[2], 0.0, "(s1 s1|g xxxz) odd-z-free zero");
    expect_pin(b12[5], 3.9665106086993402, "(s1 s2|g xxzz)");
    expect_pin(b22[14], 33.85071785368178, "(s2 s2|g zzzz)");
    expect_pin(b12[9], 0.0, "(s1 s2|g xzzz) odd-x zero");

    // Aux s shell column: (s1 s1 | s_aux), (s1 s2 | s_aux).
    std::vector<double> a11(1, 0.0), a12(1, 0.0);
    native::rys_eri_block(s1.blk, s1.blk, saux.blk, d.at(c1), a11.data());
    native::rys_eri_block(s1.blk, s2.blk, saux.blk, d.at(c1), a12.data());
    expect_pin(a11[0], 4.0019356243055375, "(s1 s1|s_aux)");
    expect_pin(a12[0], 2.0932406152946608, "(s1 s2|s_aux)");
}

// ── Root budget: a 4-center g quartet must throw cleanly (the orbital
//    basis is capped at f; only 2-/3-center paths carry g). ───────────
TEST(GShellIntegrals, FourCenterGQuartetThrows) {
    const double c0[3] = {0.0, 0.0, 0.0};
    PrimBlock g(4, 0.9, c0);
    Dummy d;
    std::vector<double> buf(15 * 15 * 15 * 15, 0.0);
    EXPECT_THROW(native::rys_eri_block(g.blk, g.blk, g.blk, g.blk, buf.data()),
                 std::runtime_error);
    // ... while the heaviest supported 3-center case (ff|g·1) runs.
    PrimBlock f1(3, 0.7, c0);
    std::vector<double> buf3(10 * 10 * 15, 0.0);
    EXPECT_NO_THROW(
        native::rys_eri_block(f1.blk, f1.blk, g.blk, d.at(c0), buf3.data()));
}

// ── Spherical auxiliary metric of one O atom vs PySCF (spherical!).
//
// Oracle command (PySCF 2.14.0):
//   auxmol = gto.M(atom=[['O',(0,0,0)]], basis='def2-universal-jkfit',
//                  unit='Bohr', cart=False)
//   V = auxmol.intor('int2c2e')      # 77 × 77
// This exercises the WHOLE aux chain — static table, g 2-center Rys
// integrals, Cartesian→spherical g transform, and the self-normalized
// aux c2s — against PySCF's spherical convention with no rescaling.
TEST(DensityFitting, OxygenSphericalMetricVsPySCF) {
    cpu::Frame f = single_atom_frame(8);
    internal::IntegralWorkspace pack{f, BasisSet::STO3G};
    internal::DensityFitting df(pack.shells, pack.atoms, AuxBasis::Def2JKFIT);

    ASSERT_EQ(df.n_aux(), 77);
    const auto& V = df.metric();
    expect_pin(V(0, 0), 0.23898023697375873, "V(0,0) s-block");
    expect_pin(V(1, 1), 1.0642468658053135, "V(1,1) s-block");
    expect_pin(V(2, 2), 2.032478556391978, "V(2,2) s-block");
    expect_pin(V(0, 1), 0.4536856867011053, "V(0,1) s-block");
    // g block: the last 9 diagonal entries are m-degenerate on a bare
    // atom — all equal 0.8244931499348 (PySCF prints them equal to
    // ~1e-13; compare each at the pin tolerance).
    for (int m = 0; m < 9; ++m)
        expect_pin(V(68 + m, 68 + m), 0.8244931499348261, "V g-block diag");
}

// ── Whitening algebra: W·V·Wᵀ = I to 1e-12. ──────────────────────────
TEST(DensityFitting, MetricWhiteningIdentity) {
    cpu::Frame f = h2_frame();
    internal::IntegralWorkspace pack{f, BasisSet::STO3G};
    internal::DensityFitting df(pack.shells, pack.atoms, AuxBasis::Def2JKFIT);

    const auto& V = df.metric();
    const auto& W = df.v_inv_sqrt();
    const RealMatrix I_test =
        xt::linalg::dot(W, RealMatrix(xt::linalg::dot(V, xt::transpose(W))));
    const int n = df.n_aux();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const double want = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(I_test(i, j), want, 1e-12)
                << "W·V·Wᵀ deviates at (" << i << "," << j << ")";
        }
}

// ── Fitted J/K vs dense exact J/K on H2/STO-3G (single Fock-level
//    calls — no SCF loop). Documented fit-error bounds: with the
//    def2-universal-jkfit auxiliary basis PySCF measures
//    ‖ΔJ‖∞ = 3.11e-5 Ha and ‖ΔK‖∞ = 2.99e-5 Ha for the Hcore-guess
//    density of H2/STO-3G at 0.74 Å (mf.get_jk vs df.DF(...).get_jk).
//    The engine fits in the identical auxiliary span, so the same
//    magnitude must appear here; the asserted ceiling 2e-4 Ha is that
//    measurement with an order-of-safety margin, and a floor check
//    keeps the test honest (a bitwise-equal "fit" would mean the dense
//    path was silently substituted). ────────────────────────────────
TEST(DensityFitting, FittedJKCloseToDenseH2) {
    cpu::Frame f = h2_frame();
    internal::IntegralWorkspace pack{f, BasisSet::STO3G, 0,
                                     /*compute_eri=*/true};
    internal::DensityFitting df(pack.shells, pack.atoms, AuxBasis::Def2JKFIT);

    // Hcore-guess density: P = 2 C_occ C_occᵀ, n_occ = 1.
    internal::EigenSolver es;
    auto [eig, C] = es.step(pack.Hcore, pack.S);
    internal::DensityBuilder db;
    db.prepare(pack.n_basis);
    const RealMatrix P = db.step(C, pack.n_occ(), pack.n_basis, 2.0);

    // Dense exact J/K from the stored ERI tensor (chemist notation):
    //   J_kl = Σ_ij (ij|kl) P_ji,  K_il = Σ_jk (ij|kl) P_jk.
    const std::vector<std::size_t> ax01{0, 1}, ax10{1, 0}, ax12{1, 2};
    const RealMatrix J_exact = xt::linalg::tensordot(pack.ERI, P, ax01, ax10);
    const RealMatrix K_exact = xt::linalg::tensordot(pack.ERI, P, ax12, ax01);

    const RealMatrix J_fit = df.j_fitted(P);
    const RealMatrix K_fit = df.k_fitted_from_density(P);

    // The fitted 4-index tensor itself must reproduce the dense ERIs to
    // fit accuracy: (μν|λσ)_fit = Σ_P B_{μν,P} B_{λσ,P}.
    {
        const auto& B = df.b_tensor();
        const int n = pack.n_basis;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    for (int l = 0; l < n; ++l) {
                        double v = 0.0;
                        for (int p = 0; p < df.n_aux(); ++p)
                            v += B(i * n + j, p) * B(k * n + l, p);
                        EXPECT_NEAR(v, pack.ERI(i, j, k, l), 2e-4)
                            << "fitted (" << i << j << "|" << k << l << ")";
                    }
    }
    double dj = 0.0, dk = 0.0;
    for (int i = 0; i < pack.n_basis; ++i)
        for (int j = 0; j < pack.n_basis; ++j) {
            dj = std::max(dj, std::fabs(J_fit(i, j) - J_exact(i, j)));
            dk = std::max(dk, std::fabs(K_fit(i, j) - K_exact(i, j)));
        }
    EXPECT_LT(dj, 2e-4) << "‖J_fit − J_exact‖∞ beyond the documented bound";
    EXPECT_LT(dk, 2e-4) << "‖K_fit − K_exact‖∞ beyond the documented bound";
    EXPECT_GT(dj, 1e-9) << "suspiciously exact J fit (dense path leak?)";
    EXPECT_GT(dk, 1e-9) << "suspiciously exact K fit (dense path leak?)";

    // Route consistency: the B half-transform on C_occ√2 must agree with
    // the density-eigendecomposition route to solver precision.
    RealMatrix C_occ = xt::zeros<double>(
        {std::size_t(pack.n_basis), std::size_t(pack.n_occ())});
    for (int i = 0; i < pack.n_basis; ++i)
        for (int o = 0; o < pack.n_occ(); ++o)
            C_occ(i, o) = std::sqrt(2.0) * C(i, o);
    const RealMatrix K_half = df.k_fitted(C_occ);
    for (int i = 0; i < pack.n_basis; ++i)
        for (int j = 0; j < pack.n_basis; ++j)
            EXPECT_NEAR(K_half(i, j), K_fit(i, j), 1e-11)
                << "k_fitted(C_occ) vs k_fitted_from_density(P) at (" << i
                << "," << j << ")";
}

// ── Near-singular auxiliary metric throws (two H at 1e-7 Å duplicate
//    every auxiliary function → Cholesky pivot collapse). ─────────────
TEST(DensityFitting, NearSingularMetricThrows) {
    cpu::Frame f = h2_frame(1e-7);
    internal::IntegralWorkspace pack{f, BasisSet::STO3G};
    EXPECT_THROW(
        internal::DensityFitting(pack.shells, pack.atoms, AuxBasis::Def2JKFIT),
        std::runtime_error);
}

#endif // ATV_HAS_HF
