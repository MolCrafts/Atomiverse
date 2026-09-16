/// @file tests/cpu/scf/test_ecp.cpp
/// @brief Unit: semi-local ECP (def2-ECP) — static table structure, the
///        EcpPotential lookup class, EcpIntegrals U-matrix elements pinned
///        against PySCF, internal::IntegralWorkspace valence wiring, and the ECP↔x2c
///        double-counting gate (hf-capability-gaps-06-ecp). No full SCF
///        solve beyond the millisecond H2/STO-3G gate check.
///
/// Pinned oracle: PySCF 2.14.0, run externally (never a test dependency):
///
///   from pyscf import gto
///   gto.basis.load_ecp("def2-svp", "I")    # and "Au" — table structure
///
/// U-matrix pins (probe A, real def2-ECP on I; engine Cartesian AO basis):
///
///   mol = gto.M(atom="I 0 0 0; H 0 0 3.0", unit="Bohr", cart=True,
///               basis={"I":[[0,[1.0,1.0]],[1,[0.9,1.0]],[2,[0.8,1.0]]],
///                      "H":[[0,[0.5,1.0]],[3,[1.1,1.0]]]},
///               ecp={"I":"def2-svp"})
///   U = mol.intor("ECPscalar")
///
/// U-matrix pins (probe B, SYNTHETIC f-projector channel — def2-ECP itself
/// carries projectors only up to l=2/d, so the high-l type-2 path is pinned
/// through a user-defined PySCF ECP):
///
///   ecpB = {"Na": [10, [[-1, [[],[],[[1.5,-3.0]],[],[],[],[]]],
///                       [3,  [[],[],[[1.0, 5.0]],[],[],[],[]]]]]}
///   mol = gto.M(atom="Na 0 0 0; H 0 0 3.2", unit="Bohr", cart=True,
///               basis={"Na":[[0,[1.0,1.0]]],
///                      "H":[[0,[0.8,1.0]],[1,[1.1,1.0]],[3,[0.9,1.0]]]},
///               ecp=ecpB)
///
/// AO-basis mapping: the engine emits one Cartesian AO per shell in libcint
/// component order with the shared per-l normalization phys_norm_l(alpha);
/// PySCF cart AOs for l >= 2 keep the bare-monomial angular norm, so the
/// PySCF matrix is rescaled per AO by D_i = sqrt(S_engine_ii / S_pyscf_ii)
/// (constant per shell) before pinning: U_pin = D U_pyscf D. Rescaled pins
/// below reproduce the engine matrix to ~3e-13 (asserted at 1e-10).

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/gto.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_data/basis_table.h"
#include "atomiverse/cpu/scf/basis_data/def2_svp.h"
#include "atomiverse/cpu/scf/basis_data/ecp_def2.h"
#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/ecp_potential.h"
#include "atomiverse/cpu/scf/internal/ecp_integrals.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;
using cpu::scf::BasisSet;
using cpu::scf::EcpPotential;
using cpu::scf::internal::IntegralWorkspace;
using cpu::scf::basis_data::AtomEcp;
using cpu::scf::basis_data::EcpChannel;
using cpu::scf::basis_data::EcpGaussianTerm;
using cpu::scf::internal::EcpCenter;
using cpu::scf::internal::EcpIntegrals;

namespace {

/// Append one shell (all Cartesian components) with raw coefficient 1.0 and
/// the engine's shared per-l physical normalization.
void add_shell(std::vector<cpu::scf::native::Shell>& shells,
               std::vector<std::vector<double>>& store, int l, double alpha,
               const double center[3]) {
    double norm = 0.0;
    switch (l) {
    case 0:
        norm = phys_norm_s(alpha);
        break;
    case 1:
        norm = phys_norm_p(alpha);
        break;
    case 2:
        norm = phys_norm_d(alpha);
        break;
    default:
        norm = phys_norm_f(alpha);
        break;
    }
    for (int cart = 0; cart < cpu::scf::n_cart_of(l); ++cart) {
        store.push_back({alpha, norm});
        cpu::scf::native::Shell s;
        s.n_prim = 1;
        s.alpha = store.back().data();
        s.coeff = store.back().data() + 1;
        s.center[0] = center[0];
        s.center[1] = center[1];
        s.center[2] = center[2];
        s.l = l;
        s.cart = cart;
        shells.push_back(s);
    }
}

cpu::Frame i2_frame() {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 53;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 53;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 2.666;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

// ── def2-ECP table structure vs PySCF gto.basis.load_ecp ──────────────

TEST(EcpTables, Def2EcpStructurePinnedVsPyscf) {
    const auto& t = cpu::scf::basis_data::ecp_def2();
    ASSERT_GE(static_cast<int>(t.size()), 87); // Z = 0..86

    // I (Z=53): ECP28MWB — 28 core electrons, s/p/d projectors + local.
    const auto& iodine = t[53];
    EXPECT_EQ(iodine.n_core, 28);
    ASSERT_EQ(iodine.channels.size(), 4u);
    EXPECT_EQ(iodine.channels[0].l, 0);
    EXPECT_EQ(iodine.channels[1].l, 1);
    EXPECT_EQ(iodine.channels[2].l, 2);
    EXPECT_EQ(iodine.channels[3].l, 3); // local U_L, stored last
    // s channel: 7 terms; first (40.015835, 49.994293),
    // last (4.884315, 0.320804); all pure Gaussians (n = 2).
    const auto& s_ch = iodine.channels[0];
    ASSERT_EQ(s_ch.terms.size(), 7u);
    EXPECT_EQ(s_ch.terms.front().n, 2);
    EXPECT_DOUBLE_EQ(s_ch.terms.front().alpha, 40.015835);
    EXPECT_DOUBLE_EQ(s_ch.terms.front().coeff, 49.994293);
    EXPECT_DOUBLE_EQ(s_ch.terms.back().alpha, 4.884315);
    EXPECT_DOUBLE_EQ(s_ch.terms.back().coeff, 0.320804);
    // local channel: 4 terms, first (19.458609, -21.84204).
    const auto& loc = iodine.channels[3];
    ASSERT_EQ(loc.terms.size(), 4u);
    EXPECT_DOUBLE_EQ(loc.terms.front().alpha, 19.458609);
    EXPECT_DOUBLE_EQ(loc.terms.front().coeff, -21.84204);

    // Au (Z=79): ECP60MWB — 60 core electrons.
    const auto& gold = t[79];
    EXPECT_EQ(gold.n_core, 60);
    ASSERT_EQ(gold.channels.size(), 4u);
    const auto& au_s = gold.channels[0];
    ASSERT_EQ(au_s.terms.size(), 4u);
    EXPECT_DOUBLE_EQ(au_s.terms.front().alpha, 13.2051);
    EXPECT_DOUBLE_EQ(au_s.terms.front().coeff, 426.8466792);
    const auto& au_loc = gold.channels[3];
    ASSERT_EQ(au_loc.terms.size(), 2u);
    EXPECT_DOUBLE_EQ(au_loc.terms.front().alpha, 4.78982);
    EXPECT_DOUBLE_EQ(au_loc.terms.front().coeff, 30.4900889);
    EXPECT_DOUBLE_EQ(au_loc.terms.back().alpha, 2.39491);
    EXPECT_DOUBLE_EQ(au_loc.terms.back().coeff, 5.17107381);

    // All-electron rows are empty: light elements, and the lanthanide gap
    // Ce-Lu (not part of the def2 family).
    EXPECT_TRUE(t[8].channels.empty());  // O
    EXPECT_TRUE(t[36].channels.empty()); // Kr (def2: Z <= 36 all-electron)
    for (int z = 58; z <= 71; ++z)
        EXPECT_TRUE(t[static_cast<std::size_t>(z)].channels.empty()) << z;
    EXPECT_FALSE(t[37].channels.empty()); // Rb
    EXPECT_FALSE(t[86].channels.empty()); // Rn
}

// ── def2-SVP heavy-element valence rows (regenerated table) ────────────

TEST(EcpTables, Def2SvpHeavyValenceRows) {
    const auto& t = cpu::scf::basis_data::def2_svp();
    ASSERT_GE(static_cast<int>(t.size()), 87);

    // I: [4s 4p 2d] valence shells, first s exponent 445.90489176.
    const auto& iodine = t[53];
    ASSERT_EQ(iodine.shells.size(), 10u);
    const std::vector<int> i_lseq{0, 0, 0, 0, 1, 1, 1, 1, 2, 2};
    for (std::size_t k = 0; k < iodine.shells.size(); ++k)
        EXPECT_EQ(iodine.shells[k].l, i_lseq[k]) << k;
    EXPECT_DOUBLE_EQ(iodine.shells.front().alpha.front(), 445.90489176);
    EXPECT_DOUBLE_EQ(iodine.shells.back().alpha.back(), 0.309);

    // Au: [6s 3p 2d 1f], f exponent 0.72482.
    const auto& gold = t[79];
    ASSERT_EQ(gold.shells.size(), 12u);
    EXPECT_EQ(gold.shells.front().l, 0);
    EXPECT_DOUBLE_EQ(gold.shells.front().alpha.front(), 20.115299);
    EXPECT_EQ(gold.shells.back().l, 3);
    EXPECT_DOUBLE_EQ(gold.shells.back().alpha.back(), 0.72482);

    // Full def2 heavy coverage Rb-La + Hf-Rn; lanthanides stay absent.
    for (int z = 37; z <= 57; ++z)
        EXPECT_FALSE(t[static_cast<std::size_t>(z)].shells.empty()) << z;
    for (int z = 58; z <= 71; ++z)
        EXPECT_TRUE(t[static_cast<std::size_t>(z)].shells.empty()) << z;
    for (int z = 72; z <= 86; ++z)
        EXPECT_FALSE(t[static_cast<std::size_t>(z)].shells.empty()) << z;
}

// ── EcpPotential lookup class ──────────────────────────────────────────

TEST(EcpPotential_, LookupHitAndMiss) {
    const EcpPotential au{molrs::Element::Au, "def2-ECP"};
    EXPECT_EQ(au.n_core, 60);
    ASSERT_EQ(au.channels.size(), 4u);
    EXPECT_EQ(au.channels.back().l, 3); // local U_L last

    // All-electron element in the family: throws, names supported entries.
    try {
        EcpPotential o{molrs::Element::O, "def2-ECP"};
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("def2-ECP"), std::string::npos) << msg;
        EXPECT_NE(msg.find("Supported"), std::string::npos) << msg;
        EXPECT_NE(msg.find("Au"), std::string::npos) << msg;
    }

    // Unknown family name: throws, names the supported family.
    try {
        EcpPotential i{molrs::Element::I, "stuttgart-rlc"};
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("def2-ECP"), std::string::npos)
            << e.what();
    }
}

// ── U matrix vs PySCF ECPscalar: probe A (real def2-ECP on I) ──────────

TEST(EcpIntegrals_, UMatrixPinnedVsPyscfDef2Iodine) {
    std::vector<cpu::scf::native::Shell> shells;
    std::vector<std::vector<double>> store;
    store.reserve(32);
    const double ci[3] = {0.0, 0.0, 0.0};
    const double ch[3] = {0.0, 0.0, 3.0}; // Bohr
    add_shell(shells, store, 0, 1.0, ci);
    add_shell(shells, store, 1, 0.9, ci);
    add_shell(shells, store, 2, 0.8, ci);
    add_shell(shells, store, 0, 0.5, ch);
    add_shell(shells, store, 3, 1.1, ch);
    ASSERT_EQ(shells.size(), 21u);

    const auto& table = cpu::scf::basis_data::ecp_def2();
    EcpIntegrals ints{shells, {EcpCenter{{0.0, 0.0, 0.0}, &table[53]}}};
    const auto U = ints.matrix();

    const double tol = 1e-10;
    // AO order: 0 s(I) | 1-3 p(I) | 4-9 d(I) | 10 s(H) | 11-20 f(H).
    EXPECT_NEAR(U(0, 0), 14.570165990955642, tol);   // on-center s|s
    EXPECT_NEAR(U(0, 3), 0.0, tol);                  // s|pz parity zero
    EXPECT_NEAR(U(3, 3), 1.3153155891567683, tol);   // pz|pz
    EXPECT_NEAR(U(4, 4), 0.3781208959791853, tol);   // dxx|dxx
    EXPECT_NEAR(U(9, 9), 0.37812089597918547, tol);  // dzz|dzz
    EXPECT_NEAR(U(0, 10), 0.11733491248147881, tol); // s(I)|s(H) off-center
    EXPECT_NEAR(U(3, 10), 0.019185187897729716, tol);
    EXPECT_NEAR(U(10, 10), 0.0013777647564709125, tol); // s(H)|s(H)
    EXPECT_NEAR(U(0, 13), -0.0013508245615027352, tol); // s(I)|f_xxz(H)
    EXPECT_NEAR(U(13, 13), 8.033850652280492e-05, tol);
    EXPECT_NEAR(U(0, 20), -0.3171136281006726, tol);  // s(I)|f_zzz(H)
    EXPECT_NEAR(U(9, 20), -0.07169738211390235, tol); // dzz(I)|f_zzz(H)
    EXPECT_NEAR(U(20, 20), 0.027673825681968182, tol);
    EXPECT_NEAR(U(6, 16), 0.003952726519762412, tol); // dxz(I)|f_xzz(H)
    EXPECT_NEAR(U(4, 11), 0.0, tol);                  // dxx|f_xxy parity zero

    // Symmetry of the assembled matrix.
    for (int i = 0; i < 21; ++i)
        for (int j = 0; j < i; ++j)
            EXPECT_DOUBLE_EQ(U(i, j), U(j, i));
}

// ── U matrix vs PySCF ECPscalar: probe B (synthetic f projector) ───────

TEST(EcpIntegrals_, UMatrixPinnedVsPyscfSyntheticFChannel) {
    std::vector<cpu::scf::native::Shell> shells;
    std::vector<std::vector<double>> store;
    store.reserve(24);
    const double cx[3] = {0.0, 0.0, 0.0};
    const double ch[3] = {0.0, 0.0, 3.2}; // Bohr
    add_shell(shells, store, 0, 1.0, cx);
    add_shell(shells, store, 0, 0.8, ch);
    add_shell(shells, store, 1, 1.1, ch);
    add_shell(shells, store, 3, 0.9, ch);
    ASSERT_EQ(shells.size(), 15u);

    // Synthetic semi-local ECP: one f (l=3) projector + local U_L, exactly
    // the user-defined PySCF ECP quoted in the file header. The local
    // channel sits last (relabeled l = L = 4).
    const AtomEcp ecp{10,
                      {EcpChannel{3, {EcpGaussianTerm{2, 1.0, 5.0}}},
                       EcpChannel{4, {EcpGaussianTerm{2, 1.5, -3.0}}}}};
    EcpIntegrals ints{shells, {EcpCenter{{0.0, 0.0, 0.0}, &ecp}}};
    const auto U = ints.matrix();

    const double tol = 1e-10;
    // AO order: 0 s(X) | 1 s(H) | 2-4 p(H) | 5-14 f(H).
    EXPECT_NEAR(U(0, 0), -1.2958781931745214, tol); // local only (on-center)
    EXPECT_NEAR(U(1, 1), 0.0004686514516159109, tol);
    EXPECT_NEAR(U(1, 4), -0.000801415612588452, tol);
    EXPECT_NEAR(U(4, 4), 0.0014194674322348753, tol); // pz|pz, f projector
    EXPECT_NEAR(U(2, 2), 0.00017798145582986112, tol);
    EXPECT_NEAR(U(5, 5), 0.003135088563530394, tol); // f_xxx|f_xxx
    EXPECT_NEAR(U(13, 13), 0.014502453251252007, tol);
    EXPECT_NEAR(U(1, 14), -0.0016804706902401598, tol); // s|f_zzz
    EXPECT_NEAR(U(4, 14), 0.004922158586548874, tol);   // pz|f_zzz
    EXPECT_NEAR(U(14, 14), -0.06850139838965047, tol);  // f_zzz|f_zzz
    EXPECT_NEAR(U(1, 13), 0.0, tol);                    // parity zeros
    EXPECT_NEAR(U(2, 7), 0.0, tol);
}

// ── internal::IntegralWorkspace wiring: valence counting, Z_val, untouched S/T ────────

TEST(EcpIntegralWorkspace, I2ValenceWiring) {
    const auto frame = i2_frame();
    const cpu::scf::internal::IntegralWorkspace ae{frame, BasisSet::Def2SVP};
    const cpu::scf::internal::IntegralWorkspace ve{frame, BasisSet::Def2SVP, 0, false, "def2-ECP"};

    // Electron and AO bookkeeping: 2 * (53 - 28) = 50 valence electrons.
    EXPECT_EQ(ae.n_electrons, 106);
    EXPECT_EQ(ve.n_electrons, 50);
    EXPECT_EQ(ve.n_occ(), 25);
    EXPECT_TRUE(ve.has_ecp);
    EXPECT_FALSE(ae.has_ecp);
    EXPECT_EQ(ve.n_basis, ae.n_basis); // same valence basis, 52 spherical
    EXPECT_EQ(ve.n_basis, 52);

    // V_nuc uses Z_val = 25: 625 / R. PySCF 2.14.0: mol.energy_nuc() =
    // 124.05692304013505 for I2 at 2.666 A (identical Bohr constant).
    const double r_bohr =
        units::convert(2.666, units::Length::Angstrom, units::Length::Bohr);
    EXPECT_NEAR(ve.V_nuc, 625.0 / r_bohr, 1e-12);
    EXPECT_NEAR(ve.V_nuc, 124.05692304013505, 1e-8);
    EXPECT_NEAR(ae.V_nuc, 53.0 * 53.0 / r_bohr, 1e-12);

    // S and T are byte-untouched by the ECP; V (and hence Hcore) differs.
    ASSERT_EQ(ae.S.shape(), ve.S.shape());
    double max_dS = 0.0, max_dT = 0.0, max_dV = 0.0;
    for (std::size_t i = 0; i < ae.S.shape()[0]; ++i) {
        for (std::size_t j = 0; j < ae.S.shape()[1]; ++j) {
            max_dS = std::max(max_dS, std::abs(ae.S(i, j) - ve.S(i, j)));
            max_dT = std::max(max_dT, std::abs(ae.T(i, j) - ve.T(i, j)));
            max_dV = std::max(max_dV, std::abs(ae.V(i, j) - ve.V(i, j)));
        }
    }
    EXPECT_EQ(max_dS, 0.0);
    EXPECT_EQ(max_dT, 0.0);
    EXPECT_GT(max_dV, 1.0); // core removal changes V by O(Z_core)
}

TEST(EcpIntegralWorkspace, LightMoleculeUnchangedAndUnknownNameThrows) {
    // H2O: every atom is all-electron in def2-ECP — the pack must be
    // identical to the empty-ecp path (per-element table, no global switch).
    cpu::Frame h2o(3);
    h2o.span(kAtom_type)[0] = 8;
    h2o.span(kAtom_x)[0] = 0.0;
    h2o.span(kAtom_y)[0] = 0.0;
    h2o.span(kAtom_z)[0] = 0.1173;
    h2o.span(kAtom_type)[1] = 1;
    h2o.span(kAtom_x)[1] = 0.0;
    h2o.span(kAtom_y)[1] = 0.7572;
    h2o.span(kAtom_z)[1] = -0.4692;
    h2o.span(kAtom_type)[2] = 1;
    h2o.span(kAtom_x)[2] = 0.0;
    h2o.span(kAtom_y)[2] = -0.7572;
    h2o.span(kAtom_z)[2] = -0.4692;
    h2o.box = cpu::Box::cubic(20.0);

    const cpu::scf::internal::IntegralWorkspace plain{h2o, BasisSet::Def2SVP};
    const cpu::scf::internal::IntegralWorkspace named{h2o, BasisSet::Def2SVP, 0, false, "def2-ECP"};
    EXPECT_FALSE(named.has_ecp);
    EXPECT_EQ(named.n_electrons, plain.n_electrons);
    EXPECT_EQ(named.V_nuc, plain.V_nuc);
    double max_dH = 0.0;
    for (std::size_t i = 0; i < plain.Hcore.shape()[0]; ++i)
        for (std::size_t j = 0; j < plain.Hcore.shape()[1]; ++j)
            max_dH = std::max(max_dH,
                              std::abs(plain.Hcore(i, j) - named.Hcore(i, j)));
    EXPECT_EQ(max_dH, 0.0);

    EXPECT_THROW(cpu::scf::internal::IntegralWorkspace(h2o, BasisSet::Def2SVP, 0, false, "no-such"),
                 std::runtime_error);
}

// ── ECP ↔ x2c per-element double-counting gate ─────────────────────────

TEST(EcpX2cGate, ThrowsForEcpElement) {
    // I2 + x2c + def2-ECP: throws before any SCF work, naming the element.
    // (e2e-in-tests wave-2: no multi-iter all-electron run under tests/.)
    auto frame = i2_frame();
    cpu::scf::RHF rhf;
    rhf.basis = BasisSet::Def2SVP;
    rhf.ecp = "def2-ECP";
    rhf.x2c = true;
    try {
        rhf.run(frame);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("I (Z=53)"), std::string::npos) << msg;
        EXPECT_NE(msg.find("x2c"), std::string::npos) << msg;
    }
}
