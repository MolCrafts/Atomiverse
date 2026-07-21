/// @file tests/cpu/scf/test_embedding.cpp
/// @brief Unit: point-charge embedding + external-field core Hamiltonian.
///
/// Machine identities (nuclear-attraction parity, classical sums, multipole
/// origin, empty-env no-op) and a small finite-field linear response. No
/// full SCF-to-external-engine e2e — that lives in regressions/.

#ifdef ATV_HAS_HF

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observer/multipole.h"
#include "atomiverse/cpu/scf/internal/external_field.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/point_charges.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/scf/native_integrals.h"

using namespace ATV;
using namespace ATV::cpu::scf;
using ATV::cpu::scf::internal::ExternalField;
using ATV::cpu::scf::internal::PointChargeEmbedding;
using ATV::cpu::scf::internal::PointChargeSite;

namespace {

cpu::Frame make_h2o() {
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.1173;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.7572;
    f.span(kAtom_z)[1] = -0.4692;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = 0.0;
    f.span(kAtom_y)[2] = -0.7572;
    f.span(kAtom_z)[2] = -0.4692;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

/// Asymmetric HHe-like fragment (Z=1,2) so the nuclear-charge center is off
/// a geometric midpoint.
cpu::Frame make_asymmetric() {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 2;
    f.span(kAtom_x)[1] = 1.0;
    f.span(kAtom_y)[1] = 0.2;
    f.span(kAtom_z)[1] = -0.3;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

TEST(PointChargeEmbedding, CoreMatrixMatchesNuclearAttractionUnitCharge) {
    auto frame = make_h2o();
    IntegralPack pack{frame, BasisSet::STO3G};

    // Unit charge at a fixed site [Å] → matches Z=1 pseudo-atom nuclear V
    // on the same shell list (the nuclear-attraction primitive).
    const double rx = 1.5, ry = -0.4, rz = 2.0;
    PointChargeEmbedding emb({{rx, ry, rz, 1.0}});
    const auto V_emb = emb.core_matrix(pack.shells);

    const int n = static_cast<int>(pack.shells.size());
    xt::xtensor<double, 2> V_ref = xt::zeros<double>({n, n});
    const native::Atom probe{1,
                             {rx * units::kAngstromInBohr,
                              ry * units::kAngstromInBohr,
                              rz * units::kAngstromInBohr}};
    const auto blocks = blocks_from_shells(pack.shells);
    std::vector<double> buf;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            const auto& A = blocks[i];
            const auto& B = blocks[j];
            buf.assign(static_cast<std::size_t>(A.n_cart * B.n_cart), 0.0);
            native::nuclear_block(A, B, &probe, 1, buf.data());
            for (int ia = 0; ia < A.n_cart; ++ia)
                for (int ib = 0; ib < B.n_cart; ++ib) {
                    const double v =
                        buf[static_cast<std::size_t>(ia * B.n_cart + ib)];
                    V_ref(A.ao_start + ia, B.ao_start + ib) = v;
                    V_ref(B.ao_start + ib, A.ao_start + ia) = v;
                }
        }
    }

    ASSERT_EQ(V_emb.shape(), V_ref.shape());
    for (std::size_t i = 0; i < V_emb.size(); ++i)
        EXPECT_NEAR(V_emb.flat(i), V_ref.flat(i), 1e-12) << "element " << i;
}

TEST(PointChargeEmbedding, ClassicalSumsExact) {
    // Two charges and two nuclei — hand-computed 1/r sums in Bohr.
    const double a2b = units::kAngstromInBohr;
    PointChargeEmbedding emb({{-1.0, 0.0, 0.0, 0.5}, {1.0, 0.0, 0.0, -0.25}});

    std::vector<native::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {0.0, a2b, 0.0}},
    };
    // QM–MM: Σ Z q / r
    const double r_O_m = 1.0 * a2b;
    const double r_O_p = 1.0 * a2b;
    const double r_H_m =
        std::sqrt((0.0 + 1.0 * a2b) * (0.0 + 1.0 * a2b) + a2b * a2b);
    const double r_H_p =
        std::sqrt((0.0 - 1.0 * a2b) * (0.0 - 1.0 * a2b) + a2b * a2b);
    const double want_qm_mm = 8.0 * 0.5 / r_O_m + 8.0 * (-0.25) / r_O_p
                             + 1.0 * 0.5 / r_H_m + 1.0 * (-0.25) / r_H_p;
    EXPECT_NEAR(emb.nuclear_energy(atoms), want_qm_mm, 1e-12);

    const double r_mm = 2.0 * a2b;
    const double want_mm = 0.5 * (-0.25) / r_mm;
    EXPECT_NEAR(emb.self_energy(), want_mm, 1e-12);
}

TEST(PointChargeEmbedding, NonFiniteThrows) {
    const PointChargeSite bad_q{0.0, 0.0, 0.0, std::nan("")};
    EXPECT_THROW(PointChargeEmbedding({bad_q}), std::invalid_argument);
    const PointChargeSite bad_r{std::numeric_limits<double>::infinity(), 0.0,
                                0.0, 1.0};
    EXPECT_THROW(PointChargeEmbedding({bad_r}), std::invalid_argument);
}

TEST(ExternalField, OriginMatchesMultipoleCenterOfNuclearCharge) {
    auto frame = make_asymmetric();
    // charge = +1 → 2 electrons (closed shell) on Z=1+2 nuclei.
    IntegralPack pack{frame, BasisSet::STO3G, /*charge=*/1};
    const std::array<double, 3> F{{1e-4, 0.0, 0.0}};
    ExternalField field(F, pack.atoms);

    // Hand center of nuclear charge [Bohr].
    double zsum = 0.0, o[3] = {};
    for (const auto& A : pack.atoms) {
        const double z = static_cast<double>(A.Z);
        zsum += z;
        o[0] += z * A.pos[0];
        o[1] += z * A.pos[1];
        o[2] += z * A.pos[2];
    }
    o[0] /= zsum;
    o[1] /= zsum;
    o[2] /= zsum;
    EXPECT_NEAR(field.origin_bohr()[0], o[0], 1e-12);
    EXPECT_NEAR(field.origin_bohr()[1], o[1], 1e-12);
    EXPECT_NEAR(field.origin_bohr()[2], o[2], 1e-12);

    // Same origin the multipole observer reports (Å) — link-03 convention.
    RHF rhf;
    rhf.basis = BasisSet::STO3G;
    rhf.charge = 1;
    rhf.tol = 1e-10;
    rhf.max_iter = 80;
    rhf.run(frame);
    ASSERT_TRUE(rhf.converged());
    cpu::MultipoleObserver obs{rhf.result_handle()};
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, frame, 0, 0};
    obs.launch(lc);
    for (int a = 0; a < 3; ++a)
        EXPECT_NEAR(field.origin_bohr()[static_cast<std::size_t>(a)]
                        * units::kBohrInAngstrom,
                    obs.origin_angstrom()[static_cast<std::size_t>(a)], 1e-12);
}

TEST(ExternalField, SmallFieldEnergyShiftMatchesDipole) {
    auto frame = make_h2o();
    RHF rhf0;
    rhf0.basis = BasisSet::STO3G;
    rhf0.tol = 1e-12;
    rhf0.max_iter = 100;
    rhf0.run(frame);
    const double e0 = rhf0.energy();

    cpu::MultipoleObserver obs{rhf0.result_handle()};
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, frame, 0, 0};
    obs.launch(lc);
    const auto mu = obs.dipole_au();

    const double Fz = 1e-4;
    RHF rhfF;
    rhfF.basis = BasisSet::STO3G;
    rhfF.tol = 1e-12;
    rhfF.max_iter = 100;
    rhfF.external_field = {0.0, 0.0, Fz};
    rhfF.run(frame);
    const double eF = rhfF.energy();

    // E(F) − E(0) ≈ −μ · F  (first order)
    const double dE = eF - e0;
    const double want = -mu[2] * Fz;
    const double rel = std::abs(dE - want) / std::max(std::abs(want), 1e-16);
    EXPECT_LT(rel, 1e-2) << "dE=" << dE << " want=" << want;
}

TEST(MeanFieldEnvironment, ForcesThrowWhenEnvironmentActive) {
    auto frame = make_h2o();
    RHF rhf;
    rhf.basis = BasisSet::STO3G;
    rhf.tol = 1e-8;
    rhf.max_iter = 50;
    rhf.point_charges = {{0.0, 0.0, 3.0, 0.5}};
    EXPECT_THROW(rhf.forces(frame), std::runtime_error);

    RHF rhf2;
    rhf2.basis = BasisSet::STO3G;
    rhf2.tol = 1e-8;
    rhf2.max_iter = 50;
    rhf2.external_field = {0.0, 0.0, 1e-4};
    EXPECT_THROW(rhf2.forces(frame), std::runtime_error);
}

TEST(MeanFieldEnvironment, EmptyEnvironmentIsNoOp) {
    auto frame = make_h2o();
    RHF a;
    a.basis = BasisSet::STO3G;
    a.tol = 1e-12;
    a.max_iter = 100;
    a.run(frame);
    const double e_vac = a.energy();

    RHF b;
    b.basis = BasisSet::STO3G;
    b.tol = 1e-12;
    b.max_iter = 100;
    b.point_charges = {}; // empty
    b.external_field = {0.0, 0.0, 0.0};
    b.run(frame);
    // Same vacuum path: allow tiny SCF noise (not bitwise; two independent runs).
    EXPECT_NEAR(b.energy(), e_vac, 1e-12);
}

#else
TEST(Embedding, SkippedWithoutHF) { GTEST_SKIP() << "ATV_HAS_HF off"; }
#endif
