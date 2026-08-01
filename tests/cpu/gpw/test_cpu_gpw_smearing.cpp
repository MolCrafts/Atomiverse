/// @file test_cpu_gpw_smearing.cpp
/// @brief Unit tests for GPW metallic occupation smearing API + fill math.
///
/// Full multi-k SCF is the regressions `reg_gpw-metallic-smearing` (RKS) and
/// `reg_gpw-uks-smearing` (UKS). Here: public members (no Config bag), build()
/// fail-fast, and reuse of pw::internal occupation fill (FD recovers Ne).

#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/planewave/internal/smearing.h"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace ATV;
using namespace ATV::cpu::gpw;
using ATV::cpu::pw::internal::SmearKind;
using ATV::cpu::pw::internal::fill_occupations;
using ATV::cpu::pw::internal::find_fermi_level;
using ATV::cpu::pw::internal::smear_f_band;
using ATV::cpu::pw::internal::smear_f_spinor;

TEST(GpwSmearing, PublicMembersNoConfigBag) {
    GpwDft m;
    m.smearing = Smearing::fermi_dirac;
    m.degauss = 0.02;
    m.mp_order = 1;
    m.extra_bands = 4;
    m.n_occ = 9;
    EXPECT_EQ(m.smearing, Smearing::fermi_dirac);
    EXPECT_DOUBLE_EQ(m.degauss, 0.02);
    EXPECT_EQ(m.extra_bands, 4);
    GpwResult r;
    EXPECT_DOUBLE_EQ(r.e_fermi, 0.0);
    EXPECT_DOUBLE_EQ(r.e_entropy, 0.0);
    EXPECT_DOUBLE_EQ(r.n_elec_occ, 0.0);
}

TEST(GpwSmearing, BuildAcceptsOtRksAndUksSmearRejectsOtUksSmear) {
    std::vector<GpwAtom> atoms;
    GpwDft empty;
    empty.n_occ = 1;
    empty.smearing = Smearing::fermi_dirac;
    empty.degauss = 0.02;
    EXPECT_THROW(empty.build(atoms), std::invalid_argument);

    GpwAtom a;
    a.z_val = 1.0;
    a.z = 1;
    atoms.push_back(a);

    // UKS + FD is productized (multi-k path); build must accept it.
    GpwDft uks;
    uks.n_alpha = 1;
    uks.n_beta = 0;
    uks.cell[0] = uks.cell[1] = uks.cell[2] = 10.0;
    uks.smearing = Smearing::fermi_dirac;
    uks.degauss = 0.02;
    uks.k_mesh[0] = uks.k_mesh[1] = uks.k_mesh[2] = 2;
    EXPECT_NO_THROW(uks.build(atoms));

    // RKS OT + FD is productized (multi-k OT + metallic occupations).
    GpwDft ot;
    ot.n_occ = 1;
    ot.cell[0] = ot.cell[1] = ot.cell[2] = 10.0;
    ot.use_ot = true;
    ot.smearing = Smearing::fermi_dirac;
    ot.degauss = 0.02;
    ot.k_mesh[0] = ot.k_mesh[1] = ot.k_mesh[2] = 2;
    ot.extra_bands = 2;
    EXPECT_NO_THROW(ot.build(atoms));

    // UKS OT + FD is productized (multi-k UKS OT + metallic occupations).
    GpwDft ot_uks;
    ot_uks.n_alpha = 1;
    ot_uks.n_beta = 0;
    ot_uks.cell[0] = ot_uks.cell[1] = ot_uks.cell[2] = 10.0;
    ot_uks.use_ot = true;
    ot_uks.smearing = Smearing::fermi_dirac;
    ot_uks.degauss = 0.02;
    ot_uks.k_mesh[0] = ot_uks.k_mesh[1] = ot_uks.k_mesh[2] = 2;
    ot_uks.extra_bands = 2;
    EXPECT_NO_THROW(ot_uks.build(atoms));

    GpwDft bad_deg;
    bad_deg.n_occ = 1;
    bad_deg.cell[0] = bad_deg.cell[1] = bad_deg.cell[2] = 10.0;
    bad_deg.smearing = Smearing::fermi_dirac;
    bad_deg.degauss = 0.0;
    EXPECT_THROW(bad_deg.build(atoms), std::invalid_argument);
}

TEST(GpwSmearing, FermiDiracFillRecoversNelec) {
    // Shared with PW unit tests: three levels, want Ne=3, band form.
    std::vector<AccReal> evals = {AccReal(-1), AccReal(0), AccReal(1),
                                  AccReal(2)};
    std::vector<AccReal> w(evals.size(), AccReal(1));
    const AccReal ne = AccReal(3);
    const AccReal deg = AccReal(0.1);
    const AccReal mu =
        find_fermi_level(evals, w, ne, deg, SmearKind::fermi_dirac, 1);
    auto fill = fill_occupations(evals, w, mu, deg, SmearKind::fermi_dirac, 1);
    EXPECT_NEAR(static_cast<double>(fill.n_occ), 3.0, 1e-8);
    EXPECT_NEAR(static_cast<double>(
                    smear_f_band(AccReal(0), AccReal(0), deg,
                                 SmearKind::fermi_dirac)),
                1.0, 1e-12);
}

TEST(GpwSmearing, SpinorFillRecoversNelecUks) {
    // Collinear spin-orbitals: spin_degen=1, f∈[0,1]; Ne=3 over 4 levels.
    std::vector<AccReal> evals = {AccReal(-1), AccReal(0), AccReal(0.5),
                                  AccReal(2)};
    std::vector<AccReal> w(evals.size(), AccReal(1));
    const AccReal ne = AccReal(3);
    const AccReal deg = AccReal(0.1);
    const AccReal mu = find_fermi_level(evals, w, ne, deg,
                                       SmearKind::fermi_dirac, 1, AccReal(1));
    auto fill = fill_occupations(evals, w, mu, deg, SmearKind::fermi_dirac, 1,
                                 AccReal(1));
    EXPECT_NEAR(static_cast<double>(fill.n_occ), 3.0, 1e-8);
    EXPECT_NEAR(static_cast<double>(smear_f_spinor(
                    AccReal(0), AccReal(0), deg, SmearKind::fermi_dirac)),
                0.5, 1e-12);
}

TEST(GpwSmearing, MethfesselPaxtonMemberAccepted) {
    GpwDft m;
    m.smearing = Smearing::methfessel_paxton;
    m.mp_order = 1;
    m.degauss = 0.02;
    m.n_occ = 1;
    m.cell[0] = m.cell[1] = m.cell[2] = 8.0;
    GpwAtom a;
    a.z_val = 1.0;
    a.z = 1;
    std::vector<GpwAtom> atoms{a};
    EXPECT_NO_THROW(m.build(atoms));
}

TEST(GpwSmearing, ResultHasFractionalPayloadFields) {
    GpwResult r;
    EXPECT_TRUE(r.f_k.empty());
    EXPECT_TRUE(r.f_k_beta.empty());
    r.f_k.push_back({2.0, 1.5, 0.1});
    EXPECT_EQ(r.f_k[0].size(), 3u);
}
