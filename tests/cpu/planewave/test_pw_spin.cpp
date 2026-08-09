/// @file tests/cpu/planewave/test_pw_spin.cpp
/// @brief Unit tests for collinear spin-polarized PW members and O2 mag.
#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

cpu::Frame o2_frame(double L_ang) {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 8;
        f.span(kAtom_mass)[i] = 15.999;
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
    }
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = 1.2075; // Å
    f.box = cpu::Box::cubic(L_ang);
    return f;
}

} // namespace

TEST(PWSpin, DefaultRestricted) {
    PW pw;
    pw.gth_name = "GTH-PBE-q6";
    EXPECT_FALSE(pw.spin_polarized);
    EXPECT_EQ(pw.multiplicity, 1);
    EXPECT_DOUBLE_EQ(pw.last_result().magnetization, 0.0);
}

TEST(PWSpin, MultiplicityParityThrows) {
    auto f = o2_frame(8.0);
    PW pw;
    pw.gth_name = "GTH-PBE-q6";
    pw.spin_polarized = true;
    pw.multiplicity = 2; // odd mag with even n_elec → invalid
    pw.ecut_ry = 8.0;
    pw.k_mesh = 1;
    pw.max_iter = 1;
    pw.build(f);
    EXPECT_THROW(pw.run(f), std::invalid_argument);
}

TEST(PWSpin, O2TripletMagnetizationIdentity) {
    // Fixed occupations: multiplicity 3 → M = 2 on PWResult (occupation
    // identity; does not require a tight SCF). Tiny ecut / short budget so
    // the unit suite stays in the seconds regime. Default xc=PBE uses true
    // collinear spin-GGA (`XCBuilder::step_spin`).
    auto f = o2_frame(6.0);
    PW pw;
    pw.gth_name = "GTH-PBE-q6";
    pw.ecut_ry = 4.0;
    pw.k_mesh = 1;
    pw.tol = 1e-3;
    pw.max_iter = 2; // occupation identity; SCF quality is irrelevant
    pw.mix_beta = 0.5;
    pw.spin_polarized = true;
    pw.multiplicity = 3;
    pw.xc = cpu::scf::XCType::PBE;

    pw.build(f);
    const auto& r = pw.run(f);
    EXPECT_TRUE(std::isfinite(r.energy));
    EXPECT_TRUE(std::isfinite(r.energy_ha));
    EXPECT_NEAR(r.magnetization, 2.0, 1e-12);
}

TEST(PWSpin, CollinearSmearNonMagneticSeedKeepsSpinSymmetry) {
    // EXACT invariant. `multiplicity = 1` seeds ρ↑ ≡ ρ↓, and every operator
    // in the collinear loop is spin-symmetric under that seed (V_H from the
    // total density, V_xc,σ from a symmetric ρ_σ pair, one shared μ). So the
    // two channels stay bit-identical and M = N↑ − N↓ is identically zero —
    // no tolerance, no convergence needed.
    //
    // This is the invariant `PW::run`'s collinear+smearing branch violated:
    // it assembled spin ↓'s V_xc/V_eff while ws.gvec/ws.n_g still pointed at
    // the *last k-point's wavefunction* G-sphere instead of the density
    // sphere, so spin ↓ saw a different Hamiltonian and the run broke spin
    // symmetry spontaneously (M ≈ −0.61 on a non-magnetic cell).
    auto f = o2_frame(6.0);
    PW pw;
    pw.gth_name = "GTH-PBE-q6";
    pw.ecut_ry = 4.0;
    pw.k_mesh = 1;
    pw.tol = 1e-4;
    pw.max_iter = 2;
    pw.mix_beta = 0.5;
    pw.xc = cpu::scf::XCType::LSDA;
    pw.spin_polarized = true;
    pw.multiplicity = 1;
    pw.smearing = Smearing::fermi_dirac;
    pw.degauss = 0.02;
    pw.extra_bands = 2;

    pw.build(f);
    const auto& r = pw.run(f);
    EXPECT_TRUE(std::isfinite(r.energy));
    EXPECT_EQ(r.magnetization, 0.0);
}

TEST(PWSpin, CollinearSmearMultiKSharedMuConservesElectrons) {
    // Two exact statements at once, over a Monkhorst–Pack mesh:
    //   (1) the k weights are normalised (Σ_k w_k = 1) and ONE chemical
    //       potential is shared by both spin channels, so
    //       Σ_{σ k b} w_k f_{σkb} = N_e exactly;
    //   (2) a non-magnetic seed keeps M ≡ 0 (see the Γ test above) — which
    //       also proves each k-point's spin-↓ Hamiltonian is the spin-↑ one.
    auto f = o2_frame(5.5);
    PW pw;
    pw.gth_name = "GTH-PBE-q6";
    pw.ecut_ry = 4.0;
    pw.k_mesh = 2;
    pw.tol = 1e-4;
    pw.max_iter = 1;
    pw.mix_beta = 0.5;
    pw.xc = cpu::scf::XCType::LSDA;
    pw.spin_polarized = true;
    pw.multiplicity = 1;
    pw.smearing = Smearing::fermi_dirac;
    pw.degauss = 0.02;
    pw.extra_bands = 2;

    pw.build(f);
    const auto& r = pw.run(f);
    EXPECT_EQ(r.n_kpts, 8); // k_mesh³ — the run is genuinely multi-k
    EXPECT_NEAR(r.n_elec_occ, 12.0, 1e-9);
    EXPECT_EQ(r.magnetization, 0.0);
}

TEST(PWSpin, O2TripletMultiKMagnetizationIdentity) {
    // k-point collinear UKS (pw-uks-kmesh-fe): fixed occupations still set
    // M = multiplicity − 1 with Monkhorst–Pack mesh ≥ 2. One SCF iter is
    // enough for the occupation identity; keep wall in the unit-test budget.
    auto f = o2_frame(5.5);
    PW pw;
    pw.gth_name = "GTH-PBE-q6";
    pw.ecut_ry = 4.0;
    pw.k_mesh = 2;
    pw.tol = 1e-1;
    pw.max_iter = 1;
    pw.mix_beta = 0.5;
    pw.spin_polarized = true;
    pw.multiplicity = 3;

    pw.build(f);
    const auto& r = pw.run(f);
    EXPECT_TRUE(std::isfinite(r.energy));
    EXPECT_TRUE(std::isfinite(r.energy_ha));
    EXPECT_NEAR(r.magnetization, 2.0, 1e-12);
}
