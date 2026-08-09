/// @file tests/cpu/semiempirical/test_dftb2.cpp
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

cpu::Frame make_h2(double r_ang = 0.74) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = static_cast<StorageReal>(r_ang);
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    return f;
}

cpu::Frame make_water_offeq() {
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 1.05;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = -0.26;
    f.span(kAtom_y)[2] = 1.0;
    f.span(kAtom_z)[2] = 0.0;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_mass)[i] = (i == 0) ? 15.999 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

/// O₂ triplet geometry (experimental-ish bond length, Å).
cpu::Frame make_o2(double r = 1.21) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = static_cast<StorageReal>(r);
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 15.999;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(30.0);
    f.meta.get(kEnergy) = 0;
    return f;
}


/// 1-D C dimer crystal: r(C–C)=1.4 Å, Lx=2.8 Å → second-neighbor images.
cpu::Frame make_c_chain_cell(double Lx = 2.8, double Ly = 12.0, double Lz = 12.0) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 6;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 1.4;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 12.011;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::orthorhombic(Lx, Ly, Lz);
    f.meta.get(kEnergy) = 0;
    return f;
}

double energy_only(cpu::Frame g) {
    g.meta.get(kEnergy) = 0;
    for (int i = 0; i < g.nlocal(); ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 d;
    d.with_forces = false;
    d.scc_tol = 1e-12;
    d.max_scc = 80;
    d.build({ctx, g});
    d.launch({ctx, g, 0, 0});
    return d.last_energy();
}

} // namespace

TEST(DFTB2, H2EnergyFinite) {
    auto f = make_h2(0.74);
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(dftb.last_energy()));
    // Closed shell: zero Mulliken spin
    ASSERT_EQ(dftb.last_mulliken_spin().size(), 2u);
    EXPECT_NEAR(dftb.last_mulliken_spin()[0], 0.0, 1e-8);
    EXPECT_NEAR(dftb.last_mulliken_spin()[1], 0.0, 1e-8);
}

TEST(DFTB2, SccConvergesIndependentOfIterationBudget) {
    // Undamped Picard on a polar system is a period-2 limit cycle: it never
    // meets scc_tol, always burns the whole budget, and the reported energy
    // is selected by the PARITY of max_scc. A mixed SCC reaches the fixed
    // point, so the budget must not touch the answer.
    double e[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        auto f = make_water_offeq();
        cpu::Context ctx;
        cpu::semiempirical::DFTB2 dftb;
        dftb.with_forces = false;
        dftb.scc_tol = 1e-10;
        dftb.max_scc = 49 + i;
        dftb.build({ctx, f});
        dftb.launch({ctx, f, 0, 0});
        EXPECT_LT(dftb.last_scc_residual(), dftb.scc_tol) << "budget " << i;
        EXPECT_LT(dftb.last_scc_iterations(), dftb.max_scc) << "budget " << i;
        e[i] = dftb.last_energy();
    }
    EXPECT_NEAR(e[0], e[1], 1e-10);
    EXPECT_NEAR(e[1], e[2], 1e-10);
}

TEST(DFTB2, ThrowsWhenSccBudgetIsExhausted) {
    // No silent fallback to the last iterate — an unconverged SCC is an error.
    auto f = make_water_offeq();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = false;
    dftb.scc_tol = 1e-10;
    dftb.max_scc = 2;
    dftb.build({ctx, f});
    EXPECT_THROW(dftb.launch({ctx, f, 0, 0}), std::runtime_error);
}

TEST(DFTB2, RejectsMixBetaOutsideUnitInterval) {
    auto f = make_h2(0.74);
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.mix_beta = 0.0;
    dftb.build({ctx, f});
    EXPECT_THROW(dftb.launch({ctx, f, 0, 0}), std::invalid_argument);
}

TEST(DFTB2, RejectsUnsupportedElement) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 9; // F not in tables
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0;
    f.span(kAtom_mass)[0] = 19.0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(10.0);
    f.meta.get(kEnergy) = 0;
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.build({ctx, f});
    EXPECT_THROW(dftb.launch({ctx, f, 0, 0}), std::runtime_error);
}

TEST(DFTB2, AnalyticForceMatchesFdH2) {
    auto f = make_h2(0.90);
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = true;
    dftb.scc_tol = 1e-12;
    dftb.max_scc = 80;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});

    constexpr double h = 1.0e-4;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            auto gp = f;
            auto gm = f;
            StorageReal* cp = (k == 0)   ? gp.span(kAtom_x)
                              : (k == 1) ? gp.span(kAtom_y)
                                         : gp.span(kAtom_z);
            StorageReal* cm = (k == 0)   ? gm.span(kAtom_x)
                              : (k == 1) ? gm.span(kAtom_y)
                                         : gm.span(kAtom_z);
            cp[a] += static_cast<StorageReal>(h);
            cm[a] -= static_cast<StorageReal>(h);
            const double fd = -(energy_only(gp) - energy_only(gm)) / (2.0 * h);
            const double ana =
                (k == 0)   ? static_cast<double>(f.span(kAtom_fx)[a])
                : (k == 1) ? static_cast<double>(f.span(kAtom_fy)[a])
                           : static_cast<double>(f.span(kAtom_fz)[a]);
            EXPECT_NEAR(ana, fd, 5e-3) << "atom " << a << " k " << k;
        }
    }
    // H2 force antisymmetry on bond axis
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_fx)[0]
                                    + f.span(kAtom_fx)[1]),
                0.0, 1e-6);
}

TEST(DFTB2, AnalyticForceMatchesFdH2O) {
    auto f = make_water_offeq();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = true;
    dftb.scc_tol = 1e-12;
    dftb.max_scc = 80;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});

    const double h = 1e-4;
    // Gate O_x and H1_y only (same dual-component gate as dftb2-analytic-forces).
    // A converged SCC-DFTB2 gradient is stationary, so this is FD-exact to the
    // central-difference truncation error — there is NO "charge response
    // residual" (that framing was masking the γ-gradient sign bug and the
    // Σ_A V_A q⁰_A double-counting error; both fixed 2026-07-28).
    constexpr double kFdTol = 1e-5; // eV/Å; measured residual ≤ 3e-7
    {
        auto fp = make_water_offeq();
        fp.span(kAtom_x)[0] += static_cast<StorageReal>(h);
        auto fm = make_water_offeq();
        fm.span(kAtom_x)[0] -= static_cast<StorageReal>(h);
        const double fd = -(energy_only(fp) - energy_only(fm)) / (2.0 * h);
        EXPECT_NEAR(static_cast<double>(f.span(kAtom_fx)[0]), fd, kFdTol);
    }
    {
        auto fp = make_water_offeq();
        fp.span(kAtom_y)[1] += static_cast<StorageReal>(h);
        auto fm = make_water_offeq();
        fm.span(kAtom_y)[1] -= static_cast<StorageReal>(h);
        const double fd = -(energy_only(fp) - energy_only(fm)) / (2.0 * h);
        EXPECT_NEAR(static_cast<double>(f.span(kAtom_fy)[1]), fd, kFdTol);
    }

    double net[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        net[0] += static_cast<double>(f.span(kAtom_fx)[i]);
        net[1] += static_cast<double>(f.span(kAtom_fy)[i]);
        net[2] += static_cast<double>(f.span(kAtom_fz)[i]);
    }
    EXPECT_NEAR(net[0], 0.0, 1e-6);
    EXPECT_NEAR(net[1], 0.0, 1e-6);
    EXPECT_NEAR(net[2], 0.0, 1e-6);
}

TEST(DFTB2, O2TripletSpinPopulation) {
    auto f = make_o2();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.multiplicity = 3;
    dftb.with_forces = false;
    dftb.scc_tol = 1e-10;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(dftb.last_energy()));
    ASSERT_EQ(dftb.last_mulliken_spin().size(), 2u);
    const double m0 = dftb.last_mulliken_spin()[0];
    const double m1 = dftb.last_mulliken_spin()[1];
    const double m_tot = m0 + m1;
    EXPECT_NEAR(m_tot, 2.0, 0.25);
    EXPECT_GT(std::fabs(m0), 0.3);
    EXPECT_GT(std::fabs(m1), 0.3);
    EXPECT_NEAR(m0, m1, 0.15);
}

TEST(DFTB2, SpinAnalyticForceMatchesFdO2) {
    // Off-eq triplet O₂ so |F| ≠ 0; collinear open-shell closed-form F.
    auto f = make_o2(1.30);
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.multiplicity = 3;
    dftb.with_forces = true;
    dftb.scc_tol = 1e-12;
    dftb.max_scc = 80;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(dftb.last_energy()));

    const double fx0 = static_cast<double>(f.span(kAtom_fx)[0]);
    const double fx1 = static_cast<double>(f.span(kAtom_fx)[1]);
    const double fy0 = static_cast<double>(f.span(kAtom_fy)[0]);
    const double fz0 = static_cast<double>(f.span(kAtom_fz)[0]);
    EXPECT_TRUE(std::isfinite(fx0));
    EXPECT_GT(fx0 * fx0 + fy0 * fy0 + fz0 * fz0, 1e-8);
    // Translation / action-reaction on bond axis
    EXPECT_NEAR(fx0 + fx1, 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_fy)[0]
                                    + f.span(kAtom_fy)[1]),
                0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_fz)[0]
                                    + f.span(kAtom_fz)[1]),
                0.0, 1e-6);

    auto energy_spin = [](cpu::Frame g) {
        g.meta.get(kEnergy) = 0;
        for (int i = 0; i < g.nlocal(); ++i)
            g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] =
                0;
        cpu::Context c;
        cpu::semiempirical::DFTB2 d;
        d.multiplicity = 3;
        d.with_forces = false;
        d.scc_tol = 1e-12;
        d.max_scc = 80;
        d.build({c, g});
        d.launch({c, g, 0, 0});
        return d.last_energy();
    };

    constexpr double h = 1.0e-4;
    // Gate F_O0_x (same-element residual should be tighter than multi-element).
    {
        auto gp = make_o2(1.30);
        auto gm = make_o2(1.30);
        gp.span(kAtom_x)[0] += static_cast<StorageReal>(h);
        gm.span(kAtom_x)[0] -= static_cast<StorageReal>(h);
        const double fd = -(energy_spin(gp) - energy_spin(gm)) / (2.0 * h);
        EXPECT_NEAR(fx0, fd, 5e-2) << "analytic=" << fx0 << " fd=" << fd;
    }
    {
        auto gp = make_o2(1.30);
        auto gm = make_o2(1.30);
        gp.span(kAtom_x)[1] += static_cast<StorageReal>(h);
        gm.span(kAtom_x)[1] -= static_cast<StorageReal>(h);
        const double fd = -(energy_spin(gp) - energy_spin(gm)) / (2.0 * h);
        EXPECT_NEAR(fx1, fd, 5e-2) << "analytic=" << fx1 << " fd=" << fd;
    }
}

TEST(DFTB2, RejectsBadMultiplicity) {
    auto f = make_o2();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.multiplicity = 4; // 12 valence e → odd unpaired not consistent
    dftb.with_forces = false;
    dftb.build({ctx, f});
    EXPECT_THROW(dftb.launch({ctx, f, 0, 0}), std::runtime_error);
}

TEST(DFTB2, PeriodicGammaChainFinite) {
    const double e = energy_only(make_c_chain_cell());
    EXPECT_TRUE(std::isfinite(e));
    EXPECT_GT(std::fabs(e), 1e-3);
}

TEST(DFTB2, PeriodicGammaDiffersFromIsolatedDimer) {
    // Dense 1-D cell includes second-neighbor images along x; isolated dimer
    // (large cubic box) does not — energies must separate.
    const double e_pbc = energy_only(make_c_chain_cell(2.8, 12.0, 12.0));
    cpu::Frame isol = make_c_chain_cell(2.8, 20.0, 20.0);
    isol.box = cpu::Box::cubic(20.0);
    const double e_mol = energy_only(std::move(isol));
    EXPECT_TRUE(std::isfinite(e_pbc));
    EXPECT_TRUE(std::isfinite(e_mol));
    EXPECT_GT(std::fabs(e_pbc - e_mol), 0.05);
}

/// Off-eq 1-D C dimer cell: r≠Lx/2 so left/right lattice images differ → |F|≠0.
cpu::Frame make_c_chain_offeq(double r_cc = 1.20, double Lx = 2.8,
                              double Ly = 12.0, double Lz = 12.0) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 6;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = static_cast<StorageReal>(r_cc);
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 12.011;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::orthorhombic(Lx, Ly, Lz);
    f.meta.get(kEnergy) = 0;
    return f;
}

TEST(DFTB2, PeriodicAnalyticForceMatchesFdCChain) {
    // Lattice-summed energy vs lattice-summed analytic F (not min-image only).
    auto f = make_c_chain_offeq();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = true;
    dftb.scc_tol = 1e-12;
    dftb.max_scc = 80;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(dftb.last_energy()));

    const double fx0 = static_cast<double>(f.span(kAtom_fx)[0]);
    const double fx1 = static_cast<double>(f.span(kAtom_fx)[1]);
    const double fy0 = static_cast<double>(f.span(kAtom_fy)[0]);
    const double fz0 = static_cast<double>(f.span(kAtom_fz)[0]);
    EXPECT_TRUE(std::isfinite(fx0));
    EXPECT_GT(fx0 * fx0 + fy0 * fy0 + fz0 * fz0, 1e-8);
    // Action-reaction / zero net force (no external field).
    EXPECT_NEAR(fx0 + fx1, 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_fy)[0] + f.span(kAtom_fy)[1]),
                0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_fz)[0] + f.span(kAtom_fz)[1]),
                0.0, 1e-6);

    constexpr double h = 1.0e-4;
    constexpr double kFdTol = 1e-5; // eV/Å; stationary SCC ⇒ FD-exact
    {
        auto gp = make_c_chain_offeq();
        auto gm = make_c_chain_offeq();
        gp.span(kAtom_x)[0] += static_cast<StorageReal>(h);
        gm.span(kAtom_x)[0] -= static_cast<StorageReal>(h);
        const double fd = -(energy_only(gp) - energy_only(gm)) / (2.0 * h);
        EXPECT_NEAR(fx0, fd, kFdTol) << "C0 Fx analytic=" << fx0
                                     << " fd=" << fd;
    }
    {
        auto gp = make_c_chain_offeq();
        auto gm = make_c_chain_offeq();
        gp.span(kAtom_x)[1] += static_cast<StorageReal>(h);
        gm.span(kAtom_x)[1] -= static_cast<StorageReal>(h);
        const double fd = -(energy_only(gp) - energy_only(gm)) / (2.0 * h);
        EXPECT_NEAR(fx1, fd, kFdTol) << "C1 Fx analytic=" << fx1
                                     << " fd=" << fd;
    }
}

TEST(DFTB2, PeriodicAnalyticForceMatchesFdWaterTightBox) {
    // Polar system under PBC: Δq≠0 so lattice γ force terms are live.
    // L=4.0 Å puts O/H images inside C/O r_cut (4.5 Å) while still finite.
    auto make_water_pbc = []() {
        auto f = make_water_offeq();
        f.box = cpu::Box::orthorhombic(4.0, 4.0, 4.0);
        return f;
    };
    auto f = make_water_pbc();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = true;
    dftb.scc_tol = 1e-12;
    dftb.max_scc = 80;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(dftb.last_energy()));

    constexpr double h = 1.0e-4;
    constexpr double kFdTol = 1e-5;
    {
        auto gp = make_water_pbc();
        auto gm = make_water_pbc();
        gp.span(kAtom_x)[0] += static_cast<StorageReal>(h);
        gm.span(kAtom_x)[0] -= static_cast<StorageReal>(h);
        const double fd = -(energy_only(gp) - energy_only(gm)) / (2.0 * h);
        EXPECT_NEAR(static_cast<double>(f.span(kAtom_fx)[0]), fd, kFdTol);
    }
    {
        auto gp = make_water_pbc();
        auto gm = make_water_pbc();
        gp.span(kAtom_y)[1] += static_cast<StorageReal>(h);
        gm.span(kAtom_y)[1] -= static_cast<StorageReal>(h);
        const double fd = -(energy_only(gp) - energy_only(gm)) / (2.0 * h);
        EXPECT_NEAR(static_cast<double>(f.span(kAtom_fy)[1]), fd, kFdTol);
    }

    double net[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        net[0] += static_cast<double>(f.span(kAtom_fx)[i]);
        net[1] += static_cast<double>(f.span(kAtom_fy)[i]);
        net[2] += static_cast<double>(f.span(kAtom_fz)[i]);
    }
    EXPECT_NEAR(net[0], 0.0, 1e-6);
    EXPECT_NEAR(net[1], 0.0, 1e-6);
    EXPECT_NEAR(net[2], 0.0, 1e-6);
}

TEST(DFTB2, MolecularLimitLargeBoxStable) {
    // Two large-box H2 runs should agree (lattice sum only sees T=0).
    const double e1 = energy_only(make_h2(0.74));
    const double e2 = energy_only(make_h2(0.74));
    EXPECT_TRUE(std::isfinite(e1));
    EXPECT_NEAR(e1, e2, 1e-10);
}

/// Affine strain helper for multi-SCC stress FD (matches product strain FD).
void apply_strain_voigt(cpu::Frame& frame, int voigt, double h) {
    Mat3 F = Mat3::identity();
    if (voigt == 0)
        F(0, 0) += static_cast<AccReal>(h);
    else if (voigt == 1)
        F(1, 1) += static_cast<AccReal>(h);
    else if (voigt == 2)
        F(2, 2) += static_cast<AccReal>(h);
    else if (voigt == 3) {
        F(0, 1) += static_cast<AccReal>(h);
        F(1, 0) += static_cast<AccReal>(h);
    } else if (voigt == 4) {
        F(0, 2) += static_cast<AccReal>(h);
        F(2, 0) += static_cast<AccReal>(h);
    } else if (voigt == 5) {
        F(1, 2) += static_cast<AccReal>(h);
        F(2, 1) += static_cast<AccReal>(h);
    }
    Mat3 cell = frame.box.cell();
    Mat3 cell_new;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            AccReal s = 0;
            for (int k = 0; k < 3; ++k)
                s += F(i, k) * cell(k, j);
            cell_new(i, j) = s;
        }
    frame.box.set_cell(cell_new);
    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    const int n = frame.nlocal();
    for (int a = 0; a < n; ++a) {
        const AccReal rx = static_cast<AccReal>(x[a]);
        const AccReal ry = static_cast<AccReal>(y[a]);
        const AccReal rz = static_cast<AccReal>(z[a]);
        x[a] = static_cast<StorageReal>(F(0, 0) * rx + F(0, 1) * ry
                                        + F(0, 2) * rz);
        y[a] = static_cast<StorageReal>(F(1, 0) * rx + F(1, 1) * ry
                                        + F(1, 2) * rz);
        z[a] = static_cast<StorageReal>(F(2, 0) * rx + F(2, 1) * ry
                                        + F(2, 2) * rz);
    }
}

TEST(DFTB2, AnalyticStressMatchesStrainFdH2) {
    // Stretched H2 so |virial| is finite; large box → near-isolated molecule.
    auto f = make_h2(0.90);
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = false;
    dftb.with_stress = true;
    dftb.scc_tol = 1e-12;
    dftb.max_scc = 80;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});

    const auto& san = dftb.last_stress();
    for (int v = 0; v < 6; ++v)
        EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(v)])) << v;

    // Bond along x → σ_xx is the leading component; gate xx and yy.
    constexpr double h = 2.0e-4;
    const double vol = f.box.volume();
    ASSERT_GT(vol, 0.0);

    auto strain_fd = [&](int voigt) {
        auto fp = make_h2(0.90);
        auto fm = make_h2(0.90);
        apply_strain_voigt(fp, voigt, h);
        apply_strain_voigt(fm, voigt, -h);
        const double dE = (energy_only(fp) - energy_only(fm)) / (2.0 * h);
        return -dE / vol;
    };

    // Fixed-Δq analytic vs re-SCC strain-FD: H2 is tight (no multi-element
    // charge-response residual of the H2O force gate class).
    EXPECT_NEAR(san[0], strain_fd(0), 5e-4); // xx
    EXPECT_NEAR(san[1], strain_fd(1), 5e-4); // yy (~0 for axial H2)

    // Frame meta accumulation. last_stress() is the intensive σ [eV/Å³];
    // kStress carries the extensive Voigt virial [eV] (core/frame_schema.h),
    // so the published value is σ·V.
    const auto& st = f.meta.get(kStress);
    EXPECT_NEAR(st[0], san[0] * f.box.volume(), 1e-14 * f.box.volume());
}

namespace {

/// Tilted diatomic O–X so all six Voigt components are non-trivial.
/// Bond direction (0.6, 0.7, √0.15) is a unit vector, so |R| = `r`.
cpu::Frame make_tilted_diatomic(int z1, double r, double box_l = 20.0) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_type)[1] = z1;
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = static_cast<StorageReal>(r * 0.6);
    f.span(kAtom_y)[1] = static_cast<StorageReal>(r * 0.7);
    f.span(kAtom_z)[1] = static_cast<StorageReal>(r * 0.3872983346207417);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 15.999;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(box_l);
    f.meta.get(kEnergy) = 0;
    for (int v = 0; v < 6; ++v)
        f.meta.get(kStress)[static_cast<std::size_t>(v)] = 0;
    return f;
}

double energy_only_mult(cpu::Frame g, int multiplicity) {
    g.meta.get(kEnergy) = 0;
    for (int i = 0; i < g.nlocal(); ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 d;
    d.multiplicity = multiplicity;
    d.with_forces = false;
    d.with_stress = false;
    d.scc_tol = 1e-12;
    d.max_scc = 200;
    d.build({ctx, g});
    d.launch({ctx, g, 0, 0});
    return d.last_energy();
}

/// Analytic σ (all six Voigt) for one tilted diatomic at one multiplicity.
std::array<double, 6> analytic_stress(int z1, double r, int multiplicity,
                                      cpu::Frame& out_frame) {
    out_frame = make_tilted_diatomic(z1, r);
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 d;
    d.multiplicity = multiplicity;
    d.with_forces = true;
    d.with_stress = true;
    d.scc_tol = 1e-12;
    d.max_scc = 200;
    d.build({ctx, out_frame});
    d.launch({ctx, out_frame, 0, 0});
    return d.last_stress();
}

/// Central affine-strain FD of the converged energy, σ = −(1/Ω)∂E/∂ε.
double strain_fd_stress(int z1, double r, int multiplicity, int voigt,
                        double h = 2.0e-4) {
    auto fp = make_tilted_diatomic(z1, r);
    auto fm = make_tilted_diatomic(z1, r);
    const double vol = fp.box.volume();
    apply_strain_voigt(fp, voigt, h);
    apply_strain_voigt(fm, voigt, -h);
    const double dE = (energy_only_mult(std::move(fp), multiplicity)
                       - energy_only_mult(std::move(fm), multiplicity))
                      / (2.0 * h);
    return -dE / vol;
}

} // namespace

TEST(DFTB2, SpinAnalyticStressMatchesStrainFdO2Triplet) {
    // Collinear open shell: closed-form spin-resolved pair virial vs the
    // multi-SCC affine strain-FD it replaced. Stationarity of the converged
    // SCC functional makes this FD-exact to the truncation error.
    cpu::Frame f;
    const auto san = analytic_stress(/*z1=*/8, /*r=*/1.30, /*mult=*/3, f);

    constexpr double kTol = 1e-8; // eV/Å³
    for (int v = 0; v < 6; ++v) {
        ASSERT_TRUE(std::isfinite(san[static_cast<std::size_t>(v)])) << v;
        const double fd = strain_fd_stress(8, 1.30, 3, v);
        EXPECT_NEAR(san[static_cast<std::size_t>(v)], fd, kTol)
            << "voigt=" << v;
    }

    // Frame meta accumulation is the same buffer, in the extensive kStress
    // convention: σ·V [eV] against last_stress()'s intensive σ [eV/Å³].
    const auto& st = f.meta.get(kStress);
    const double vol_a3 = f.box.volume();
    for (int v = 0; v < 6; ++v)
        EXPECT_NEAR(st[static_cast<std::size_t>(v)],
                    san[static_cast<std::size_t>(v)] * vol_a3,
                    1e-14 * vol_a3);

    // Translational invariance of the open-shell gradient is unchanged.
    double net[3] = {0, 0, 0};
    for (int i = 0; i < 2; ++i) {
        net[0] += static_cast<double>(f.span(kAtom_fx)[i]);
        net[1] += static_cast<double>(f.span(kAtom_fy)[i]);
        net[2] += static_cast<double>(f.span(kAtom_fz)[i]);
    }
    for (int d = 0; d < 3; ++d)
        EXPECT_NEAR(net[d], 0.0, 1e-6) << "cartesian=" << d;
}

TEST(DFTB2, SpinAnalyticStressPolarOpenShellMatchesStrainFdNo) {
    // NO doublet is the DISCRIMINATING open-shell case: homonuclear O₂ has
    // Δq ≡ 0 (γ virial multiplied by zero) *and* a spin shift ±W_A m_A that is
    // uniform over atoms. NO has Δq ≠ 0 and W_N m_N ≠ W_O m_O, so both the SCC
    // γ virial and the spin half of the Pulay ∂S term are exercised.
    cpu::Frame f;
    const auto san = analytic_stress(/*z1=*/7, /*r=*/1.20, /*mult=*/2, f);

    constexpr double kTol = 1e-8; // eV/Å³
    for (int v = 0; v < 6; ++v) {
        ASSERT_TRUE(std::isfinite(san[static_cast<std::size_t>(v)])) << v;
        const double fd = strain_fd_stress(7, 1.20, 2, v);
        EXPECT_NEAR(san[static_cast<std::size_t>(v)], fd, kTol)
            << "voigt=" << v;
    }
}

TEST(DFTB2, SpinAnalyticStressClosedShellLimitMatchesRestricted) {
    // Equivalence guard for folding the restricted assembly into the
    // spin-resolved one (C^α = C^β, ε^α = ε^β, m ≡ 0 ⇒ P^α = P^β = P/2).
    // Compare live: multiplicity=1 spin path vs default restricted path on
    // the same geometry (not a fossil sketch-era number).
    cpu::Frame f_spin, f_rest;
    const auto san = analytic_stress(/*z1=*/8, /*r=*/1.30, /*mult=*/1, f_spin);
    f_rest = make_tilted_diatomic(8, 1.30);
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 d;
    d.with_forces = true;
    d.with_stress = true;
    d.scc_tol = 1e-12;
    d.max_scc = 200;
    d.build({ctx, f_rest});
    d.launch({ctx, f_rest, 0, 0});
    const auto& rest = d.last_stress();
    for (int v = 0; v < 6; ++v)
        EXPECT_NEAR(san[static_cast<std::size_t>(v)],
                    rest[static_cast<std::size_t>(v)], 1e-10)
            << "voigt=" << v;
}

namespace {

double energy_kmesh(cpu::Frame g, std::array<int, 3> mesh,
                    bool time_reversal = true) {
    g.meta.get(kEnergy) = 0;
    for (int i = 0; i < g.nlocal(); ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 d;
    d.with_forces = false;
    d.scc_tol = 1e-12;
    d.max_scc = 80;
    d.k_mesh = mesh;
    d.k_time_reversal = time_reversal;
    d.build({ctx, g});
    d.launch({ctx, g, 0, 0});
    return d.last_energy();
}

} // namespace

TEST(DFTB2, KMesh111MatchesGamma) {
    // Default multi-k entry with {1,1,1} must stay on the real Γ path.
    auto f = make_c_chain_cell();
    const double e_gamma = energy_only(make_c_chain_cell());
    const double e_mesh = energy_kmesh(std::move(f), {{1, 1, 1}});
    EXPECT_TRUE(std::isfinite(e_mesh));
    EXPECT_NEAR(e_mesh, e_gamma, 1e-8);
}

TEST(DFTB2, KMesh41x1DiffersFromGamma) {
    const double e_gamma = energy_only(make_c_chain_cell());
    const double e_k = energy_kmesh(make_c_chain_cell(), {{4, 1, 1}});
    EXPECT_TRUE(std::isfinite(e_k));
    EXPECT_GT(std::fabs(e_k - e_gamma), 0.01);
}

namespace {

/// Analytic multi-k forces [eV/Å] as a flat (3·N) vector, Frame in / Frame out.
std::vector<double> forces_kmesh(cpu::Frame g, std::array<int, 3> mesh,
                                 bool time_reversal = true) {
    g.meta.get(kEnergy) = 0;
    for (int i = 0; i < g.nlocal(); ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 d;
    d.with_forces = true;
    d.scc_tol = 1e-12;
    d.max_scc = 80;
    d.k_mesh = mesh;
    d.k_time_reversal = time_reversal;
    d.build({ctx, g});
    d.launch({ctx, g, 0, 0});
    std::vector<double> out;
    for (int i = 0; i < g.nlocal(); ++i) {
        out.push_back(static_cast<double>(g.span(kAtom_fx)[i]));
        out.push_back(static_cast<double>(g.span(kAtom_fy)[i]));
        out.push_back(static_cast<double>(g.span(kAtom_fz)[i]));
    }
    return out;
}

/// −dE/dx of the multi-k energy at atom `atom`, central difference (h in Å).
double fd_force_x_kmesh(std::array<int, 3> mesh, int atom, double h = 1.0e-4) {
    auto gp = make_c_chain_offeq();
    auto gm = make_c_chain_offeq();
    gp.span(kAtom_x)[atom] += static_cast<StorageReal>(h);
    gm.span(kAtom_x)[atom] -= static_cast<StorageReal>(h);
    return -(energy_kmesh(std::move(gp), mesh)
             - energy_kmesh(std::move(gm), mesh))
           / (2.0 * h);
}

} // namespace

TEST(DFTB2, KMeshForcesMatchFdMesh211) {
    const std::array<int, 3> mesh{{2, 1, 1}};
    const auto f = forces_kmesh(make_c_chain_offeq(), mesh);
    // mio-1-1 radial tables + longer cutoffs: FD noise ~2e-6 eV/Å.
    constexpr double kTol = 5e-6;
    EXPECT_NEAR(f[0], fd_force_x_kmesh(mesh, 0), kTol);
    EXPECT_NEAR(f[3], fd_force_x_kmesh(mesh, 1), kTol);
}

TEST(DFTB2, KMeshForcesMatchFdMesh221) {
    const std::array<int, 3> mesh{{2, 2, 1}};
    const auto f = forces_kmesh(make_c_chain_offeq(), mesh);
    constexpr double kTol = 5e-6;
    EXPECT_NEAR(f[0], fd_force_x_kmesh(mesh, 0), kTol);
    EXPECT_NEAR(f[3], fd_force_x_kmesh(mesh, 1), kTol);
}

TEST(DFTB2, KMeshForcesSumToZero) {
    const auto f = forces_kmesh(make_c_chain_offeq(), {{4, 1, 1}});
    for (int k = 0; k < 3; ++k)
        EXPECT_NEAR(f[k] + f[3 + k], 0.0, 1e-10);
}

TEST(DFTB2, KMeshForcesAreNonTrivialAtOffEqGeometry) {
    const auto f = forces_kmesh(make_c_chain_offeq(), {{4, 1, 1}});
    EXPECT_GT(std::fabs(f[0]), 1e-3);
}

TEST(DFTB2, KMesh111ForcesMatchGammaForces) {
    // {1,1,1} must keep routing to the real Γ analytic gradient, bit for bit.
    const auto f_mesh = forces_kmesh(make_c_chain_offeq(), {{1, 1, 1}});
    cpu::Frame g = make_c_chain_offeq();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 d;
    d.with_forces = true;
    d.scc_tol = 1e-12;
    d.max_scc = 80;
    d.build({ctx, g});
    d.launch({ctx, g, 0, 0});
    for (int i = 0; i < 2; ++i)
        EXPECT_NEAR(f_mesh[3 * i],
                    static_cast<double>(g.span(kAtom_fx)[i]), 1e-10);
}

TEST(DFTB2, KMeshForcesAreTimeReversalInvariant) {
    // Folding ±k is exact (P(−k) = conj P(k)); the unfolded run additionally
    // arms the imaginary-residue gate inside the multi-k gradient. The 3×1×1
    // mesh contains Γ, which is self-paired and gated in both runs.
    const auto folded = forces_kmesh(make_c_chain_offeq(), {{3, 1, 1}}, true);
    const auto full = forces_kmesh(make_c_chain_offeq(), {{3, 1, 1}}, false);
    for (std::size_t i = 0; i < folded.size(); ++i)
        EXPECT_NEAR(folded[i], full[i], 1e-12);
}

TEST(DFTB2, PeriodicAnalyticStressMatchesStrainFdCChain) {
    // Γ-PBC cell whose OWN images sit inside r_cut (Lx = 2.8 Å < 4.5 Å): the
    // self-image A–(A+T) bonds carry no nuclear force but do scale with the
    // cell, so the pair virial must include them.
    auto f = make_c_chain_offeq();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = false;
    dftb.with_stress = true;
    dftb.scc_tol = 1e-12;
    dftb.max_scc = 80;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});

    const auto& san = dftb.last_stress();
    constexpr double h = 2.0e-4;
    const double vol = f.box.volume();
    ASSERT_GT(vol, 0.0);
    auto strain_fd = [&](int voigt) {
        auto fp = make_c_chain_offeq();
        auto fm = make_c_chain_offeq();
        apply_strain_voigt(fp, voigt, h);
        apply_strain_voigt(fm, voigt, -h);
        return -(energy_only(fp) - energy_only(fm)) / (2.0 * h) / vol;
    };
    EXPECT_NEAR(san[0], strain_fd(0), 1e-6); // xx — chain axis
    EXPECT_NEAR(san[1], strain_fd(1), 1e-6); // yy — vacuum axis
}

TEST(DFTB2, KMeshStressMatchesStrainFd) {
    // The MP mesh lives in fractional coordinates, so an affine strain touches
    // only the image bonds — the multi-k virial is the same pair sum.
    const std::array<int, 3> mesh{{2, 1, 1}};
    auto f = make_c_chain_offeq();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 d;
    d.with_forces = false;
    d.with_stress = true;
    d.scc_tol = 1e-12;
    d.max_scc = 80;
    d.k_mesh = mesh;
    d.build({ctx, f});
    d.launch({ctx, f, 0, 0});

    const auto& san = d.last_stress();
    constexpr double h = 2.0e-4;
    const double vol = f.box.volume();
    ASSERT_GT(vol, 0.0);
    auto strain_fd = [&](int voigt) {
        auto fp = make_c_chain_offeq();
        auto fm = make_c_chain_offeq();
        apply_strain_voigt(fp, voigt, h);
        apply_strain_voigt(fm, voigt, -h);
        return -(energy_kmesh(std::move(fp), mesh)
                 - energy_kmesh(std::move(fm), mesh))
               / (2.0 * h) / vol;
    };
    EXPECT_NEAR(san[0], strain_fd(0), 1e-6); // xx — chain axis
    EXPECT_NEAR(san[1], strain_fd(1), 1e-6); // yy — vacuum axis
}

TEST(DFTB2, KMeshOpenShellEnergyFinite) {
    // Collinear open shell + multi-k (dual α/β H(k)).
    auto f = make_c_chain_cell();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 d;
    d.k_mesh = {{4, 1, 1}};
    d.with_forces = false;
    d.multiplicity = 3;
    d.scc_tol = 1e-8;
    d.build({ctx, f});
    d.launch({ctx, f, 0, 0});
    const double e = f.meta.get(kEnergy);
    EXPECT_TRUE(std::isfinite(e));
    EXPECT_LT(e, 0.0);
    // Total Mulliken spin ≈ n_alpha − n_beta = 2 for multiplicity 3.
    double m = 0;
    for (double mi : d.last_mulliken_spin())
        m += mi;
    EXPECT_NEAR(m, 2.0, 0.5);
}

TEST(DFTB2, KMeshOpenShellForceMatchesFd) {
    auto f = make_c_chain_cell();
    // Off-eq: stretch the C–C bond slightly so |F| ≠ 0.
    f.span(kAtom_x)[1] = static_cast<StorageReal>(1.45);
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 d;
    d.k_mesh = {{2, 1, 1}};
    d.with_forces = true;
    d.multiplicity = 3;
    d.scc_tol = 1e-10;
    d.build({ctx, f});
    d.launch({ctx, f, 0, 0});
    const double fx0 = static_cast<double>(f.span(kAtom_fx)[0]);
    auto energy_at = [&](double h) {
        auto g = make_c_chain_cell();
        g.span(kAtom_x)[1] = static_cast<StorageReal>(1.45);
        g.span(kAtom_x)[0] = static_cast<StorageReal>(h);
        g.meta.get(kEnergy) = 0;
        for (int i = 0; i < 2; ++i)
            g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
        cpu::semiempirical::DFTB2 dd;
        dd.k_mesh = {{2, 1, 1}};
        dd.with_forces = false;
        dd.multiplicity = 3;
        dd.scc_tol = 1e-10;
        cpu::Context c2;
        dd.build({c2, g});
        dd.launch({c2, g, 0, 0});
        return g.meta.get(kEnergy);
    };
    constexpr double h = 1e-4;
    const double fd = -(energy_at(+h) - energy_at(-h)) / (2.0 * h);
    EXPECT_NEAR(fx0, fd, 5e-4) << "analytic=" << fx0 << " fd=" << fd;
}

TEST(DFTB2, KMeshRejectsBadMesh) {
    auto f = make_c_chain_cell();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 d;
    d.k_mesh = {{0, 1, 1}};
    d.with_forces = false;
    d.build({ctx, f});
    EXPECT_THROW(d.launch({ctx, f, 0, 0}), std::invalid_argument);
}

TEST(DFTB2, MioAnalyticStressFossilGuard) {
    // The restricted-MioSketch closed-form pair virial must not drift when the
    // ThreeOb virial is added beside it (`dftb2-analytic-stress-threeob`).
    // Re-pinned with consistent γ_AB (units::coulomb); see
    // .claude/notes/dftb-gamma-units-defect.md. Leading digits also track
    // regressions/semiempirical/dftb2-analytic-stress after the same re-pin.
    auto f = make_water_offeq();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = false;
    dftb.with_stress = true;
    dftb.scc_tol = 1e-10;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    const auto& s = dftb.last_stress();
    // Re-pinned after DFTB+ Elstner γ (was Ohno–Klopman; 2026-07-29).
    const double pin[6] = {
        -4.157623027391737e-04, -3.0093632837246975e-04, 0.0,
        1.4547104687076456e-04, 0.0, 0.0,
    };
    // (xx,yy,zz,xy,xz,yz); xz/yz ~0 for planar water in xy.
    for (int v = 0; v < 6; ++v)
        EXPECT_NEAR(s[static_cast<std::size_t>(v)],
                    pin[static_cast<std::size_t>(v)], 1e-12)
            << "voigt " << v;
}
