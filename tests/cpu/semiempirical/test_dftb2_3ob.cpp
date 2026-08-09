/// @file tests/cpu/semiempirical/test_dftb2_3ob.cpp
/// Unit tests for DFTB2 ThreeOb static SK tables (H/C/N/O).
#include <array>
#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_3ob_tables.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;
using namespace ATV::cpu::semiempirical;

namespace {

cpu::Frame make_ch4() {
    cpu::Frame f(5);
    f.span(kAtom_type)[0] = 6;
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    const double s = 1.09 / std::sqrt(3.0);
    const double hpos[4][3] = {
        {s, s, s}, {s, -s, -s}, {-s, s, -s}, {-s, -s, s},
    };
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_type)[i + 1] = 1;
        f.span(kAtom_x)[i + 1] = hpos[i][0];
        f.span(kAtom_y)[i + 1] = hpos[i][1];
        f.span(kAtom_z)[i + 1] = hpos[i][2];
    }
    for (int i = 0; i < 5; ++i) {
        f.span(kAtom_mass)[i] = (i == 0) ? 12.0 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

} // namespace

TEST(DFTB23ob, TablesCoverHCNO) {
    using namespace basis_data;
    EXPECT_EQ(kDftb3obNElements, 4u);
    int mask = 0;
    for (std::size_t i = 0; i < kDftb3obNElements; ++i)
        mask |= (1 << kDftb3obElements[i].z);
    EXPECT_TRUE(mask & (1 << 1));
    EXPECT_TRUE(mask & (1 << 6));
    EXPECT_TRUE(mask & (1 << 7));
    EXPECT_TRUE(mask & (1 << 8));
    EXPECT_GE(kDftb3obNPairs, 16u);
}

TEST(DFTB23ob, CH4EnergyFinite) {
    auto f = make_ch4();
    cpu::Context ctx;
    DFTB2 dftb;
    dftb.sk_set = DftbSkSet::ThreeOb;
    dftb.with_forces = false;
    dftb.scc_tol = 1e-8;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(dftb.last_energy()));
    EXPECT_NEAR(f.meta.get(kEnergy), dftb.last_energy(), 1e-12);
    // Bound organic: total energy well below zero (eV).
    EXPECT_LT(dftb.last_energy(), -50.0);
}

TEST(DFTB23ob, UnsupportedElementThrows) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 26;
    f.box = cpu::Box::cubic(10.0);
    f.meta.get(kEnergy) = 0;
    cpu::Context ctx;
    DFTB2 dftb;
    dftb.sk_set = DftbSkSet::ThreeOb;
    dftb.build({ctx, f});
    EXPECT_THROW(dftb.launch({ctx, f, 0, 0}), std::runtime_error);
}

TEST(DFTB23ob, DistinctFromMioSketch) {
    auto f1 = make_ch4();
    auto f2 = make_ch4();
    cpu::Context ctx;
    DFTB2 a;
    a.sk_set = DftbSkSet::ThreeOb;
    a.with_forces = false;
    a.build({ctx, f1});
    a.launch({ctx, f1, 0, 0});
    DFTB2 b;
    b.sk_set = DftbSkSet::MioSketch;
    b.with_forces = false;
    b.build({ctx, f2});
    b.launch({ctx, f2, 0, 0});
    // Both paths use real tables now; residual is SK-set difference only.
    EXPECT_GT(std::fabs(a.last_energy() - b.last_energy()), 0.05);
}

namespace {

cpu::Frame make_water_offeq() {
    // Off-eq water (Å) so |F| ≠ 0 — same geometry as dftb2-analytic-forces.
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

double energy_3ob_only(cpu::Frame g) {
    g.meta.get(kEnergy) = 0;
    for (int i = 0; i < g.nlocal(); ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Context ctx;
    DFTB2 d;
    d.sk_set = DftbSkSet::ThreeOb;
    d.with_forces = false;
    d.scc_tol = 1e-10;
    d.max_scc = 80;
    d.build({ctx, g});
    d.launch({ctx, g, 0, 0});
    return g.meta.get(kEnergy);
}

} // namespace

TEST(DFTB23ob, AnalyticForcesMatchFdWater) {
    auto f = make_water_offeq();
    cpu::Context ctx;
    DFTB2 dftb;
    dftb.sk_set = DftbSkSet::ThreeOb;
    dftb.with_forces = true;
    dftb.scc_tol = 1e-10;
    dftb.max_scc = 80;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});

    EXPECT_TRUE(std::isfinite(dftb.last_energy()));
    double f2 = 0;
    double net[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        const double fxi = static_cast<double>(f.span(kAtom_fx)[i]);
        const double fyi = static_cast<double>(f.span(kAtom_fy)[i]);
        const double fzi = static_cast<double>(f.span(kAtom_fz)[i]);
        EXPECT_TRUE(std::isfinite(fxi));
        EXPECT_TRUE(std::isfinite(fyi));
        EXPECT_TRUE(std::isfinite(fzi));
        f2 += fxi * fxi + fyi * fyi + fzi * fzi;
        net[0] += fxi;
        net[1] += fyi;
        net[2] += fzi;
    }
    EXPECT_GT(f2, 1e-8);
    for (int d = 0; d < 3; ++d)
        EXPECT_NEAR(net[d], 0.0, 1e-6);

    constexpr double h = 1e-4;
    // A converged SCC-DFTB2 gradient is stationary, so fixed-Δq analytic is
    // FD-exact to the central-difference truncation error. The old 1.0 eV/Å
    // "charge response" gate was masking the γ-gradient sign bug (measured
    // residual 8.4e-1 → 4.4e-7 eV/Å when fixed, 2026-07-28).
    constexpr double tol = 1e-5;

    {
        auto fp = make_water_offeq();
        fp.span(kAtom_x)[0] += static_cast<StorageReal>(h);
        auto fm = make_water_offeq();
        fm.span(kAtom_x)[0] -= static_cast<StorageReal>(h);
        const double fd = -(energy_3ob_only(fp) - energy_3ob_only(fm)) / (2.0 * h);
        const double fx0 = static_cast<double>(f.span(kAtom_fx)[0]);
        EXPECT_NEAR(fx0, fd, tol) << "F_O_x analytic=" << fx0 << " fd=" << fd;
    }
    {
        auto fp = make_water_offeq();
        fp.span(kAtom_y)[1] += static_cast<StorageReal>(h);
        auto fm = make_water_offeq();
        fm.span(kAtom_y)[1] -= static_cast<StorageReal>(h);
        const double fd = -(energy_3ob_only(fp) - energy_3ob_only(fm)) / (2.0 * h);
        const double fy1 = static_cast<double>(f.span(kAtom_fy)[1]);
        EXPECT_NEAR(fy1, fd, tol) << "F_H1_y analytic=" << fy1 << " fd=" << fd;
    }
}

namespace {

/// Affine strain of cell + nuclei, Voigt 0..5 (off-diagonals symmetric:
/// ε_αβ = ε_βα = h) — the deformation the mechanical stress differentiates.
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
    for (int a = 0; a < frame.nlocal(); ++a) {
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

/// Asymmetric CH4: four distinct C–H bond lengths break the tetrahedral
/// cancellation that makes every shear component of symmetric CH4 identically
/// zero, so all six Voigt components are exercised. Radii are placed mid-way
/// between 3ob radial-grid knots (grid 0.02 Bohr) — the SK tables are linearly
/// interpolated, so a finite-difference interval that straddles a knot compares
/// two different interpolant slopes and is not a valid oracle.
cpu::Frame make_ch4_asymmetric() {
    cpu::Frame f(5);
    const double inv_sqrt3 = 1.0 / std::sqrt(3.0);
    const double dir[4][3] = {
        {inv_sqrt3, inv_sqrt3, inv_sqrt3},
        {inv_sqrt3, -inv_sqrt3, -inv_sqrt3},
        {-inv_sqrt3, inv_sqrt3, -inv_sqrt3},
        {-inv_sqrt3, -inv_sqrt3, inv_sqrt3},
    };
    const double r_ch[4] = {1.09540, 1.12714, 1.06364, 1.10598}; // Å
    f.span(kAtom_type)[0] = 6;
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_type)[i + 1] = 1;
        f.span(kAtom_x)[i + 1] = static_cast<StorageReal>(r_ch[i] * dir[i][0]);
        f.span(kAtom_y)[i + 1] = static_cast<StorageReal>(r_ch[i] * dir[i][1]);
        f.span(kAtom_z)[i + 1] = static_cast<StorageReal>(r_ch[i] * dir[i][2]);
    }
    for (int i = 0; i < 5; ++i) {
        f.span(kAtom_mass)[i] = (i == 0) ? 12.0 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    for (int v = 0; v < 6; ++v)
        f.meta.get(kStress)[static_cast<std::size_t>(v)] = 0;
    return f;
}

/// Converged ThreeOb stress of `f` (Voigt eV/Å³), forces off.
std::array<double, 6> stress_3ob(cpu::Frame& f) {
    cpu::Context ctx;
    DFTB2 d;
    d.sk_set = DftbSkSet::ThreeOb;
    d.with_forces = false;
    d.with_stress = true;
    d.scc_tol = 1e-10;
    d.max_scc = 80;
    d.build({ctx, f});
    d.launch({ctx, f, 0, 0});
    return d.last_stress();
}

/// σ_v from a central affine-strain difference of the converged 3ob energy.
/// h = 1e-5 keeps the strained C–H / H–H distances inside one SK grid interval.
double strain_fd_stress(const cpu::Frame& ref, int voigt, double vol) {
    constexpr double h = 1e-5;
    cpu::Frame fp = ref;
    cpu::Frame fm = ref;
    apply_strain_voigt(fp, voigt, h);
    apply_strain_voigt(fm, voigt, -h);
    const double dE = (energy_3ob_only(fp) - energy_3ob_only(fm)) / (2.0 * h);
    return -dE / vol;
}

} // namespace

TEST(DFTB23ob, AnalyticStressMatchesStrainFdCh4) {
    auto ref = make_ch4_asymmetric();
    auto f = make_ch4_asymmetric();
    const double vol = f.box.volume();
    ASSERT_GT(vol, 0.0);
    const std::array<double, 6> san = stress_3ob(f);

    // Frame meta carries the same tensor the method reports, in the extensive
    // kStress convention: σ·V [eV] against the method's intensive σ [eV/Å³].
    for (int v = 0; v < 6; ++v)
        EXPECT_NEAR(f.meta.get(kStress)[static_cast<std::size_t>(v)],
                    san[static_cast<std::size_t>(v)] * vol, 1e-15 * vol)
            << "voigt " << v;

    double norm2 = 0;
    for (int v = 0; v < 6; ++v) {
        ASSERT_TRUE(std::isfinite(san[static_cast<std::size_t>(v)])) << v;
        norm2 += san[static_cast<std::size_t>(v)]
                 * san[static_cast<std::size_t>(v)];
    }
    // Asymmetric CH4: no component (shears included) is trivially zero.
    EXPECT_GT(norm2, 1e-14);

    for (int v = 0; v < 6; ++v) {
        const double fd = strain_fd_stress(ref, v, vol);
        EXPECT_NEAR(san[static_cast<std::size_t>(v)], fd, 1e-8)
            << "voigt " << v << " analytic=" << san[static_cast<std::size_t>(v)]
            << " fd=" << fd;
    }
}

TEST(DFTB23ob, AnalyticStressMatchesStrainFdWater) {
    // Polar Δq ≠ 0 case — the class of geometry that exposed the γ-gradient
    // sign defect the ThreeOb force gate now pins.
    auto ref = make_water_offeq();
    auto f = make_water_offeq();
    const double vol = f.box.volume();
    const std::array<double, 6> san = stress_3ob(f);

    for (int v = 0; v < 6; ++v) {
        const double fd = strain_fd_stress(ref, v, vol);
        EXPECT_NEAR(san[static_cast<std::size_t>(v)], fd, 1e-8)
            << "voigt " << v << " analytic=" << san[static_cast<std::size_t>(v)]
            << " fd=" << fd;
    }
}

TEST(DFTB23ob, AnalyticStressIsotropicScalingIdentity) {
    // Isotropic scaling r → (1+ε)r gives Ω(ε) = (1+ε)³Ω₀, so dΩ/dε|₀ = 3Ω₀ and
    // dE/dε = Σ_α ∂E/∂ε_αα = −Ω₀ tr(σ). Hence the pressure identity
    //     tr(σ)/3 = −dE/dΩ = −(1/3Ω₀) dE/dε|₀ .
    auto f = make_ch4_asymmetric();
    const double vol = f.box.volume();
    const std::array<double, 6> san = stress_3ob(f);
    const double trace_third = (san[0] + san[1] + san[2]) / 3.0;

    constexpr double h = 1e-5;
    auto scaled = [&](double eps) {
        cpu::Frame g = make_ch4_asymmetric();
        // Isotropic: all three diagonal Voigt strains at once.
        Mat3 cell = g.box.cell();
        Mat3 cell_new;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cell_new(i, j) = cell(i, j) * static_cast<AccReal>(1.0 + eps);
        g.box.set_cell(cell_new);
        for (int a = 0; a < g.nlocal(); ++a) {
            g.span(kAtom_x)[a] = static_cast<StorageReal>(
                static_cast<double>(g.span(kAtom_x)[a]) * (1.0 + eps));
            g.span(kAtom_y)[a] = static_cast<StorageReal>(
                static_cast<double>(g.span(kAtom_y)[a]) * (1.0 + eps));
            g.span(kAtom_z)[a] = static_cast<StorageReal>(
                static_cast<double>(g.span(kAtom_z)[a]) * (1.0 + eps));
        }
        return energy_3ob_only(g);
    };
    const double dE_deps = (scaled(h) - scaled(-h)) / (2.0 * h);
    const double minus_dE_dvol = -dE_deps / (3.0 * vol);

    EXPECT_NEAR(trace_third, minus_dE_dvol, 1e-8)
        << "tr(σ)/3=" << trace_third << " −dE/dΩ=" << minus_dE_dvol;
}
