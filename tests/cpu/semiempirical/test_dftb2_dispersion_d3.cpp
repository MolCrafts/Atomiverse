/// @file tests/cpu/semiempirical/test_dftb2_dispersion_d3.cpp
/// @brief Unit: DFTB2::d3 folds D3(BJ) into energy, forces, and stress.
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

cpu::Frame water_dimer() {
    // Spec geometry (Å); cubic box 20 Å (isolated).
    cpu::Frame f(6);
    const double xyz[6][3] = {
        {0.000, 0.000, 0.000}, {0.757, 0.000, 0.586}, {-0.757, 0.000, 0.586},
        {0.000, 0.000, 2.900}, {0.757, 0.000, 3.486}, {-0.757, 0.000, 3.486},
    };
    const int Z[6] = {8, 1, 1, 8, 1, 1};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_type)[i] = Z[i];
        f.span(kAtom_x)[i] = static_cast<StorageReal>(xyz[i][0]);
        f.span(kAtom_y)[i] = static_cast<StorageReal>(xyz[i][1]);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(xyz[i][2]);
        f.span(kAtom_mass)[i] = (Z[i] == 8) ? 15.999 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

void zero_forces(cpu::Frame& f) {
    const int n = f.nlocal();
    for (int a = 0; a < n; ++a)
        f.span(kAtom_fx)[a] = f.span(kAtom_fy)[a] = f.span(kAtom_fz)[a] = 0;
}

void zero_stress(cpu::Frame& f) {
    auto& st = f.meta.get(kStress);
    for (int v = 0; v < 6; ++v)
        st[static_cast<std::size_t>(v)] = 0;
}

double e_disp_at(cpu::Frame f) {
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.d3 = true;
    dftb.with_forces = false;
    dftb.with_stress = false;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    return dftb.last_e_disp();
}

double e_disp_periodic_at(cpu::Frame f) {
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.d3 = true;
    dftb.d3_periodic = true;
    dftb.with_forces = false;
    dftb.with_stress = false;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    return dftb.last_e_disp();
}

/// One water molecule in a cubic cell of edge `L` [Å], `rep`³ replicated.
cpu::Frame water_lattice(double L, int rep) {
    const double xyz[3][3] = {
        {0.000, 0.000, 0.000}, {0.757, 0.000, 0.586}, {-0.757, 0.000, 0.586}};
    const int Z[3] = {8, 1, 1};
    const int n = 3 * rep * rep * rep;
    cpu::Frame f(n);
    int i = 0;
    for (int cx = 0; cx < rep; ++cx)
        for (int cy = 0; cy < rep; ++cy)
            for (int cz = 0; cz < rep; ++cz)
                for (int a = 0; a < 3; ++a, ++i) {
                    f.span(kAtom_type)[i] = Z[a];
                    f.span(kAtom_x)[i] =
                        static_cast<StorageReal>(xyz[a][0] + cx * L);
                    f.span(kAtom_y)[i] =
                        static_cast<StorageReal>(xyz[a][1] + cy * L);
                    f.span(kAtom_z)[i] =
                        static_cast<StorageReal>(xyz[a][2] + cz * L);
                    f.span(kAtom_mass)[i] = (Z[a] == 8) ? 15.999 : 1.008;
                    f.span(kAtom_tag)[i] = i + 1;
                    f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] =
                        f.span(kAtom_fz)[i] = 0;
                }
    f.box = cpu::Box::cubic(L * rep);
    f.meta.get(kEnergy) = 0;
    return f;
}

cpu::Frame water_dimer_in_box(double L) {
    cpu::Frame f = water_dimer();
    f.box = cpu::Box::cubic(L);
    return f;
}

} // namespace

TEST(DFTB2DispersionD3, OffMeansZeroDisp) {
    auto f = water_dimer();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.d3 = false;
    dftb.with_forces = false;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    EXPECT_NEAR(dftb.last_e_disp(), 0.0, 1e-15);
    EXPECT_TRUE(std::isfinite(dftb.last_energy()));
}

TEST(DFTB2DispersionD3, OnAddsAttractiveDisp) {
    auto f = water_dimer();
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 dftb;
    dftb.d3 = true;
    dftb.with_forces = false;
    dftb.build({ctx, f});
    dftb.launch({ctx, f, 0, 0});
    EXPECT_LT(dftb.last_e_disp(), 0.0);
    EXPECT_TRUE(std::isfinite(dftb.last_e_disp()));
    EXPECT_NEAR(f.meta.get(kEnergy), dftb.last_energy(), 1e-12);
}

TEST(DFTB2DispersionD3, TotDiffEqualsEDisp) {
    cpu::Context ctx;
    auto f_off = water_dimer();
    cpu::semiempirical::DFTB2 off;
    off.d3 = false;
    off.with_forces = false;
    off.build({ctx, f_off});
    off.launch({ctx, f_off, 0, 0});
    const double e_off = off.last_energy();

    auto f_on = water_dimer();
    cpu::semiempirical::DFTB2 on;
    on.d3 = true;
    on.with_forces = false;
    on.build({ctx, f_on});
    on.launch({ctx, f_on, 0, 0});
    const double e_on = on.last_energy();
    const double ed = on.last_e_disp();

    EXPECT_NEAR(e_on - e_off, ed, 1e-9);
}

TEST(DFTB2DispersionD3, ForcesMatchCentralFDOfEDisp) {
    // Isolate F_d3 = F(d3 on) − F(d3 off); compare to −∂E_disp/∂r (central FD).
    cpu::Context ctx;
    auto f_off = water_dimer();
    zero_forces(f_off);
    cpu::semiempirical::DFTB2 off;
    off.d3 = false;
    off.with_forces = true;
    off.build({ctx, f_off});
    off.launch({ctx, f_off, 0, 0});

    auto f_on = water_dimer();
    zero_forces(f_on);
    cpu::semiempirical::DFTB2 on;
    on.d3 = true;
    on.with_forces = true;
    on.build({ctx, f_on});
    on.launch({ctx, f_on, 0, 0});

    const int n = f_on.nlocal();
    std::vector<std::array<double, 3>> f_d3(static_cast<std::size_t>(n));
    for (int a = 0; a < n; ++a) {
        f_d3[static_cast<std::size_t>(a)] = {
            static_cast<double>(f_on.span(kAtom_fx)[a]
                                - f_off.span(kAtom_fx)[a]),
            static_cast<double>(f_on.span(kAtom_fy)[a]
                                - f_off.span(kAtom_fy)[a]),
            static_cast<double>(f_on.span(kAtom_fz)[a]
                                - f_off.span(kAtom_fz)[a])};
    }

    constexpr double h = 1.0e-4; // Å
    double max_err = 0.0;
    for (int a = 0; a < n; ++a) {
        for (int k = 0; k < 3; ++k) {
            auto fp = water_dimer();
            auto fm = water_dimer();
            StorageReal* sp =
                (k == 0) ? fp.span(kAtom_x)
                         : ((k == 1) ? fp.span(kAtom_y) : fp.span(kAtom_z));
            StorageReal* sm =
                (k == 0) ? fm.span(kAtom_x)
                         : ((k == 1) ? fm.span(kAtom_y) : fm.span(kAtom_z));
            sp[a] = static_cast<StorageReal>(static_cast<double>(sp[a]) + h);
            sm[a] = static_cast<StorageReal>(static_cast<double>(sm[a]) - h);
            const double ed_p = e_disp_at(std::move(fp));
            const double ed_m = e_disp_at(std::move(fm));
            const double f_fd = -(ed_p - ed_m) / (2.0 * h);
            const double err = std::abs(
                f_d3[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)]
                - f_fd);
            max_err = std::max(max_err, err);
        }
    }
    EXPECT_LT(max_err, 1e-6) << "max |F_d3 − (−∂E_disp/∂r)| [eV/Å]";
}

TEST(DFTB2DispersionD3, StressIncludesD3) {
    cpu::Context ctx;

    auto f_off = water_dimer();
    zero_stress(f_off);
    cpu::semiempirical::DFTB2 off;
    off.d3 = false;
    off.with_forces = false;
    off.with_stress = true;
    off.build({ctx, f_off});
    off.launch({ctx, f_off, 0, 0});
    const auto st_off = off.last_stress();

    auto f_on = water_dimer();
    zero_stress(f_on);
    cpu::semiempirical::DFTB2 on;
    on.d3 = true;
    on.with_forces = false;
    on.with_stress = true;
    on.build({ctx, f_on});
    on.launch({ctx, f_on, 0, 0});
    const auto st_on = on.last_stress();
    const auto& st_meta = f_on.meta.get(kStress);

    // D3 must change the stress and land on kStress.
    double max_diff = 0.0;
    for (int v = 0; v < 6; ++v) {
        max_diff =
            std::max(max_diff, std::abs(st_on[static_cast<std::size_t>(v)]
                                        - st_off[static_cast<std::size_t>(v)]));
        // kStress is the extensive Voigt virial [eV]; last_stress() is the
        // intensive σ [eV/Å³] (core/frame_schema.h).
        EXPECT_NEAR(st_meta[static_cast<std::size_t>(v)],
                    st_on[static_cast<std::size_t>(v)] * f_on.box.volume(),
                    1e-14 * f_on.box.volume());
    }
    EXPECT_GT(max_diff, 1e-12) << "σ(d3=true) should differ from σ(d3=false)";

    // σ_d3 = σ_on − σ_off; isotropic volume gate:
    // tr(σ) = −(3/V) dE/dlnV for pure pair forces under affine scale.
    std::array<double, 6> sig_d3{};
    for (int v = 0; v < 6; ++v)
        sig_d3[static_cast<std::size_t>(v)] =
            st_on[static_cast<std::size_t>(v)]
            - st_off[static_cast<std::size_t>(v)];
    const double tr = sig_d3[0] + sig_d3[1] + sig_d3[2];

    constexpr double h = 1.0e-4;
    auto scale_frame = [](cpu::Frame f, double s) {
        const int n = f.nlocal();
        for (int a = 0; a < n; ++a) {
            f.span(kAtom_x)[a] = static_cast<StorageReal>(
                static_cast<double>(f.span(kAtom_x)[a]) * s);
            f.span(kAtom_y)[a] = static_cast<StorageReal>(
                static_cast<double>(f.span(kAtom_y)[a]) * s);
            f.span(kAtom_z)[a] = static_cast<StorageReal>(
                static_cast<double>(f.span(kAtom_z)[a]) * s);
        }
        f.box.scale_isotropic(s); // h ← s h (edges × s for orthorhombic)
        return f;
    };
    const double ed_p = e_disp_at(scale_frame(water_dimer(), 1.0 + h));
    const double ed_m = e_disp_at(scale_frame(water_dimer(), 1.0 - h));
    // dE/ds at s=1; d lnV = 3 d ln s ⇒ dE/dlnV = (1/3) dE/ds.
    const double dE_ds = (ed_p - ed_m) / (2.0 * h);
    const double dE_dlnV = dE_ds / 3.0;
    const double vol = water_dimer().box.volume();
    ASSERT_GT(vol, 0.0);
    // tr(σ) + 3/V * dE/dlnV ≈ 0
    EXPECT_NEAR(tr + 3.0 / vol * dE_dlnV, 0.0, 1e-6)
        << "tr(σ_d3)=" << tr << " dE_disp/dlnV=" << dE_dlnV;
}

// ---------------------------------------------------------------------------
// d3_periodic: Γ-PBC lattice-image D3(BJ) sum
// ---------------------------------------------------------------------------

TEST(DFTB2DispersionD3Periodic, BeyondCutoffBoxEqualsMolecular) {
    // The D3 pair cutoff is 94.2 Bohr ~ 49.9 Å. In a 120 Å cell every image
    // pair is past the cutoff, so the lattice sum must collapse *exactly* onto
    // the isolated-cluster sum -- no residual image term, no double counting
    // of the home cell.
    const double ed_mol = e_disp_at(water_dimer_in_box(120.0));
    const double ed_pbc = e_disp_periodic_at(water_dimer_in_box(120.0));
    EXPECT_NEAR(ed_pbc, ed_mol, 1e-12) << "E_disp [eV]";
}

TEST(DFTB2DispersionD3Periodic, ImagesAddAttraction) {
    // At 12 Å the nearest images are well inside the cutoff, so the lattice
    // sum must be strictly more attractive than the isolated cluster.
    const double ed_mol = e_disp_at(water_dimer_in_box(12.0));
    const double ed_pbc = e_disp_periodic_at(water_dimer_in_box(12.0));
    EXPECT_LT(ed_pbc, ed_mol) << "images must lower E_disp";
    // ...but by a physically small amount for a 12 Å cell (C6/R^6 tail).
    EXPECT_LT(std::abs(ed_pbc - ed_mol), 0.1) << "eV";
}

TEST(DFTB2DispersionD3Periodic, SupercellEnergyIsExtensive) {
    // Decisive Bravais-sum identity: the same infinite crystal described by a
    // 1x1x1 cell and by its 2x2x2 supercell must give the same energy *per
    // primitive cell*. A wrong image count, a half-counted home cell, or a
    // non-image-aware coordination number all break this.
    const double L = 7.0;
    const double e1 = e_disp_periodic_at(water_lattice(L, 1));
    const double e2 = e_disp_periodic_at(water_lattice(L, 2));
    ASSERT_LT(e1, 0.0) << "dispersion must be attractive";
    EXPECT_NEAR(e2, 8.0 * e1, 1e-8 * std::abs(8.0 * e1))
        << "E(2x2x2)=" << e2 << " vs 8*E(1x1x1)=" << 8.0 * e1 << " [eV]";
}

TEST(DFTB2DispersionD3Periodic, ForcesMatchCentralFDOfEDisp) {
    // F_d3 = F(d3 on) - F(d3 off) under PBC vs -dE_disp/dr of the *periodic*
    // sum: the analytic gradient must differentiate the functional actually
    // summed, images included.
    constexpr double kBox = 12.0;
    cpu::Context ctx;

    auto f_off = water_dimer_in_box(kBox);
    zero_forces(f_off);
    cpu::semiempirical::DFTB2 off;
    off.d3 = false;
    off.with_forces = true;
    off.build({ctx, f_off});
    off.launch({ctx, f_off, 0, 0});

    auto f_on = water_dimer_in_box(kBox);
    zero_forces(f_on);
    cpu::semiempirical::DFTB2 on;
    on.d3 = true;
    on.d3_periodic = true;
    on.with_forces = true;
    on.build({ctx, f_on});
    on.launch({ctx, f_on, 0, 0});

    const int n = f_on.nlocal();
    constexpr double h = 1.0e-4; // Å
    double max_err = 0.0;
    for (int a = 0; a < n; ++a) {
        for (int k = 0; k < 3; ++k) {
            const double f_an = static_cast<double>(
                (k == 0)   ? f_on.span(kAtom_fx)[a] - f_off.span(kAtom_fx)[a]
                : (k == 1) ? f_on.span(kAtom_fy)[a] - f_off.span(kAtom_fy)[a]
                           : f_on.span(kAtom_fz)[a] - f_off.span(kAtom_fz)[a]);
            auto fp = water_dimer_in_box(kBox);
            auto fm = water_dimer_in_box(kBox);
            StorageReal* sp = (k == 0)   ? fp.span(kAtom_x)
                              : (k == 1) ? fp.span(kAtom_y)
                                         : fp.span(kAtom_z);
            StorageReal* sm = (k == 0)   ? fm.span(kAtom_x)
                              : (k == 1) ? fm.span(kAtom_y)
                                         : fm.span(kAtom_z);
            sp[a] = static_cast<StorageReal>(static_cast<double>(sp[a]) + h);
            sm[a] = static_cast<StorageReal>(static_cast<double>(sm[a]) - h);
            const double f_fd = -(e_disp_periodic_at(std::move(fp))
                                  - e_disp_periodic_at(std::move(fm)))
                                / (2.0 * h);
            max_err = std::max(max_err, std::abs(f_an - f_fd));
        }
    }
    EXPECT_LT(max_err, 1e-6) << "max |F_d3^pbc - (-dE_disp/dr)| [eV/Å]";
}

TEST(DFTB2DispersionD3Periodic, StressTraceMatchesVolumeDerivative) {
    // Image-aware virial: tr(sigma_d3) = -(3/V) dE_disp/dlnV under an affine
    // scaling of *both* the cell and the atom positions. The molecular gate
    // scales positions only; here the lattice vectors must scale too, which is
    // exactly what a non-image-aware virial would get wrong.
    constexpr double kBox = 12.0;
    cpu::Context ctx;

    auto stress_of = [&](bool d3_on) {
        auto f = water_dimer_in_box(kBox);
        zero_stress(f);
        cpu::semiempirical::DFTB2 m;
        m.d3 = d3_on;
        m.d3_periodic = d3_on;
        m.with_forces = false;
        m.with_stress = true;
        m.build({ctx, f});
        m.launch({ctx, f, 0, 0});
        return m.last_stress();
    };
    const auto st_off = stress_of(false);
    const auto st_on = stress_of(true);
    const double tr = (st_on[0] - st_off[0]) + (st_on[1] - st_off[1])
                      + (st_on[2] - st_off[2]);

    auto scaled = [](double s) {
        auto f = water_dimer_in_box(kBox);
        const int n = f.nlocal();
        for (int a = 0; a < n; ++a) {
            f.span(kAtom_x)[a] = static_cast<StorageReal>(
                static_cast<double>(f.span(kAtom_x)[a]) * s);
            f.span(kAtom_y)[a] = static_cast<StorageReal>(
                static_cast<double>(f.span(kAtom_y)[a]) * s);
            f.span(kAtom_z)[a] = static_cast<StorageReal>(
                static_cast<double>(f.span(kAtom_z)[a]) * s);
        }
        f.box.scale_isotropic(s);
        return f;
    };
    constexpr double h = 1.0e-4;
    const double dE_ds = (e_disp_periodic_at(scaled(1.0 + h))
                          - e_disp_periodic_at(scaled(1.0 - h)))
                         / (2.0 * h);
    const double dE_dlnV = dE_ds / 3.0;
    const double vol = water_dimer_in_box(kBox).box.volume();
    ASSERT_GT(vol, 0.0);
    EXPECT_NEAR(tr + 3.0 / vol * dE_dlnV, 0.0, 1e-6)
        << "tr(σ_d3^pbc)=" << tr << " dE_disp/dlnV=" << dE_dlnV;
}

TEST(DFTB2DispersionD3Periodic, NonOrthorhombicBoxThrows) {
    // The lattice sum only knows orthorhombic edge lengths; a triclinic cell
    // must fail loudly rather than silently sum the wrong images.
    auto f = water_dimer_in_box(12.0);
    ATV::Mat3 cell{};
    cell(0, 0) = 12.0;
    cell(1, 0) = 3.0; // triclinic tilt
    cell(1, 1) = 12.0;
    cell(2, 2) = 12.0;
    f.box.set_cell(cell);
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 m;
    m.d3 = true;
    m.d3_periodic = true;
    m.build({ctx, f});
    EXPECT_THROW(m.launch({ctx, f, 0, 0}), std::runtime_error);
}

TEST(DFTB2DispersionD3Periodic, CutoffTruncatesTheLatticeTail) {
    // d3_cutoff defines what a lattice sum *is*: the tail past it is dropped.
    // A shorter cutoff must therefore keep strictly less attraction, and the
    // sum must approach a limit as the cutoff grows. The isolated-cluster
    // energy, by contrast, cannot depend on it at all for a 6-atom molecule.
    auto pbc_at_cutoff = [](double cut_ang) {
        auto f = water_dimer_in_box(12.0);
        cpu::Context ctx;
        cpu::semiempirical::DFTB2 m;
        m.d3 = true;
        m.d3_periodic = true;
        m.d3_cutoff = cut_ang;
        m.build({ctx, f});
        m.launch({ctx, f, 0, 0});
        return m.last_e_disp();
    };
    const double e40 = pbc_at_cutoff(40.0 * units::kBohrInAngstrom);
    const double e60 = pbc_at_cutoff(60.0 * units::kBohrInAngstrom);
    const double e94 = pbc_at_cutoff(94.2 * units::kBohrInAngstrom);
    EXPECT_GT(e40, e60) << "a shorter cutoff must drop attractive tail";
    EXPECT_GT(e60, e94);
    // Converging, not diverging: each extension adds less than the last.
    EXPECT_LT(std::abs(e94 - e60), std::abs(e60 - e40));

    // The default member value is exactly the historical 94.2 Bohr cutoff.
    auto f = water_dimer_in_box(12.0);
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 m;
    m.d3 = true;
    m.d3_periodic = true;
    m.build({ctx, f});
    m.launch({ctx, f, 0, 0});
    EXPECT_NEAR(m.last_e_disp(), e94, 1e-15);
}

TEST(DFTB2DispersionD3Periodic, NonPositiveCutoffThrows) {
    auto f = water_dimer_in_box(12.0);
    cpu::Context ctx;
    cpu::semiempirical::DFTB2 m;
    m.d3 = true;
    m.d3_cutoff = 0.0;
    m.build({ctx, f});
    EXPECT_THROW(m.launch({ctx, f, 0, 0}), std::runtime_error);
}
