/// @file regressions/semiempirical/dftb2-analytic-stress-threeob/reg_dftb2-analytic-stress-threeob.cpp
/// @brief CH4 / 3ob-3-1 SCC-DFTB2 closed-form cell stress vs affine strain-FD.
///
/// Domain: Elstner et al., Phys. Rev. B 58, 7260 (1998); 3ob-3-1 SK tables from
/// Gaus et al., J. Chem. Theory Comput. 9, 338 (2013). Stress theorem /
/// mechanical convention: Nielsen & Martin, Phys. Rev. B 32, 3780 (1985);
/// DFTB+ stress module boundary: Hourahine et al., J. Chem. Phys. 152, 124101
/// (2020) (family oracle DFTB+ 24.1 — formula-level, no binary on pin host).
///
/// Exit non-zero on failure. Public API only. <= 60 s.
#include <array>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

/// Asymmetric CH4: four distinct C-H bonds so every Voigt component (shears
/// included) is non-zero — symmetric tetrahedral CH4 has identically zero
/// shear. Bond lengths sit mid-way between 3ob radial-grid knots (grid
/// 0.02 Bohr) so the finite-difference oracle never straddles a knot of the
/// linearly interpolated SK tables.
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
        f.span(kAtom_mass)[i] = (i == 0) ? 12.011 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    for (int v = 0; v < 6; ++v)
        f.meta.get(kStress)[static_cast<std::size_t>(v)] = 0;
    return f;
}

/// Affine strain of cell + nuclei; off-diagonal Voigt is symmetric
/// (eps_ab = eps_ba = h) — the deformation sigma = -(1/Omega) dE/deps means.
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

double energy_only(cpu::Frame g) {
    g.meta.get(kEnergy) = 0;
    for (int i = 0; i < g.nlocal(); ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Evaluate ev;
    cpu::semiempirical::DFTB2 d;
    d.sk_set = cpu::semiempirical::DftbSkSet::ThreeOb;
    d.with_forces = false;
    d.with_stress = false;
    d.scc_tol = 1e-10;
    d.max_scc = 80;
    ev.on(evaluate::Phase::evaluate).add(std::move(d));
    ev.build();
    ev.run(g);
    return g.meta.get(kEnergy);
}

/// Isotropic scaling r -> (1+eps) r of nuclei and cell.
double energy_scaled(double eps) {
    cpu::Frame g = make_ch4_asymmetric();
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
    return energy_only(g);
}

} // namespace

int main() {
    auto frame = make_ch4_asymmetric();

    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb;
    dftb.sk_set = cpu::semiempirical::DftbSkSet::ThreeOb;
    dftb.with_forces = false;
    dftb.with_stress = true;
    dftb.scc_tol = 1e-10;
    dftb.max_scc = 80;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    std::printf("DFTB2 ThreeOb analytic stress CH4: E=%.8f eV\n", e);
    if (!std::isfinite(e) || std::fabs(e) < 1e-3) {
        std::printf("FAIL energy not finite/bound\n");
        return 1;
    }

    // kStress is the extensive Voigt virial [eV] (core/frame_schema.h);
    // these gates are written against the intensive stress W/V [eV/A^3].
    const double vol_a3 = frame.box.volume();
    const auto& w_ext = frame.meta.get(kStress);
    double st[6];
    for (int v = 0; v < 6; ++v)
        st[v] = w_ext[static_cast<std::size_t>(v)] / vol_a3;
    double norm2 = 0;
    for (int v = 0; v < 6; ++v) {
        const double s = st[static_cast<std::size_t>(v)];
        std::printf("  sigma[%d]=%.10e eV/A3\n", v, s);
        if (!std::isfinite(s)) {
            std::printf("FAIL non-finite stress\n");
            return 1;
        }
        norm2 += s * s;
    }
    // Asymmetric CH4 -> no component is trivially zero (the gate would be
    // vacuous on symmetric CH4, whose shears vanish identically).
    if (!(norm2 > 1e-14)) {
        std::printf("FAIL stress tensor is (numerically) null: |sigma|^2=%.3e\n",
                    norm2);
        return 1;
    }

    // h = 1e-5 keeps every strained C-H / H-H distance inside one 3ob SK grid
    // interval (grid 0.02 Bohr); the tables are linearly interpolated, so an FD
    // interval straddling a knot would compare two different slopes.
    constexpr double h = 1e-5;
    constexpr double tol = 1e-8; // eV/A3
    const double vol = frame.box.volume();

    for (int voigt = 0; voigt < 6; ++voigt) {
        auto fp = make_ch4_asymmetric();
        auto fm = make_ch4_asymmetric();
        apply_strain_voigt(fp, voigt, h);
        apply_strain_voigt(fm, voigt, -h);
        const double dE = (energy_only(fp) - energy_only(fm)) / (2.0 * h);
        const double s_fd = -dE / vol;
        const double s_an = st[static_cast<std::size_t>(voigt)];
        std::printf("  sigma_v%d analytic=%.10e  fd=%.10e  |d|=%.3e eV/A3\n",
                    voigt, s_an, s_fd, std::fabs(s_an - s_fd));
        if (std::fabs(s_an - s_fd) > tol) {
            std::printf("FAIL sigma[%d] vs strain-FD (tol %.3e eV/A3)\n", voigt,
                        tol);
            return 1;
        }
    }

    // Isotropic identity: Omega(eps) = (1+eps)^3 Omega0 so dOmega/deps|0 = 3
    // Omega0, and dE/deps = sum_a dE/deps_aa = -Omega0 tr(sigma). Hence
    //     tr(sigma)/3 = -dE/dOmega = -(1/(3 Omega0)) dE/deps|0 .
    const double trace_third = (st[0] + st[1] + st[2]) / 3.0;
    const double dE_deps = (energy_scaled(h) - energy_scaled(-h)) / (2.0 * h);
    const double minus_dE_dvol = -dE_deps / (3.0 * vol);
    std::printf("  tr(sigma)/3=%.10e  -dE/dOmega=%.10e  |d|=%.3e eV/A3\n",
                trace_third, minus_dE_dvol,
                std::fabs(trace_third - minus_dE_dvol));
    if (std::fabs(trace_third - minus_dE_dvol) > tol) {
        std::printf("FAIL isotropic trace identity (tol %.3e eV/A3)\n", tol);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
