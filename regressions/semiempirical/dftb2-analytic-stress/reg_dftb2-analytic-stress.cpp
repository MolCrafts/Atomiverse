/// @file regressions/semiempirical/dftb2-analytic-stress/reg_dftb2-analytic-stress.cpp
/// @brief H2O SCC-DFTB2 analytic cell stress vs multi-SCC affine strain-FD.
///
/// Domain: Elstner et al., Phys. Rev. B 58, 7260 (1998); DFTB+ stress module
/// boundary (Hourahine et al., J. Chem. Phys. 152, 124101 (2020)). In-tree
/// mio-style SK tables — not absolute mio-1-1 DFTB+ numbers.
///
/// Exit non-zero on failure. Public API only. <= 60 s.
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

cpu::Frame make_water() {
    // Off-equilibrium water (Å) so |virial| ≠ 0.
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
    for (int v = 0; v < 6; ++v)
        f.meta.get(kStress)[static_cast<std::size_t>(v)] = 0;
    return f;
}

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
    for (int a = 0; a < 3; ++a) {
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
    for (int i = 0; i < 3; ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Evaluate ev;
    cpu::semiempirical::DFTB2 d;
    d.with_forces = false;
    d.with_stress = false;
    d.scc_tol = 1e-10;
    ev.on(evaluate::Phase::evaluate).add(std::move(d));
    ev.build();
    ev.run(g);
    return g.meta.get(kEnergy);
}

} // namespace

int main() {
    auto frame = make_water();

    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = false;
    dftb.with_stress = true;
    dftb.scc_tol = 1e-10;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    std::printf("DFTB2 analytic stress H2O: E=%.8f eV\n", e);
    if (!std::isfinite(e) || std::fabs(e) < 1e-3) {
        std::printf("FAIL energy not finite/bound\n");
        return 1;
    }

    // kStress is the extensive Voigt virial [eV] (core/frame_schema.h); this
    // gate is written against the intensive stress W/V [eV/A^3].
    const double vol_a3 = frame.box.volume();
    const auto& w = frame.meta.get(kStress);
    double st[6];
    for (int v = 0; v < 6; ++v)
        st[v] = w[static_cast<std::size_t>(v)] / vol_a3;
    for (int v = 0; v < 6; ++v) {
        std::printf("  σ[%d]=%.8e eV/A3\n", v, st[static_cast<std::size_t>(v)]);
        if (!std::isfinite(st[static_cast<std::size_t>(v)])) {
            std::printf("FAIL non-finite stress\n");
            return 1;
        }
    }

    constexpr double h = 2e-4;
    // Same stationarity argument as the force gate: measured residual
    // 1.1e-10 / 8.9e-11 eV/Å³ (2026-07-28).
    constexpr double tol = 1e-8; // eV/Å³
    const double vol = frame.box.volume();

    // Gate σ_xx and σ_yy (leading in-plane components for planar H2O).
    for (int voigt : {0, 1}) {
        auto fp = make_water();
        auto fm = make_water();
        apply_strain_voigt(fp, voigt, h);
        apply_strain_voigt(fm, voigt, -h);
        const double dE = (energy_only(fp) - energy_only(fm)) / (2.0 * h);
        const double s_fd = -dE / vol;
        const double s_an = st[static_cast<std::size_t>(voigt)];
        std::printf("  σ_v%d analytic=%.8e  fd=%.8e  |d|=%.3e eV/A3\n", voigt,
                    s_an, s_fd, std::fabs(s_an - s_fd));
        if (std::fabs(s_an - s_fd) > tol) {
            std::printf("FAIL σ[%d] vs strain-FD (tol %.3e eV/A3)\n", voigt, tol);
            return 1;
        }
    }

    std::printf("PASS\n");
    return 0;
}
