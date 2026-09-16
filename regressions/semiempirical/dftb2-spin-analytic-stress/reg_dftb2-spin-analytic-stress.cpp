/// @file regressions/semiempirical/dftb2-spin-analytic-stress/reg_dftb2-spin-analytic-stress.cpp
/// @brief Collinear open-shell SCC-DFTB2 analytic cell stress vs strain-FD.
///
/// Domain: collinear spin-DFTB (Köhler, Seifert, Frauenheim; Elstner et al.,
/// Phys. Rev. B 58, 7260 (1998)); DFTB+ spin + stress module boundary
/// (Hourahine et al., J. Chem. Phys. 152, 124101 (2020)).
/// oracle: dftb+ 24.1 for energy/forces
/// (tools/tb_oracle/inputs/dftbplus/dftb2-spin-analytic-stress/); cluster
/// DFTB+ has no cell stress — σ is analytic vs strain-FD.
///
/// Headline case: O2 triplet. Discriminator: NO doublet — O2 is homonuclear, so
/// dq = 0 and the spin shift +/- W_A m_A is uniform across atoms; NO carries
/// both a nonzero SCC gamma virial and a non-uniform spin shift.
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

/// Tilted O-X diatomic; direction (0.6, 0.7, sqrt(0.15)) is a unit vector, so
/// every Voigt component of the virial is non-trivial and |R| = r_ang.
cpu::Frame make_tilted_diatomic(int z1, double r_ang) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_type)[1] = z1;
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = static_cast<StorageReal>(r_ang * 0.6);
    f.span(kAtom_y)[1] = static_cast<StorageReal>(r_ang * 0.7);
    f.span(kAtom_z)[1] = static_cast<StorageReal>(r_ang * 0.3872983346207417);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 15.999;
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

double energy_only(cpu::Frame g, int multiplicity) {
    g.meta.get(kEnergy) = 0;
    for (int i = 0; i < g.nlocal(); ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Evaluate ev;
    cpu::semiempirical::DFTB2 d;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    d.boundary = cpu::semiempirical::DftbBoundary::cluster;
    d.multiplicity = multiplicity;
    d.spin_w_ha = cpu::semiempirical::basis_data::kDftbSpinConstantsPbeHa;
    d.with_forces = false;
    d.with_stress = false;
    d.scc_tol = 1e-12;
    d.max_scc = 200;
    ev.on(evaluate::Phase::evaluate).add(std::move(d));
    ev.build();
    ev.run(g);
    return g.meta.get(kEnergy);
}

/// Returns 0 on success, 1 on failure.
int check_case(const char* label, int z1, double r_ang, int multiplicity) {
    auto frame = make_tilted_diatomic(z1, r_ang);

    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
    dftb.multiplicity = multiplicity;
    dftb.spin_w_ha = cpu::semiempirical::basis_data::kDftbSpinConstantsPbeHa;
    dftb.with_forces = true;
    dftb.with_stress = true;
    dftb.scc_tol = 1e-12;
    dftb.max_scc = 200;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    std::printf("%s: E=%.8f eV\n", label, e);
    if (!std::isfinite(e) || std::fabs(e) < 1e-3) {
        std::printf("FAIL %s energy not finite/bound\n", label);
        return 1;
    }

    // Translational invariance of the open-shell gradient.
    double net[3] = {0, 0, 0};
    for (int i = 0; i < 2; ++i) {
        net[0] += static_cast<double>(frame.span(kAtom_fx)[i]);
        net[1] += static_cast<double>(frame.span(kAtom_fy)[i]);
        net[2] += static_cast<double>(frame.span(kAtom_fz)[i]);
    }
    for (int d = 0; d < 3; ++d)
        if (!std::isfinite(net[d]) || std::fabs(net[d]) > 1e-6) {
            std::printf("FAIL %s sum(F)[%d] = %.3e eV/A\n", label, d, net[d]);
            return 1;
        }

    constexpr double h = 2e-4;
    // A converged SCC-DFTB2 gradient is stationary and the virial is the pair
    // contraction of the same bond terms, so this is FD-exact to the
    // central-difference truncation error.
    constexpr double tol = 1e-8; // eV/A^3
    const double vol = frame.box.volume();
    // kStress is the extensive Voigt virial [eV] (core/frame_schema.h);
    // these gates are written against the intensive stress W/V [eV/A^3].
    const auto& w_ext = frame.meta.get(kStress);
    double st[6];
    for (int v = 0; v < 6; ++v)
        st[v] = w_ext[static_cast<std::size_t>(v)] / vol;

    for (int voigt = 0; voigt < 6; ++voigt) {
        const double s_an = st[static_cast<std::size_t>(voigt)];
        if (!std::isfinite(s_an)) {
            std::printf("FAIL %s non-finite sigma[%d]\n", label, voigt);
            return 1;
        }
        auto fp = make_tilted_diatomic(z1, r_ang);
        auto fm = make_tilted_diatomic(z1, r_ang);
        apply_strain_voigt(fp, voigt, h);
        apply_strain_voigt(fm, voigt, -h);
        const double dE = (energy_only(std::move(fp), multiplicity)
                           - energy_only(std::move(fm), multiplicity))
                          / (2.0 * h);
        const double s_fd = -dE / vol;
        std::printf("  sigma[%d] analytic=%.8e  fd=%.8e  |d|=%.3e eV/A3\n",
                    voigt, s_an, s_fd, std::fabs(s_an - s_fd));
        if (std::fabs(s_an - s_fd) > tol) {
            std::printf("FAIL %s sigma[%d] vs strain-FD (tol %.3e eV/A3)\n",
                        label, voigt, tol);
            return 1;
        }
    }
    return 0;
}

} // namespace

int main() {
    // Headline: O2 triplet (Z=8/8, multiplicity 3 -> n_alpha=7, n_beta=5).
    if (check_case("spin-DFTB2 analytic stress O2 triplet", 8, 1.30, 3) != 0)
        return 1;
    // Discriminator: NO doublet (Z=8/7, multiplicity 2 -> n_alpha=6, n_beta=5)
    // -> dq != 0 and a non-uniform spin shift.
    if (check_case("spin-DFTB2 analytic stress NO doublet", 7, 1.20, 2) != 0)
        return 1;

    std::printf("PASS\n");
    return 0;
}
