/// @file regressions/semiempirical/dftb2-spin-analytic-forces/reg_dftb2-spin-analytic-forces.cpp
/// @brief O₂ triplet collinear spin-DFTB2 analytic forces vs re-SCC FD.
///
/// Method: collinear unrestricted SCC-DFTB2 (Elstner + Köhler spin channel).
/// Force boundary: Hourahine et al. JCP 152, 124101 (2020) / DFTB+ spin module.
/// oracle: dftb+ 24.1 (tools/tb_oracle/inputs/dftbplus/dftb2-spin-analytic-forces/).
///
/// Exit non-zero on failure. Public API only. <= 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

cpu::Frame make_o2_offeq(double r_ang = 1.30) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = r_ang;
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

double energy_only_triplet(cpu::Frame g) {
    g.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Evaluate ev;
    cpu::semiempirical::DFTB2 d;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    d.boundary = cpu::semiempirical::DftbBoundary::cluster;
    d.multiplicity = 3;
    d.spin_w_ha = cpu::semiempirical::basis_data::kDftbSpinConstantsPbeHa;
    d.with_forces = false;
    d.scc_tol = 1e-10;
    d.max_scc = 80;
    ev.on(evaluate::Phase::evaluate).add(std::move(d));
    ev.build();
    ev.run(g);
    return g.meta.get(kEnergy);
}

} // namespace

int main() {
    auto frame = make_o2_offeq();

    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
    dftb.multiplicity = 3;
    dftb.spin_w_ha = cpu::semiempirical::basis_data::kDftbSpinConstantsPbeHa;
    dftb.with_forces = true;
    dftb.scc_tol = 1e-10;
    dftb.max_scc = 80;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    std::printf("spin-DFTB2 analytic F O2: E=%.8f eV\n", e);
    if (!std::isfinite(e) || std::fabs(e) < 1e-3) {
        std::printf("FAIL energy not finite/bound\n");
        return 1;
    }

    double f2 = 0;
    double net[3] = {0, 0, 0};
    for (int i = 0; i < 2; ++i) {
        const double fxi = static_cast<double>(frame.span(kAtom_fx)[i]);
        const double fyi = static_cast<double>(frame.span(kAtom_fy)[i]);
        const double fzi = static_cast<double>(frame.span(kAtom_fz)[i]);
        std::printf("  F[%d]=(%.8f, %.8f, %.8f) eV/A\n", i, fxi, fyi, fzi);
        f2 += fxi * fxi + fyi * fyi + fzi * fzi;
        net[0] += fxi;
        net[1] += fyi;
        net[2] += fzi;
        if (!std::isfinite(fxi) || !std::isfinite(fyi) || !std::isfinite(fzi)) {
            std::printf("FAIL non-finite force\n");
            return 1;
        }
    }
    if (f2 < 1e-8) {
        std::printf("FAIL all forces ~0 at off-eq geometry\n");
        return 1;
    }
    for (int d = 0; d < 3; ++d) {
        if (std::fabs(net[d]) > 1e-6) {
            std::printf("FAIL net force[%d]=%.3e eV/A\n", d, net[d]);
            return 1;
        }
    }

    constexpr double h = 1e-4;
    constexpr double tol = 0.05; // eV/Å — same-element O–O

    // F_O0_x
    {
        auto fp = make_o2_offeq();
        fp.span(kAtom_x)[0] += static_cast<StorageReal>(h);
        auto fm = make_o2_offeq();
        fm.span(kAtom_x)[0] -= static_cast<StorageReal>(h);
        const double fd =
            -(energy_only_triplet(fp) - energy_only_triplet(fm)) / (2.0 * h);
        const double fx0 = static_cast<double>(frame.span(kAtom_fx)[0]);
        std::printf("  F_0_x analytic=%.8f  fd=%.8f  |d|=%.3e eV/A\n", fx0, fd,
                    std::fabs(fx0 - fd));
        if (std::fabs(fx0 - fd) > tol) {
            std::printf("FAIL F_0_x vs FD (tol %.3f eV/A)\n", tol);
            return 1;
        }
    }
    // F_O1_x
    {
        auto fp = make_o2_offeq();
        fp.span(kAtom_x)[1] += static_cast<StorageReal>(h);
        auto fm = make_o2_offeq();
        fm.span(kAtom_x)[1] -= static_cast<StorageReal>(h);
        const double fd =
            -(energy_only_triplet(fp) - energy_only_triplet(fm)) / (2.0 * h);
        const double fx1 = static_cast<double>(frame.span(kAtom_fx)[1]);
        std::printf("  F_1_x analytic=%.8f  fd=%.8f  |d|=%.3e eV/A\n", fx1, fd,
                    std::fabs(fx1 - fd));
        if (std::fabs(fx1 - fd) > tol) {
            std::printf("FAIL F_1_x vs FD (tol %.3f eV/A)\n", tol);
            return 1;
        }
    }

    std::printf("PASS\n");
    return 0;
}
