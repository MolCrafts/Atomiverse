/// @file regressions/multiscale/qmmm_opt_h2_ar/reg_qmmm_opt_h2_ar.cpp
/// @brief END-TO-END: QM/MM geometry optimization via composition
///        (Minimize + QmmmMechanical + DFTB2), AIMD-shaped set_evaluator.
///
/// H2 is QM (DFTB2); Ar is MM with repulsive H–Ar LJ at short range. Start with
/// Ar too close; L-BFGS must push it out and meet ForceConverged. Literature
/// pin is the LJ force direction (repulsive for r < r*), not an absolute energy.
/// Public headers only. See REFERENCE.md.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/column.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"
#include "atomiverse/cpu/multiscale/qmmm_mechanical.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

int main() {
    const double eps = 0.01;  // eV
    const double sigma = 3.4; // Å
    const double cutoff = 12.0;
    const double r_star = sigma * std::pow(2.0, 1.0 / 6.0); // ~3.81 Å
    const double r0 = 2.2; // Å — deep in repulsive wall of H–Ar LJ

    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_x)[2] = r0; // Ar along x, near H2
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_y)[i] = 0;
        f.span(kAtom_z)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = (i < 2) ? 1.0 : 40.0;
        f.span(kAtom_charge)[i] = 0;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 18;
    f.box = cpu::Box::cubic(40.0);
    f.blocks.at("atoms").insert(
        "qm_z", cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                                kDynamicFieldId});
    auto qm = f.span(cpu::kAtomQmZ);
    qm[0] = 1;
    qm[1] = 1;
    qm[2] = 0;

    cpu::QmmmMechanical mech{{{1, 1, eps, sigma}, {18, 18, eps, sigma}},
                             cutoff};
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = true;
    mech.set_engine(std::move(dftb));

    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::LBFGS{8, 0.05, 20});
    opt.add_criterion(cpu::ForceConverged{0.05, 0.03, 0.1, 0.05, 1e-4});
    opt.set_evaluator(std::move(mech));
    opt.set_threads(1);
    opt.build();
    opt.run(f, 80);

    const double x_ar = f.span(kAtom_x)[2];
    const double r_h0 = std::hypot(x_ar - f.span(kAtom_x)[0],
                                   f.span(kAtom_y)[2], f.span(kAtom_z)[2]);
    const double r_h1 = std::hypot(x_ar - f.span(kAtom_x)[1],
                                   f.span(kAtom_y)[2], f.span(kAtom_z)[2]);
    const double r_ar = std::min(r_h0, r_h1);
    double fmax = 0;
    for (int i = 0; i < 3; ++i) {
        fmax =
            std::max(fmax, std::hypot(f.span(kAtom_fx)[i], f.span(kAtom_fy)[i],
                                      f.span(kAtom_fz)[i]));
    }
    const double E = f.meta.get(kEnergy);

    std::printf("qmmm_opt_h2_ar r0=%.3f r_ar=%.4f r*=%.3f E=%.6e fmax=%.3e "
                "converged=%d reason=%s\n",
                r0, r_ar, r_star, E, fmax, (int)opt.converged(),
                opt.stop_reason().c_str());

    if (!std::isfinite(E) || !std::isfinite(r_ar) || !std::isfinite(fmax)) {
        std::printf("FAIL: non-finite result\n");
        return 1;
    }
    // Repulsive wall: Ar must move outward toward / past the LJ well.
    if (!(r_ar > r0 + 0.3)) {
        std::printf("FAIL: Ar did not move out of the repulsive wall "
                    "(r_ar=%.4f, r0=%.3f)\n",
                    r_ar, r0);
        return 1;
    }
    if (!(r_ar > 2.8)) {
        std::printf("FAIL: r_ar=%.4f still far below LJ minimum scale\n", r_ar);
        return 1;
    }
    if (!opt.converged() && !(fmax < 0.08)) {
        std::printf("FAIL: not converged and residual force large "
                    "(fmax=%.3e reason=%s)\n",
                    fmax, opt.stop_reason().c_str());
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
