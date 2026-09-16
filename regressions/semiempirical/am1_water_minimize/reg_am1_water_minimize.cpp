/// @file regressions/semiempirical/am1_water_minimize/reg_am1_water_minimize.cpp
/// @brief AM1 water geometry step: Minimize + forces lowers energy and max|F|.
///
/// Method: AM1 (Dewar et al., J. Am. Chem. Soc. 107, 3902 (1985)).
/// Public API only: `cpu::minimizer::Minimize` + `AM1Evaluator` + LBFGS.
/// Exit non-zero on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"

namespace {

ATV::cpu::Frame make_distorted_water() {
    ATV::cpu::Frame f(3);
    f.span(ATV::kAtom_type)[0] = 8;
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_x)[1] = 1.05;
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_x)[2] = -0.26;
    f.span(ATV::kAtom_y)[2] = 1.0;
    f.span(ATV::kAtom_z)[2] = 0.0;
    for (int i = 0; i < 3; ++i) {
        f.span(ATV::kAtom_tag)[i] = i + 1;
        f.span(ATV::kAtom_mass)[i] =
            (f.span(ATV::kAtom_type)[i] == 8) ? 15.999 : 1.008;
    }
    f.box = ATV::cpu::Box::cubic(20.0);
    return f;
}

double max_force(const ATV::cpu::Frame& f) {
    double m = 0.0;
    for (int i = 0; i < f.nlocal(); ++i) {
        const double fx = f.span(ATV::kAtom_fx)[i];
        const double fy = f.span(ATV::kAtom_fy)[i];
        const double fz = f.span(ATV::kAtom_fz)[i];
        m = std::max(m, std::hypot(fx, fy, fz));
    }
    return m;
}

void evaluate_ef(ATV::cpu::Frame& frame) {
    ATV::cpu::Evaluate eval;
    eval.on(ATV::evaluate::Phase::evaluate)
        .add(ATV::cpu::semiempirical::AM1Evaluator{}.with_gradient(true));
    eval.build();
    eval.run(frame);
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = make_distorted_water();
    evaluate_ef(frame);
    const double e0 = frame.meta.get(kEnergy);
    const double f0 = max_force(frame);
    if (!(std::isfinite(e0) && e0 < 0.0) || !(f0 > 1.0e-4)) {
        std::fprintf(stderr, "FAIL: bad start state E=%g eV  max|F|=%g eV/Å\n",
                     e0, f0);
        return 1;
    }

    cpu::minimizer::Minimize opt;
    // Modest LBFGS: small steps so AM1 SCF stays stable; force gate loose.
    opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.02, 8});
    opt.add_criterion(cpu::ForceConverged{0.5, 0.3, 0.5, 0.3, 1e-3});
    opt.on(minimizer::Phase::force)
        .add(cpu::semiempirical::AM1Evaluator{}.with_gradient(true));
    opt.build();
    opt.run(frame, 15);

    evaluate_ef(frame);
    const double e1 = frame.meta.get(kEnergy);
    const double f1 = max_force(frame);

    std::printf("AM1 water minimize: E0=%.6f → E1=%.6f eV   "
                "max|F| %.4f → %.4f eV/Å  converged=%d  reason=%s\n",
                e0, e1, f0, f1, (int)opt.converged(),
                opt.stop_reason().c_str());

    if (!std::isfinite(e1) || !std::isfinite(f1)) {
        std::fprintf(stderr, "FAIL: non-finite final E or force\n");
        return 1;
    }
    // Geometry optimization must improve energy or residual force (or both).
    const bool energy_down = e1 < e0 - 1.0e-4;
    const bool force_down = f1 < f0 * 0.95;
    if (!energy_down && !force_down) {
        std::fprintf(stderr,
                     "FAIL: neither energy nor max|F| improved "
                     "(ΔE=%g, f1/f0=%g)\n",
                     e1 - e0, f1 / f0);
        return 1;
    }

    std::printf("PASS: AM1 Minimize improves water geometry "
                "(energy_down=%d force_down=%d)\n",
                (int)energy_down, (int)force_down);
    return 0;
}
