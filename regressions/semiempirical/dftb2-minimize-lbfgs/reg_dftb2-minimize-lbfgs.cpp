/// @file regressions/semiempirical/dftb2-minimize-lbfgs/reg_dftb2-minimize-lbfgs.cpp
/// @brief SCC-DFTB2 forces driving L-BFGS to the fmax = 0.05 eV/Å gate.
///
/// Domain: L-BFGS on the analytic SCC-DFTB2 gradient (Elstner et al., Phys.
/// Rev. B 58, 7260 (1998)) in a periodic cell, so the Ewald γ gradient is on
/// the optimisation path and not just on a single-point one.
///
/// The convergence threshold is the one the DFTB+ protocol in Hooven et al.,
/// J. Chem. Theory Comput. 22, 6917 (2026) specifies for its n=8 relaxation
/// (`GeometryOptimisation { Optimiser = LBFGS {}; Convergence = { GradElem
/// [eV/angstrom] = 0.05 } }`), which is the reason this number and not another.
///
/// Not a DFTB+ geometry pin: two L-BFGS implementations reach different points
/// on the same basin. What is gated is that the loop converges by its own
/// criterion and that the criterion means what it says — `ForceConverged`
/// measures the max force *vector magnitude*, not the max Cartesian component.
/// Public API only. <= 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

// 2 waters in a 6.0 Å cell, bond lengths and angles pushed off equilibrium.
constexpr int kNat = 6;
constexpr int kZnum[] = {8, 1, 1, 8, 1, 1};
constexpr double kPx[] = {0.00, 1.06, -0.30, 3.00, 4.06, 2.70};
constexpr double kPy[] = {0.00, 0.00, 1.02, 0.00, 0.00, 1.02};
constexpr double kPz[] = {0.00, 0.15, -0.10, 2.20, 2.35, 2.10};

/// The protocol's gate [eV/Å].
constexpr double kFmax = 0.05;

cpu::Frame make_cell() {
    cpu::Frame f(kNat);
    for (int i = 0; i < kNat; ++i) {
        f.span(kAtom_type)[i] = kZnum[i];
        f.span(kAtom_x)[i] = kPx[i];
        f.span(kAtom_y)[i] = kPy[i];
        f.span(kAtom_z)[i] = kPz[i];
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = (kZnum[i] == 8) ? 15.999 : 1.008;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(6.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

cpu::semiempirical::DFTB2 make_method() {
    cpu::semiempirical::DFTB2 d;
    d.boundary = cpu::semiempirical::DftbBoundary::lattice;
    d.scc_tol = 1e-9;
    d.max_scc = 200;
    d.with_forces = true;
    return d;
}

/// Max force **vector magnitude** [eV/Å] — the quantity `ForceConverged`
/// judges (`lbfgs.cpp` publishes sqrt(fx²+fy²+fz²), not max|f_component|).
double max_force(const cpu::Frame& f) {
    double m = 0;
    for (int i = 0; i < f.nlocal(); ++i) {
        const double fx = f.span(kAtom_fx)[i];
        const double fy = f.span(kAtom_fy)[i];
        const double fz = f.span(kAtom_fz)[i];
        m = std::max(m, std::sqrt(fx * fx + fy * fy + fz * fz));
    }
    return m;
}

/// Independent single point, so the reported forces are not the optimizer's
/// own last internal evaluation.
void evaluate_ef(cpu::Frame& f) {
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < f.nlocal(); ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add(make_method());
    eval.build();
    eval.run(f);
}

} // namespace

int main() {
    cpu::Frame frame = make_cell();
    evaluate_ef(frame);
    const double e0 = frame.meta.get(kEnergy);
    const double f0 = max_force(frame);
    if (!(std::isfinite(e0) && e0 < 0.0) || !(f0 > kFmax)) {
        std::printf("FAIL bad start state: E = %g eV, max|F| = %g eV/A"
                    " (must start above the %.2f eV/A gate)\n",
                    e0, f0, kFmax);
        return 1;
    }

    auto dftb = make_method();
    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.02, 10});
    // The protocol's force gate. The displacement / energy legs of
    // ForceConverged are left at values loose enough that the *force* is what
    // stops the loop — this case is about the fmax criterion, not about
    // whichever leg happens to trip first.
    opt.add_criterion(cpu::ForceConverged{kFmax, kFmax, 1.0, 1.0, 1e-2});
    opt.set_evaluator_ref(dftb);
    opt.build();
    const int iters = opt.run(frame, 60);

    evaluate_ef(frame);
    const double e1 = frame.meta.get(kEnergy);
    const double f1 = max_force(frame);

    std::printf("dftb2-minimize-lbfgs: 2 H2O, periodic 6.0 A, fmax gate"
                " %.2f eV/A\n",
                kFmax);
    std::printf("  E     %.6f -> %.6f eV   (dE = %+.6f)\n", e0, e1, e1 - e0);
    std::printf("  max|F| %.4f -> %.4f eV/A\n", f0, f1);
    std::printf("  iterations = %d   converged = %d   reason = %s\n", iters,
                static_cast<int>(opt.converged()), opt.stop_reason().c_str());

    if (!std::isfinite(e1) || !std::isfinite(f1)) {
        std::printf("FAIL non-finite final energy or force\n");
        return 1;
    }
    if (!opt.converged()) {
        std::printf("FAIL loop did not converge within 60 iterations\n");
        return 1;
    }
    // The criterion has to mean what it says: re-measured independently, the
    // residual force must be under the gate it claimed to reach.
    if (!(f1 <= kFmax)) {
        std::printf("FAIL converged() true but max|F| = %.4f > %.2f eV/A\n", f1,
                    kFmax);
        return 1;
    }
    if (!(e1 < e0)) {
        std::printf("FAIL energy did not decrease\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
