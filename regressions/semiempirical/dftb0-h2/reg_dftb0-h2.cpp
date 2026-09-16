/// @file regressions/semiempirical/dftb0-h2/reg_dftb0-h2.cpp
/// @brief H2 non-SCC DFTB0 total energy vs offline DFTB+ 24.1 / mio-1-1 pin.
///
/// oracle: dftb+  oracle_version: 24.1
/// Public API only. Exit non-zero on failure. <= 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb0.h"

using namespace ATV;

// Offline DFTB+ 24.1 + mio-1-1 Total energy [eV] at r=0.74 Å
// (detailed.out: -0.6749509324 H × 27.211386245988; band -0.6807883427 H,
// repulsive +0.0058374104 H, SCC 0 because Δq = 0).
//
// This is an SCC-DFTB2 run used as a **DFTB0** oracle: H2 is homonuclear, so
// the Mulliken fluctuations vanish by symmetry (q_A = q_B = 0), the DFTB2
// second-order term is identically zero, and DFTB2 reduces exactly to
// E = Σ_i 2ε_i + E_rep — the DFTB0 expression. See REFERENCE.md.
static constexpr double kE_H2_074_eV = -18.36635052;
static constexpr double kE_tol_eV = 1.0e-3;

static double energy_at(double r, bool forces) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(30.0);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;

    cpu::semiempirical::DFTB0 dftb0;
    dftb0.with_forces = forces;
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb0));
    eval.build();
    eval.run(f);
    return f.meta.get(kEnergy);
}

int main() {
    const double e_eq = energy_at(0.74, false);
    const double e_far = energy_at(3.0, false);
    std::printf("DFTB0 H2: E(0.74)=%.10f E(3.0)=%.10f eV\n", e_eq, e_far);

    if (!std::isfinite(e_eq) || !std::isfinite(e_far)) {
        std::printf("FAIL non-finite\n");
        return 1;
    }
    if (std::fabs(e_eq - e_far) < 1e-6) {
        std::printf("FAIL energies identical at two separations\n");
        return 1;
    }
    std::printf("DFTB0 H2 r=0.74 A: E_ATV=%.10f eV  E_pin=%.8f eV  |dE|=%.3e eV\n",
                e_eq, kE_H2_074_eV, std::fabs(e_eq - kE_H2_074_eV));
    if (std::fabs(e_eq - kE_H2_074_eV) > kE_tol_eV) {
        std::printf("FAIL energy pin: got %.10f expected %.10f ± %.1e eV\n",
                    e_eq, kE_H2_074_eV, kE_tol_eV);
        return 1;
    }

    // Force skeleton: non-zero along bond at stretched geometry.
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 0.90;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(30.0);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    cpu::semiempirical::DFTB0 dftb0;
    dftb0.with_forces = true;
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb0));
    eval.build();
    eval.run(f);
    const double f2 = static_cast<double>(f.span(kAtom_fx)[0] * f.span(kAtom_fx)[0]
                                          + f.span(kAtom_fy)[0] * f.span(kAtom_fy)[0]
                                          + f.span(kAtom_fz)[0] * f.span(kAtom_fz)[0]);
    if (!(f2 > 0.0) || !std::isfinite(f2)) {
        std::printf("FAIL force skeleton |F|^2=%g\n", f2);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
