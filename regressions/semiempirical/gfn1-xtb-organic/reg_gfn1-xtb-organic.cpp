/// @file regressions/semiempirical/gfn1-xtb-organic/reg_gfn1-xtb-organic.cpp
/// @brief CH4 GFN1-xTB energy vs offline xtb 6.7.1 --gfn 1 pin.
///
/// Oracle: xtb 6.7.1. See REFERENCE.md. Public API only. <= 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/gfn1.h"

using namespace ATV;

namespace {

// Offline xtb 6.7.1 --gfn 1, tetrahedral CH4 r(C-H)=1.09 Å (Ha * CODATA eV).
constexpr double kPin_eV = -116.30778398;
// 20 meV floor: 10.2 meV measured vs xtb 6.7.1 (see REFERENCE.md).
constexpr double kTol_eV = 0.02;

cpu::Frame make_ch4() {
    cpu::Frame f(5);
    f.span(kAtom_type)[0] = 6;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    const double s = 1.09 / std::sqrt(3.0);
    const double hpos[4][3] = {
        {s, s, s},
        {s, -s, -s},
        {-s, s, -s},
        {-s, -s, s},
    };
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_type)[i + 1] = 1;
        f.span(kAtom_x)[i + 1] = hpos[i][0];
        f.span(kAtom_y)[i + 1] = hpos[i][1];
        f.span(kAtom_z)[i + 1] = hpos[i][2];
    }
    for (int i = 0; i < 5; ++i) {
        f.span(kAtom_mass)[i] = (i == 0) ? 12.011 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

} // namespace

int main() {
    auto frame = make_ch4();

    cpu::Evaluate eval;
    cpu::semiempirical::GFN1 gfn1;
    gfn1.with_forces = false;
    gfn1.scc_tol = 1e-8;
    gfn1.max_scc = 80;
    eval.on(evaluate::Phase::evaluate).add_ref(gfn1);
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    std::printf("GFN1 CH4: E=%.8f eV  pin=%.8f eV  |dE|=%.6e\n", e, kPin_eV,
                std::fabs(e - kPin_eV));
    std::printf("  SCC residual=%.3e in %d iterations\n",
                gfn1.last_scc_residual(), gfn1.last_scc_iterations());
    if (!std::isfinite(e)) {
        std::printf("FAIL non-finite energy\n");
        return 1;
    }
    if (!(gfn1.last_scc_residual() < gfn1.scc_tol)) {
        std::printf("FAIL SCC residual above scc_tol\n");
        return 1;
    }
    if (std::fabs(e - kPin_eV) > kTol_eV) {
        std::printf("FAIL |E - pin| > %.3f eV\n", kTol_eV);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
