/// @file regressions/semiempirical/gfn2-xtb-water/reg_gfn2-xtb-water.cpp
/// @brief H2O GFN2-xTB energy vs archived xtb 6.7.1.
///
/// Domain: Bannwarth, Ehlert, Grimme, JCTC 15, 1652 (2019).
/// Oracle family: xtb. Public API only. Exit non-zero on failure. ≤ 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/gfn2.h"

using namespace ATV;

namespace {

// xtb 6.7.1 (edcfbbe) `mol.xyz --gfn 2 --chrg 0 --uhf 0`
// tools/tb_oracle/inputs/xtb/h2o_gfn2/xtb.out
constexpr double kPinEnergyEh = -5.070383095219;
// Transcript prints 12 decimals Eh; ATV writes eV with CODATA-2018 Hartree.
// 2 meV floor: missing quadrupole AES + D3 vs dexp CN (see REFERENCE.md).
constexpr double kPinTolEV = 2.0e-3;

cpu::Frame make_water() {
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.96;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = -0.24;
    f.span(kAtom_y)[2] = 0.93;
    f.span(kAtom_z)[2] = 0.0;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_mass)[i] = (i == 0) ? 15.999 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

} // namespace

int main() {
    auto frame = make_water();

    cpu::Evaluate eval;
    cpu::semiempirical::GFN2 gfn2;
    gfn2.with_forces = false;
    gfn2.scc_tol = 1e-8;
    eval.on(evaluate::Phase::evaluate).add_ref(gfn2);
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    const double pin_ev = kPinEnergyEh * units::kHartreeInEv;
    std::printf("GFN2 H2O: E=%.12f eV  pin=%.12f eV  dE=%.6e eV\n", e, pin_ev,
                e - pin_ev);
    std::printf("  SCC residual=%.3e in %d iterations\n",
                gfn2.last_scc_residual(), gfn2.last_scc_iterations());
    if (!std::isfinite(e)) {
        std::printf("FAIL non-finite energy\n");
        return 1;
    }
    if (std::fabs(e) < 1.0) {
        std::printf("FAIL energy near zero\n");
        return 1;
    }
    if (!(gfn2.last_scc_residual() < gfn2.scc_tol)) {
        std::printf("FAIL SCC residual above scc_tol\n");
        return 1;
    }
    if (std::fabs(e - pin_ev) > kPinTolEV) {
        std::printf("FAIL energy pin: got %.12f expected %.12f (tol %g eV)\n",
                    e, pin_ev, kPinTolEV);
        return 1;
    }
    std::printf("PIN match: %.12f eV (tol %g)\n", pin_ev, kPinTolEV);
    std::printf("PASS\n");
    return 0;
}
