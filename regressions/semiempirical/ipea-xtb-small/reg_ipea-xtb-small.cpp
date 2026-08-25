/// @file regressions/semiempirical/ipea-xtb-small/reg_ipea-xtb-small.cpp
/// @brief CH4 IPEA-xTB energy vs archived xtb 6.7.1.
///
/// Domain: Ásgeirsson, Bauer, Grimme, Chem. Sci. 8, 4879 (2017).
/// Oracle family: xtb. Public API only. Exit non-zero on failure. ≤ 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/ipea.h"

using namespace ATV;

namespace {

// xtb 6.7.1 (2026-06-09) IPEA-xTB: XTBPATH with param_gfn1-xtb.txt ←
// param_ipea-xtb.txt; `mol.xyz --gfn 1 --chrg 0 --uhf 0 --sp`
// tools/tb_oracle/inputs/xtb/ch4_ipea/xtb.out.txt
constexpr double kPinEnergyEh = -4.670476737294;
constexpr double kPinTolEV = 0.05;

cpu::Frame make_ch4() {
    const double s = 1.09 / std::sqrt(3.0);
    cpu::Frame f(5);
    const double x[5] = {0, s, s, -s, -s};
    const double y[5] = {0, s, -s, s, -s};
    const double z[5] = {0, s, -s, -s, s};
    const int Z[5] = {6, 1, 1, 1, 1};
    for (int i = 0; i < 5; ++i) {
        f.span(kAtom_type)[i] = Z[i];
        f.span(kAtom_x)[i] = x[i];
        f.span(kAtom_y)[i] = y[i];
        f.span(kAtom_z)[i] = z[i];
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
    cpu::semiempirical::IPEA ipea;
    ipea.with_forces = false;
    ipea.scc_tol = 1e-8;
    eval.on(evaluate::Phase::evaluate).add_ref(ipea);
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    const double pin_ev = kPinEnergyEh * units::kHartreeInEv;
    std::printf("IPEA CH4: E=%.12f eV  pin=%.12f eV  dE=%.6e eV\n", e, pin_ev,
                e - pin_ev);
    std::printf("  SCC residual=%.3e in %d iterations\n",
                ipea.last_scc_residual(), ipea.last_scc_iterations());
    if (!std::isfinite(e)) {
        std::printf("FAIL non-finite energy\n");
        return 1;
    }
    if (std::fabs(e) < 1.0) {
        std::printf("FAIL energy near zero\n");
        return 1;
    }
    if (!(ipea.last_scc_residual() < ipea.scc_tol)) {
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
