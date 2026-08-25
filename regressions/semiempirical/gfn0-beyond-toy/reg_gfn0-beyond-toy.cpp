/// @file regressions/semiempirical/gfn0-beyond-toy/reg_gfn0-beyond-toy.cpp
/// @brief Methanol GFN0-xTB total energy vs offline xtb 6.7.1 `--gfn 0`.
///
/// oracle: xtb  oracle_version: 6.7.1
/// Public API only. Exit non-zero on failure. <= 60 s.
#include <chrono>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/gfn0.h"

using namespace ATV;

// Offline xtb 6.7.1 `--gfn 0 --chrg 0 --uhf 0 --acc 1.0` on mol.xyz
// (tools/tb_oracle/inputs/xtb/ch3oh_gfn0/). TOTAL ENERGY -7.597234498685 Eh.
static constexpr double kE_pin_Ha = -7.597234498685;
static constexpr double kE_tol_eV = 1.0e-3;

static cpu::Frame make_ch3oh() {
    // Same geometry as the oracle mol.xyz [Å].
    cpu::Frame f(6);
    const double xyz[6][3] = {
        {0.0000000000, 0.0000000000, 0.0000000000},  // C
        {1.4270000000, 0.0000000000, 0.0000000000},  // O
        {-0.3639000000, 1.0297000000, 0.0000000000}, // H
        {-0.3639000000, -0.5148500000, 0.8917500000},
        {-0.3639000000, -0.5148500000, -0.8917500000},
        {1.7505000000, 0.0000000000, 0.9059000000},
    };
    const int Z[6] = {6, 8, 1, 1, 1, 1};
    const double mass[6] = {12.011, 15.999, 1.008, 1.008, 1.008, 1.008};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_type)[i] = Z[i];
        f.span(kAtom_x)[i] = xyz[i][0];
        f.span(kAtom_y)[i] = xyz[i][1];
        f.span(kAtom_z)[i] = xyz[i][2];
        f.span(kAtom_mass)[i] = mass[i];
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(30.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

int main() {
    cpu::Frame f = make_ch3oh();
    cpu::semiempirical::GFN0 gfn0;
    gfn0.with_forces = false;
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add(std::move(gfn0));
    eval.build();
    const auto t0 = std::chrono::steady_clock::now();
    eval.run(f);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    const double e_ev = f.meta.get(kEnergy);
    const double e_ha = e_ev / units::kHartreeInEv;
    const double pin_ev = kE_pin_Ha * units::kHartreeInEv;
    std::printf("GFN0 CH3OH: E_ATV=%.10f eV (%.12f Eh)  E_pin=%.12f Eh  "
                "|dE|=%.3e eV  wall=%.3f ms\n",
                e_ev, e_ha, kE_pin_Ha, std::fabs(e_ev - pin_ev), ms);
    if (!std::isfinite(e_ev)) {
        std::printf("FAIL non-finite\n");
        return 1;
    }
    if (std::fabs(e_ev - pin_ev) > kE_tol_eV) {
        std::printf("FAIL energy pin: got %.10f expected %.10f ± %.1e eV\n",
                    e_ev, pin_ev, kE_tol_eV);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
