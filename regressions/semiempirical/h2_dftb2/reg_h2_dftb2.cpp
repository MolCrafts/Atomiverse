/// @file regressions/semiempirical/h2_dftb2/reg_h2_dftb2.cpp
/// @brief H₂ SCC-DFTB2 absolute energy vs offline DFTB+ 24.1 (mio-1-1).
///
/// Same pin as reg_dftb2-dftbplus-energy-pin; keeps a short Evaluate-path
/// smoke case for the historical h2_dftb2 slug. See REFERENCE.md.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

// DFTB+ 24.1 Total Energy @ r=0.74 Å, mio-1-1 H–H, SCC (re-verified 2026-07-29).
constexpr double kPin_eV = -18.36635052;
constexpr double kTol_eV = 1e-6;

double energy_at(double r) {
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

    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
    dftb.with_forces = false;
    dftb.scc_tol = 1e-10;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(f);
    return f.meta.get(kEnergy);
}

} // namespace

int main() {
    const double e_eq = energy_at(0.74);
    const double e_far = energy_at(3.0);
    std::printf("DFTB2 H2: E(0.74)=%.10f E(3.0)=%.10f eV  pin=%.10f\n", e_eq,
                e_far, kPin_eV);
    if (!std::isfinite(e_eq) || !std::isfinite(e_far)) {
        std::printf("FAIL non-finite\n");
        return 1;
    }
    if (std::fabs(e_eq - kPin_eV) > kTol_eV) {
        std::printf("FAIL vs DFTB+ pin |dE|=%.3e eV (tol %g)\n",
                    std::fabs(e_eq - kPin_eV), kTol_eV);
        return 1;
    }
    if (std::fabs(e_eq - e_far) < 1e-6) {
        std::printf("FAIL energies identical at two separations\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
