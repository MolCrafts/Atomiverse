/// @file regressions/semiempirical/dftb2-3ob-organic/reg_dftb2-3ob-organic.cpp
/// @brief CH4 SCC-DFTB2 with static 3ob-3-1 SK — energy vs offline DFTB+ pin.
///
/// Oracle: DFTB+ 24.1, SCC-DFTB2 (no ThirdOrder), 3ob-3-1 SK. See REFERENCE.md.
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

// Offline DFTB+ 24.1 pin (Ha * 27.211386245988) — see REFERENCE.md.
constexpr double kPin_eV = -87.86856831;
constexpr double kTol_eV = 0.05;

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
    cpu::semiempirical::DFTB2 dftb;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
    dftb.sk_set = cpu::semiempirical::DftbSkSet::ThreeOb;
    dftb.with_forces = false;
    dftb.scc_tol = 1e-10;
    dftb.max_scc = 80;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    std::printf("DFTB2 ThreeOb CH4: E=%.8f eV  pin=%.8f eV  |dE|=%.6e\n", e,
                kPin_eV, std::fabs(e - kPin_eV));
    if (!std::isfinite(e)) {
        std::printf("FAIL non-finite energy\n");
        return 1;
    }
    if (std::fabs(e - kPin_eV) > kTol_eV) {
        std::printf("FAIL |E - pin| > %.3f eV\n", kTol_eV);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
