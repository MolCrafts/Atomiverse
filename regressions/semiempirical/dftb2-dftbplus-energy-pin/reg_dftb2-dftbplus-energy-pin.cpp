/// @file regressions/semiempirical/dftb2-dftbplus-energy-pin/reg_dftb2-dftbplus-energy-pin.cpp
/// @brief H2 SCC-DFTB2 total energy vs offline DFTB+ 24.1 / mio-1-1 pin.
///
/// oracle: dftb+  oracle_version: 24.1
/// Public API only. Exit non-zero on failure. <= 60 s.
#include <chrono>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

// Offline DFTB+ 24.1 + mio-1-1 Total energy [eV] at r=0.74 Å
// (detailed.out: -0.6749509324 H × 27.211386245988).
constexpr double kPin_eV = -18.36635052;
constexpr double kTol_eV = 1.0e-3;

cpu::Frame make_h2(double r_A) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r_A;
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
    return f;
}

} // namespace

int main() {
    auto frame = make_h2(0.74);

    const auto t0 = std::chrono::steady_clock::now();
    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
    dftb.with_forces = false;
    dftb.scc_tol = 1e-8;
    dftb.max_scc = 50;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(frame);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();

    const double e = frame.meta.get(kEnergy);
    const double err = std::fabs(e - kPin_eV);
    std::printf(
        "DFTB2 H2 r=0.74 A: E_ATV=%.8f eV  E_pin=%.8f eV  |dE|=%.3e eV\n", e,
        kPin_eV, err);
    std::printf(
        "wall_atv_s=%.6f  (oracle cold ~0.15 s; ratio informative only)\n",
        wall_s);

    if (!std::isfinite(e)) {
        std::printf("FAIL non-finite energy\n");
        return 1;
    }
    if (err > kTol_eV) {
        std::printf("FAIL |E-pin| > %.1e eV\n", kTol_eV);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
