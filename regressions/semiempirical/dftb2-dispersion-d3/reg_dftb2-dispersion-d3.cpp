/// @file regressions/semiempirical/dftb2-dispersion-d3/reg_dftb2-dispersion-d3.cpp
/// @brief Water-dimer DFTB2 + D3(BJ) — E_disp vs offline s-dftd3 pin.
///
/// Oracle: Simple-DFTD3 1.2.1 (literature D3 model; see REFERENCE.md).
/// Public API only. Exit non-zero on failure. ≤ 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

// Offline s-dftd3 1.2.1 --bj dftb3 on the geometry below [eV].
constexpr double kEDispPinEv = -0.04596788515696838;
constexpr double kEDispTolEv = 1e-4;

cpu::Frame make_dimer() {
    cpu::Frame f(6);
    const double xyz[6][3] = {
        {0.000, 0.000, 0.000}, {0.757, 0.000, 0.586}, {-0.757, 0.000, 0.586},
        {0.000, 0.000, 2.900}, {0.757, 0.000, 3.486}, {-0.757, 0.000, 3.486},
    };
    const int Z[6] = {8, 1, 1, 8, 1, 1};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_type)[i] = Z[i];
        f.span(kAtom_x)[i] = static_cast<StorageReal>(xyz[i][0]);
        f.span(kAtom_y)[i] = static_cast<StorageReal>(xyz[i][1]);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(xyz[i][2]);
        f.span(kAtom_mass)[i] = (Z[i] == 8) ? 15.999 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

} // namespace

int main() {
    auto f_off = make_dimer();
    cpu::Evaluate eval_off;
    cpu::semiempirical::DFTB2 d_off;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    d_off.boundary = cpu::semiempirical::DftbBoundary::cluster;
    d_off.d3 = false;
    d_off.with_forces = false;
    d_off.scc_tol = 1e-10;
    eval_off.on(evaluate::Phase::evaluate).add_ref(d_off);
    eval_off.build();
    eval_off.run(f_off);
    const double e_off = f_off.meta.get(kEnergy);
    std::printf("DFTB2 water dimer (no D3): E=%.10f eV\n", e_off);

    auto f_on = make_dimer();
    cpu::Evaluate eval_on;
    cpu::semiempirical::DFTB2 d_on;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    d_on.boundary = cpu::semiempirical::DftbBoundary::cluster;
    d_on.d3 = true;
    d_on.with_forces = false;
    d_on.scc_tol = 1e-10;
    eval_on.on(evaluate::Phase::evaluate).add_ref(d_on);
    eval_on.build();
    eval_on.run(f_on);
    const double e_on = f_on.meta.get(kEnergy);
    const double e_disp = d_on.last_e_disp();
    std::printf("DFTB2+D3 water dimer: E=%.10f eV  E_disp=%.12f eV\n", e_on,
                e_disp);
    std::printf("  pin E_disp=%.12f eV  |Δ|=%.3e eV\n", kEDispPinEv,
                std::fabs(e_disp - kEDispPinEv));

    if (!std::isfinite(e_on) || !std::isfinite(e_disp)
        || !std::isfinite(e_off)) {
        std::printf("FAIL non-finite energy\n");
        return 1;
    }
    if (std::fabs(e_on) < 1e-3) {
        std::printf("FAIL total energy near zero\n");
        return 1;
    }
    if (e_disp >= 0.0) {
        std::printf("FAIL E_disp not attractive\n");
        return 1;
    }
    if (std::fabs(e_disp - kEDispPinEv) > kEDispTolEv) {
        std::printf("FAIL E_disp vs s-dftd3 pin (tol %.1e eV)\n", kEDispTolEv);
        return 1;
    }
    const double de = e_on - e_off;
    if (std::fabs(de - e_disp) > 1e-8) {
        std::printf("FAIL E_tot(d3)-E_tot(off)=%.12e != E_disp=%.12e\n", de,
                    e_disp);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
