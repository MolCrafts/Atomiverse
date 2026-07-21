/// @file regressions/semiempirical/rm1_water_energy/reg_rm1_water_energy.cpp
/// @brief END-TO-END: RM1 water energy converges and differs from AM1.
///
/// See REFERENCE.md. Public API only.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"

using namespace ATV;
using namespace ATV::cpu::semiempirical;

static cpu::Frame make_water() {
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_mass)[0] = 15.999;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_mass)[2] = 1.008;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.9572;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_x)[2] = -0.239987;
    f.span(kAtom_y)[2] = 0.926627;
    f.span(kAtom_z)[2] = 0.0;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

int main() {
    auto frame = make_water();

    AM1Solver am1;
    am1.build(frame);
    const auto& ra = am1.run(frame);
    if (!ra.converged || !std::isfinite(ra.total_energy_ev)) {
        std::printf("FAIL: AM1 water did not converge\n");
        return 1;
    }

    RM1Solver rm1;
    rm1.build(frame);
    const auto& rr = rm1.run(frame);
    if (!rr.converged || !std::isfinite(rr.total_energy_ev)) {
        std::printf("FAIL: RM1 water did not converge\n");
        return 1;
    }
    if (!std::isfinite(rr.heat_of_formation_kcal_mol)) {
        std::printf("FAIL: RM1 ΔHf non-finite\n");
        return 1;
    }
    const double de = std::abs(rr.total_energy_ev - ra.total_energy_ev);
    if (!(de > 0.05)) {
        std::printf("FAIL: |E_RM1-E_AM1|=%.6g eV (expected > 0.05)\n", de);
        return 1;
    }

    std::printf(
        "ok: E_RM1=%.6g eV  E_AM1=%.6g eV  |dE|=%.4g  dHf=%.4g kcal/mol\n",
        rr.total_energy_ev, ra.total_energy_ev, de,
        rr.heat_of_formation_kcal_mol);
    std::printf("PASS: rm1 water energy (converge + method difference)\n");
    return 0;
}
