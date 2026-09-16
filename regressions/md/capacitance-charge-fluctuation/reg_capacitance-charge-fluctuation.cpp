/// @file regressions/md/capacitance-charge-fluctuation/reg_capacitance-charge-fluctuation.cpp
/// @brief E2E: C_diff / C_int vs literature charge-fluctuation pin (parallel plate).
///
/// Public API only. Synthetic electrode-Q sample series (proxy for const-pot
/// MD charge time series); CapacitanceChargeFluctuation estimates capacitance.
///
/// Pin: REFERENCE.md — Limmer / Scalfi / MetalWalls fluctuation formula.
///   ⟨Q⟩=0.5 e, Var=0.01 e², C_diff=0.01/(kB·300) e/V, C_int=0.5 e/V
/// Oracle: literature (no MW/LAMMPS twin binary).

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/fix_addforce.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/capacitance_charge_fluctuation.h"

using namespace ATV;

namespace {

// Offline pin (REFERENCE.md).
constexpr double kPinMeanQ = 0.5;
constexpr double kPinVarQ = 0.01;
constexpr double kPinCInt = 0.5;
constexpr double kTol = 1e-12;

// Q_L samples [e] from REFERENCE.md
constexpr double kQL[] = {0.4, 0.6, 0.4, 0.6};
constexpr int kNSamples = 4;

cpu::Frame make_parallel_plate() {
    cpu::Frame f(8);
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = static_cast<double>(i);
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = 0.0;
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_charge)[i] = 0.0;
        f.span(kAtom_mass)[i] = 12.0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    for (int i = 4; i < 8; ++i) {
        f.span(kAtom_x)[i] = static_cast<double>(i - 4);
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = 10.0;
        f.span(kAtom_type)[i] = 2;
        f.span(kAtom_charge)[i] = 0.0;
        f.span(kAtom_mass)[i] = 12.0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    return f;
}

void set_wall_total_Q(cpu::Frame& f, double QL, double QR) {
    const double qL = QL / 4.0;
    const double qR = QR / 4.0;
    for (int i = 0; i < 4; ++i)
        f.span(kAtom_charge)[i] = qL;
    for (int i = 4; i < 8; ++i)
        f.span(kAtom_charge)[i] = qR;
}

bool check(const char* name, double got, double want, double tol) {
    if (std::abs(got - want) > tol) {
        std::cerr << "FAIL " << name << ": got " << got << " want " << want
                  << " tol " << tol << '\n';
        return false;
    }
    std::cout << "PASS " << name << ": " << got << " (tol " << tol << ")\n";
    return true;
}

} // namespace

int main() {
    constexpr double T = 300.0;
    constexpr double dPsi = 1.0;
    const double kPinCDiff = kPinVarQ / (units::kB_ev_per_k * T);

    auto frame = make_parallel_plate();
    cpu::CapacitanceChargeFluctuation obs({1, 2}, T, dPsi);

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_evaluator(cpu::md::FixAddForce{0.0, 0.0, 0.0, /*type=*/1});
    sim.set_observer_ref(obs);
    sim.build();
    for (int s = 0; s < kNSamples; ++s) {
        set_wall_total_Q(frame, kQL[s], -kQL[s]);
        sim.run(frame, 1);
    }

    bool ok = true;
    ok = check("n_samples", static_cast<double>(obs.n_samples()),
               static_cast<double>(kNSamples), 0.0)
         && ok;
    ok = check("mean_Q", obs.mean_charge(), kPinMeanQ, kTol) && ok;
    ok = check("var_Q", obs.charge_variance(), kPinVarQ, kTol) && ok;
    ok = check("C_diff", obs.differential_capacitance(), kPinCDiff, kTol) && ok;
    ok = check("C_int", obs.integral_capacitance(), kPinCInt, kTol) && ok;
    ok = check("meta C_diff", frame.meta.get(kCapacitanceDiff), kPinCDiff, kTol)
         && ok;
    ok = check("meta C_int", frame.meta.get(kCapacitanceIntegral), kPinCInt,
               kTol)
         && ok;

    // Cross-check pin numeric against REFERENCE.md printed constant.
    constexpr double kRefPrinted = 0.38681727072485284;
    ok = check("C_diff vs REFERENCE printed", kPinCDiff, kRefPrinted, 1e-14)
         && ok;

    if (!ok) {
        std::cerr << "capacitance-charge-fluctuation: FAILED\n";
        return 1;
    }
    std::cout << "capacitance-charge-fluctuation: all checks passed\n";
    return 0;
}
