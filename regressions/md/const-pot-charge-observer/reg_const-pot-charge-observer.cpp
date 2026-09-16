/// @file regressions/md/const-pot-charge-observer/reg_const-pot-charge-observer.cpp
/// @brief E2E: electrode Q_tot vs offline LAMMPS reduce-sum pin (parallel plate).
///
/// Public API only. Fixed charges on two walls (constant-charge control of the
/// E1 foothold); ElectrodeChargeObserver measures Q_L and Q_R.
///
/// Pin: tools/metalwalls_oracle/inputs/lammps/const_pot_charge_observer.in
///   PIN_QL = +0.5 e, PIN_QR = -0.5 e
/// Oracle: LAMMPS 30 Mar 2026 (develop). See REFERENCE.md.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_addforce.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/electrode_charge.h"

using namespace ATV;

namespace {

// Offline pin (REFERENCE.md) — elementary charge [e].
constexpr double kPinQL = 0.5;
constexpr double kPinQR = -0.5;
constexpr double kTol = 1e-12;

cpu::Frame make_parallel_plate() {
    // Geometry twin of the LAMMPS input (10 atoms, box ~20 Å).
    cpu::Frame f(10);
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    auto type = f.span(kAtom_type);
    auto q = f.span(kAtom_charge);
    auto mass = f.span(kAtom_mass);
    auto tag = f.span(kAtom_tag);

    const double wall_x[4] = {2.0, 6.0, 10.0, 14.0};
    for (int i = 0; i < 4; ++i) {
        x[i] = wall_x[i];
        y[i] = 2.0;
        z[i] = 1.0;
        type[i] = 1;
        q[i] = 0.125;
        mass[i] = 12.0;
        tag[i] = i + 1;
    }
    for (int i = 0; i < 4; ++i) {
        const int j = 4 + i;
        x[j] = wall_x[i];
        y[j] = 2.0;
        z[j] = 9.0;
        type[j] = 2;
        q[j] = -0.125;
        mass[j] = 12.0;
        tag[j] = j + 1;
    }
    x[8] = 5.0;
    y[8] = 5.0;
    z[8] = 5.0;
    type[8] = 3;
    q[8] = 1.0;
    mass[8] = 23.0;
    tag[8] = 9;
    x[9] = 15.0;
    y[9] = 5.0;
    z[9] = 5.0;
    type[9] = 3;
    q[9] = 1.0;
    mass[9] = 23.0;
    tag[9] = 10;

    f.box = cpu::Box::orthorhombic(20.0, 10.0, 10.0);
    return f;
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
    auto frame = make_parallel_plate();
    cpu::ElectrodeChargeObserver obs({1, 2});

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_evaluator(cpu::md::FixAddForce{0.0, 0.0, 0.0, /*type=*/1});
    sim.set_observer_ref(obs);
    sim.build();
    sim.run(frame, 1);

    bool ok = true;
    ok = check("Q_L", obs.total_charge(0), kPinQL, kTol) && ok;
    ok = check("Q_R", obs.total_charge(1), kPinQR, kTol) && ok;
    ok = check("meta Q_L", frame.meta.get(kElectrodeCharge)[0], kPinQL, kTol)
         && ok;
    ok = check("meta Q_R", frame.meta.get(kElectrodeCharge)[1], kPinQR, kTol)
         && ok;
    // Neutrality of the two electrodes alone (electrolyte excluded).
    ok = check("Q_L+Q_R", obs.total_charge(0) + obs.total_charge(1), 0.0, kTol)
         && ok;

    if (!ok) {
        std::cerr << "const-pot-charge-observer: FAILED\n";
        return 1;
    }
    std::cout << "const-pot-charge-observer: all checks passed\n";
    return 0;
}
