/// @file regressions/md/potential-profile-observer/reg_potential-profile-observer.cpp
/// @brief E2E: planar Ψ(z) vs literature Poisson pin (parallel-plate sheets).
///
/// Public API only. Synthetic two-sheet capacitor; PotentialProfile integrates
/// planar Poisson for ρ(z) → E_z(z), Ψ(z).
///
/// Pin: REFERENCE.md — E_gap = 4π ke σ, Ψ_drop = −E_gap × 9 Å.
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
#include "atomiverse/cpu/observe/potential_profile.h"

using namespace ATV;

namespace {

// Offline pin (REFERENCE.md).
constexpr double kPinEGap = 1.80951275785204;
constexpr double kPinPsiDrop = -16.28561482066836;
constexpr double kPinRhoLeft = 0.01;
constexpr double kPinRhoRight = -0.01;
constexpr double kTol = 1e-12;

cpu::Frame make_two_sheet() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 5.0;
    f.span(kAtom_y)[0] = 5.0;
    f.span(kAtom_z)[0] = 5.5;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_mass)[0] = 12.0;
    f.span(kAtom_tag)[0] = 1;

    f.span(kAtom_x)[1] = 5.0;
    f.span(kAtom_y)[1] = 5.0;
    f.span(kAtom_z)[1] = 14.5;
    f.span(kAtom_type)[1] = 2;
    f.span(kAtom_charge)[1] = -1.0;
    f.span(kAtom_mass)[1] = 12.0;
    f.span(kAtom_tag)[1] = 2;

    f.box = cpu::Box::orthorhombic(10.0, 10.0, 20.0);
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
    // Continuum identity cross-check vs units::coulomb.
    const double E_cont = 4.0 * units::pi * units::coulomb * (1.0 / 100.0);
    bool ok =
        check("continuum E_gap vs REFERENCE printed", E_cont, kPinEGap, 1e-14);

    auto frame = make_two_sheet();
    cpu::PotentialProfile obs(/*n_bins=*/20, /*z_min=*/0.0, /*z_max=*/20.0);

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_evaluator(cpu::md::FixAddForce{0.0, 0.0, 0.0, /*type=*/1});
    sim.set_observer_ref(obs);
    sim.build();
    sim.run(frame, 1);

    ok = check("E mid-gap bin9", obs.field()[9], kPinEGap, kTol) && ok;
    ok = check("E vacuum left", obs.field()[0], 0.0, kTol) && ok;
    ok = check("E vacuum right", obs.field()[19], 0.0, kTol) && ok;
    ok = check("psi bin0", obs.psi()[0], 0.0, kTol) && ok;
    ok = check("psi bin19", obs.psi()[19], kPinPsiDrop, kTol) && ok;
    ok = check("potential_drop", obs.potential_drop(), kPinPsiDrop, kTol) && ok;
    ok = check("meta potential_drop", frame.meta.get(kPotentialDrop),
               kPinPsiDrop, kTol)
         && ok;
    ok = check("rho left bin5", obs.rho()[5], kPinRhoLeft, kTol) && ok;
    ok = check("rho right bin14", obs.rho()[14], kPinRhoRight, kTol) && ok;
    ok = check("psi_drop = -E*9", obs.potential_drop(), -kPinEGap * 9.0, kTol)
         && ok;

    if (!ok) {
        std::cerr << "potential-profile-observer: FAILED\n";
        return 1;
    }
    std::cout << "potential-profile-observer: all checks passed\n";
    return 0;
}
