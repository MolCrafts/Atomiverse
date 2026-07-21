/// @file examples/12_delta_scf.cpp
/// @brief ΔSCF vertical IP: neutral RHF vs cation UHF at fixed geometry.

#include <iomanip>
#include <iostream>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

static cpu::Frame make_water() {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.7572;
    f.span(kAtom_x)[2] = -0.7572;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_y)[2] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = -0.5860;
    f.span(kAtom_z)[2] = -0.5860;
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_mass)[0] = 15.999;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_mass)[2] = 1.008;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

int main() {
    auto frame = make_water();

    cpu::scf::RHF neutral;
    neutral.basis = cpu::scf::BasisSet::STO3G;
    neutral.tol = 1e-10;
    neutral.run(frame);
    const double e_n = neutral.energy();

    cpu::scf::UHF cation(/*n_alpha=*/5, /*n_beta=*/4); // H2O+ : 9 e-
    cation.basis = cpu::scf::BasisSet::STO3G;
    cation.charge = 1;
    cation.tol = 1e-10;
    cation.run(frame);
    const double e_c = cation.energy();

    const double ip_ev = (e_c - e_n) * units::kHartreeInEv;
    std::cout << std::setprecision(6) << "E_neutral = " << e_n << " Ha\n"
              << "E_cation  = " << e_c << " Ha\n"
              << "ΔSCF IP   = " << ip_ev << " eV\n";
    return 0;
}
