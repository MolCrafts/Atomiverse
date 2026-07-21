/// @file examples/13_mp2.cpp
/// @brief Closed-shell MP2 correlation energy on H2O/STO-3G via RHF method.

#include <iomanip>
#include <iostream>

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
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    auto mp2 = rhf.mp2(frame);
    std::cout << std::setprecision(10) << "E_SCF = " << rhf.energy()
              << " Ha\n"
              << "E_corr = " << mp2.e_corr << " Ha\n"
              << "E_MP2 = " << mp2.e_tot << " Ha\n";
    return 0;
}
