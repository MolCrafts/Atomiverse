/// @file examples/14_fci.cpp
/// @brief Tiny full CI on H2/STO-3G via RHF method.

#include <iomanip>
#include <iostream>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

int main() {
    cpu::Frame frame(2);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_x)[1] = 0.74;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_z)[1] = 0.0;
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_mass)[0] = 1.008;
    frame.span(kAtom_mass)[1] = 1.008;
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0, /*tol=*/1e-12};
    auto fci = rhf.fci(frame);
    std::cout << std::setprecision(10) << "E_SCF = " << rhf.energy() << " Ha\n"
              << "E_FCI = " << fci.e_fci << " Ha\n"
              << "E_corr = " << fci.e_corr << " Ha\n";
    return 0;
}
