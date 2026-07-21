/// @file examples/03_uhf.cpp
/// @brief Unrestricted Hartree-Fock via concrete `cpu::scf::UHF`.

#include <iomanip>
#include <iostream>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

static cpu::Frame make_h() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

int main() {
    auto frame = make_h();
    cpu::scf::UHF uhf(/*n_alpha=*/1, /*n_beta=*/0);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.tol = 1e-8;
    uhf.run(frame);
    std::cout << std::setprecision(10)
              << "H atom UHF converged=" << uhf.converged()
              << " E=" << uhf.energy() << " Ha\n";
    return uhf.converged() ? 0 : 1;
}
