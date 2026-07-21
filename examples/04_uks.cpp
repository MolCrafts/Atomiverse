/// @file examples/04_uks.cpp
/// @brief Unrestricted Kohn–Sham via concrete `cpu::scf::UKS`.

#include <iomanip>
#include <iostream>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

static cpu::Frame make_he() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_type)[0] = 2;
    f.span(kAtom_mass)[0] = 4.0026;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

int main() {
    auto frame = make_he();
    cpu::scf::UKS uks(/*n_alpha=*/1, /*n_beta=*/1, cpu::scf::XCType::LSDA);
    uks.basis = cpu::scf::BasisSet::STO3G;
    uks.run(frame);
    std::cout << std::setprecision(10)
              << "He UKS-LSDA converged=" << uks.converged()
              << " E=" << uks.energy() << " Ha\n";
    return uks.converged() ? 0 : 1;
}
