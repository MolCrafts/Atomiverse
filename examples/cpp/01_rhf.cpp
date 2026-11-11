/// @file examples/cpp/01_rhf.cpp
/// @brief First program: restricted Hartree–Fock single point on H2.
///
/// One concept: a method is a concrete object whose parameters are
/// constructor arguments, and the `cpu::Frame` is its only input and output.
/// `RHF::run` writes the energy [eV] into the frame's metadata; the method
/// keeps the Hartree energy and the converged determinant for inspection.
///
/// Build:  -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON, target `01_rhf`
/// Expect: E ≈ −1.1168 Hartree (Szabo & Ostlund, Table 3.7)

#include <iomanip>
#include <iostream>

#include "atomiverse/cpu/scf/methods.h"
#include "molecules.h"

using namespace ATV;

int main() {
    cpu::Frame frame = examples::h2(0.74);

    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0, /*tol=*/1e-10,
                      /*max_iter=*/100};
    rhf.run(frame);

    std::cout << "H2 / STO-3G / RHF\n"
              << "converged : " << (rhf.converged() ? "yes" : "no") << " ("
              << rhf.iterations() << " iterations)\n"
              << std::fixed << std::setprecision(10)
              << "E_total   : " << rhf.energy() << " Hartree\n";

    using Spin = cpu::scf::SlaterDeterminant::Spin;
    const auto& eps = rhf.determinant().orbital_energies(Spin::alpha);
    std::cout << std::setprecision(6);
    for (int i = 0; i < static_cast<int>(eps.size()); ++i)
        std::cout << "epsilon[" << i << "] : " << std::setw(12) << eps(i)
                  << " Hartree\n";
    return rhf.converged() ? 0 : 1;
}
