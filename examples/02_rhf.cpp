/// @file examples/02_rhf.cpp
/// @brief Tutorial example for restricted Hartree-Fock on molecular hydrogen.
///
/// End-to-end RHF workflow on CPU:
///   1. Build H2 geometry in cpu::Frame
///   2. Run concrete method `cpu::scf::RHF` (integrals are private workspace)
///   3. Print results and compare with textbook
///
/// Build:
///   cmake -B build -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON
///   cmake --build build --target 02_rhf
///
/// Expected total energy: about -1.1168 Hartree (Szabo & Ostlund, Table 3.7)

#include <cmath>
#include <iomanip>
#include <iostream>

#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

int main() {
    cpu::Frame frame(2);
    frame.span(ATV::kAtom_x)[0] = double(0);
    frame.span(ATV::kAtom_y)[0] = double(0);
    frame.span(ATV::kAtom_z)[0] = double(0);
    frame.span(ATV::kAtom_x)[1] = double(0.74);
    frame.span(ATV::kAtom_y)[1] = double(0);
    frame.span(ATV::kAtom_z)[1] = double(0);
    frame.span(ATV::kAtom_mass)[0] = double(1.008);
    frame.span(ATV::kAtom_mass)[1] = double(1.008);
    frame.span(ATV::kAtom_type)[0] = 1;
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.box = ATV::cpu::Box::cubic(100.0);

    std::cout << "=== H2 / STO-3G / RHF (CPU) ===\n"
              << "Geometry: H-H at 0.74 Angstrom\n\n";

    // Concrete method — parameters are constructor arguments; Frame is the
    // only I/O.
    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0,
                      /*tol=*/(sizeof(double) == sizeof(float)) ? double(1e-5)
                                                                : double(1e-10),
                      /*max_iter=*/100};

    rhf.run(frame); // eV energy on frame.meta[kEnergy]; Hartree on energy()

    std::cout << "=== RHF Result ===\n"
              << "Converged:        " << (rhf.converged() ? "yes" : "no")
              << " (" << rhf.iterations() << " iterations)\n"
              << std::fixed << std::setprecision(10)
              << "E_total:          " << rhf.energy() << " Hartree\n\n";

    using Spin = cpu::scf::SlaterDeterminant::Spin;
    const auto& eps = rhf.determinant().orbital_energies(Spin::alpha);
    std::cout << "Orbital energies:\n" << std::setprecision(6);
    for (int i = 0; i < static_cast<int>(eps.size()); ++i)
        std::cout << "  epsilon[" << i << "] = " << std::setw(12)
                  << static_cast<double>(eps(i)) << " Hartree\n";

    std::cout << "\n--- Textbook comparison (Szabo & Ostlund) ---\n"
              << "  Reference: -1.1168 Hartree\n"
              << std::setprecision(4) << "  This:      " << rhf.energy()
              << " Hartree\n";

    std::cout << "\n=== Done ===\n";
    return 0;
}
