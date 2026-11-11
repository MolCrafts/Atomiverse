/// @file examples/cpp/03_ionization.cpp
/// @brief Vertical ionization energy of water: Koopmans vs ΔSCF.
///
/// One concept: two ways to get an ionization energy from SCF.
///   - Koopmans: IP ≈ −ε_HOMO of the neutral RHF determinant (frozen orbitals).
///   - ΔSCF:     IP = E(cation, UHF) − E(neutral, RHF) at the same geometry,
///               which lets the orbitals relax after the electron is removed.
/// Orbital relaxation makes the ΔSCF IP smaller than the Koopmans estimate.
///
/// Build:  -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON, target `03_ionization`

#include <iomanip>
#include <iostream>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/methods.h"
#include "molecules.h"

using namespace ATV;

int main() {
    cpu::Frame frame = examples::water();

    cpu::scf::RHF neutral{cpu::scf::BasisSet::STO3G, /*charge=*/0,
                          /*tol=*/1e-10};
    neutral.run(frame);

    using Spin = cpu::scf::SlaterDeterminant::Spin;
    const auto& det = neutral.determinant();
    const int homo = det.n_electrons(Spin::alpha) - 1;
    const double koopmans_ev =
        -det.orbital_energies(Spin::alpha)(homo) * units::kHartreeInEv;

    // H2O+ has 9 electrons: 5 alpha, 4 beta.
    cpu::scf::UHF cation{cpu::scf::BasisSet::STO3G, /*n_alpha=*/5, /*n_beta=*/4,
                         /*charge=*/1};
    cation.tol = 1e-10;
    cation.run(frame);
    const double delta_scf_ev =
        (cation.energy() - neutral.energy()) * units::kHartreeInEv;

    std::cout << std::fixed << std::setprecision(4)
              << "H2O / STO-3G vertical ionization energy\n"
              << "Koopmans (-e_HOMO) : " << koopmans_ev << " eV\n"
              << "Delta-SCF          : " << delta_scf_ev << " eV\n";
    return (neutral.converged() && cation.converged()) ? 0 : 1;
}
