/// @file examples/cpp/04_correlation.cpp
/// @brief Electron correlation on top of RHF: MP2 and full CI on H2.
///
/// One concept: post-Hartree–Fock methods wrap a converged reference method.
/// `MP2<RHF>` adds second-order perturbation theory; `FCI<RHF>` is exact
/// within the basis. For two electrons in STO-3G, FCI recovers the whole
/// correlation energy and MP2 most of it.
///
/// Build:  -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON, target `04_correlation`

#include <iomanip>
#include <iostream>

#include "atomiverse/cpu/scf/methods.h"
#include "molecules.h"

using namespace ATV;

int main() {
    cpu::Frame frame = examples::h2(0.74);

    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0, /*tol=*/1e-12};
    const auto mp2 = cpu::scf::MP2<cpu::scf::RHF>{rhf}.run(frame);
    const auto fci = cpu::scf::FCI<cpu::scf::RHF>{rhf}.run(frame);

    std::cout << std::fixed << std::setprecision(8)
              << "H2 / STO-3G               E_total        E_corr\n"
              << "RHF   " << std::setw(22) << rhf.energy() << "\n"
              << "MP2   " << std::setw(22) << mp2.e_tot << std::setw(14)
              << mp2.e_corr << "\n"
              << "FCI   " << std::setw(22) << fci.e_fci << std::setw(14)
              << fci.e_corr << "\n";
    return rhf.converged() ? 0 : 1;
}
