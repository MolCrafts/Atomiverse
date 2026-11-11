/// @file examples/cpp/02_open_shell.cpp
/// @brief Open-shell methods: UHF and UKS on the hydrogen atom.
///
/// One concept: an open-shell method takes the α/β electron counts
/// explicitly. The same doublet (one α electron, no β) is solved with
/// unrestricted Hartree–Fock and with unrestricted Kohn–Sham DFT (LSDA);
/// the only difference between the two objects is the exchange-correlation
/// model.
///
/// Build:  -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON, target `02_open_shell`
/// Expect: UHF/STO-3G E(H) ≈ −0.4666 Hartree (the STO-3G 1s energy); the
///         LSDA energy is higher, the well-known self-interaction error.

#include <iomanip>
#include <iostream>

#include "atomiverse/cpu/scf/methods.h"
#include "molecules.h"

using namespace ATV;

int main() {
    cpu::Frame frame = examples::hydrogen_atom();

    cpu::scf::UHF uhf{cpu::scf::BasisSet::STO3G, /*n_alpha=*/1, /*n_beta=*/0};
    uhf.tol = 1e-8;
    uhf.run(frame);

    cpu::scf::UKS uks{cpu::scf::BasisSet::STO3G, /*n_alpha=*/1, /*n_beta=*/0,
                      cpu::xc::XCType::LSDA};
    uks.run(frame);

    std::cout << std::fixed << std::setprecision(8)
              << "H atom / STO-3G (doublet)\n"
              << "UHF      : " << uhf.energy() << " Hartree"
              << (uhf.converged() ? "" : "  (not converged)") << "\n"
              << "UKS-LSDA : " << uks.energy() << " Hartree"
              << (uks.converged() ? "" : "  (not converged)") << "\n";
    return (uhf.converged() && uks.converged()) ? 0 : 1;
}
