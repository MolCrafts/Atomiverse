/// @file cpu/scf/internal/dhf.h
/// @brief Four-component Dirac–Hartree–Fock cycle (private).
///
/// Transplants PySCF 2.14.0 `pyscf/scf/dhf.py` (`DHF.get_hcore`,
/// `get_ovlp`, `get_occ`, `energy_elec`, `get_jk_coulomb` via
/// `_call_veff_llll` and `_call_veff_ssll`, `DHF.kernel`) and
/// `pyscf/scf/hf.py` (`kernel`, `get_fock`, `eig`, `make_rdm1`,
/// `energy_tot`) on l ≤ 1 shells. The j-adapted coefficients are
/// `pyscf.symm.sph.sph2spinor` for l = 0 and l = 1. The driver walks
/// the cycle; this type does not decide convergence.
#pragma once

#include <complex>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/internal/complex_ghf.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief One Dirac–Hartree–Fock SCF state.
///
/// Construction is the core-Hamiltonian density (`init_guess_by_1e`:
/// `eig` of `get_hcore`, then `get_occ` / `make_rdm1`). Each later call
/// is one SCF stage. This type does not decide convergence.
struct DhfCycle {
    /// @brief Assemble `get_hcore`, `get_ovlp`, the (LL|LL)+(SS|LL)
    ///        integrals, and the core-Hamiltonian density.
    /// @param frame Geometry [Å], species in `kAtom_type`.
    /// @param basis Gaussian basis (l ≤ 1).
    /// @param charge Molecular charge [e].
    /// @throws std::runtime_error naming `dhf_ao_lgt1` when a shell has l > 1.
    DhfCycle(const Frame& frame, BasisSet basis, int charge);

    /// @brief `get_jk_coulomb` and `energy_elec` + `energy_tot` for the
    ///        current density. Stores the total energy [Hartree].
    void fock();

    /// @brief CDIIS mix of the current Fock (`hf.get_fock` with DIIS on).
    void mix();

    /// @brief `hf.kernel` `eig`: generalized Hermitian diagonalization.
    void diagonalize();

    /// @brief `DHF.get_occ` + `hf.make_rdm1` from the orbitals just diagonalized.
    void density();

    /// @brief |ΔE| [Hartree] against the previous fock energy, or +∞ on the
    ///        first call. Records the current energy as the next baseline.
    ///        Does not decide convergence.
    double delta();

    ComplexMatrix H;
    ComplexMatrix S4;
    ComplexMatrix P;
    ComplexMatrix F;
    GenEigen orbitals;
    HermitianDiis diis;
    std::vector<std::complex<double>> ell;
    std::vector<std::complex<double>> ess;
    int n2 = 0;
    int n_elec = 0;
    double c = 0.0;
    double scale = 0.0;
    double v_nuc = 0.0;
    /// @brief Last total energy [Hartree], nuclear repulsion included.
    double energy = 0.0;
    double previous = 0.0;
    bool have_previous = false;
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
