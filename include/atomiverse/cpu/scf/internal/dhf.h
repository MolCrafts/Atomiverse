/// @file cpu/scf/internal/dhf.h
/// @brief Four-component Dirac–Hartree–Fock cycle (private).
///
/// Transplants PySCF 2.14.0 `pyscf/scf/dhf.py` (`DHF.get_hcore`,
/// `get_ovlp`, `get_occ`, `energy_elec`, `get_jk_coulomb` via
/// `_call_veff_llll` and `_call_veff_ssll`, `DHF.kernel`) and
/// `pyscf/scf/hf.py` (`kernel`, `get_fock`, `eig`, `make_rdm1`,
/// `energy_tot`) on l ≤ 1 shells. The j-adapted coefficients are
/// `pyscf.symm.sph.sph2spinor` for l = 0, 1. The driver walks the
/// cycle; this type does not decide convergence.
#pragma once

#include <memory>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief One Dirac–Hartree–Fock SCF state.
struct DhfCycle;

/// @brief Destroy a cycle allocated by `dhf_cycle_open`.
void dhf_cycle_delete(DhfCycle* cycle);

/// @brief Owning pointer. The deleter is a function so callers need not
///        see the cycle body.
using DhfCycleOwner = std::unique_ptr<DhfCycle, void (*)(DhfCycle*)>;

/// @brief Assemble `get_hcore`, `get_ovlp`, the (LL|LL)+(SS|LL) integrals,
///        and the core-Hamiltonian density (`init_guess_by_1e`).
/// @param frame Geometry [Å], species in `kAtom_type`.
/// @param basis Gaussian basis (l ≤ 1).
/// @param charge Molecular charge [e].
/// @throws std::runtime_error naming `dhf_ao_lgt1` when a shell has l > 1.
DhfCycleOwner dhf_cycle_open(const Frame& frame, BasisSet basis, int charge);

/// @brief `get_jk_coulomb` and `energy_elec` + `energy_tot` for the
///        current density. Stores the total energy [Hartree].
void dhf_cycle_fock(DhfCycle& cycle);

/// @brief CDIIS mix of the current Fock (`hf.get_fock` with DIIS on).
void dhf_cycle_mix(DhfCycle& cycle);

/// @brief `hf.kernel` `eig`: generalized Hermitian diagonalization.
void dhf_cycle_diagonalize(DhfCycle& cycle);

/// @brief `DHF.get_occ` + `hf.make_rdm1` from the orbitals just diagonalized.
void dhf_cycle_density(DhfCycle& cycle);

/// @brief Last total energy [Hartree], nuclear repulsion included.
double dhf_cycle_energy(const DhfCycle& cycle);

/// @brief |ΔE| [Hartree] against the previous fock energy, or +∞ on the
///        first call. Records the current energy as the next baseline.
///        Does not decide convergence.
double dhf_cycle_delta(DhfCycle& cycle);

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
