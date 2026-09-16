/// @file cpu/scf/basis_data/basis_table.h
/// @brief Generic container types for checked-in basis-set data.
///
/// Each basis-set header under `basis_data/` declares one inline
/// function returning a `const std::vector<AtomBasis>&` indexed by
/// atomic number Z (position 0 is unused; empty AtomBasis marks an
/// unsupported element).
///
/// The types model the same "raw" / "unnormalized" contraction
/// convention used by PySCF's `mol._basis`: `coeff[p]` is the Pople
/// contraction coefficient for the primitive with exponent `alpha[p]`,
/// and the physical GTO normalization (`phys_norm_s(alpha)` etc.) is
/// applied by the shell-list builder.
///
/// @see cpu::scf::ShellList for the constructor that consumes these
///      tables.
/// @see tools/gen_basis_data.py for the generator that emits them.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace scf {
namespace basis_data {

/// @brief One contracted Gaussian shell.
///
/// For `l = 1` (p), the same `alpha/coeff` pair is reused across the
/// three Cartesian components px/py/pz. For `l = 2` (d), likewise
/// across the six Cartesian components d_xx..d_yz (Stage B2b).
struct ContractedShell {
    int l;                     ///< Angular momentum (0 = s, 1 = p, 2 = d).
    std::vector<double> alpha; ///< Gaussian exponents, length `n_prim`.
    std::vector<double>
        coeff; ///< Contraction coefficients (unnormalized), length `n_prim`.
};

/// @brief Angular-function convention a basis set is DEFINED with.
///
/// Pople sets (6-31G family) are defined with 6 Cartesian d components;
/// def2/cc- sets are defined with real spherical harmonics (5d/7f).
/// The ShellList/internal::IntegralWorkspace pipeline reads this per-set flag to pick the
/// AO dimension; there is no runtime override (the convention is part of
/// the basis-set definition, not a user knob).
enum class AngularConvention {
    Cartesian, ///< (l+1)(l+2)/2 components per shell (6d/10f).
    Spherical, ///< 2l+1 components per shell (5d/7f).
};

/// @brief All shells belonging to one element.
///
/// `shells` is empty for atomic numbers that the basis set does not
/// cover. The shell-list builder throws `std::runtime_error` when it
/// encounters such an atom.
struct AtomBasis {
    std::vector<ContractedShell> shells;
};

/// @brief One Gaussian term of a semi-local ECP channel (Kahn form):
///        `coeff * r^(n-2) * exp(-alpha * r^2)`.
///
/// Kahn, Baybutt, Truhlar, J. Chem. Phys. 65, 3826 (1976). All published
/// def2-ECP terms use `n = 2` (pure Gaussians).
struct EcpGaussianTerm {
    int n;        ///< Radial power convention: the term carries r^(n-2).
    double alpha; ///< Gaussian exponent [Bohr^-2].
    double coeff; ///< Linear coefficient d_k [Hartree].
};

/// @brief One angular-momentum channel of a semi-local ECP.
struct EcpChannel {
    int l;                              ///< Angular momentum (0 = s, 1 = p, …).
    std::vector<EcpGaussianTerm> terms; ///< Gaussian expansion of the channel.
};

/// @brief Semi-local effective core potential of one element:
///        U = U_L + Σ_{l<L} Σ_m |lm⟩ (U_l − U_L) ⟨lm|.
///
/// `channels` is sorted by ascending l; the LAST channel is the local part
/// U_L (relabeled from PySCF's l = -1 to l = L = 1 + max projector l) and
/// applies to every angular momentum. The preceding projector channels
/// store U_l − U_L directly (PySCF/def2 storage convention). An empty
/// `AtomEcp` (no channels) marks an all-electron element.
struct AtomEcp {
    int n_core = 0;                   ///< Core electrons replaced by the ECP.
    std::vector<EcpChannel> channels; ///< Projector channels, then local U_L.
};

} // namespace basis_data
} // namespace scf
} // namespace cpu
} // namespace ATV
