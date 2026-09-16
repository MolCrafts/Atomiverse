/// @file cpu/scf/internal/periodic_hf.h
/// @brief Γ-point truncated lattice sums for periodic RHF (v1 foothold).
///
/// Image half-counts follow the GPW `image_counts` rule
/// (`cpu/gpw/internal/periodic.h`): |t_d| ≤ ceil(rcut / L_d). The same
/// image set is used for S, T, V, E_nn, J and K so the electrostatic
/// convention is internally consistent. No Ewald: a cell larger than
/// the Gaussian pair reach has only T = 0 and recovers molecular RHF.
#pragma once

#include <array>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

struct IntegralWorkspace;

/// @brief Orthorhombic Γ-point image window.
///
/// Images run t_d ∈ [-m[d], m[d]] with lattice vectors T_d = t_d L[d].
/// `L` is in Bohr. `m = {0,0,0}` is the molecular (T = 0 only) window.
struct LatticeSpec {
    int m[3] = {0, 0, 0};
    double L[3] = {0, 0, 0}; ///< Orthorhombic edges [Bohr].

    /// @brief True when any extra (T ≠ 0) image is requested.
    [[nodiscard]] bool has_extra() const {
        return m[0] > 0 || m[1] > 0 || m[2] > 0;
    }
};

/// @brief Per-axis half-count covering a pair reach `rcut`.
/// @param L    Orthorhombic edges [Bohr].
/// @param rcut Gaussian pair reach [Bohr].
/// @return LatticeSpec with m[d] = floor(rcut / L[d]) (0 if L[d] ≤ 0
///         or rcut < L[d]).
LatticeSpec lattice_spec(const double L[3], double rcut);

/// @brief Pair reach from the most diffuse primitive.
///
/// r_shell = √(log_eps / α_min), rcut = 2 r_shell. Default log_eps =
/// −ln(10⁻⁸) so a 20 Å box around STO-3G H₂ has no extra images.
/// @param shells Contracted shells (Bohr centres).
/// @param log_eps Positive −ln(threshold) used as α r².
/// @return Pair reach [Bohr].
double shell_pair_rcut(const std::vector<native::Shell>& shells,
                       double log_eps = 18.420680743952367);

/// @brief Lattice-summed exchange K (includes T = 0).
///
/// K_μν = Σ_λσ P_λσ Σ_T (μλ | ν^T σ^T) with |t_d| ≤ m[d]. No Schwarz
/// screening — v1 is for tiny Cartesian systems.
/// @param blocks Shell blocks (home-cell centres, Bohr).
/// @param P      Total RHF density [n × n].
/// @param spec   Image window.
/// @param[out] K Bare exchange (same normalization as `ShellBlockPairList::k`).
/// @param omega  If > 0, use erfc(ω r)/r = 1/r − erf(ω r)/r so the
///               lattice sum is cutoff-stable (AC-001). ω = 0 is bare 1/r.
void exchange_lattice(const std::vector<native::ShellBlock>& blocks,
                      const RealMatrix& P, const LatticeSpec& spec,
                      RealMatrix& K, double omega = 0.0);

/// @brief Fill a dense chemist-notation ERI including every image T.
///
/// ERI(i,j,k,l) = Σ_T (ij | k^T l^T). Used when extra images exist so the
/// stored-ERI Fock path sees both J and K lattice sums.
void fill_eri_lattice(const std::vector<native::ShellBlock>& blocks,
                      const LatticeSpec& spec, xt::xtensor<double, 4>& ERI);

/// @brief Add T ≠ 0 contributions to S, T, V and E_nn.
///
/// T = 0 is assumed already present (molecular fill). No-op when
/// `!has_extra()`.
void apply_gamma_images(IntegralWorkspace& ws, const LatticeSpec& spec);

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
