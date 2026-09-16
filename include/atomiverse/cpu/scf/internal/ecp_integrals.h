/// @file cpu/scf/internal/ecp_integrals.h
/// @brief **PRIVATE** EcpIntegrals workspace — semi-local ECP core-Hamiltonian
///        integrals ⟨μ|U|ν⟩ over the Cartesian engine AO basis.
///
/// Not a user type: `internal::IntegralWorkspace` builds it from Frame-derived shells and
/// adds the resulting U matrix into the nuclear-attraction block of Hcore
/// BEFORE any Cartesian→spherical sandwich. Never included from a
/// facade-public header (tests may include it).
///
/// Method: McMurchie, Davidson, J. Comput. Phys. 44, 289 (1981) — type-1
/// (local channel U_L, plane-wave-like expansion about the ECP center) and
/// type-2 (semi-local projector channels U_l − U_L, per-side expansions)
/// integrals. Radial integrands carry modified spherical Bessel functions
/// i_l(kr); the implementation uses the SCALED form M_l(x) = e^{−x} i_l(x)
/// (small-x series, large-x normalized downward recursion) so the Gaussian
/// prefactor combines to exp(−α(r−|A'|)² − β(r−|B'|)² − ζr²) ≤ 1 with no
/// catastrophic cancellation at high l. Radial quadrature: per-term
/// 128-point Gauss–Legendre on [max(0, r* − 9σ), r* + 9σ] with
/// r* = (k_A + k_B)/(2γ_tot), σ = 1/√γ_tot — validated to ~3e-13 absolute
/// against PySCF 2.14.0 `mol.intor('ECPscalar')`. Angular parts use the
/// shared `ATV::real_spherical_harmonics_normalized` polynomials and the
/// exact `sphere_monomial_integral`.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/real_harmonics.h"
#include "atomiverse/cpu/scf/basis_data/basis_table.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief One ECP-carrying atom: position + its semi-local potential row.
struct EcpCenter {
    double pos[3]; ///< Center position [Bohr].
    const basis_data::AtomEcp*
        ecp; ///< Borrowed table row (static-table or caller-owned lifetime).
};

/// @brief Private workspace computing the ECP matrix U over Cartesian AOs.
///
/// Constructed per geometry from the engine shell list (one `native::Shell`
/// per Cartesian AO) and the list of ECP-carrying centers; `matrix()`
/// assembles the full symmetric U. Angular tables (real-harmonic sphere
/// integrals) are memoized across shell pairs inside the object.
class EcpIntegrals {
  public:
    /// @brief Bind the AO basis and the ECP centers.
    /// @param shells  Engine shell list (centers in Bohr); borrowed — must
    ///                outlive this object.
    /// @param centers ECP centers (positions in Bohr + table rows).
    EcpIntegrals(const std::vector<native::Shell>& shells,
                 std::vector<EcpCenter> centers);

    /// @brief Assemble U(i,j) = ⟨μ_i|U|ν_j⟩ over all ECP centers.
    /// @return Symmetric (n_ao_cart × n_ao_cart) matrix [Hartree].
    xt::xtensor<double, 2> matrix();

  private:
    const std::vector<native::Shell>& shells_;
    std::vector<EcpCenter> centers_;
    int lmax_harm_ = 0; ///< Highest harmonic order any sum can reach.

    /// Real spherical harmonics r^l Y_lm as Cartesian polynomials, [l][m].
    std::vector<std::vector<RealHarmonic>> harm_;

    /// Memoized type-1 angular integrals ∫ Y_λμ x̂^i ŷ^j ẑ^k dΩ, per μ.
    std::unordered_map<std::int64_t, std::vector<double>> omega1_;
    /// Memoized type-2 angular integrals ∫ Y_lm Y_λμ x̂^i ŷ^j ẑ^k dΩ,
    /// flattened [m*(2λ+1)+μ].
    std::unordered_map<std::int64_t, std::vector<double>> omega2_;

    const std::vector<double>& omega1(int lam, int i, int j, int k);
    const std::vector<double>& omega2(int l, int lam, int i, int j, int k);

    /// One AO pair against one ECP center: type-1 (local) + type-2 sum.
    double pair_center(const native::Shell& a, const native::Shell& b,
                       const EcpCenter& center);
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
