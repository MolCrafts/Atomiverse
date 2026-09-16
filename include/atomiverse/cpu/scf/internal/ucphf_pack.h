/// @file cpu/scf/internal/ucphf_pack.h
/// @brief **PRIVATE** unrestricted CPHF linear-response solver + analytic
///        UHF nuclear-Hessian assembly.
///
/// Real nuclear perturbations x on a canonical UHF reference (Hartree/Bohr
/// internals). Coupled α/β orbital-rotation responses:
///
///   (A+B) · [U^{xα}; U^{xβ}] = −[B^{xα}; B^{xβ}]
///
/// with the unrestricted orbital Hessian (same MO-ERI contractions as the
/// SOSCF UHF energy Hessian without the energy-gradient factor of 2 —
/// Helgaker/Jørgensen/Olsen ch. 10; Yamaguchi et al., *A New Dimension to
/// Quantum Chemistry*, 1994):
///
///   (A+B)^{σσ}_{ai,bj} = δ_ab δ_ij (ε_a−ε_i)
///                      + 2(ai|bj) − (ab|ij) − (aj|bi)
///   (A+B)^{αβ}_{ai,bj} = 2(ai|bj)
///
/// Right-hand sides (skeleton Fock + occ–occ orthonormality):
///
///   B^σ_ai = F^{σ(x)}_{ai} − ε_i S^{(x)}_{ai}
///          − Σ_{kl∈σ} S^{(x)}_{kl} [(ai|kl) − (ak|il)]
///          − Σ_{kl∈τ} S^{(x)}_{kl} (ai|kl)
///
/// Hessian assembly (convention-free form, matching `CphfPack` for RHF):
///
///   E^{xy} = Tr[P h^{(xy)}] + ½ Σ P P ĵ^{xy} − ½ Σ_σ P^σ P^σ k̂^{xy}
///          + V_nuc^{xy} − Tr[W S^{(xy)}]
///          + Σ_σ Tr[P^{σ,y} F^{σ(x)}] − Tr[W^y S^{(x)}]
///
/// with P = P^α + P^β, W = W^α + W^β, W^σ = C_o^σ diag(ε_o^σ) C_o^{σ⊤},
/// and W^y the derivative of Σ_σ P^σ F^σ P^σ.
///
/// References: Pople et al., Int. J. Quantum Chem. Symp. 13, 225 (1979);
/// Yamaguchi et al. (1994); Helgaker, Jørgensen, Olsen response chapter.
#pragma once

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Unrestricted CPHF for all 3N nuclear perturbations + analytic
///        UHF Hessian assembly. Constructor-wired private workspace: all
///        inputs are borrowed const references the caller keeps alive.
class UCphfPack {
  public:
    /// @brief Wire the solver to a converged unrestricted reference.
    /// @param det    Converged unrestricted determinant (C^σ, ε^σ [Hartree]).
    /// @param pack   Integral pack of the same geometry (S, Hcore [Hartree]).
    /// @param eri_ao Dense AO ERI tensor (μν|λσ) [Hartree], working AO dim.
    /// @param dS_ao  Skeleton overlap derivatives [n_atoms, 3, n, n] [1/Bohr].
    /// @param dFa_ao Skeleton α Fock derivatives F^{α(x)} [n_atoms,3,n,n]
    ///               [Hartree/Bohr] = h^{(x)} + J^{(x)}[P] − K^{(x)}[P^α].
    /// @param dFb_ao Skeleton β Fock derivatives (same layout).
    /// @throws std::invalid_argument on a restricted determinant, a missing
    ///         virtual space on either spin, or dimension mismatches.
    UCphfPack(const SlaterDeterminant& det, const internal::IntegralView& pack,
              const xt::xtensor<double, 4>& eri_ao,
              const xt::xtensor<double, 4>& dS_ao,
              const xt::xtensor<double, 4>& dFa_ao,
              const xt::xtensor<double, 4>& dFb_ao);

    /// @brief Relative CG residual tolerance ‖r‖/max(1, ‖b‖).
    double cg_tol = 1e-12;
    /// @brief CG iteration budget per right-hand side.
    int cg_max_iter = 400;
    /// @brief Dense direct solve when n_rot ≤ this (0 forces CG).
    int dense_threshold = 256;

    /// @brief Apply the unrestricted (A+B) orbital Hessian to a trial
    ///        vector concatenated [α; β] of length n_rot.
    /// @param v Trial amplitudes (dimensionless).
    /// @return (A+B)·v [Hartree].
    RealVector apply_lhs(const RealVector& v) const;

    /// @brief Solve (A+B) U^x = −B^x for every nuclear coordinate x.
    /// @throws std::runtime_error when CG fails to reach `cg_tol`.
    void solve();

    /// @brief RHS B^x [3·n_atoms, n_rot] [Hartree/Bohr], α block then β.
    const xt::xtensor<double, 2>& rhs() const { return B_; }

    /// @brief Responses U^x [3·n_atoms, n_rot] (dimensionless).
    /// @pre `solve()` has run.
    const xt::xtensor<double, 2>& response() const { return U_; }

    /// @brief Number of right-hand sides solved by the iterative (PCG) path
    ///        (0 for the dense path). The shared `linalg::solve_pcg` returns
    ///        only the solution, so this counts iteratively-solved
    ///        perturbations rather than summed CG iterations.
    int iterations() const { return iterations_; }

    /// @brief Spin-α AO density response ∂P^α/∂x (dimensionless).
    /// @param x Coordinate index 3·atom + axis.
    /// @pre `solve()` has run.
    RealMatrix density_response_alpha(int x) const;

    /// @brief Spin-β AO density response ∂P^β/∂x (dimensionless).
    /// @param x Coordinate index 3·atom + axis.
    /// @pre `solve()` has run.
    RealMatrix density_response_beta(int x) const;

    /// @brief Assemble the analytic Hessian E^{xy} [Hartree/Bohr²], shape
    ///        [3N, 3N], symmetrized.
    /// @pre `solve()` has run.
    xt::xtensor<double, 2> hessian() const;

  private:
    /// @brief AO Fock response F^{σ(x)} + G^σ[P^{α,x}, P^{β,x}].
    void fock_response_(int x, const RealMatrix& dPa, const RealMatrix& dPb,
                        RealMatrix& dFa, RealMatrix& dFb) const;

    /// @brief Diagonal of (A+B) for CG preconditioning [Hartree].
    RealVector diagonal_lhs_() const;

    /// @brief Dense (A+B) matrix [n_rot, n_rot] [Hartree].
    RealMatrix dense_lhs_() const;

    const SlaterDeterminant& det_;
    internal::IntegralView pack_;
    const xt::xtensor<double, 4>& eri_;
    const xt::xtensor<double, 4>& dS_ao_;
    const xt::xtensor<double, 4>& dFa_ao_;
    const xt::xtensor<double, 4>& dFb_ao_;

    int n_ = 0;
    int natm_ = 0;
    int no_a_ = 0;
    int nv_a_ = 0;
    int no_b_ = 0;
    int nv_b_ = 0;
    int rot_a_ = 0; ///< n_ov α = nv_a * no_a
    int rot_b_ = 0;
    int n_rot_ = 0;

    RealMatrix Co_a_, Cv_a_, Co_b_, Cv_b_;
    RealVector eps_oa_, eps_va_, eps_ob_, eps_vb_;

    /// MO-ERI blocks [Hartree]: ovov = (ai|bj), oovv = (ij|ab).
    xt::xtensor<double, 4> ovov_aa_, oovv_aa_, ovov_bb_, oovv_bb_, ovov_ab_;
    /// (ai|kl) for RHS occ–occ contractions [nv, no, no, no].
    xt::xtensor<double, 4> vooo_aa_, vooo_ab_, vooo_bb_, vooo_ba_;

    xt::xtensor<double, 3> S_vo_a_, S_oo_a_, S_vo_b_, S_oo_b_;
    xt::xtensor<double, 2> B_;
    xt::xtensor<double, 2> U_;
    bool solved_ = false;
    int iterations_ = 0; ///< RHS solved by the iterative PCG path.
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
