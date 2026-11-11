/// @file cpu/scf/internal/cphf_pack.h
/// @brief **PRIVATE** coupled-perturbed Hartree-Fock (CPHF) linear-response
///        solver + analytic RHF nuclear-Hessian assembly.
///
/// The CPHF equations for real nuclear perturbations x (closed-shell RHF,
/// canonical MOs, Hartree/Bohr internally; Pople, Krishnan, Schlegel,
/// Binkley, Int. J. Quantum Chem. Symp. 13, 225 (1979),
/// DOI 10.1002/qua.560160825):
///
///   ¹(A+B) · U^x = −B^x,
///   ¹(A+B)_{ai,bj} = δ_ab δ_ij (ε_a − ε_i) + 4(ai|bj) − (ab|ij) − (aj|ib)
///   B^x_{ai} = F^{(x)}_{ai} − ε_i S^{(x)}_{ai}
///              − Σ_kl S^{(x)}_{kl} [2(ai|kl) − (ak|il)]
///
/// The left-hand operator is exactly the singlet σ₊ channel of the shared
/// `OrbitalHessian` kernel (link 08: stability eigenvalues; link 09: SOSCF
/// Newton; this link: CPHF linear equations — its third configuration).
/// The solve is preconditioned conjugate-gradient per right-hand side
/// (diagonal ε_a − ε_i preconditioner) with a dense direct solve via
/// `xt::linalg` below `dense_threshold`.
///
/// Hessian assembly (convention-free form; equivalent to the classic
/// 4 Σ_ai U^y_ai B^x_ai coupling plus its occ-occ overlap companions — see
/// the 实现审计注记 in the owning spec):
///
///   E^{xy} = Tr[P h^{(xy)}] + ½ Σ P P ĝ^{(xy)} + V_nuc^{xy}
///          − Tr[W S^{(xy)}] + Tr[P^y F^{(x)}] − Tr[W^y S^{(x)}]
///
/// with P^y / W^y the density / energy-weighted-density responses built
/// from U^y (occ-occ rotations fixed by −½S^{(y)}, which is the only
/// combination the symmetric responses depend on).
///
/// References: Pople et al. 1979 (above); Pulay, Mol. Phys. 17, 197
/// (1969); Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure
/// Theory*, response-theory chapter.
#pragma once

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief CPHF linear-response solve for all 3N nuclear perturbations plus
///        the analytic RHF Hessian assembly. Constructor-wired private
///        workspace (mp2_pack precedent): all inputs are borrowed const
///        references the caller keeps alive.
class CphfPack {
  public:
    /// @brief Wire the solver to a converged closed-shell reference.
    /// @param det    Converged restricted determinant (C, ε [Hartree],
    ///               integer occupations).
    /// @param oh     The shared link-08 orbital-Hessian kernel built from
    ///               `det` and `eri_ao` — its singlet σ₊ IS the CPHF LHS.
    /// @param pack   Integral pack of the same geometry (S, Hcore [Hartree],
    ///               Cartesian convention).
    /// @param eri_ao Dense AO ERI tensor (μν|λσ) [Hartree], working AO
    ///               dimension.
    /// @param dS_ao  Skeleton overlap derivatives [n_atoms, 3, n, n]
    ///               [1/Bohr] (`IntegralDerivPack::dS`).
    /// @param dF_ao  Skeleton Fock derivatives F^{(x)} = h^{(x)} + G^{(x)}[P]
    ///               [n_atoms, 3, n, n] [Hartree/Bohr].
    /// @throws std::invalid_argument on an unrestricted determinant or
    ///         dimension mismatches (no fallback).
    CphfPack(const SlaterDeterminant& det, const OrbitalHessian& oh,
             const internal::IntegralView& pack,
             const xt::xtensor<double, 4>& eri_ao,
             const xt::xtensor<double, 4>& dS_ao,
             const xt::xtensor<double, 4>& dF_ao);

    /// @brief Relative CG residual tolerance ‖r‖/max(1, ‖b‖)
    ///        (dimensionless).
    double cg_tol = 1e-12;
    /// @brief CG iteration budget per right-hand side (dimensionless).
    int cg_max_iter = 400;
    /// @brief Use the dense direct solve when n_ov ≤ this (dimensionless);
    ///        0 forces the iterative path, a large value forces dense.
    int dense_threshold = 256;

    /// @brief The CPHF LHS action — literally the link-08 kernel's singlet
    ///        σ₊, no reimplementation (`oh.sigma_plus(v, Channel::singlet)`).
    /// @param v Trial vector, length n_ov (dimensionless amplitudes).
    /// @return ¹(A+B)·v [Hartree].
    RealVector apply_lhs(const RealVector& v) const;

    /// @brief Solve ¹(A+B) U^x = −B^x for every nuclear coordinate x
    ///        (3·n_atoms right-hand sides).
    /// @throws std::runtime_error when CG fails to reach `cg_tol` within
    ///         `cg_max_iter` iterations (message reports the `cg_max_iter`
    ///         budget; no fallback).
    void solve();

    /// @brief CPHF right-hand sides B^x [3·n_atoms, n_ov] [Hartree/Bohr],
    ///        composite column index a·n_occ + i.
    const xt::xtensor<double, 2>& rhs() const { return B_; }

    /// @brief Occ-virt responses U^x [3·n_atoms, n_ov] (dimensionless).
    /// @pre `solve()` has run.
    const xt::xtensor<double, 2>& response() const { return U_; }

    /// @brief Number of right-hand sides solved by the iterative (PCG) path
    ///        (0 for the dense path). The per-solve CG iteration total is no
    ///        longer tracked: the shared `linalg::solve_pcg` returns only the
    ///        solution, so this counts the iteratively-solved perturbations.
    int iterations() const { return iterations_; }

    /// @brief AO density-matrix response ∂P/∂x for one nuclear coordinate
    ///        (dimensionless) — the convention-free response observable
    ///        (FD-checkable against the converged density).
    /// @param x Coordinate index 3·atom + axis.
    /// @pre `solve()` has run.
    RealMatrix density_response(int x) const;

    /// @brief Assemble the analytic Hessian E^{xy} [Hartree/Bohr²],
    ///        shape [3N, 3N], symmetrized. Computes the skeleton
    ///        second-derivative pieces (one-electron `IntegralSecondDerivPack`,
    ///        the two-electron direct contraction, V_nuc^{xy}, −Tr[W S^{xy}])
    ///        and adds the CPHF response coupling.
    /// @pre `solve()` has run.
    xt::xtensor<double, 2> hessian() const;

  private:
    /// @brief Fock response for one perturbation: F^{(x)} + G[P^x]
    ///        [Hartree/Bohr] (G = Coulomb − ½ exchange over `eri_`).
    RealMatrix fock_response_(int x, const RealMatrix& Py) const;

    const SlaterDeterminant& det_;      ///< Converged reference (borrowed).
    const OrbitalHessian& oh_;          ///< Shared σ-vector kernel (borrowed).
    internal::IntegralView pack_;       ///< Geometry + S/Hcore (borrowed).
    const xt::xtensor<double, 4>& eri_; ///< Dense AO ERI (borrowed).
    const xt::xtensor<double, 4>& dS_ao_; ///< Skeleton dS (borrowed).
    const xt::xtensor<double, 4>& dF_ao_; ///< Skeleton dF (borrowed).

    int n_ = 0;    ///< AO dimension.
    int natm_ = 0; ///< Atom count.
    int no_ = 0;   ///< Occupied MO count.
    int nv_ = 0;   ///< Virtual MO count.

    RealMatrix Co_;       ///< Occupied MO columns [n × n_occ].
    RealMatrix Cv_;       ///< Virtual MO columns [n × n_virt].
    RealVector eps_occ_;  ///< ε_i [Hartree].
    RealVector eps_virt_; ///< ε_a [Hartree].

    xt::xtensor<double, 4> vooo_; ///< (ai|kl) MO block [nv,no,no,no] [Ha].
    xt::xtensor<double, 3> S_vo_; ///< S^{(x)} virt-occ [3N,nv,no] [1/Bohr].
    xt::xtensor<double, 3> S_oo_; ///< S^{(x)} occ-occ [3N,no,no] [1/Bohr].
    xt::xtensor<double, 2> B_;    ///< RHS B^x [3N, n_ov] [Hartree/Bohr].
    xt::xtensor<double, 2> U_;    ///< Response U^x [3N, n_ov].
    bool solved_ = false;         ///< `solve()` completed.
    int iterations_ = 0;          ///< RHS solved by the iterative PCG path.
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
