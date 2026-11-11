/// @file cpu/scf/internal/x2c.h
/// @brief internal::X2cWorkspace — one-step spin-free x2c-1e decoupling.
///
/// **Internal SCF helper.** Does NOT inherit `cpu::Component`. Builds the
/// exact two-component one-electron core Hamiltonian h^X2C from {S, T, V, W}
/// in the contracted AO basis (no uncontract refinement). Used only when
/// `MeanFieldMethod::x2c == true`.
///
/// Working equations (Kutzelnigg–Liu; Peng–Reiher review): modified Dirac
/// matrix h = [[V, T], [T, W/(4c²)−T]] with metric M = [[S, 0], [0, T/(2c²)]];
/// X = C_S C_L⁻¹ from the positive-energy block; renormalisation
/// R = S^{−1/2}(S^{−1/2}S̃S^{−1/2})^{−1/2}S^{1/2} with S̃ = S + X†TX/(2c²);
/// h^X2C = R†(V + TX + X†T + X†(W/(4c²)−T)X)R.
///
/// @see hf-capability-gaps-18-x2c
#pragma once

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Speed of light in atomic units — pinned to PySCF
///        `lib.param.LIGHT_SPEED` (137.03599967994). At the 1e-5 Ha Ar
///        acceptance tolerance the sensitivity to the trailing digits is
///        ~1e-8 Ha and can be ignored.
inline constexpr double kLightSpeedAU = 137.03599967994;

/// @brief One-step spin-free exact-two-component (x2c-1e) core Hamiltonian.
///
/// Construction performs the full Dirac → X → R → h^X2C pipeline. Matrices
/// are n×n in the contracted AO basis (Cartesian or spherical — whatever
/// the internal::IntegralWorkspace published). The speed of light is injectable so unit
/// tests can drive the non-relativistic (c → ∞) limit.
class X2cWorkspace {
  public:
    /// @brief Assemble h^X2C from one-electron integral blocks.
    /// @param S Overlap matrix (dimensionless), n × n, symmetric.
    /// @param T Kinetic-energy matrix [Hartree], n × n, symmetric.
    /// @param V Nuclear-attraction matrix [Hartree], n × n, symmetric.
    /// @param W Spin-free nuclear pVp matrix [Hartree/Bohr²], n × n,
    ///          symmetric (PySCF `int1e_pnucp`).
    /// @param c Speed of light [a.u.]; default `kLightSpeedAU`.
    /// @throws std::invalid_argument on shape mismatch.
    /// @throws std::runtime_error when the large-component block C_L is
    ///         singular or the metric is not positive-definite.
    X2cWorkspace(const RealMatrix& S, const RealMatrix& T, const RealMatrix& V,
                 const RealMatrix& W, double c = kLightSpeedAU);

    /// @brief Exact two-component one-electron core Hamiltonian.
    /// @return h^X2C [Hartree], n × n, symmetric.
    [[nodiscard]] const RealMatrix& hcore() const { return hcore_; }

    /// @brief Lowest positive-energy eigenvalue of the modified Dirac
    ///        matrix (ε > −c² branch). Used by the exact-decoupling
    ///        identity pin: equals the lowest eigenvalue of h^X2C for a
    ///        one-electron atom.
    /// @return Energy [Hartree].
    [[nodiscard]] double lowest_positive_dirac() const {
        return lowest_positive_dirac_;
    }

  private:
    RealMatrix hcore_;
    double lowest_positive_dirac_ = 0.0;
};

/// @brief Nuclear derivative of the spin-free x2c-1e core Hamiltonian.
///
/// Transplant of `pyscf.x2c.sfx2c1e_grad.gen_sf_hfw` /
/// `_gen_first_order_quantities` / `_get_r1` at `approx='1E'` in the
/// contracted AO basis (`hcore_grad_generator` with `xuncontract=False`,
/// so the contraction coefficient is absent). The two-electron integrals
/// and the overlap Pulay term are not part of this matrix — callers
/// replace only Tr[P · (dT+dV)] by Tr[P · dh^X2C]
/// (`pyscf.grad.rhf.grad_elec`).
///
/// @param S Overlap [dimensionless], n × n.
/// @param T Kinetic energy [Hartree], n × n.
/// @param V Nuclear attraction [Hartree], n × n.
/// @param W Spin-free pVp [Hartree/Bohr²], n × n.
/// @param dS dS/dR [1/Bohr], [n_atoms, 3, n, n].
/// @param dT dT/dR [Hartree/Bohr], same layout.
/// @param dV dV/dR [Hartree/Bohr], same layout (basis + operator).
/// @param dW dW/dR [Hartree/Bohr³], same layout.
/// @param c Speed of light [a.u.]; default `kLightSpeedAU`.
/// @return dh^X2C/dR [Hartree/Bohr], [n_atoms, 3, n, n], symmetric in
///         the AO pair.
/// @throws std::invalid_argument on a shape mismatch.
/// @throws std::runtime_error when the Dirac spectrum does not split into
///         n negative and n positive states, the overlap is not positive
///         definite, or the large-component block is singular.
xt::xtensor<double, 4> x2c_hcore_gradient(
    const xt::xtensor<double, 2>& S, const xt::xtensor<double, 2>& T,
    const xt::xtensor<double, 2>& V, const xt::xtensor<double, 2>& W,
    const xt::xtensor<double, 4>& dS, const xt::xtensor<double, 4>& dT,
    const xt::xtensor<double, 4>& dV, const xt::xtensor<double, 4>& dW,
    double c = kLightSpeedAU);

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
