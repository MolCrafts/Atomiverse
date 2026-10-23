/// @file cpu/scf/internal/casscf_cycle.h
/// @brief One closed-shell CASSCF macro iteration. Not a public type.
///
/// The product loop is `cpu::scf::CASSCF` (`Driver<mc::Phase>`). This type
/// is the engine that loop calls. The step order is PySCF
/// `pyscf/mcscf/mc1step.py` `kernel`: `rotate_mo`, then `casci`, then the
/// orbital gradient `gen_g_hop` packs from the generalized Fock. The
/// rotation generator is the existing diagonal Newton step (trust-clamped),
/// not `rotate_orb_cc`'s augmented-Hessian micro-iterations.
///
/// Nothing here sets a converged flag.
#pragma once

#include <utility>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/internal/ci_engine.h"
#include "atomiverse/cpu/scf/internal/integral_view.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Orbital state for one closed-shell CASSCF macro cycle.
///
/// Construction validates the active space and copies the reference MO
/// coefficients. Each later call is one stage. This type does not decide
/// convergence.
struct CasscfCycle {
    /// @brief Validate the active space and copy the reference orbitals.
    /// @param view Integral view with the ERI tensor materialized.
    /// @param det Converged closed-shell determinant.
    /// @param e_scf_ha SCF total energy [Hartree], stored for `e_corr`.
    /// @param ncas_in Active spatial orbitals (dimensionless).
    /// @param nelecas_in Active electrons (even, closed-shell).
    /// @throws std::invalid_argument on open-shell, missing ERI, or an
    ///         active space that does not fit the reference.
    CasscfCycle(const internal::IntegralView& view,
                const SlaterDeterminant& det, double e_scf_ha, int ncas_in,
                int nelecas_in);

    /// @brief Apply the orbital step stored by the previous `gradient`.
    ///        The first call is the identity (`rotate_mo` with u = 1).
    void rotate();

    /// @brief Active-space CI at the current orbitals (`CASSCF.casci`).
    ///        Caches `e_cas` [Hartree].
    void casci();

    /// @brief Generalized-Fock orbital gradient (`gen_g_hop`). Stores the
    ///        diagonal-Newton step for the next `rotate`. Sets `gmax`.
    void gradient();

    /// @brief max |g_pq| from the last `gradient` [Hartree].
    /// @return Orbital-gradient residual. A large sentinel before the
    ///         first gradient, so a missed stage does not look converged.
    double residual() const { return gmax; }

    internal::IntegralView ints;
    xt::xtensor<double, 2> C;
    xt::xtensor<double, 2> K;
    xt::xtensor<double, 2> h_mo;
    xt::xtensor<double, 4> g_mo;
    CiSolution ci;
    std::vector<std::pair<int, int>> pairs;
    /// @brief SCF total stored at construction [Hartree].
    double e_scf = 0.0;
    /// @brief Last active-space CI energy [Hartree].
    double e_cas = 0.0;
    /// @brief max |g_pq| [Hartree]. Sentinel until `gradient` runs.
    double gmax = 1.0;
    int ncas = 0;
    int nelecas = 0;
    int ncore = 0;
    int n_mo = 0;
    int n_act = 0;
    bool have_step = false;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
