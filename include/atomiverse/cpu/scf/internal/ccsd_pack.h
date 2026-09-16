/// @file cpu/scf/internal/ccsd_pack.h
/// @brief Private pack-based closed-shell CCSD / CCSD(T) / EOM amplitude entry.
#pragma once

#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/cc2.h"
#include "atomiverse/cpu/scf/ccsd.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Converged spin-orbital CCSD amplitudes and antisymmetrized MO blocks
///        shared with EOM-EE-CCSD and CCSD(T) (private).
struct CCSDSolution {
    CCSDResult result;
    /// Spin-orbital T₁ [no, nv] and T₂ [no, no, nv, nv] (physicist ⟨⟩).
    xt::xtensor<double, 2> t1;
    xt::xtensor<double, 4> t2;
    /// Antisymmetrized spin-orbital ERI blocks ⟨pq||rs⟩.
    xt::xtensor<double, 4> oooo;
    xt::xtensor<double, 4> ooov;
    xt::xtensor<double, 4> oovv;
    xt::xtensor<double, 4> ovov;
    xt::xtensor<double, 4> ovvv;
    xt::xtensor<double, 4> vvvv;
    /// Canonical Fock blocks (diagonal ε in spin-orbital basis).
    xt::xtensor<double, 2> foo;
    xt::xtensor<double, 2> fvv;
    xt::xtensor<double, 2> fov;
    /// Spin-orbital orbital energies [no+nv].
    std::vector<double> eps_so;
    int no = 0;       ///< Occupied spin-orbitals.
    int nv = 0;       ///< Virtual spin-orbitals.
    int n_occ_sp = 0; ///< Occupied spatial MOs (closed-shell: n_α = n_β).
    int n_vir_sp = 0; ///< Virtual spatial MOs (closed-shell).
    /// Open-shell (UHF) channel sizes for EOM-SF root selection. Layout of
    /// spin-orbitals: occ = [α₀…α_{nα−1} | β₀…β_{nβ−1}],
    /// vir = [α₀…α_{nva−1} | β₀…β_{nvb−1}]. For closed-shell RHF these match
    /// `n_occ_sp` / `n_vir_sp` on both spins.
    int n_alpha = 0;
    int n_beta = 0;
    int n_vir_a = 0;
    int n_vir_b = 0;
};

/// @brief Closed-shell CCSD correlation energy from a converged RHF reference.
///
/// Expands the spatial MO ERI tensor to spin-orbitals, builds Gauss–Stanton
/// intermediates, and iterates T₁/T₂ by plain Jacobi updates until
/// max|Δt| < `tol` or `max_iter` is exhausted.
///
/// Closed-shell RHF routes to the spatial-orbital fast path
/// (`ccsd_energy_spatial`); the spin-orbital kernel stays the oracle and the
/// backbone for EOM/(T)/CC2/gradient/UCCSD.
/// @param ints     Integral pack with the dense ERI tensor materialized.
/// @param det      Converged closed-shell determinant (MO coeffs, ε, n_e).
/// @param e_scf_ha Converged SCF total energy [Hartree].
/// @param tol      Amplitude residual tolerance (dimensionless max-norm).
/// @param max_iter Jacobi/DIIS iteration budget (dimensionless).
/// @param use_diis When true, accelerate with `CcAmplitudeDiis`; false → plain
///                 Jacobi (same fixed point).
/// @return CCSD correlation and total energies [Hartree].
/// @throws std::invalid_argument on open-shell / missing ERI / non-aufbau
///         denominators / non-convergence (no silent fallback).
CCSDResult ccsd_energy(const internal::IntegralView& ints, const SlaterDeterminant& det,
                       double e_scf_ha, double tol = 1e-10, int max_iter = 100,
                       bool use_diis = true);

/// @brief Closed-shell CCSD(T): CCSD amplitudes + non-iterative (T).
///
/// Same CCSD path as `ccsd_energy`, then the spin-orbital (T) formula of
/// Watts–Gauss–Bartlett (J. Chem. Phys. **98**, 8718 (1993); PySCF
/// `gccsd_t.kernel`). Matches PySCF `cc.CCSD(...).ccsd_t()` on RHF.
///
/// @param ints     Integral pack with the dense ERI tensor materialized.
/// @param det      Converged closed-shell determinant (MO coeffs, ε, n_e).
/// @param e_scf_ha Converged SCF total energy [Hartree].
/// @param tol      Amplitude residual tolerance (dimensionless max-norm).
/// @param max_iter Jacobi iteration budget (dimensionless).
/// @return CCSD(T) correlation pieces and total energy [Hartree].
/// @throws std::invalid_argument as for `ccsd_energy`.
CCSDTResult ccsd_t_energy(const internal::IntegralView& ints,
                          const SlaterDeterminant& det, double e_scf_ha,
                          double tol = 1e-10, int max_iter = 100);

/// @brief Same as `ccsd_energy` but also returns amplitudes and SO integral
///        blocks for EOM-EE-CCSD (private). Closed-shell RHF only.
/// @param cc2 When true, use the CC2 doubles residual (Christiansen 1995)
///            instead of full CCSD; singles residual stays full CCSD.
CCSDSolution ccsd_solve(const internal::IntegralView& ints, const SlaterDeterminant& det,
                        double e_scf_ha, double tol = 1e-10, int max_iter = 100,
                        bool cc2 = false);

/// @brief Closed-shell CC2 correlation energy (Christiansen–Koch–Jørgensen).
///
/// Same spin-orbital infrastructure as `ccsd_energy` with the CC2 T₂
/// truncation. Matches PySCF `cc.RCCSD(mf)` with `cc2=True`.
///
/// @param ints     Integral pack with the dense ERI tensor materialized.
/// @param det      Converged closed-shell determinant (MO coeffs, ε, n_e).
/// @param e_scf_ha Converged SCF total energy [Hartree].
/// @param tol      Amplitude residual tolerance (dimensionless max-norm).
/// @param max_iter Jacobi iteration budget (dimensionless).
/// @return CC2 correlation and total energies [Hartree].
/// @throws std::invalid_argument as for `ccsd_energy`.
CC2Result cc2_energy(const internal::IntegralView& ints, const SlaterDeterminant& det,
                     double e_scf_ha, double tol = 1e-10, int max_iter = 100);

/// @brief Unrestricted (UHF) spin-orbital CCSD — same Gauss–Stanton
///        equations as `ccsd_solve`, with independent α/β MO coefficients
///        and occupations. Matches PySCF `cc.UCCSD` / `cc.GCCSD` on UHF.
///
/// @param ints     Integral pack with dense ERI materialized.
/// @param det      Converged unrestricted determinant (n_α, n_β may differ).
/// @param e_scf_ha Converged UHF total energy [Hartree].
/// @param tol      Amplitude residual tolerance (dimensionless max-norm).
/// @param max_iter Jacobi iteration budget (dimensionless).
/// @return Spin-orbital amplitudes + SO integral blocks for EOM-SF.
/// @throws std::invalid_argument on restricted det / missing ERI / GHF /
///         non-aufbau denominators / non-convergence (no silent fallback).
CCSDSolution uccsd_solve(const internal::IntegralView& ints, const SlaterDeterminant& det,
                         double e_scf_ha, double tol = 1e-10,
                         int max_iter = 100);

} // namespace scf
} // namespace cpu
} // namespace ATV
