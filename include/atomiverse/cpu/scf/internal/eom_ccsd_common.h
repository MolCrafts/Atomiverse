/// @file cpu/scf/internal/eom_ccsd_common.h
/// @brief **PRIVATE** shared spin-orbital EOM-CCSD pieces: the τ amplitude, the
///        full Gauss–Stanton one-/two-body intermediates, and the dense dgeev
///        glue used by EOM-EE / EA / IP / CC2 / SF.
///
/// Sits beside `internal/ccsd_pack.h` (namespace `ATV::cpu::scf`). The
/// intermediates are built by the computing constructor
/// `EomImds(const CCSDSolution&)` — there is no free `build_imds` factory. The
/// FULL intermediate set (including `Wvvvo`/`Wvvvv`/`Wvovv`) is always built;
/// EE/EA read every field, IP reads all but those three (values are bit-identical
/// to the former IP subset build).
///
/// References: Gauss & Stanton, J. Chem. Phys. 103, 3561 (1995) Table III
/// (EOM section (b)); PySCF `cc.gintermediates`.
#pragma once

#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/internal/ccsd_pack.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief τ_ij^ab = t2 + ½·fac·P(ij)P(ab) t1_ia t1_jb (PySCF
///        gintermediates.make_tau). `fac=1` → τ, `fac=0.5` → τ̃.
/// @param t2  Spin-orbital doubles amplitude [no, no, nv, nv] (dimensionless).
/// @param t1  Spin-orbital singles amplitude [no, nv] (dimensionless).
/// @param fac Amplitude factor (dimensionless): 1 for τ, 0.5 for τ̃.
/// @return τ [no, no, nv, nv].
xt::xtensor<double, 4> make_tau(const xt::xtensor<double, 4>& t2,
                                const xt::xtensor<double, 2>& t1, double fac);

/// @brief Full EOM-CCSD one-/two-body intermediates (Gauss–Stanton
///        gintermediates). EE/EA consume every field; IP consumes all but
///        Wvovv/Wvvvv/Wvvvo.
struct EomImds {
    xt::xtensor<double, 2>
        Foo; ///< F_mi one-body intermediate (occ-occ) [Hartree]
    xt::xtensor<double, 2> Fvv;   ///< F_ae (vir-vir) [Hartree]
    xt::xtensor<double, 2> Fov;   ///< F_me (occ-vir) [Hartree]
    xt::xtensor<double, 4> Wovvo; ///< W_mbej (ovvo sector)
    xt::xtensor<double, 4> Woovv; ///< ⟨mn||ef⟩ antisymmetrized oovv block
    xt::xtensor<double, 4> Woooo; ///< W_mnij
    xt::xtensor<double, 4> Wooov; ///< W_mnie
    xt::xtensor<double, 4> Wovoo; ///< W_mbij
    xt::xtensor<double, 4> Wvovv; ///< W_amef
    xt::xtensor<double, 4> Wvvvv; ///< W_abef
    xt::xtensor<double, 4> Wvvvo; ///< W_abei
    xt::xtensor<double, 4>
        t2; ///< converged CCSD amplitudes (dimensionless), copied through
    xt::xtensor<double, 2>
        t1;     ///< converged CCSD amplitudes (dimensionless), copied through
    int no = 0; ///< occupied spin-orbital count
    int nv = 0; ///< virtual spin-orbital count

    /// @brief Compute the full Gauss–Stanton one-/two-body intermediates from a
    ///        converged CCSD solution.
    /// @param sol Converged spin-orbital CCSD amplitudes + antisymmetrized MO
    ///            blocks (`internal/ccsd_pack.h`).
    explicit EomImds(const CCSDSolution& sol);
};

/// @brief Dense right-eigenpair of a real nonsymmetric matrix (LAPACK dgeev via
///        `linalg::eig_nonsymmetric`). `vr` is empty unless eigenvectors were
///        requested.
struct EigPair {
    double wr = 0.0; ///< Eigenvalue real part.
    double wi =
        0.0; ///< Eigenvalue imaginary part (≈0 for the real roots callers keep).
    std::vector<double> vr; ///< Right eigenvector (real part when wi≈0).
};

/// @brief Real nonsymmetric dense eigenvalues only (JOBVR='N'). Column-major
///        fill reproduces the former hand-rolled buffer bit-for-bit.
/// @param H     Row-of-rows dense matrix (n × n).
/// @param label Diagnostic tag folded into the dgeev throw message.
/// @return Per-column eigenpairs (.vr left empty); complex conjugate pairs
///         occupy adjacent entries in dgeev order.
std::vector<EigPair> eig_dense(const std::vector<std::vector<double>>& H,
                               const char* label);

/// @brief Real nonsymmetric dense eigenvalues + right eigenvectors (JOBVR='V').
/// @param H            Row-of-rows dense matrix (n × n).
/// @param label        Diagnostic tag folded into the dgeev throw message.
/// @param want_vectors Must be true to select this overload (fills each `.vr`).
/// @return Per-column eigenpairs, .vr filled with the right-eigenvector column
///         (real part when wi≈0).
std::vector<EigPair> eig_dense(const std::vector<std::vector<double>>& H,
                               const char* label, bool want_vectors);

} // namespace scf
} // namespace cpu
} // namespace ATV
