/// @file cpu/scf/internal/mp2_pack.h
/// @brief Private pack-based MP2 entry (unit tests / method implementation).
#pragma once

#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/mp2.h"
#include "atomiverse/cpu/scf/slater_determinant.h" // SlaterDeterminant

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell canonical MP2 energy from a converged reference.
/// @param ints     Integral pack with the ERI tensor materialized.
/// @param det      Converged closed-shell reference determinant (MO
///                 coefficients, orbital energies, electron counts).
/// @param e_scf_ha Converged SCF total energy [Hartree] (reported as the
///                 reference part of `MP2Result::e_tot`).
/// @param n_frozen Number of lowest occupied spatial MOs excluded from the
///                 correlation sum (frozen core). 0 = all occupied correlated.
///                 Matches PySCF `mp.MP2(mf, frozen=n_frozen)`.
/// @param os_scale Opposite-spin scale \(c_{\mathrm{OS}}\) (1 = plain MP2;
///                 1.2 = SCS-MP2). Dimensionless.
/// @param ss_scale Same-spin scale \(c_{\mathrm{SS}}\) (1 = plain MP2;
///                 1/3 = SCS-MP2). Dimensionless.
/// @return MP2 correlation and total energies [Hartree]; unscaled OS/SS on
///         `e_corr_os` / `e_corr_ss`.
MP2Result mp2_energy(const internal::IntegralView& ints,
                     const SlaterDeterminant& det, double e_scf_ha,
                     int n_frozen = 0, double os_scale = 1.0,
                     double ss_scale = 1.0);

/// @brief Unrestricted canonical MP2 (UMP2) on a UHF/UKS-style reference.
///
/// Spin-orbital second-order energy decomposed into same-spin αα/ββ
/// (antisymmetrized integrals, factor ¼) and opposite-spin αβ channels:
///
/// \f[
///   E^{(2)}_{\sigma\sigma}
///     = \tfrac14\sum_{ijab}
///       \frac{|(ia|jb)-(ib|ja)|^2}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b},
/// \quad
///   E^{(2)}_{\alpha\beta}
///     = \sum_{i_\alpha j_\beta a_\alpha b_\beta}
///       \frac{(i_\alpha a_\alpha|j_\beta b_\beta)^2}
///            {\varepsilon_{i\alpha}+\varepsilon_{j\beta}
///             -\varepsilon_{a\alpha}-\varepsilon_{b\beta}}.
/// \f]
///
/// Matches raw (unprojected) PySCF `mp.UMP2`. No frozen core.
/// Optional `os_scale` / `ss_scale` apply Grimme SCS/SOS scaling to the
/// unscaled channels (defaults 1 / 1 = plain UMP2).
/// @param ints     Integral pack with the ERI tensor materialized.
/// @param det      Converged unrestricted determinant (independent α/β MOs).
/// @param e_scf_ha Converged UHF total energy [Hartree].
/// @param os_scale Opposite-spin scale (dimensionless).
/// @param ss_scale Same-spin scale (dimensionless).
/// @return UMP2 correlation and total energies [Hartree].
MP2Result ump2_energy(const internal::IntegralView& ints,
                      const SlaterDeterminant& det, double e_scf_ha,
                      double os_scale = 1.0, double ss_scale = 1.0);

/// @brief ROHF-MP2 (PySCF convention): UMP2 on ROHF spatial MOs with
///        semi-canonical spin-Fock diagonal orbital energies
///        \(\varepsilon_i^\sigma = (C^\dagger F_\sigma C)_{ii}\).
///
/// Matches `mp.UMP2(scf.ROHF(...))` / `mp.MP2` on an ROHF reference, which
/// converts to UHF (shared \(C\), spin-resolved \(\varepsilon\)) then runs
/// raw unprojected UMP2. No frozen core.
/// @param ints     Integral pack with ERI materialized.
/// @param det      Converged ROHF determinant (shared spatial \(C\); Roothaan
///                 \(\varepsilon\) on the det are ignored).
/// @param F_alpha  True α Fock matrix in the AO basis [Hartree] (ROHF
///                 uncoupled spin Fock, not \(F_{\mathrm{eff}}\)).
/// @param F_beta   True β Fock matrix in the AO basis [Hartree].
/// @param e_scf_ha Converged ROHF total energy [Hartree].
/// @param os_scale Opposite-spin scale (dimensionless).
/// @param ss_scale Same-spin scale (dimensionless).
/// @return ROMP2 correlation and total energies [Hartree].
MP2Result romp2_energy(const internal::IntegralView& ints,
                       const SlaterDeterminant& det, const RealMatrix& F_alpha,
                       const RealMatrix& F_beta, double e_scf_ha,
                       double os_scale = 1.0, double ss_scale = 1.0);

/// @brief Closed-shell RI-MP2 correlation energy via Coulomb-metric density
///        fitting (same `DensityFitting` workspace as SCF RI-J/JK). Does not
///        require a materialized 4-center ERI tensor.
///
/// The fitted amplitudes B_ov(ia,P) = Σ_μν C_μi C_νa B_μν^P are assembled
/// BLAS-shaped from the frozen `DensityFitting::b_tensor()` output (Coulomb
/// metric V=(P|Q), Cholesky-whitened W=L⁻¹ ⇒ V⁻¹), then the (ia|jb) table is
/// formed with a single dgemm (ia|jb) = Σ_P B_ov(ia,P) B_ov(jb,P). Gaussian
/// centers are in Bohr; all energies in Hartree.
/// @param ints     Integral pack (shells + atoms, centers in Bohr).
/// @param det      Converged closed-shell RHF determinant.
/// @param e_scf_ha SCF energy [Hartree].
/// @param aux      Auxiliary basis (def2 universal JKFIT).
/// @param n_frozen Lowest occupied spatial MOs frozen (same convention as
///                 `mp2_energy`).
/// @param os_scale Opposite-spin scale (dimensionless; default 1).
/// @param ss_scale Same-spin scale (dimensionless; default 1).
/// @return MP2Result with RI (ia|jb)≈Σ_P B_ia^P B_jb^P [Hartree].
MP2Result ri_mp2_energy(const internal::IntegralView& ints,
                        const SlaterDeterminant& det, double e_scf_ha,
                        AuxBasis aux, int n_frozen = 0, double os_scale = 1.0,
                        double ss_scale = 1.0);

} // namespace scf
} // namespace cpu
} // namespace ATV
