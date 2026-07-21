/// @file cpu/scf/xc_functional.h
/// @brief Exchange-correlation functional evaluator for KS / UKS.
///
/// Supports LSDA (Slater exchange + VWN5 correlation), PBE GGA, and the
/// B3LYP / PBE0 global hybrids. A global hybrid mixes a functional-specific
/// fraction of exact Hartree-Fock exchange (see `hybrid_exchange_fraction`)
/// on top of a semi-local exchange-correlation part; `evaluate_xc` returns
/// only the semi-local part, the HF-exchange admixture is applied by the
/// Fock builder.
///
/// Evaluates XC energy density and potential derivatives at a single grid point.
///
/// References:
///   - Slater: Dirac, Proc. Cambridge Phil. Soc. 26, 376 (1930)
///   - VWN5: Vosko, Wilk, Nusair, Can. J. Phys. 58, 1200 (1980), parameterization V
///   - PBE: Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996)
///   - B3LYP: Stephens, Devlin, Chabalowski, Frisch, J. Phys. Chem. 98, 11623 (1994)
///   - PBE0: Adamo, Barone, J. Chem. Phys. 110, 6158 (1999)
#pragma once

namespace ATV {
namespace cpu {
namespace scf {

/// @brief XC functional type selector.
///
/// `LSDA` and `PBE` are pure (semi-local) functionals. `B3LYP` and `PBE0`
/// are global hybrids: their semi-local part is evaluated by `evaluate_xc`
/// while a fixed fraction of exact HF exchange (`hybrid_exchange_fraction`)
/// is added by the Fock builder.
enum class XCType { LSDA, PBE, B3LYP, PBE0 };

/// @brief Fraction of exact Hartree-Fock exchange (`alpha_hf`) for a functional.
///
/// Returns the global-hybrid exact-exchange admixture: `0.20` for B3LYP,
/// `0.25` for PBE0, and `0.0` for every pure functional (`LSDA`, `PBE`).
/// A `0.0` return is the signal "pure functional, no exchange-matrix build".
///
/// @param type XC functional type.
/// @return alpha_hf, the dimensionless exact-exchange fraction in [0, 1].
double hybrid_exchange_fraction(XCType type);

/// @brief Whether a functional has a gradient (GGA) dependence.
///
/// True for `PBE`, `B3LYP` and `PBE0` (all carry a semi-local GGA term);
/// false for the pure-LDA `LSDA`. Used to decide whether the XC grid engine
/// must build basis-function gradients.
///
/// @param type XC functional type.
/// @return `true` if the semi-local part needs density gradients.
bool is_gga(XCType type);

/// @brief Output from XC evaluation at a single grid point.
struct XCOutput {
    double
        exc; ///< XC energy density per electron f(rho_alpha, rho_beta) [Hartree].
    double vrho_alpha; ///< df/drho_alpha (XC potential for alpha).
    double vrho_beta;  ///< df/drho_beta (XC potential for beta).
    // GGA only:
    double vsigma_aa; ///< df/dsigma_alpha_alpha.
    double vsigma_ab; ///< df/dsigma_alpha_beta.
    double vsigma_bb; ///< df/dsigma_beta_beta.
};

/// @brief Evaluate XC functional at a single grid point.
///
/// Returns the XC energy density f such that E_xc = integral f(rho) dtau,
/// along with potential derivatives for V_xc matrix construction.
///
/// Applies density cutoff: if rho_total < 1e-12, returns zero for all outputs.
///
/// @param type  Functional type (LSDA, PBE, B3LYP or PBE0).
/// @param rho_a Alpha density at grid point [electrons/Bohr^3].
/// @param rho_b Beta density at grid point [electrons/Bohr^3].
/// @param sigma_aa |grad rho_alpha|^2 (GGA only, ignored for LSDA).
/// @param sigma_ab grad rho_alpha . grad rho_beta (GGA only).
/// @param sigma_bb |grad rho_beta|^2 (GGA only).
/// @return XCOutput with energy density and potential derivatives.
XCOutput evaluate_xc(XCType type, double rho_a, double rho_b,
                     double sigma_aa = 0, double sigma_ab = 0,
                     double sigma_bb = 0);

/// @brief Batched XC evaluation over an array of grid points.
///
/// Processes `n` grid points at once, enabling SIMD auto-vectorization of
/// transcendental functions (pow, log, exp).
///
/// @param type     Functional type (LSDA, PBE, B3LYP or PBE0).
/// @param n        Number of grid points.
/// @param rho_a    Alpha densities [n].
/// @param rho_b    Beta densities [n].
/// @param sigma_aa |grad rho_alpha|^2 array [n] (GGA only, may be nullptr for LDA).
/// @param sigma_ab grad rho_alpha . grad rho_beta array [n] (GGA only).
/// @param sigma_bb |grad rho_beta|^2 array [n] (GGA only).
/// @param[out] exc     XC energy density per electron [n].
/// @param[out] vrho_a  df/drho_alpha [n].
/// @param[out] vrho_b  df/drho_beta [n].
/// @param[out] vsigma_aa df/dsigma_alpha_alpha [n] (GGA only, may be nullptr for LDA).
/// @param[out] vsigma_ab df/dsigma_alpha_beta [n] (GGA only).
/// @param[out] vsigma_bb df/dsigma_beta_beta [n] (GGA only).
void evaluate_xc_batch(XCType type, int n, const double* rho_a,
                       const double* rho_b, const double* sigma_aa,
                       const double* sigma_ab, const double* sigma_bb,
                       double* exc, double* vrho_a, double* vrho_b,
                       double* vsigma_aa, double* vsigma_ab, double* vsigma_bb);

} // namespace scf
} // namespace cpu
} // namespace ATV
