/// @file cpu/scf/xc_functional.h
/// @brief Exchange-correlation functional evaluator for KS / UKS.
///
/// Supports LSDA (Slater exchange + VWN5 correlation), pure GGAs (PBE,
/// PBEsol, revPBE, RPBE, BLYP), the B3LYP / PBE0 global hybrids, the
/// CAM-B3LYP range-separated hybrid, and the B2PLYP double hybrid.
/// A global hybrid mixes a functional-specific fraction of exact
/// Hartree-Fock exchange (see `hybrid_exchange_fraction`) on top of a
/// semi-local exchange-correlation part; `evaluate_xc` returns only the
/// semi-local part, the HF-exchange admixture is applied by the Fock
/// builder. Double hybrids add a PT2/MP2 correlation scale
/// (`hybrid_pt2_fraction`) applied after the KS SCF by `MP2` on the RKS reference
/// (`MP2::os_scale` / `ss_scale` = a_c), not a mean-field member.
///
/// Evaluates XC energy density and potential derivatives at a single grid point.
///
/// References:
///   - Slater: Dirac, Proc. Cambridge Phil. Soc. 26, 376 (1930)
///   - VWN5: Vosko, Wilk, Nusair, Can. J. Phys. 58, 1200 (1980), parameterization V
///   - PBE: Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996)
///   - PBEsol: Perdew et al., PRL 100, 136406 (2008) — solid-state μ=10/81,
///     β=0.046 (libxc / CP2K XC_FUNCTIONAL PBEsol convention)
///   - revPBE: Zhang & Yang, PRL 80, 890 (1998) — PBE form with κ=1.245
///     exchange (libxc GGA_X_PBE_R + GGA_C_PBE / CP2K XC_FUNCTIONAL revPBE)
///   - RPBE: Hammer, Hansen, Nørskov, PRB 59, 7413 (1999) — exponential
///     exchange enhancement \(F_x=1+\kappa(1-e^{-\mu s^2/\kappa})\) with
///     PBE κ/μ + PBE correlation (libxc GGA_X_RPBE + GGA_C_PBE /
///     CP2K XC_FUNCTIONAL RPBE). Distinct from revPBE.
///   - BLYP: B88 (Becke, Phys. Rev. A 38, 3098 (1988)) + LYP (Lee-Yang-Parr,
///     Phys. Rev. B 37, 785 (1988); Miehlich et al., Chem. Phys. Lett. 157,
///     200 (1989))
///   - OptB88VdW semi-local: optimized B88 exchange (Klimeš et al., J. Phys.:
///     Condens. Matter 22, 022201 (2010); libxc GGA_X_OPTB88_VDW) + PW92 LDA
///     correlation. Pair with GPW `vdw_df` (DRSLL) for full optB88-vdW.
///   - B3LYP: Stephens, Devlin, Chabalowski, Frisch, J. Phys. Chem. 98, 11623 (1994)
///   - PBE0: Adamo, Barone, J. Chem. Phys. 110, 6158 (1999)
///   - B2PLYP: Grimme, J. Chem. Phys. 124, 034108 (2006) — a_x=0.53 HF,
///     (1−a_x) B88, (1−a_c) LYP with a_c=0.27, plus a_c · E_MP2 on the KS
///     orbitals (double hybrid; PT2 is not inside `evaluate_xc`).
#pragma once

#include "atomiverse/electronic/xc_functional.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief XC functional type selector.
///
/// `LSDA`, `PBE`, `PBEsol`, `RevPBE`, `RPBE`, `BLYP` and `OptB88VdW` are pure
/// (semi-local) functionals. `B3LYP` and `PBE0` are global hybrids: their semi-local part
/// is evaluated by `evaluate_xc` while a fixed fraction of exact HF exchange
/// (`hybrid_exchange_fraction`) is added by the Fock builder. `CAMB3LYP` is
/// a range-separated hybrid: .35 B88 + .46 short-range ITYH-B88 +
/// .19 VWN5 + .81 LYP, plus
/// \(\alpha K[1/r] + \beta K[\mathrm{erf}(\mu r)/r]\) (see
/// `rsh_exchange_params`). `B2PLYP` is a **double hybrid**: same hybrid-KS
/// build with a_x=0.53 plus a PT2 correlation scale a_c=0.27 applied by
/// the MP2 path (see `hybrid_pt2_fraction`).
enum class XCType {
    LSDA,
    PBE,
    PBEsol,
    RevPBE,
    RPBE,
    BLYP,
    OptB88VdW, ///< optB88 exchange + PW92 LDA-C (use with GPW vdw_df for full).
    B3LYP,
    PBE0,
    B2PLYP,
    CAMB3LYP, ///< Yanai CAM-B3LYP; RSH params via `rsh_exchange_params`.
    Custom ///< Constructor-injected semi-local provider; not a built-in kernel.
};

/// @brief Range-separated hybrid (RSH) exact-exchange parameters.
///
/// The Fock builder mixes
/// \(K = \alpha\,K[1/r] + \beta\,K[\mathrm{erf}(\mu r)/r]\).
/// Pure functionals and global hybrids return `beta = 0` and `mu = 0`
/// (single-operator path). CAM-B3LYP returns the Yanai / PySCF / Gaussian
/// set \((\alpha,\beta,\mu) = (0.19, 0.46, 0.33\,\mathrm{Bohr}^{-1})\).
struct RshExchangeParams {
    double alpha = 0; ///< Bare-Coulomb HF weight (dimensionless).
    double beta = 0;  ///< erf-Coulomb HF weight (dimensionless).
    double mu = 0;    ///< Range-separation parameter [1/Bohr].
};

/// @brief RSH exact-exchange parameters for a functional.
///
/// @param type XC functional type.
/// @return CAM-B3LYP: (0.19, 0.46, 0.33); every other type: (α_hf, 0, 0)
///         with α_hf from `hybrid_exchange_fraction`.
RshExchangeParams rsh_exchange_params(XCType type);

/// @brief Fraction of exact Hartree-Fock exchange (`alpha_hf`) for a functional.
///
/// Returns the global-hybrid exact-exchange admixture: `0.20` for B3LYP,
/// `0.25` for PBE0, `0.53` for B2PLYP, and `0.0` for every pure functional
/// (`LSDA`, `PBE`, `PBEsol`, `RevPBE`, `RPBE`, `BLYP`, `OptB88VdW`). A `0.0` return is the
/// signal "pure functional, no exchange-matrix build".
///
/// @param type XC functional type.
/// @return alpha_hf, the dimensionless exact-exchange fraction in [0, 1].
double hybrid_exchange_fraction(XCType type);

/// @brief PT2 / MP2 correlation scale (`a_c`) for double-hybrid functionals.
///
/// Returns `0.27` for B2PLYP (Grimme 2006) and `0.0` for every pure or
/// single-hybrid functional. When non-zero, the total double-hybrid energy
/// is \(E_{\mathrm{KS}} + a_c\,E_{\mathrm{MP2}}\) on the converged KS
/// orbitals; callers set `MP2::os_scale = MP2::ss_scale = a_c` and
/// `MP2{rks}.run(frame)`.
///
/// @param type XC functional type.
/// @return a_c, the dimensionless PT2 coefficient in [0, 1].
double hybrid_pt2_fraction(XCType type);

/// @brief Whether a functional has a gradient (GGA) dependence.
///
/// True for `PBE`, `PBEsol`, `RevPBE`, `RPBE`, `BLYP`, `OptB88VdW`, `B3LYP`,
/// `PBE0`, `CAMB3LYP` and `B2PLYP` (all carry a semi-local GGA term); false for the pure-LDA
/// `LSDA`. Used to decide whether the XC grid engine must build
/// basis-function gradients.
///
/// @param type XC functional type.
/// @return `true` if the semi-local part needs density gradients.
bool is_gga(XCType type);

/// @brief Output from XC evaluation at a single grid point.
struct XCOutput {
    double energy_density; ///< f_xc [Ha/Bohr^3], E_xc = integral f_xc dr.
    double vrho_alpha;     ///< df/drho_alpha (XC potential for alpha).
    double vrho_beta;      ///< df/drho_beta (XC potential for beta).
    // GGA only:
    double vsigma_aa; ///< df/dsigma_alpha_alpha.
    double vsigma_ab; ///< df/dsigma_alpha_beta.
    double vsigma_bb; ///< df/dsigma_beta_beta.
};

/// @brief Evaluate XC functional at a single grid point.
///
/// Returns f_xc per volume, E_xc = integral f_xc dr, and derivatives
/// of that same energy density for V_xc construction.
///
/// Applies density cutoff: if rho_total < 1e-12, returns zero for all outputs.
///
/// @param type  Functional type (LSDA, PBE, PBEsol, RevPBE, RPBE, BLYP,
///              OptB88VdW, B3LYP, PBE0, CAMB3LYP or B2PLYP).
/// @param rho_a Alpha density at grid point [electrons/Bohr^3].
/// @param rho_b Beta density at grid point [electrons/Bohr^3].
/// @param sigma_aa |grad rho_alpha|^2 (GGA only, ignored for LSDA).
/// @param sigma_ab grad rho_alpha . grad rho_beta (GGA only).
/// @param sigma_bb |grad rho_beta|^2 (GGA only).
/// @return XCOutput with energy density and potential derivatives.
XCOutput evaluate_xc(XCType type, double rho_a, double rho_b,
                     double sigma_aa = 0, double sigma_ab = 0,
                     double sigma_bb = 0);

/// @brief Built-in semi-local XC provider, using the existing native math kernels.
class BuiltinXc final : public ATV::electronic::XcFunctional {
  public:
    /// @brief Select a built-in functional's semi-local part.
    /// @param type Built-in identity; Custom and invalid identities reject.
    explicit BuiltinXc(XCType type);
    /// @brief Built-in identity, including its separately applied exact exchange.
    /// @return Selected built-in functional.
    XCType type() const noexcept { return type_; }
    ATV::electronic::XcIngredients ingredients() const noexcept override;
    bool supports(ATV::electronic::XcSpin spin,
                  ATV::electronic::XcOrder order) const noexcept override;

  protected:
    void evaluate_batch(const ATV::electronic::XcBatch& in,
                        ATV::electronic::XcOrder order,
                        ATV::electronic::XcOutput& out) const override;

  private:
    XCType type_;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
