/// @file cpu/scf/internal/pcm.h
/// @brief Polarizable continuum models (C-PCM / COSMO / IEF-PCM) as
///        SCF-private workspace.
///
/// Cavity = union of atom-centered spheres (Bondi radii × scale); surface
/// discretized by Lebedev points with hard bury-delete (a point inside any
/// other sphere is discarded — the cavity surface is discontinuous in nuclear
/// coordinates; SWIG smoothing is intentionally out of scope).
///
/// Conductor-like models (C-PCM / COSMO):
///
///   S q = −f(ε) V,   f(ε) = (ε−1)/(ε+x)
///
/// with x=0 (C-PCM, Barone–Cossi / PySCF default) or x=0.5 (original COSMO,
/// Klamt).
///
/// IEF-PCM (Cancès–Mennucci–Tomasi; PySCF `method='IEF-PCM'`):
///
///   f(ε) = (ε−1)/(ε+1)
///   DA = D ⊙ A  (column-scale by tessera areas)
///   K = S − f/(2π) DA S
///   R = −f (I − DA/(2π))
///   K q = R V
///
/// with q symmetrized for a real free energy: q_sym = (q + q̃)/2 where
/// Kᵀ v_K = V and q̃ = Rᵀ v_K. Electrostatic free energy
/// ΔG_el = ½ q_sym · V [Hartree].
///
/// S uses the Lange–Herbert Gaussian-smeared Coulomb form (erf(ξr)/r, ξ from
/// the Lebedev XI table) when the order is tabulated, else the classical
/// point-charge S with Klamt–Schüürmann diagonal 1.0694 √(4π/a_k). D is the
/// corresponding double-layer operator (Lange & Herbert, JCP 133, 244111
/// (2010)). Non-electrostatic terms are **not** included.
///
/// Refs: Barone & Cossi, J. Phys. Chem. A 102, 1995 (1998),
/// DOI 10.1021/jp9716997; Klamt & Schüürmann, J. Chem. Soc. Perkin Trans. 2
/// (1993) 799, DOI 10.1039/P29930000799; Cancès, Mennucci & Tomasi,
/// J. Chem. Phys. 107, 3032 (1997), DOI 10.1063/1.474659; Born, Z. Phys. 1,
/// 45 (1920).
///
/// No free-function factories — construct `Pcm{…}` only.
#pragma once

#include <array>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/scf_enums.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Bondi van der Waals radius [Å] (Bondi, J. Phys. Chem. 68, 441
///        (1964)), with the common PCM **modified Bondi** hydrogen radius
///        of 1.1 Å (PySCF / Q-Chem). Unsupported Z throws — no fallback.
/// @param Z Atomic number (dimensionless).
/// @return Radius [Å].
/// @throws std::invalid_argument when Z is outside the table.
[[nodiscard]] double bondi_radius_ang(int Z);

/// @brief PCM reaction-field workspace (geometry-scoped).
///
/// Positions and matrices are internal Hartree–Bohr units after construction
/// (atoms enter in Bohr, as in `internal::IntegralWorkspace::atoms`).
class Pcm {
  public:
    /// @brief Result of one density-dependent reaction-field update.
    struct Result {
        xt::xtensor<double, 2> V_pcm; ///< Reaction-field AO matrix [Hartree].
        double delta_g_el = 0;        ///< Electrostatic free energy ½ q·V
                                      ///< [Hartree].
    };

    /// @brief Build the cavity from nuclear geometry and solvent parameters.
    /// @param atoms         Nuclei (Z, positions [Bohr]).
    /// @param eps           Solvent dielectric constant ε (dimensionless);
    ///                      must be > 0 when the model is active.
    /// @param radii_scale   Multiplier on Bondi radii (dimensionless;
    ///                      default 1.2, Q-Chem / PySCF convention).
    /// @param x             Dielectric scaling in f(ε)=(ε−1)/(ε+x) for
    ///                      conductor models (0 → C-PCM, 0.5 → COSMO).
    ///                      Ignored for `SolventModel::IEFPCM`.
    /// @param lebedev_n     Lebedev points per sphere before bury-delete
    ///                      (default 302, algebraic degree 29).
    /// @param model         Continuum model (default C-PCM).
    /// @throws std::invalid_argument on empty atoms, non-positive ε, unknown
    ///         Lebedev order, or unsupported element.
    Pcm(std::vector<native::Atom> atoms, double eps, double radii_scale = 1.2,
        double x = 0.0, int lebedev_n = 302,
        SolventModel model = SolventModel::CPCM);

    /// @brief Generate tesserae, factor the PCM kernel, and cache unit-charge
    ///        AO potential matrices ⟨μ|1/|r−s_k||ν⟩ via `nuclear_block` (Z=1)
    ///        at each tessera. Call once per geometry / basis.
    /// @param shells Cartesian AO shell list (engine convention).
    /// @param c2s    Optional Cartesian→spherical transform (n_cart × n_sph);
    ///               empty ⇒ keep matrices in the Cartesian AO basis.
    void prepare(const std::vector<native::Shell>& shells,
                 const xt::xtensor<double, 2>& c2s = {});

    /// @brief Solve the continuum equation for the current total density and
    ///        contract the apparent surface charges into V_pcm.
    /// @param P_total Total density matrix P^α+P^β (closed-shell: the
    ///                occupied-scaled total P) in the same AO basis as the
    ///                prepared potential matrices [e].
    /// @return V_pcm [Hartree] and ΔG_el [Hartree].
    /// @pre `prepare` has been called (or the pure-nuclear path with no AOs
    ///      is used via empty shells).
    [[nodiscard]] Result update(const xt::xtensor<double, 2>& P_total) const;

    /// @brief Pure-nuclear update (P = 0) — Born-ion / cavity-only tests.
    /// @return V_pcm (zero if no AOs prepared) and ΔG_el [Hartree].
    [[nodiscard]] Result update_nuclear_only() const;

    /// @brief Dielectric scaling f(ε) used by the active model (dimensionless).
    [[nodiscard]] double f_epsilon() const;

    /// @brief Active continuum model.
    [[nodiscard]] SolventModel model() const { return model_; }

    /// @brief Number of exposed tesserae after bury-delete (dimensionless).
    [[nodiscard]] int n_tesserae() const {
        return static_cast<int>(pos_.size());
    }

    /// @brief Apparent surface charges from the most recent `update` [e].
    ///        Empty until the first update.
    [[nodiscard]] const std::vector<double>& last_charges() const {
        return last_q_;
    }

    /// @brief Tessera positions [Bohr] (const view of the cavity surface).
    [[nodiscard]] const std::vector<std::array<double, 3>>& positions() const {
        return pos_;
    }

    /// @brief Tessera areas a_k [Bohr²].
    [[nodiscard]] const std::vector<double>& areas() const { return area_; }

  private:
    std::vector<native::Atom> atoms_; ///< Nuclei [Bohr].
    double eps_ = 0;
    double x_ = 0;
    double f_eps_ = 0;
    SolventModel model_ = SolventModel::CPCM;

    std::vector<std::array<double, 3>> pos_; ///< Tessera centres [Bohr].
    std::vector<std::array<double, 3>> nrm_; ///< Outward unit normals.
    std::vector<double> area_;               ///< Tessera areas [Bohr²].
    std::vector<double> R_sphere_;           ///< Sphere radius of parent
                                             ///< atom [Bohr] (per tessera).
    std::vector<double> xi_;                 ///< Lange–Herbert Gaussian ξ.
    std::vector<double> v_nuc_;              ///< Nuclear potential at tesserae
                                             ///< [Hartree/e].
    xt::xtensor<double, 2> S_;               ///< Single-layer Coulomb matrix
                                             ///< [Hartree/e²].
    /// Inverse of S for conductor models (C-PCM / COSMO). Empty for IEF-PCM.
    xt::xtensor<double, 2> S_inv_;
    /// IEF-PCM kernel K and right-hand-side matrix R (empty for conductor).
    xt::xtensor<double, 2> K_;
    xt::xtensor<double, 2> R_;

    /// Cached unit-charge nuclear-attraction matrices at each tessera
    /// (nuclear_block Z=1 = −⟨μ|1/r|ν⟩), AO basis after optional c2s
    /// sandwich [Hartree/e].
    std::vector<xt::xtensor<double, 2>> V_unit_;
    int n_ao_ = 0;

    mutable std::vector<double> last_q_; ///< Last solved ASC [e].

    Result solve_and_contract_(const std::vector<double>& V) const;
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
