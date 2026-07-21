/// @file cpu/scf/internal/pcm.h
/// @brief Conductor-like polarizable continuum model (C-PCM / COSMO) as
///        SCF-private workspace.
///
/// Cavity = union of atom-centered spheres (Bondi radii × scale); surface
/// discretized by Lebedev points with hard bury-delete (a point inside any
/// other sphere is discarded — the cavity surface is discontinuous in nuclear
/// coordinates; SWIG smoothing is intentionally out of scope). Conductor
/// equation on tesserae:
///
///   S q = −f(ε) V,   f(ε) = (ε−1)/(ε+x)
///
/// with x=0 (C-PCM, Barone–Cossi / PySCF default) or x=0.5 (original COSMO,
/// Klamt). S uses the Lange–Herbert Gaussian-smeared Coulomb form
/// (erf(ξr)/r, ξ from the Lebedev XI table) when the order is tabulated,
/// else the classical point-charge S with Klamt–Schüürmann diagonal
/// 1.0694 √(4π/a_k). Electrostatic free energy ΔG_el = ½ Σ_k q_k V_k
/// [Hartree]. Non-electrostatic terms (cavitation, dispersion, repulsion)
/// are **not** included.
///
/// Refs: Barone & Cossi, J. Phys. Chem. A 102, 1995 (1998),
/// DOI 10.1021/jp9716997; Klamt & Schüürmann, J. Chem. Soc. Perkin Trans. 2
/// (1993) 799, DOI 10.1039/P29930000799; Born, Z. Phys. 1, 45 (1920).
///
/// No free-function factories — construct `Pcm{…}` only.
#pragma once

#include <array>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/native_integrals.h"

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

/// @brief C-PCM / COSMO reaction-field workspace (geometry-scoped).
///
/// Positions and matrices are internal Hartree–Bohr units after construction
/// (atoms enter in Bohr, as in `IntegralPack::atoms`).
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
    /// @param x             Dielectric scaling convention in f(ε)=(ε−1)/(ε+x):
    ///                      0 → C-PCM, 0.5 → COSMO (dimensionless).
    /// @param lebedev_n     Lebedev points per sphere before bury-delete
    ///                      (default 302, algebraic degree 29).
    /// @throws std::invalid_argument on empty atoms, non-positive ε, unknown
    ///         Lebedev order, or unsupported element.
    Pcm(std::vector<native::Atom> atoms, double eps,
        double radii_scale = 1.2, double x = 0.0, int lebedev_n = 302);

    /// @brief Generate tesserae, factor S, and cache unit-charge AO potential
    ///        matrices ⟨μ|1/|r−s_k||ν⟩ via `nuclear_block` (Z=1) at each
    ///        tessera. Call once per geometry / basis.
    /// @param shells Cartesian AO shell list (engine convention).
    /// @param c2s    Optional Cartesian→spherical transform (n_cart × n_sph);
    ///               empty ⇒ keep matrices in the Cartesian AO basis.
    void prepare(const std::vector<native::Shell>& shells,
                 const xt::xtensor<double, 2>& c2s = {});

    /// @brief Solve the conductor equation for the current total density and
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

    /// @brief Dielectric scaling f(ε) = (ε−1)/(ε+x) (dimensionless).
    [[nodiscard]] double f_epsilon() const;

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

    std::vector<std::array<double, 3>> pos_; ///< Tessera centres [Bohr].
    std::vector<double> area_;               ///< Tessera areas [Bohr²].
    std::vector<double> v_nuc_;              ///< Nuclear potential at tesserae
                                             ///< [Hartree/e].
    xt::xtensor<double, 2> S_;               ///< Conductor Coulomb matrix
                                             ///< [Hartree/e²].
    /// Inverse of S (or LU-solve substitute) [e²/Hartree], for repeated
    /// updates. Built in the constructor after the cavity is assembled.
    xt::xtensor<double, 2> S_inv_;

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
