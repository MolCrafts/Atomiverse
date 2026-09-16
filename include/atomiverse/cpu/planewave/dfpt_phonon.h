/// @file cpu/planewave/dfpt_phonon.h
/// @brief DFPT phonons for the plane-wave method (Γ and finite q; CPU).
///
/// Analytic dynamical matrix from density-functional perturbation theory
/// (self-consistent Sternheimer response) instead of `6N` finite-displacement
/// SCF solves. Pure LSDA **or pure GGA** (PBE, PBEsol, revPBE, RPBE, BLYP)
/// with a UPF norm-conserving pseudopotential. **Fixed occupations** or
/// **metallic smearing** (restricted or collinear spin) (`method.smearing` = Fermi–Dirac /
/// Methfessel–Paxton / Gaussian with `degauss` [Ry] and `extra_bands`) at
/// Γ, commensurate finite \(\mathbf{q}\), and **direct** finite-\(q\)
/// Sternheimer. **Collinear spin** (`method.spin_polarized`) is supported
/// at Γ, commensurate finite \(\mathbf{q}\) (supercell Γ path), and
/// **incommensurate** \(\mathbf{q}\) (direct Sternheimer) with fixed
/// occupations or metallic smear: spin-resolved response with spinor
/// \(f\in[0,1]\) and a spin XC kernel (local \(f_{\mathrm{xc}}^{\sigma\sigma'}\)
/// for LDA; FD of `XCBuilder::step_spin` for GGA). **Restricted metallic
/// smearing** is allowed at Γ, commensurate finite \(\mathbf{q}\), and
/// direct finite-\(q\) Sternheimer. **Collinear spin + metallic smearing**
/// (shared \(\mu\), spinor \(f\)) is allowed at Γ, commensurate
/// \(\mathbf{q}\) (supercell Γ path), and **direct** finite-\(q\) Sternheimer
/// (no \(\delta\mu\) at \(q\neq 0\)). Hybrids are rejected. Default
/// `q_frac = {0,0,0}` is the Γ path. Finite \(\mathbf{q}\) that is
/// **commensurate** with a small supercell (e.g. \(X=(1/2,0,0)\) →
/// \(2\times1\times1\)) is evaluated as Γ-DFPT on that supercell plus IFC
/// Fourier (exactly equivalent for rational \(q\)). **Incommensurate** (or
/// oversize-supercell) \(\mathbf{q}\) uses the direct finite-\(q\) Sternheimer
/// assembly (`internal::dfpt_q_force_constants`) on the primitive cell
/// (restricted fixed-occ / smear, collinear spin fixed-occ or spin+smear).
/// GGA \(K_{\mathrm{xc}}\) is the second functional derivative of the same
/// `XCBuilder` operator used in the ground-state SCF (central FD of
/// \(V_{\mathrm{xc}}[n]\) or `step_spin`); pure LDA keeps a local
/// \(f_{\mathrm{xc}}\) table.
/// Parameters are members — no public `*Config` bag (`Bands` precedent).
///
/// Reference: Baroni, de Gironcoli, Dal Corso, Giannozzi,
/// *Rev. Mod. Phys.* **73**, 515 (2001).
///
/// @code
///   cpu::pw::PW pw;
///   pw.upf_path = "Si.upf";
///   pw.ecut_ry = 15.0;
///   pw.k_mesh = 2;
///   pw.xc = cpu::scf::XCType::PBE;   // pure GGA kernel OK
///
///   cpu::pw::DfptPhonon dfpt;
///   dfpt.q_frac = {0.3, 0.0, 0.0};   // incommensurate along Γ–X
///   auto r = dfpt.compute(pw, frame); // ω [cm⁻¹] at q
/// @endcode
#pragma once

#include <array>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

namespace ATV {
namespace cpu {
namespace pw {

/// @brief Result of a DFPT phonon calculation (Γ or finite q).
///
/// Frame purity: spectra and force constants live here, never on Frame.
struct DfptPhononResult {
    /// @brief Wavenumbers [cm⁻¹], length `3N`, ascending; imaginary modes
    ///        (λ < 0) are reported as negative values.
    xt::xtensor<double, 1> frequencies;

    /// @brief Real part of the Cartesian force-constant matrix [eV/Å²],
    ///        shape `[3N, 3N]`, index `3·I + α`. At Γ this is the full
    ///        (symmetric) matrix; at finite q it is \(\mathrm{Re}\,C(q)\).
    xt::xtensor<double, 2> force_constants;

    /// @brief Imaginary part of \(C(q)\) [eV/Å²], shape `[3N, 3N]`.
    ///        Zero (to roundoff) at Γ.
    xt::xtensor<double, 2> force_constants_imag;

    /// @brief Whether the ground-state SCF converged.
    bool scf_converged = false;

    /// @brief Whether every one of the `3N` response SCFs converged.
    bool response_converged = false;

    /// @brief Max |C − C†|/2 before Hermitian projection [eV/Å²].
    double max_asymmetry = 0;

    /// @brief Atoms in the cell.
    int n_atoms = 0;
};

/// @brief DFPT phonon calculator over a `PW` ground state (Γ or finite q).
///
/// `compute` runs one ground-state SCF, then `3N` self-consistent
/// Sternheimer responses at `q_frac`, and diagonalizes the mass-weighted
/// Hermitian dynamical matrix (masses from the IUPAC table by atomic
/// number, the same convention as `cpu::vib`). All workspace is private
/// and rebuilt per call; the input Frame is not mutated.
class DfptPhonon {
  public:
    /// @brief Response-density L2 convergence per perturbation
    ///        [electrons/Bohr³ per Bohr, volume-weighted grid norm].
    double response_tol = 1e-8;
    /// @brief Response SCF iteration cap per perturbation (dimensionless).
    int response_max_iter = 80;
    /// @brief Sternheimer CG relative residual tolerance (dimensionless).
    double cg_tol = 1e-10;
    /// @brief Sternheimer CG iteration cap (dimensionless).
    int cg_max_iter = 200;
    /// @brief Linear mixing factor for the response density in (0, 1].
    double mix_beta = 0.5;
    /// @brief Phonon wavevector in fractional crystal coordinates of the
    ///        simulation cell (dimensionless). Default `{0,0,0}` = Γ.
    ///        Rational values with supercell product ≤ 8 use the
    ///        commensurate Γ-supercell path; other values use direct
    ///        finite-\(q\) Sternheimer on the primitive cell.
    std::array<double, 3> q_frac{{0.0, 0.0, 0.0}};

    /// @brief DFPT phonons at `q_frac` for the geometry in `frame`.
    ///
    /// Throws `std::invalid_argument` unless `method.xc` is pure LSDA or
    /// pure GGA (`hybrid_exchange_fraction == 0`) and `method.upf_path` is
    /// set. Restricted metallic smearing is allowed at Γ, commensurate
    /// \(\mathbf{q}\), and direct finite-\(q\) Sternheimer. Collinear spin
    /// (fixed occupations or metallic smear) is allowed at Γ, commensurate
    /// \(\mathbf{q}\), and incommensurate \(\mathbf{q}\).
    ///
    /// @param method Plane-wave method (members only; not mutated).
    /// @param frame  Atom positions [Å], types (Z), simulation cell.
    /// @return Owned result with ω [cm⁻¹] and force constants [eV/Å²].
    DfptPhononResult compute(const PW& method, const cpu::Frame& frame) const;
};

} // namespace pw
} // namespace cpu
} // namespace ATV
