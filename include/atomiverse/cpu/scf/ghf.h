/// @file cpu/scf/ghf.h
/// @brief Generalized Hartree–Fock (GHF) concrete mean-field method — real
///        two-component spinor orbitals, noncollinear spin.
#pragma once

#include <string>

#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Generalized Hartree–Fock — the mean-field method whose molecular
///        orbitals are two-component real spinors φ_i = (φ_i^α; φ_i^β), each
///        component a full AO expansion, so the spin quantization axis may
///        vary in space (noncollinear magnetism). The density and Fock
///        matrices are 2n×2n with 2×2 spin blocks; the Coulomb term sees
///        only the charge density P^αα + P^ββ while the exchange acts on all
///        four blocks: F^{σσ'} = δ_{σσ'}(h + J[P^αα+P^ββ]) − K[P^{σσ'}].
///
/// GHF is the last rung of the constraint-removal ladder RHF → UHF → ROHF →
/// GHF (Fukutome's classification of the eight broken-symmetry HF classes:
/// Int. J. Quantum Chem. 20, 955 (1981), DOI 10.1002/qua.560200502; general
/// spin orbitals: Helgaker–Jørgensen–Olsen, *Molecular Electronic-Structure
/// Theory*, §10). Without spin–orbit terms every UHF solution is a GHF
/// stationary point, so on spin-unfrustrated systems the GHF minimum is
/// collinear and E_GHF = E_UHF (collapse theorem — the correctness
/// sentinel); only spin-frustrated systems (e.g. the equilateral H3
/// triangle) gain energy through a genuinely noncollinear spin arrangement.
///
/// The initial guess must break collinearity explicitly
/// (`guess_spin_rotation`): a collinear guess is preserved by the SCF map,
/// and DIIS then converges onto the collinear UHF stationary point — a
/// saddle on frustrated systems (test-pinned trap).
///
/// This link is real-algebra only (planar noncollinear spin; no complex
/// coefficients, no spin–orbit coupling), energy only. Not supported
/// (throws, no fallback): analytic forces, `mp2()`, `fci()`, `soscf`,
/// `level_shift`, EDIIS/ADIIS mixers, the ASPC predictor, and non-Hcore
/// initial guesses. Requires the stored dense ERI (`compute_eri`, on by
/// default here).
class GHF : public MeanFieldMethod<GHF> {
  public:
    /// @brief Total electron count N (dimensionless; must be set before
    ///        `run`, else std::invalid_argument — the spinor aufbau fills
    ///        the lowest N of the 2n spinors with occupation 1 each).
    int n_electrons = 0;

    /// @brief Collinearity-breaking initial-guess rotation [radians]. The
    ///        guess is the noncollinear-magnetism standard "charge density
    ///        + per-site moments" construction: ½P_c⊗1₂ plus a
    ///        full-strength local moment on each AO μ pointing along
    ///        (sin θ_μ, 0, cos θ_μ) in the real-algebra x-z spin plane,
    ///        θ_μ = `guess_spin_rotation`·μ. The default is a small nonzero
    ///        angle so the guess is never exactly collinear; 0 keeps the
    ///        moments parallel (collinear) and lands on a collinear UHF
    ///        stationary point (the saddle trap on frustrated systems).
    ///        Strongly frustrated systems want the geometry-adapted angle
    ///        (2π/3 for the equilateral H3 triangle) to escape the
    ///        saddle's basin.
    double guess_spin_rotation = 0.2;

    /// @brief Construct with the GHF defaults (generalized HF; Hcore initial
    ///        guess; 0.7 density damping; DIIS capped at 6 vectors and
    ///        started at iteration 5; energy tolerance 1e-8 Ha; stored dense
    ///        ERI on; no nested-force write). Set `n_electrons` before
    ///        `run`.
    GHF();
    /// @brief Construct with an explicit electron count.
    /// @param n_electrons Total electrons N (dimensionless).
    explicit GHF(int n_electrons);
    std::string name() const { return "ghf"; }

  protected:
    /// @brief Occupation from the `n_electrons` member: the spinor channel
    ///        carries all N electrons (β channel unused).
    /// @throws std::invalid_argument when `n_electrons` was not set (≤ 0).
    void resolve_occupation(int n_electrons) override;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
