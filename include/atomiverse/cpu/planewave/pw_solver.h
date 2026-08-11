/// @file cpu/pw/pw_solver.h
/// @brief cpu::pw::PW — concrete plane-wave DFT method (CPU Driver node).
///
/// Parameters are public members (RHF pattern). Frame is the only I/O.
/// Self-consistency is a **Driver loop**: `PW` derives
/// `cpu::Driver<ATV::pw::Phase>`. All product SCF paths (restricted
/// fixed-occupation, collinear fixed-occupation, restricted metallic
/// smearing, collinear + metallic smearing) are owned as stages
/// (`veff → diagonalize → density → mix → check`) with path-local StepFns.
/// Workflow entry: `PWEvaluator` or direct `build`/`run`.
///
/// @code
///   cpu::pw::PW pw;
///   pw.ecut_ry = 30.0;
///   pw.k_mesh = 4;
///   pw.build(frame);
///   const auto& result = pw.run(frame); // [Ry]/Ha on result
/// @endcode
#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/stages/density_stage.h"
#include "atomiverse/cpu/planewave/stages/diagonalize_stage.h"
#include "atomiverse/cpu/planewave/stages/mix_stage.h"
#include "atomiverse/cpu/planewave/stages/pw_converged.h"
#include "atomiverse/cpu/planewave/stages/veff_stage.h"
#include "atomiverse/cpu/scf/xc_functional.h"

namespace ATV {
namespace cpu {
namespace pw {

namespace internal {
class PoissonSolver;
class XCBuilder;
class VeffBuilder;
class DavidsonSolver;
class DensityBuilder;
class ConvergenceCheck;
} // namespace internal

/// @brief Result of a plane-wave DFT total energy calculation.
struct PWResult {
    double energy = 0;      ///< Total energy [Ry] (QE-comparable).
    double energy_ha = 0;   ///< Total energy [Hartree].
    bool converged = false; ///< Whether SCF converged within max_iter.
    int iterations = 0;     ///< Actual SCF iterations taken.
    /// @brief Collinear magnetization N↑ − N↓ [electrons] after SCF.
    ///        Zero when `spin_polarized == false`. Fixed occupations set this
    ///        exactly to `multiplicity - 1`. With metallic smearing it is the
    ///        self-consistent \(\sum w(f_\uparrow - f_\downarrow)\).
    double magnetization = 0;
    double e_fermi = 0;    ///< Fermi level [Ry] (0 if fixed occupations).
    double n_elec_occ = 0; ///< \(\sum w f\) after last iteration.
    double e_entropy = 0;  ///< \(-TS\) contribution included in energy [Ry].
    /// @brief Total Monkhorst–Pack k-points of this run (`k_mesh`³ set).
    int n_kpts = 0;
    /// @brief k-points owned by THIS rank under the round-robin MPI
    ///        distribution. Serial (or MPI inactive): equals `n_kpts`.
    int n_kpts_local = 0;
};

/// @brief Concrete plane-wave DFT method — `cpu::Driver<pw::Phase>` loop node.
///
/// Construction → set members → `build(reference)` → `run(frame)`.
/// `last_result()` is valid after each `run()`. Geometry-dependent
/// structures rebuild inside each `run()` (AIMD / optimize safe).
/// All product SCF paths are driven by the same Driver stages with
/// path-local StepFn bodies (restricted / collinear × fixed / smear).
enum class Smearing : int {
    none = 0,              ///< Fixed closed-shell occupations.
    fermi_dirac = 1,       ///< Fermi–Dirac.
    methfessel_paxton = 2, ///< Methfessel–Paxton of order `mp_order`.
    gaussian = 3,          ///< Gaussian (cold/Marzari–Vanderbilt not here).
};

class PW : public cpu::Driver<ATV::pw::Phase> {
  public:
    /// Un-hide `cpu::Driver::build()` (Plan compile) alongside `build(frame)`.
    using cpu::Driver<ATV::pw::Phase>::build;

    /// @brief Norm-conserving UPF pseudopotential file. Mutually exclusive
    ///        with `gth_name`; exactly one must be set.
    std::string upf_path;
    /// @brief In-tree analytic Goedecker-Teter-Hutter parameterization, e.g.
    ///        "GTH-PBE-q4" or "GTH-PADE-q6". Needs no external file: the
    ///        published parameters ship in `cpu/gpw/basis_data/gpw_tables.h`
    ///        (Goedecker-Teter-Hutter, PRB 54, 1703 (1996); Krack, TCA 114,
    ///        145 (2005)). Mutually exclusive with `upf_path`.
    ///
    /// A plane-wave run has **no default pseudopotential** — leaving both
    /// empty throws. There is no built-in substitute for a real
    /// pseudopotential: a bare valence-charge Coulomb potential is not
    /// representable in a plane-wave basis at any finite cutoff.
    std::string gth_name;
    /// @brief Wavefunction cutoff [Ry].
    double ecut_ry = 30.0;
    /// @brief Monkhorst-Pack mesh size per dimension (dimensionless).
    int k_mesh = 4;
    /// @brief SCF tolerance on the density-change L2 norm (dimensionless).
    double tol = 1e-6;
    /// @brief Maximum SCF iterations (dimensionless).
    int max_iter = 50;
    /// @brief Density mixing parameter (QE mixing_beta, dimensionless).
    double mix_beta = 0.7;
    /// @brief Pulay mixing history window (dimensionless).
    int pulay_depth = 5;
    /// @brief Exchange–correlation functional (default true GGA PBE).
    ///
    /// `XCType::PBE` uses spectral density gradients (true GGA) for both
    /// restricted and collinear spin-polarized paths. Pure LDA is
    /// `XCType::LSDA`. Parameters are members — no public `*Config` bag.
    cpu::scf::XCType xc = cpu::scf::XCType::PBE;
    /// @brief Collinear spin-polarized KS (nspin=2). Default false = restricted.
    ///        Compatible with metallic `smearing` (shared μ, self-consistent mag).
    ///        With `xc = PBE` uses true collinear spin-GGA (spectral ∇ρ_σ).
    bool spin_polarized = false;
    /// @brief Spin multiplicity 2S+1 (used only when `spin_polarized`).
    ///        Fixed occupations: N↑−N↓ = multiplicity−1. With metallic smearing
    ///        this only seeds the initial ρ↑/ρ↓ split (mag is free in SCF).
    int multiplicity = 1;
    /// @brief Metallic smearing kind (default: fixed occupations).
    ///        May be combined with `spin_polarized` (collinear metals).
    Smearing smearing = Smearing::none;
    /// @brief Smearing width [Ry] (QE `degauss`). Required > 0 when smearing is on.
    double degauss = 0.02;
    /// @brief Methfessel–Paxton order \(N\) (used when `smearing == methfessel_paxton`).
    int mp_order = 1;
    /// @brief Extra bands beyond ceil(n_e/2) when smearing is on (per spin if collinear).
    int extra_bands = 4;

    /// @brief Default construct (helpers allocated on first build/run).
    PW();

    ~PW();
    PW(PW&&) noexcept;
    PW& operator=(PW&&) noexcept;
    PW(const PW&) = delete;
    PW& operator=(const PW&) = delete;

    /// @brief One-time setup: parses the UPF (if any) and validates Z.
    /// @param reference Frame used only to validate atom types.
    void build(const cpu::Frame& reference);

    /// @brief Plane-wave SCF at the frame's current geometry.
    /// @param frame Atom positions [Å], types (Z), and simulation cell.
    /// @return Owned `PWResult` (valid until next `run()`).
    PWResult& run(const cpu::Frame& frame);

    /// @brief Most recent SCF result.
    const PWResult& last_result() const { return result_; }

  private:
    /// @brief Wire path-local stage StepFns + compile Plan (rebuilds each drive).
    void ensure_driver_built_(VeffStage::StepFn veff, DiagonalizeStage::StepFn diag,
                              DensityStage::StepFn dens, MixStage::StepFn mix);

    /// @brief Drop wired stages / Plan / criteria so the next ensure rebinds.
    void reset_built_driver_();

    /// @brief Stages fixed by ensure; nothing user-suppliable.
    void validate_components() const override {}

    /// @brief Shared Driver loop: setup_pass → step_once × N → poll_criteria.
    /// @param max_iter_in SCF iteration budget (dimensionless).
    /// @param res Result buffer (writes iterations / converged).
    /// @return true if the residual gate fired (Driver converged).
    bool drive_stage_loop_(int max_iter_in, PWResult& res);

    bool built_ = false;
    std::unique_ptr<internal::PoissonSolver> poisson_;
    std::unique_ptr<internal::XCBuilder> xc_;
    std::unique_ptr<internal::VeffBuilder> veff_;
    std::unique_ptr<internal::DensityBuilder> density_;
    std::unique_ptr<internal::DavidsonSolver> davidson_;
    int davidson_bands_ = 0;
    PWResult result_;

    // ── SCF iteration residual flag (wired into PwConverged) ─────────
    bool loop_converged_ = false;
    bool driver_built_ = false;
    const void* wired_self_ = nullptr;

    std::unique_ptr<VeffStage> veff_stage_;
    std::unique_ptr<DiagonalizeStage> diag_stage_;
    std::unique_ptr<DensityStage> density_stage_;
    std::unique_ptr<MixStage> mix_stage_;
    std::unique_ptr<PwConverged> converged_stage_;
};

/// @brief Finite-difference ionic forces [eV/Å] (oracle for analytic).
std::vector<std::array<double, 3>>
pw_forces_fd(const cpu::Frame& frame, const PW& method, double h_ang = 1e-3);

/// @brief Analytic ionic forces [eV/Å] (local HF + Ewald + KB nonlocal).
std::vector<std::array<double, 3>> pw_forces_analytic(const cpu::Frame& frame,
                                                      const PW& method);

/// @brief Nielsen–Martin analytic cell stress σ [eV/Å³], Voigt xx,yy,zz,xy,xz,yz.
///
/// One SCF + Hellmann–Feynman cell derivatives (kinetic, Hartree, local PP,
/// Ewald, XC, KB nonlocal). Matches strain-FD of the total energy for fixed
/// occupations, or of the Mermin free energy \(F=E_{\mathrm{KS}}-TS\) when
/// restricted metallic `smearing` is set. Mechanical / CP2K sign:
/// \(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\) (positive when
/// compressed — drives `IsotropicCellRelax` to expand). Restricted paths
/// only; collinear spin analytic stress is a separate product row.
///
/// @param energy_ry_out Optional: free/total energy [Ry] from the production
///        SCF (avoids a second SCF when the evaluator also needs energy).
std::array<double, 6> pw_stress(const cpu::Frame& frame, const PW& method,
                                double* energy_ry_out = nullptr);

/// @brief Strain finite-difference cell stress oracle [eV/Å³] (Voigt).
///
/// Central differences of SCF energy under symmetric strain at fixed
/// fractional coordinates (12 SCF solves). Dual-gate oracle for
/// `pw_stress`; not the production path.
///
/// @param frame Geometry [Å]; not mutated.
/// @param method Plane-wave method members.
/// @param h Dimensionless strain step (must be > 0).
std::array<double, 6> pw_stress_fd(const cpu::Frame& frame, const PW& method,
                                   double h = 2e-3);

} // namespace pw
} // namespace cpu
} // namespace ATV
