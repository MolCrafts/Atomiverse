/// @file cpu/planewave/pw_solver.h
/// @brief cpu::pw::PW — concrete plane-wave DFT method (CPU Driver node).
///
/// Pseudopotential, cutoff, k-mesh and functional are constructor arguments;
/// SCF knobs are public members (RHF pattern). Frame is the only I/O.
/// Self-consistency is a **Driver loop**: `PW` derives
/// `cpu::Driver<ATV::pw::Phase>`. All product SCF paths (restricted
/// fixed-occupation, collinear fixed-occupation, restricted metallic
/// smearing, collinear + metallic smearing) are owned as stages
/// (`veff → diagonalize → density → mix → check`) with path-local StepFns.
/// Workflow entry: mount it with `outer.on(phase).add(pw)` or call
/// `build`/`run` directly.
///
/// @code
///   cpu::pw::PW pw{cpu::pw::GthPseudopotential{"GTH-PBE-q4"},
///                  /*ecut_ry=*/30.0, /*k_mesh=*/4};
///   pw.build(frame);
///   const auto& result = pw.run(frame); // [Ry]/Ha on result
/// @endcode
#pragma once

#include <array>
#include <functional>
#include <memory>
#include <stdexcept>
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
struct PWProblem;
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

/// @brief Norm-conserving UPF pseudopotential file.
struct UpfPseudopotential {
    std::string path; ///< UPF file path.
};

/// @brief In-tree analytic Goedecker-Teter-Hutter pseudopotential, e.g.
///        "GTH-PBE-q4" or "GTH-PADE-q6" (published parameters in
///        `cpu/gpw/basis_data/gpw_tables.h`; Goedecker-Teter-Hutter, PRB 54,
///        1703 (1996); Krack, TCA 114, 145 (2005)).
struct GthPseudopotential {
    std::string name; ///< Table entry name.
};

class PW : public cpu::Driver<ATV::pw::Phase> {
  public:
    bool requires_full_frame() const override { return true; }
    /// Un-hide `cpu::Driver::build()` (Plan compile) alongside `build(frame)`.
    using cpu::Driver<ATV::pw::Phase>::build;

    // ── Identity (constructor arguments, fixed after construction) ────
    /// @brief UPF file path; empty when a GTH table entry is used.
    const std::string upf_path;
    /// @brief GTH table entry; empty when a UPF file is used.
    const std::string gth_name;
    /// @brief Wavefunction cutoff [Ry].
    const double ecut_ry;
    /// @brief Monkhorst-Pack mesh size per dimension (dimensionless).
    const int k_mesh;
    /// @brief Exchange–correlation functional. `XCType::PBE` uses spectral
    ///        density gradients (true GGA) for restricted and collinear spin
    ///        paths; pure LDA is `XCType::LSDA`.
    const cpu::scf::XCType xc;

    // ── Knobs ──────────────────────────────────────────────────────────
    /// @brief SCF tolerance on the density-change L2 norm (dimensionless).
    double tol = 1e-6;
    /// @brief Maximum SCF iterations (dimensionless).
    int max_iter = 50;
    /// @brief Density mixing parameter (QE mixing_beta, dimensionless).
    double mix_beta = 0.7;
    /// @brief Pulay mixing history window (dimensionless).
    int pulay_depth = 5;
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

    /// @brief Plane-wave DFT from a UPF pseudopotential file.
    /// @param pp Pseudopotential file.
    /// @param ecut_ry Wavefunction cutoff [Ry].
    /// @param k_mesh Monkhorst-Pack mesh per dimension (dimensionless).
    /// @param xc Exchange–correlation functional.
    PW(UpfPseudopotential pp, double ecut_ry, int k_mesh,
       cpu::scf::XCType xc = cpu::scf::XCType::PBE);
    /// @brief Plane-wave DFT from an in-tree GTH pseudopotential.
    /// @param pp GTH table entry.
    /// @param ecut_ry Wavefunction cutoff [Ry].
    /// @param k_mesh Monkhorst-Pack mesh per dimension (dimensionless).
    /// @param xc Exchange–correlation functional.
    PW(GthPseudopotential pp, double ecut_ry, int k_mesh,
       cpu::scf::XCType xc = cpu::scf::XCType::PBE);

    ~PW();
    PW(PW&&);
    PW& operator=(PW&&) = delete;
    PW(const PW&) = delete;
    PW& operator=(const PW&) = delete;

    /// @brief One-time setup: allocate the step helpers and resolve the
    ///        pseudopotential up front, so a bad file or an unknown GTH name
    ///        fails here rather than mid-SCF. `run()` calls it lazily.
    /// @param reference Frame used only to validate atom types (the first
    ///        atom's Z selects the GTH entry).
    /// @throws std::invalid_argument when neither or both of `upf_path` /
    ///         `gth_name` are set — a plane-wave run has no default
    ///         pseudopotential — or when the named potential does not
    ///         resolve.
    void build(const cpu::Frame& reference);

    /// @brief Plane-wave SCF at the frame's current geometry.
    ///
    /// Lazily calls `build(frame)` when it has not run yet. Exhausting
    /// `max_iter` is **not** an error: the returned result carries
    /// `converged == false` and the last density's energy, so callers must
    /// read the flag.
    ///
    /// @param frame Atom positions [Å], types (Z), and simulation cell.
    /// @return Owned `PWResult` (valid until next `run()`).
    /// @throws std::invalid_argument when the restricted, fixed-occupation
    ///         path is handed an odd valence-electron count
    ///         (`internal::require_closed_shell`, the same gate
    ///         `forces()` and `stress()` apply — an energy the
    ///         gradient paths refuse is not one this path may report); when
    ///         `spin_polarized` is set with `multiplicity < 1` or with a
    ///         multiplicity whose parity contradicts the electron count; and
    ///         from `build()` when the pseudopotential source is missing,
    ///         doubly set, or unresolvable.
    PWResult& run(const cpu::Frame& frame);

    /// @brief `run(const Frame&)`, then writes the total (or Mermin free)
    ///        energy [eV] to `frame.meta[kEnergy]`.
    /// @param frame Atom positions [Å], types (Z), and simulation cell.
    /// @return Owned `PWResult` (valid until the next `run()` / `forces()` /
    ///         `stress()`).
    /// @throws as `run(const Frame&)`.
    PWResult& run(Frame& frame);

    /// @brief SCF + analytic ionic forces (local Hellmann–Feynman + Ewald +
    ///        Kleinman–Bylander nonlocal) at the Frame's geometry.
    ///
    /// Writes `kAtom_fx/fy/fz` [eV/Å] (overwritten, not accumulated) and
    /// `frame.meta[kEnergy]` [eV] — the energy of the same SCF the forces are
    /// the gradient of. A non-converged SCF still yields forces at the last
    /// density (with a warning); read `converged` on the returned result.
    /// @param frame Atom positions [Å], types (Z), and simulation cell.
    /// @return Owned `PWResult` of that SCF (also `last_result()`).
    /// @throws std::invalid_argument on an empty frame; on `degauss <= 0`
    ///         with smearing on; on a `multiplicity` incompatible with the
    ///         electron count; and, on the restricted fixed-occupation path,
    ///         on an odd valence-electron count.
    PWResult& forces(Frame& frame);

    /// @brief SCF + Nielsen–Martin analytic cell stress at the Frame's
    ///        geometry.
    ///
    /// One SCF + Hellmann–Feynman cell derivatives (kinetic, Hartree, local
    /// PP, Ewald, XC, KB nonlocal) as central differences of the
    /// non-self-consistent energy under symmetric strain at the converged
    /// density. Matches strain-FD of the total energy for fixed occupations,
    /// or of the Mermin free energy under restricted metallic `smearing`.
    /// Mechanical / CP2K sign: \(\sigma=-(1/\Omega)\partial E/\partial
    /// \varepsilon\). Writes the extensive virial \(W=\sigma\Omega\) [eV]
    /// to `frame.meta[kStress]` (overwritten) and the energy [eV] to
    /// `kEnergy`. Collinear UKS with smearing has no analytic path — use
    /// `pw_stress_fd`.
    /// @param frame Atom positions [Å], types (Z), and simulation cell.
    /// @return Voigt stress \((\sigma_{xx},\sigma_{yy},\sigma_{zz},
    ///         \sigma_{xy},\sigma_{xz},\sigma_{yz})\) [eV/Å³].
    /// @throws std::invalid_argument on an empty frame or non-positive cell
    ///         volume; on `spin_polarized` combined with `smearing`; on
    ///         `degauss <= 0` with smearing on; on an incompatible
    ///         `multiplicity`; and on an odd valence count on the restricted
    ///         fixed-occupation path.
    std::array<double, 6> stress(Frame& frame);

    /// @brief Display name in plan dumps and binding diagnostics.
    /// @return `"pw"`.
    std::string name() const { return "pw"; }
    /// @brief Energy source for an outer loop's validation.
    /// @return true.
    bool provides_energy() const { return true; }
    /// @brief Force source for an outer loop's validation.
    /// @return true.
    bool provides_force() const { return true; }

    /// @brief Nested mount: remember the parent Runtime the SCF runs on.
    /// @param rt Parent Runtime.
    void build_on(cpu::Runtime& rt) override;
    /// @brief One nested solve at the frame's geometry: the SCF stage plan
    ///        runs on the parent Runtime, then energy [eV], forces [eV/Å]
    ///        (when the phase requests them) and the virial σΩ [eV] (when it
    ///        requests stress) are accumulated onto the frame — one SCF
    ///        serves all three.
    /// @param ctx Parent Context (requests read from `ctx.exec`).
    /// @param frame Parent Frame.
    /// @param budget Ignored; the SCF runs to convergence or `max_iter`.
    /// @return SCF iterations.
    int run_nested_erased(cpu::Context& ctx, cpu::Frame& frame,
                          int budget) override;
    /// @brief Detach from the parent Runtime.
    /// @param rt Parent Runtime.
    void destroy_on(cpu::Runtime& rt) override;

    /// @brief Most recent SCF result.
    const PWResult& last_result() const { return result_; }

  private:
    void evaluate_(const Frame& frame,
                   std::vector<std::array<double, 3>>* forces,
                   std::array<double, 6>* stress);
    std::unique_ptr<internal::PWProblem> problem_;
    /// @brief Wire path-local stage StepFns + compile Plan (rebuilds each drive).
    void ensure_driver_built_(VeffStage::StepFn veff,
                              DiagonalizeStage::StepFn diag,
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
    /// Parent Runtime while nested (`build_on`); null for standalone entries.
    cpu::Runtime* host_runtime_ = nullptr;
    /// True during a nested solve: `drive_stage_loop_` steps on the host.
    bool nested_active_ = false;

    std::unique_ptr<VeffStage> veff_stage_;
    std::unique_ptr<DiagonalizeStage> diag_stage_;
    std::unique_ptr<DensityStage> density_stage_;
    std::unique_ptr<MixStage> mix_stage_;
    std::unique_ptr<PwConverged> converged_stage_;
};

/// @brief Finite-difference ionic forces [eV/Å] (oracle for analytic).
///
/// Central differences of the SCF total energy in each Cartesian coordinate:
/// \(F_{i\alpha} = -[E(R_{i\alpha}+h) - E(R_{i\alpha}-h)] / 2h\), with \(E\)
/// converted from the solver's Ry to eV by `qe_units::rydberg_to_ev` so the
/// result is in the engine's eV/Å. That is \(6N\) full SCF solves for \(N\)
/// atoms, so this is the correctness oracle for `PW::forces`, never
/// the production path.
///
/// @param frame  Geometry [Å]; not mutated (each solve runs on a displaced
///               copy).
/// @param method Plane-wave method members; copied per displacement via
///               `internal::copy_pw_params`.
/// @param h_ang  Displacement step [Å], default 1e-3. Must be > 0: too small
///               and the SCF tolerance dominates, too large and the cubic
///               truncation error does.
/// @return Per-atom force triples \((F_x, F_y, F_z)\) [eV/Å], one per atom
///         in Frame order.
/// @throws std::invalid_argument when @p h_ang <= 0 or the frame has no
///         atoms, plus anything the \(6N\) `PW::run` calls themselves throw.
std::vector<std::array<double, 3>>
pw_forces_fd(const cpu::Frame& frame, const PW& method, double h_ang = 1e-3);

/// @brief Strain finite-difference cell stress oracle [eV/Å³] (Voigt).
///
/// Central differences of SCF energy under symmetric strain at fixed
/// fractional coordinates (12 SCF solves). Dual-gate oracle for
/// `PW::stress`; not the production path.
///
/// @param frame Geometry [Å]; not mutated.
/// @param method Plane-wave method members.
/// @param h Dimensionless strain step (must be > 0).
std::array<double, 6> pw_stress_fd(const cpu::Frame& frame, const PW& method,
                                   double h = 2e-3);

} // namespace pw
} // namespace cpu
} // namespace ATV
