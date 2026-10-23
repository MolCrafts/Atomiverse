/// @file cpu/scf/post_hf.h
/// @brief Post-HF correlation types. They are not methods of `MeanFieldMethod`.
///
/// Boundary transplanted from PySCF (not re-derived):
/// - `pyscf.mp.mp2.MP2(mf)` / `kernel` — correlation on a mean-field reference
/// - `pyscf.cc.ccsd.CCSD` / `ccsd_t` — CCSD and the (T) correction live on the
///   CC object, not on the SCF class. `eomee_ccsd_singlet`, `ipccsd`,
///   `eaccsd`, and `eomsf_ccsd` are methods of that CC object — not of
///   `RHF` / `UHF`. EOM-CC2 is `CC2::eom_cc2`.
/// - `pyscf.mcscf.CASCI` / `CASSCF(mf, ncas, nelecas)` (`mc1step.kernel`)
///   and `mrpt.NEVPT(mc)`. CASSCF's macro iteration is `Driver<mc::Phase>`.
///
/// Each type holds a reference to a concrete mean-field method (`RHF`, `UHF`,
/// `ROHF`, `RKS`, …) and drives `run` itself. `MeanFieldMethod` keeps SCF
/// `run` / `forces` only.
#pragma once

#include <memory>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/casci.h"
#include "atomiverse/cpu/scf/casscf.h"
#include "atomiverse/cpu/scf/cc2.h"
#include "atomiverse/cpu/scf/ccsd.h"
#include "atomiverse/cpu/scf/eom_cc2.h"
#include "atomiverse/cpu/scf/eom_ea_ccsd.h"
#include "atomiverse/cpu/scf/eom_ee_ccsd.h"
#include "atomiverse/cpu/scf/eom_ip_ccsd.h"
#include "atomiverse/cpu/scf/eom_sf_ccsd.h"
#include "atomiverse/cpu/scf/fci.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/mp2.h"
#include "atomiverse/cpu/scf/mp3.h"
#include "atomiverse/cpu/scf/mp4.h"
#include "atomiverse/cpu/scf/nevpt2.h"
#include "atomiverse/cpu/scf/scf_guess.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Canonical MP2 on a mean-field reference.
///
/// Mirrors `pyscf.mp.mp2.MP2(mf)`: `frozen` and the Grimme OS/SS scales are
/// parameters of this object (`kernel` → `run`). Closed-shell `RHF`/`RKS` →
/// RMP2 or RI-MP2; `UHF` → UMP2; `ROHF` → ROMP2. `forces` is the nuclear
/// gradient of E_HF + E_corr (PySCF `nuc_grad_method().kernel()`).
/// @tparam Method Concrete mean-field type (`RHF`, `UHF`, `ROHF`, `RKS`, …).
template <typename Method>
class MP2 {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method (not copied).
    explicit MP2(Method& mf);

    /// @brief Lowest occupied spatial MOs excluded from the correlation sum.
    ///        PySCF `MP2(mf, frozen=N)`. 0 = correlate every occupied MO.
    int frozen = 0;

    /// @brief Opposite-spin scale (dimensionless). 1 = plain MP2; SCS = 1.2;
    ///        SOS = 1.3; B2PLYP = `hybrid_pt2_fraction`.
    double os_scale = 1.0;

    /// @brief Same-spin scale (dimensionless). 1 = plain MP2; SCS = 1/3;
    ///        SOS = 0; B2PLYP = `hybrid_pt2_fraction`.
    double ss_scale = 1.0;

    /// @brief Fresh SCF on `mf`, then the MP2 correction.
    /// @param frame Geometry [Å].
    /// @param guess Optional density guess for the reference SCF.
    /// @return Correlation and total energy [Hartree].
    MP2Result run(Frame& frame, const SCFInitialGuess* guess = nullptr);

    /// @brief Analytic nuclear forces on E_HF + E_corr [eV/Å].
    /// @param frame Geometry; forces written to fx/fy/fz, energy to meta.
    /// @param guess Optional density guess for the reference SCF.
    /// @return Forces [eV/Å] and total MP2 energy [eV].
    GradientResult forces(Frame& frame, const SCFInitialGuess* guess = nullptr);

  private:
    Method* mf_ = nullptr;
};

/// @brief Closed-shell MP3. Mirrors a post-HF object on an RHF reference
///        (PySCF 2.x has no `mp.MP3`; the energy kernel is unchanged).
template <typename Method>
class MP3 {
  public:
    explicit MP3(Method& mf);
    /// @brief Frozen occupied spatial MOs (0 = full). Same convention as `MP2`.
    int frozen = 0;
    /// @param frame Geometry [Å].
    /// @param guess Optional density guess for the reference SCF.
    /// @return MP3 correlation pieces [Hartree].
    MP3Result run(Frame& frame, const SCFInitialGuess* guess = nullptr);

  private:
    Method* mf_ = nullptr;
};

/// @brief Closed-shell MP4(SDTQ) on an RHF reference.
template <typename Method>
class MP4 {
  public:
    explicit MP4(Method& mf);
    /// @brief Frozen occupied spatial MOs (0 = full).
    int frozen = 0;
    /// @param frame Geometry [Å].
    /// @param guess Optional density guess for the reference SCF.
    /// @return MP4 correlation pieces [Hartree].
    MP4Result run(Frame& frame, const SCFInitialGuess* guess = nullptr);

  private:
    Method* mf_ = nullptr;
};

/// @brief Tiny full CI on an RHF reference (`pyscf.fci.FCI(mf)` shape).
template <typename Method>
class FCI {
  public:
    explicit FCI(Method& mf);
    /// @param frame Geometry [Å]. Drives a fresh SCF, then the CI.
    /// @return Full-CI total and correlation energy [Hartree].
    FCIResult run(Frame& frame);

  private:
    Method* mf_ = nullptr;
};

/// @brief Closed-shell CASCI. Mirrors `pyscf.mcscf.CASCI(mf, ncas, nelecas)`.
template <typename Method>
class CASCI {
  public:
    explicit CASCI(Method& mf);
    /// @param frame Geometry [Å].
    /// @param ncas Active spatial orbitals.
    /// @param nelecas Active electrons (even, closed-shell).
    /// @return CASCI total and correlation energy [Hartree].
    CASCIResult run(Frame& frame, int ncas, int nelecas);

  private:
    Method* mf_ = nullptr;
};

/// @brief Closed-shell CASSCF. Mirrors `pyscf.mcscf.CASSCF(mf, ncas, nelecas)`.
///
/// The orbital macro-iteration is `cpu::Driver<mc::Phase>` (`rotate →
/// casci → gradient → check`), the same preset shape as `cpu::md::MD`:
/// stages are mounted with `on(phase).add` in the constructor, max |g|
/// is published on `ConvergenceMetrics`, and `OrbitalGradientConverged`
/// decides. This type does not assign convergence.
///
/// PySCF 2.14.0: `pyscf/mcscf/__init__.py` `CASSCF` and
/// `pyscf/mcscf/mc1step.py` `kernel` / `casci` / `gen_g_hop` / `rotate_mo`.
/// The rotation generator is the existing diagonal Newton step, not
/// `rotate_orb_cc`.
///
/// Parameters are constructor arguments, fixed after construction.
/// `run(Frame&)` drives the reference SCF, then overwrites `kEnergy`
/// with the correlated total [eV]. Hartree stays on `CASSCFResult`.
/// Nuclear forces are not implemented.
/// @tparam Method Concrete mean-field type (`RHF`, …). The reference is
///         not owned and must outlive `*this`.
template <typename Method>
class CASSCF : public cpu::Driver<ATV::mc::Phase> {
  public:
    /// @brief Bind a mean-field reference. `orb_tol` is
    ///        `max(mf.tol, 1e-8)` [Hartree] and the macro cap is 50.
    /// @param mf Mean-field method (not copied).
    /// @param ncas_in Active spatial orbitals.
    /// @param nelecas_in Active electrons (even, closed-shell).
    explicit CASSCF(Method& mf, int ncas_in, int nelecas_in);

    /// @brief Bind a mean-field reference with an explicit gradient
    ///        tolerance and macro-iteration cap.
    /// @param mf Mean-field method (not copied).
    /// @param ncas_in Active spatial orbitals.
    /// @param nelecas_in Active electrons (even, closed-shell).
    /// @param orb_tol_in max |g_pq| tolerance [Hartree].
    /// @param max_macro_in Macro-iteration cap [dimensionless].
    CASSCF(Method& mf, int ncas_in, int nelecas_in, double orb_tol_in,
           int max_macro_in);

    /// @brief Drop the orbital plan while its stage storage is alive.
    ~CASSCF() override;

    CASSCF(const CASSCF&) = delete;
    CASSCF& operator=(const CASSCF&) = delete;
    CASSCF(CASSCF&&) = delete;
    CASSCF& operator=(CASSCF&&) = delete;

    /// @brief Active spatial orbitals. Fixed at construction.
    const int ncas;
    /// @brief Active electrons (even, closed-shell). Fixed at construction.
    const int nelecas;
    /// @brief max |g_pq| tolerance [Hartree].
    const double orb_tol;
    /// @brief Macro-iteration cap [dimensionless].
    const int max_macro;

    /// @brief This product contributes a total energy.
    bool provides_energy() const override { return true; }

    /// @brief Nuclear forces are not implemented.
    bool provides_force() const override { return false; }

    /// @brief Whole-molecule evaluator.
    bool requires_full_frame() const override { return true; }

    /// @brief Nested budget is the macro-iteration cap.
    /// @return `max_macro` [dimensionless].
    int default_budget() const override { return max_macro; }

    /// @brief Orbital stages do not assemble nuclear forces.
    /// @return False.
    bool phase_wants_forces(ATV::mc::Phase) const override { return false; }

    /// @brief Fresh SCF on the reference, then the orbital macro-iteration.
    /// @param frame Geometry [Å]. Overwrites `kEnergy` with E_CASSCF [eV].
    /// @return Total and correlation energy [Hartree].
    /// @throws std::invalid_argument if the reference is not closed-shell
    ///         RHF, the SCF did not converge, or the orbital gradient did
    ///         not fall under `orb_tol`.
    CASSCFResult run(Frame& frame);

    /// @brief Same solve, without writing `frame`.
    /// @param frame Geometry [Å], read only.
    /// @return Total and correlation energy [Hartree].
    CASSCFResult run(const Frame& frame);

    /// @brief One CASSCF on the parent's Runtime.
    ///
    /// Energy is accumulated onto `kEnergy` [eV]. A parent that requests
    /// forces is refused: this method has no nuclear gradient.
    /// @param ctx Parent context. Loop status is restored on exit.
    /// @param frame Parent frame.
    /// @param budget Macro iterations; `<= 0` uses `max_macro`.
    /// @return Iterations the orbital driver walked.
    int run_nested_erased(cpu::Context& ctx, cpu::Frame& frame,
                          int budget) override;

  protected:
    /// @brief The method is the energy source; no extra component is required.
    void validate_components() const override {}

  private:
    enum class Write { None, Overwrite, Accumulate };
    struct Loop;
    struct PriorEnergy {
        bool energy = false;
        double energy_v = 0.0;
        bool electronic = false;
        double electronic_v = 0.0;
    };

    /// @brief SCF total that was on the frame before the reference `run`.
    static PriorEnergy capture(const Frame& frame);
    /// @brief Write the correlated total [eV] over or onto the saved value.
    static void store_energy(Frame& frame, double e_ev, Write write,
                             const PriorEnergy& prior);

    /// @brief Orbital driver. A null `nested` walks this object's Runtime;
    ///        otherwise `Driver::run_nested` on the parent. The caller has
    ///        already converged the reference SCF.
    CASSCFResult solve(Frame& frame, Write write, const PriorEnergy& prior,
                       int cap, cpu::Context* nested);
    /// @brief Read the cycle the criterion accepted and write energy if asked.
    CASSCFResult finish(Frame& frame, int executed, int cap, Write write,
                        const PriorEnergy& prior);

    Method* mf_ = nullptr;
    std::unique_ptr<Loop> loop_;
    CASSCFResult last_{};
    int executed_ = 0;
};

/// @brief Closed-shell SC-NEVPT2. Drives the same RHF→CASSCF→PT2 sequence as
///        the previous `MeanFieldMethod::nevpt2` entry (PySCF
///        `mrpt.NEVPT(mc).kernel()` on that CASSCF reference).
template <typename Method>
class NEVPT2 {
  public:
    explicit NEVPT2(Method& mf);
    /// @param frame Geometry [Å].
    /// @param ncas Active spatial orbitals.
    /// @param nelecas Active electrons (even, closed-shell).
    /// @return NEVPT2 total and PT2 correlation [Hartree].
    NEVPT2Result run(Frame& frame, int ncas, int nelecas);

  private:
    Method* mf_ = nullptr;
};

/// @brief Closed-shell CCSD. Mirrors `pyscf.cc.ccsd.CCSD(mf)`.
///
/// The amplitude iteration is `cpu::Driver<cc::Phase>` (`update → mix →
/// energy → check`), the same preset shape as `cpu::md::MD`: stages are
/// mounted with `on(phase).add` in the constructor, the residual is
/// published on `ConvergenceMetrics`, and `AmplitudeConverged` decides.
/// This type does not assign convergence.
///
/// PySCF 2.14.0: `pyscf/cc/ccsd.py` `kernel` / `CCSD.run_diis` / `energy`,
/// and `pyscf/cc/rccsd.py` `update_amps` / `energy` (the spatial residual).
/// `ccsd_t` is `CCSD.ccsd_t`. EOM entries stay methods of this object.
///
/// Parameters are constructor arguments, fixed after construction.
/// `run(Frame&)` drives the reference SCF, then overwrites `kEnergy` with
/// the correlated total [eV]. Hartree stays on `CCSDResult`.
/// @tparam Method Concrete mean-field type (`RHF`, …). The reference is not
///         owned and must outlive `*this`.
template <typename Method>
class CCSD : public cpu::Driver<ATV::cc::Phase> {
  public:
    /// @brief Bind a mean-field reference.
    /// @param mf Mean-field method (not copied).
    /// @param diis_in Amplitude DIIS. False = Jacobi, same fixed point.
    /// @param frozen_in Frozen-core gate for `forces`. Nonzero throws —
    ///        frozen-core CCSD gradients are not implemented. The energy
    ///        path does not freeze orbitals.
    /// @param tol_in Amplitude residual tolerance, max|Δt| [dimensionless].
    /// @param max_iter_in Amplitude iteration cap [dimensionless].
    explicit CCSD(Method& mf, bool diis_in = true, int frozen_in = 0,
                  double tol_in = 1e-10, int max_iter_in = 100);

    /// @brief Drop the amplitude plan while its stage storage is alive.
    ~CCSD() override;

    CCSD(const CCSD&) = delete;
    CCSD& operator=(const CCSD&) = delete;
    CCSD(CCSD&&) = delete;
    CCSD& operator=(CCSD&&) = delete;

    /// @brief Amplitude DIIS. Fixed at construction.
    const bool diis;
    /// @brief Frozen-core gate for `forces` (0 = correlate every occupied MO).
    const int frozen;
    /// @brief max|Δt| tolerance [dimensionless].
    const double tol;
    /// @brief Amplitude iteration cap [dimensionless].
    const int max_iter;

    /// @brief This product contributes a total energy.
    bool provides_energy() const override { return true; }

    /// @brief Nuclear forces are `forces(Frame&)`.
    bool provides_force() const override { return true; }

    /// @brief Whole-molecule evaluator.
    bool requires_full_frame() const override { return true; }

    /// @brief Nested budget is the amplitude iteration cap.
    /// @return `max_iter` [dimensionless].
    int default_budget() const override { return max_iter; }

    /// @brief Amplitude stages do not assemble nuclear forces.
    /// @return False. `run_nested_erased` writes forces when the parent asks.
    bool phase_wants_forces(ATV::cc::Phase) const override { return false; }

    /// @brief Fresh SCF on the reference, then the amplitude solve.
    /// @param frame Geometry [Å]. Overwrites `kEnergy` with E_HF + E_corr [eV].
    /// @return Correlation and total energy [Hartree].
    CCSDResult run(Frame& frame);

    /// @brief Same solve, without writing `frame`.
    /// @param frame Geometry [Å], read only.
    /// @return Correlation and total energy [Hartree].
    CCSDResult run(const Frame& frame);

    /// @brief (T) correction. Mirrors `pyscf.cc.ccsd.CCSD.ccsd_t`.
    /// @param frame Geometry [Å].
    /// @return CCSD(T) correlation pieces [Hartree].
    CCSDTResult ccsd_t(Frame& frame);

    /// @brief Nuclear forces on E_HF + E_CCSD [eV/Å].
    /// @param frame Geometry; forces written to fx/fy/fz, energy to meta [eV].
    /// @return Forces [eV/Å] and total CCSD energy [eV].
    GradientResult forces(Frame& frame);

    /// @brief Singlet EOM-EE-CCSD. Mirrors
    ///        `pyscf.cc.ccsd.CCSD.eomee_ccsd_singlet` (which builds
    ///        `eom_rccsd.EOMEESinglet(cc)`). Not a member of `RHF`.
    /// @param frame Geometry [Å]. Drives a fresh SCF, then CCSD, then EOM.
    /// @param n_states Number of lowest singlet roots (≥ 1).
    /// @return Singlet ω [eV] plus CCSD energetics [Hartree].
    /// @throws std::invalid_argument on a non-restricted reference, missing
    ///         virtuals, a non-converged SCF or CCSD, or n_states out of range.
    EOMEECCSDResult eom_ee_ccsd(Frame& frame, int n_states = 1);

    /// @brief EOM-IP-CCSD ionization potentials. Mirrors
    ///        `pyscf.cc.ccsd.CCSD.ipccsd` (`eom_gccsd.EOMIP`). Not a member
    ///        of `RHF`.
    /// @param frame Geometry [Å]. Drives a fresh SCF, then CCSD, then EOM.
    /// @param n_states Number of lowest unique ionization roots (≥ 1).
    /// @return Ionization ω [eV] plus CCSD energetics [Hartree].
    /// @throws std::invalid_argument on a non-restricted reference, a
    ///         non-converged SCF or CCSD, or n_states out of range.
    EOMIPCCSDResult eom_ip_ccsd(Frame& frame, int n_states = 1);

    /// @brief EOM-EA-CCSD electron-attachment energies. Mirrors
    ///        `pyscf.cc.ccsd.CCSD.eaccsd` (`eom_gccsd.EOMEA`). Not a member
    ///        of `RHF`.
    /// @param frame Geometry [Å]. Drives a fresh SCF, then CCSD, then EOM.
    /// @param n_states Number of lowest unique attachment roots (≥ 1).
    /// @return Attachment ω [eV] plus CCSD energetics [Hartree].
    /// @throws std::invalid_argument on a non-restricted reference, a
    ///         non-converged SCF or CCSD, or n_states out of range.
    EOMEACCSDResult eom_ea_ccsd(Frame& frame, int n_states = 1);

    /// @brief EOM-SF-CCSD spin-flip excitations. Mirrors
    ///        `pyscf.cc.uccsd.UCCSD.eomsf_ccsd`. Not a member of `UHF`.
    /// @param frame Geometry [Å]. Drives a fresh UHF, then UCCSD, then SF-EOM.
    /// @param n_states Number of lowest SF roots (≥ 1).
    /// @param guess Optional density guess for the reference UHF.
    /// @return SF excitation energies [eV] plus UCCSD energetics [Hartree].
    /// @throws std::invalid_argument on a restricted reference, a
    ///         non-converged SCF, or an empty SF manifold.
    EOMSFCCSDResult eom_sf_ccsd(Frame& frame, int n_states = 1,
                                const SCFInitialGuess* guess = nullptr);

    /// @brief One CCSD on the parent's Runtime.
    ///
    /// Energy is accumulated onto `kEnergy` [eV]. When the parent requests
    /// forces, `kAtom_f*` is accumulated too [eV/Å].
    /// @param ctx Parent context. Loop status is restored on exit.
    /// @param frame Parent frame.
    /// @param budget Amplitude iterations; `<= 0` uses `max_iter`.
    /// @return Iterations the amplitude driver walked.
    int run_nested_erased(cpu::Context& ctx, cpu::Frame& frame,
                          int budget) override;

  protected:
    /// @brief The method is the energy source; no extra component is required.
    void validate_components() const override {}

  private:
    enum class Write { None, Overwrite, Accumulate };
    struct Loop;
    struct PriorEnergy {
        bool energy = false;
        double energy_v = 0.0;
        bool electronic = false;
        double electronic_v = 0.0;
    };

    /// @brief SCF total that was on the frame before the reference `run`.
    static PriorEnergy capture(const Frame& frame);
    /// @brief Write the correlated total [eV] over or onto the saved value.
    static void store_energy(Frame& frame, double e_ev, Write write,
                             const PriorEnergy& prior);

    /// @brief Amplitude driver. A null `nested` walks this object's Runtime;
    ///        otherwise `Driver::run_nested` on the parent. The caller has
    ///        already converged the reference SCF.
    CCSDResult solve_amplitudes(Frame& frame, Write write,
                                const PriorEnergy& prior, int cap,
                                cpu::Context* nested);
    /// @brief Read the cycle the criterion accepted and write energy if asked.
    CCSDResult finish(Frame& frame, int executed, int cap, Write write,
                      const PriorEnergy& prior);
    /// @brief Reference SCF plus finite-difference forces.
    GradientResult compute_forces(Frame& frame, Write write,
                                  cpu::Context* nested);

    Method* mf_ = nullptr;
    std::unique_ptr<Loop> loop_;
    CCSDResult last_{};
    int executed_ = 0;
};

/// @brief Closed-shell CC2. Mirrors PySCF `cc.RCCSD` with `cc2=True`.
template <typename Method>
class CC2 {
  public:
    explicit CC2(Method& mf);
    /// @param frame Geometry [Å].
    /// @return CC2 correlation and total energy [Hartree].
    CC2Result run(Frame& frame);

    /// @brief Singlet EOM-CC2. Mirrors PySCF `cc.RCCSD` with `cc2=True`
    ///        followed by `eomee_ccsd_singlet`. Not a member of `RHF`.
    /// @param frame Geometry [Å]. Drives a fresh SCF, then CC2, then EOM.
    /// @param n_states Number of lowest singlet roots (≥ 1).
    /// @return Singlet ω [eV] plus CC2 energetics [Hartree].
    /// @throws std::invalid_argument on a non-restricted reference, a
    ///         non-converged SCF or CC2, or n_states out of range.
    EOMCC2Result eom_cc2(Frame& frame, int n_states = 1);

  private:
    Method* mf_ = nullptr;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
