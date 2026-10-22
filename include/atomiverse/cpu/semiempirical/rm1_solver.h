/// @file cpu/semiempirical/rm1_solver.h
/// @brief RM1 concrete method — public members + Frame build/run.
#pragma once

#include <memory>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_solver.h"
#include "atomiverse/cpu/semiempirical/nddo_method.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

using RM1Result = NDDOResult;

/// @brief RM1 concrete semiempirical method.
///
/// Same public shape as `AM1`: parameters as members, `run` / `forces` on a
/// Frame, and `cpu::QmEngine` for multiscale couplings (no embedding path).
/// The self-consistency loop is `NddoMethod`.
class RM1 : public NddoMethod, public cpu::QmEngine {
  public:
    int charge = 0;
    int multiplicity = 1;
    int max_iter = 200;
    double density_tol = 1.0e-8;
    double energy_tol = 1.0e-10;
    /// @brief Finite-difference displacement \(h\) [Å] of `forces()`;
    ///        truncation error \(O(h^{2})\).
    double force_step = 1.0e-4;

    /// @brief Default parameters (charge 0, singlet, max_iter 200, …).
    RM1();
    /// @brief Construct with SCF parameters assigned immediately.
    /// @param charge Total molecular charge [e].
    /// @param multiplicity Spin multiplicity 2S+1 (dimensionless).
    /// @param max_iter SCF iteration cap (dimensionless).
    /// @param density_tol Density residual tolerance (dimensionless).
    /// @param energy_tol Energy residual tolerance [eV].
    RM1(int charge, int multiplicity, int max_iter = 200,
        double density_tol = 1.0e-8, double energy_tol = 1.0e-10);
    ~RM1();
    RM1(RM1&&) noexcept;
    RM1& operator=(RM1&&) noexcept;
    RM1(const RM1&) = delete;
    RM1& operator=(const RM1&) = delete;

    void build(const cpu::Frame& reference);
    RM1Result& run(const cpu::Frame& frame);
    /// @brief `run(const Frame&)`, then writes the total energy [eV] to
    ///        `frame.meta[kEnergy]`.
    /// @param frame Geometry [Å]; energy out.
    /// @return `last_result()`.
    /// @throws std::runtime_error if the SCF does not converge in `max_iter`.
    RM1Result& run(cpu::Frame& frame);

    /// @brief SCF + nuclear forces \(F = -\partial E/\partial R\) [eV/Å] by
    ///        central finite difference of the energy at the converged
    ///        density (step `force_step`).
    ///
    /// See `NDDOSolver::compute_forces` for why holding the density fixed is
    /// exact here and what the difference step costs in accuracy. Writes
    /// `kAtom_fx/fy/fz` [eV/Å] (overwritten, not accumulated) and
    /// `frame.meta[kEnergy]` [eV].
    /// @param frame Geometry [Å]; forces and energy out.
    /// @return `last_result()` with `forces_ev_a` filled (3 · nlocal).
    /// @throws std::runtime_error if the SCF does not converge.
    RM1Result& forces(cpu::Frame& frame);

    const RM1Result& last_result() const;
    bool built() const noexcept;
    bool has_density() const noexcept;
    const NDDOSystem& system() const;
    const NDDOScfState& state() const;

    /// @brief Engine label.
    /// @return `"rm1"`.
    /// @brief Display name in plan dumps and binding diagnostics.
    /// @return `"rm1"`.
    std::string name() const { return "rm1"; }
    std::string qm_engine_name() const override { return "rm1"; }
    void prepare(const cpu::BuildContext& qm_ctx) override;
    void evaluate(const cpu::LaunchContext& qm_ctx) override;
    void reset_history() override;

    /// @brief Run the SCF and write every result channel onto the frame.
    ///
    /// Same contract as `AM1::publish`: total energy [eV] to `kEnergy` and
    /// `kEnergyElectronic`, Mulliken charges [e] to `kAtom_charge`, and —
    /// when asked — forces [eV/Å] to `kAtom_f{x,y,z}`.
    /// @param frame Target frame; geometry in, results out.
    /// @param forces_requested Also compute finite-difference forces.
    /// @param accumulate true → `+=` energy/forces; false → `=`. Charges
    ///        always overwrite.
    void publish(cpu::Frame& frame, bool forces_requested,
                 bool accumulate) override;

  private:
    void sync_engine_();
    std::unique_ptr<NDDOSolver> engine_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
