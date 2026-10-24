/// @file cpu/semiempirical/pm3_solver.h
/// @brief PM3 concrete method — public members + Frame build/run.
#pragma once

#include <memory>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/nddo_method.h"
#include "atomiverse/semiempirical/nddo_result.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

using PM3Result = ATV::semiempirical::NDDOResult;

/// @brief PM3 molecular CHNO semiempirical method.
/// Current envelope: neutral closed-shell molecules, analytic energy/charges/forces.
/// Unsupported elements, charge, and multiplicity are rejected before solving.
///
/// Same public shape as `AM1`: parameters as members, `run` / `forces` on a
/// Frame, and `cpu::QmEngine` for multiscale couplings (no embedding path).
/// The self-consistency loop is `NddoMethod`.
class PM3 : public NddoMethod, public cpu::QmEngine {
  public:
    const int charge;
    const int multiplicity;
    int max_iter = 200;
    double density_tol = 1.0e-8;
    double energy_tol = 1.0e-10;

    /// @brief Construct with SCF parameters assigned immediately.
    /// @param charge Total molecular charge [e].
    /// @param multiplicity Spin multiplicity 2S+1 (dimensionless).
    /// @param max_iter SCF iteration cap (dimensionless).
    /// @param density_tol Density residual tolerance (dimensionless).
    /// @param energy_tol Energy residual tolerance [eV].
    explicit PM3(int charge = 0, int multiplicity = 1, int max_iter = 200,
                 double density_tol = 1.0e-8, double energy_tol = 1.0e-10);
    ~PM3();
    PM3(PM3&&);
    PM3& operator=(PM3&&) = delete;
    PM3(const PM3&) = delete;
    PM3& operator=(const PM3&) = delete;

    void build(const cpu::Frame& reference);
    PM3Result& run(const cpu::Frame& frame);
    /// @brief Write total energy [eV] and Mulliken charges [e], preserving forces.
    /// @param frame Geometry [Å]; energy and charges out (overwritten).
    /// @return `last_result()`.
    /// @throws std::runtime_error if the SCF does not converge in `max_iter`.
    PM3Result& run(cpu::Frame& frame);

    /// @brief SCF + analytic nuclear forces \(F = -\partial E/\partial R\) [eV/Å]
    ///        contracted from pair-integral derivatives at converged density.
    ///
    /// See `NDDOSolver::compute_forces` for the stationary-density derivative. Writes
    /// `kAtom_fx/fy/fz` [eV/Å] (overwritten, not accumulated) and
    /// `frame.meta[kEnergy]` [eV].
    /// @param frame Geometry [Å]; forces and energy out.
    /// @return `last_result()` with `forces_ev_a` filled (3 · nlocal).
    /// @throws std::runtime_error if the SCF does not converge.
    PM3Result& forces(cpu::Frame& frame);

    const PM3Result& last_result() const;
    bool built() const noexcept;
    bool has_density() const noexcept;

    /// @brief Engine label.
    /// @return `"pm3"`.
    /// @brief Display name in plan dumps and binding diagnostics.
    /// @return `"pm3"`.
    std::string name() const { return "pm3"; }
    std::string qm_engine_name() const override { return "pm3"; }
    void prepare(const cpu::BuildContext& qm_ctx) override;
    void evaluate(const cpu::LaunchContext& qm_ctx) override;
    void reset_history() override;

    /// @brief Run the SCF and write every result channel onto the frame.
    ///
    /// Same contract as `AM1::publish`: total energy [eV] to `kEnergy` and
    /// `kEnergyElectronic`, Mulliken charges [e] to `kAtom_charge`, and —
    /// when asked — forces [eV/Å] to `kAtom_f{x,y,z}`.
    /// @param frame Target frame; geometry in, results out.
    /// @param forces_requested Also compute analytic forces.
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
