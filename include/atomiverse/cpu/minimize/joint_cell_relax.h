/// @file cpu/minimize/joint_cell_relax.h
/// @brief Joint atom + isotropic cell steepest-descent Optimizer (CPU).
///
/// Each iteration: (1) steepest-descent atom step along forces,
/// (2) isotropic volume scale from virial pressure toward P_target.
/// Compose ForceConverged + PressureConverged for termination.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/optimizer.h"

namespace ATV {
namespace cpu {
namespace minimizer {

/// @brief Joint isotropic cell + atom steepest-descent Optimizer.
class JointIsotropicRelax : public Optimizer {
  public:
    double P_target = 0.0;   ///< Target pressure [eV/Å³].
    double atom_step = 0.01; ///< Atom step along unit force [Å].
    double alpha0 = 0.05;    ///< Relative scale gain (dimensionless).
    double max_scale = 0.05; ///< Cap on |s−1| per step.
    double bulk_mod = 1.0;   ///< Soft bulk modulus [eV/Å³].

    JointIsotropicRelax() = default;

    /// @param P_target_ev_a3 Target pressure [eV/Å³].
    /// @param atom_step_a Atom step length [Å].
    explicit JointIsotropicRelax(double P_target_ev_a3,
                                 double atom_step_a = 0.01);

    std::string name() const override { return "joint_isotropic_relax"; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::minimizer::Phase::search_direction, 0),
                ATV::make_binding(ATV::minimizer::Phase::line_search, 0),
                ATV::make_binding(ATV::minimizer::Phase::convergence_check, 0)};
    }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    double pressure() const { return pressure_; }
    double pressure_residual() const { return p_residual_; }
    double max_force() const { return max_force_; }

  private:
    static double pressure_from_frame_(const Frame& f);
    void apply_scale_(Frame& f, double s) const;

    double pressure_ = 0;
    double p_residual_ = 0;
    double max_force_ = 0;
    double scale_step_ = 1.0;
    double energy_prev_ = 0;
    bool has_prev_ = false;
    int step_ = 0;
};

} // namespace minimizer
} // namespace cpu
} // namespace ATV
