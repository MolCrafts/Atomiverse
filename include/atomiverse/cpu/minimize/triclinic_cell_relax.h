/// @file cpu/minimize/triclinic_cell_relax.h
/// @brief Full-cell (including shear) stress-driven relax Optimizer (CPU).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/unit_system.h"
#include "atomiverse/cpu/component/optimizer.h"

namespace ATV {
namespace cpu {
namespace minimizer {

/// @brief Strain cell by ε ∝ σ toward zero stress (0 K geometry).
///
/// Uses full Voigt stress; updates `box` via \(h \leftarrow (I+ε)h\) and
/// affinely maps atom positions. Textbook skeleton — not a PR barostat.
class TriclinicCellRelax : public Optimizer {
  public:
    double alpha0 = 0.02;     ///< Strain gain (dimensionless).
    double max_strain = 0.05; ///< Cap on |ε_αβ| per step.
    double bulk_mod = units::
        kStyleDefault; ///< Soft modulus [style pressure]; `units::kStyleDefault`
    ///< (the default) is 1 eV/Å³ in the run's style (1.6e6 bar, 1.58e6 atm;
    ///< 1 in lj).

    TriclinicCellRelax() = default;
    explicit TriclinicCellRelax(double alpha,
                                double bulk = units::kStyleDefault);

    std::string name() const override { return "triclinic_cell_relax"; }
    /// @brief Reads the system stress (kStress) every iteration.
    /// @return True.
    bool requires_system_stress() const override { return true; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::minimizer::Phase::search_direction, 0),
                ATV::make_binding(ATV::minimizer::Phase::line_search, 0),
                ATV::make_binding(ATV::minimizer::Phase::convergence_check, 0)};
    }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    double stress_residual() const { return residual_; }

  private:
    Mat3 eps_{}; // strain to apply
    double residual_ = 0;
    double energy_prev_ = 0;
    BigInt step_ = 0;
    bool has_prev_ = false;
};

} // namespace minimizer
} // namespace cpu
} // namespace ATV
