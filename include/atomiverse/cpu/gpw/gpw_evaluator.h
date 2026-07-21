/// @file cpu/gpw/gpw_evaluator.h
/// @brief GPWEvaluator — GPW DFT energy component for CPU drivers.
///
/// Mirrors `cpu::pw::PWEvaluator` / `cpu::scf::SCF`: a self-contained
/// Evaluator that wraps an owned inner `GPWSolver`. `build()` validates the
/// per-element basis/pseudopotential templates, assembles the `GpwConfig`
/// from the build-time frame (cell [Bohr], n_occ, the ATV_GPW_OT diagnostic
/// switch) and builds the solver; each `launch()` assembles the atoms at the
/// frame's current geometry (Angstrom -> Bohr), runs the solver, and writes
/// the total energy to `frame.meta.get(kEnergy)` [eV]. GPW does not support a
/// variable cell: a `launch()` under a box that differs from the build-time
/// cell throws `std::runtime_error`.
///
/// `role() == Evaluator`, `capabilities() == Energy` (plus `Force` with
/// `compute_forces`). Plugs into the Evaluate / MD / Minimize drivers exactly
/// like the other ES evaluators.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief One element's basis + pseudopotential — CP2K's `&KIND` concept.
///        Element identity is owned by `molrs::Element` (see
///        atomiverse/molrs.hpp) and remains an explicit field. `Kind` is a
///        composition object, not an element alias or implicit adapter.
struct Kind {
    molrs::Element
        element; ///< chemical element, matched against the atoms type column (Z).
    GaussianBasis basis; ///< contracted Gaussian basis for this element.
    GthPotential pot;    ///< GTH pseudopotential for this element.

    /// @brief One element's basis + potential in a single expression.
    /// @param element Chemical element (molrs periodic-table enum).
    /// @param basis   Contracted Gaussian basis for this element.
    /// @param pot     GTH pseudopotential for this element.
    Kind(molrs::Element element, GaussianBasis basis, GthPotential pot)
        : element(element), basis(std::move(basis)), pot(std::move(pot)) {}
};

/// @brief Configuration for the GPW evaluator (grid/XC/electrostatics + kinds).
struct GpwEvaluatorConfig {
    double cutoff_ry = 300.0;
    int max_iter = 200;
    double tol = 1e-7;
    double mix = 0.3;
    XcKind xc = XcKind::PADE;
    bool nonperiodic = false;
    bool compute_forces =
        false; ///< also compute analytic forces (one SCF + one gradient pass).
    double fd_step = 1e-3; ///< FD displacement [Bohr] for forces.
    bool d3 = false;       ///< add Grimme DFT-D3 (zero damping, PBE) dispersion
    ///< from the in-tree tables (basis_data/dftd3_tables.h).
    std::vector<Kind> kinds; ///< one per distinct element in the system.
};

/// @brief GPW total-energy evaluator component.
class GPWEvaluator : public cpu::GPW {
  public:
    explicit GPWEvaluator(GpwEvaluatorConfig config)
        : config_(std::move(config)) {}

    std::string name() const override { return "gpw_evaluator"; }
    /* role via Potential/kind base */

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Last total energy [Hartree] (frame carries the eV value).
    double last_energy_hartree() const { return last_.e_total; }
    const GpwResult& last_result() const { return last_; }

  private:
    GpwEvaluatorConfig config_;
    GpwResult last_{};
    std::unique_ptr<GPWSolver> solver_;  ///< Owned inner solver (built once).
    std::array<double, 3> built_cell_{}; ///< Build-time cell [Bohr] (guard).
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
