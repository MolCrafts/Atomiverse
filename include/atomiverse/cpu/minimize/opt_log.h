/// @file cpu/minimize/opt_log.h
/// @brief OptLog -- per-step optimization summary written as TSV.
///
/// Records `step`, `E[eV]`, `max_force[eV/A]`, `rms_force[eV/A]`, and
/// `step_norm[Angstrom]` to a tab-separated file (and stdout if requested).
///
/// Usage:
/// @code
///   opt.on(minimizer::Phase::end_of_step)
///      .add(cpu::minimizer::OptLog{"aspirin_opt.tsv"});
/// @endcode
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/recorder.h"

namespace ATV {
namespace cpu {
namespace minimizer {

/// @brief TSV logger for geometry-optimization diagnostics.
///
/// Reads forces and positions directly from the canonical atom columns,
/// so it works with any force evaluator (classical, SCF-based, ...).
/// Computes step_norm = ||x - x_prev|| between successive launches.
///
/// MPI-aware: every rank computes owned-row partials and joins the
/// cross-rank reduction (the launch is collective under multi-rank);
/// only rank 0 writes the file / stdout row. After an atom-migration
/// reshuffle the step_norm reference is invalid and the row reports 0.
class OptLog : public cpu::Recorder {
  public:
    /// @brief Construct.
    /// @param filename   Output TSV path.
    /// @param also_stdout If true, mirror each row to stdout.
    explicit OptLog(std::string filename, bool also_stdout = false);

    std::string name() const override { return "opt_log"; }
    /* role via Recorder base */

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::minimizer::Phase::end_of_step, 0)};
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;
    void destroy(const cpu::DestroyContext& ctx) override;

    /// @brief Number of rows written (excluding header).
    int n_rows() const { return n_rows_; }

  private:
    std::string filename_;
    bool also_stdout_;
    std::ofstream os_;
    std::vector<double> x_prev_;
    std::vector<double> y_prev_;
    std::vector<double> z_prev_;
    bool has_prev_ = false;
    int n_rows_ = 0;
    /// @brief Last observed `dd->neighbor_generation` (row-reshuffle signal).
    std::uint64_t last_generation_ = ~std::uint64_t{0};
};

} // namespace minimizer
} // namespace cpu
} // namespace ATV
