/// @file cpu/vib/frequency_report.h
/// @brief cpu::vib::FrequencyReport — tab-separated vibrational-frequency
///        summary writer.
///
/// Bound at `vib::Phase::end_of_step`, this Recorder fires once on the
/// finalize pass (`ctx.step == 6N`), after `Thermochem` has populated
/// the thermochemistry fields of the shared workspace. It writes a
/// machine-parseable TSV: one data row per normal-mode frequency
/// (index, wavenumber, real/imaginary flag) followed by a trailer
/// summarizing the imaginary-mode count, the zero-point energy, and the
/// Gibbs free energy correction.
///
/// Single responsibility (CLAUDE.md): this component performs I/O only —
/// it computes nothing. All numbers come straight from the workspace.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/component/recorder.h"
#include "atomiverse/cpu/vib/hessian_workspace.h"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Recorder that writes the vibrational-frequency TSV summary.
///
/// `role() == ComponentRole::Recorder` — I/O only; `capabilities()` is
/// empty (it writes no `cpu::Frame` field).
class FrequencyReport : public cpu::Recorder {
  public:
    /// @brief Construct bound to a driver-owned workspace.
    /// @param ws   Non-owning pointer to the shared Hessian workspace.
    /// @param path Output TSV path (default `"frequencies.tsv"`).
    explicit FrequencyReport(HessianWorkspace* ws,
                             std::string path = "frequencies.tsv")
        : ws_(ws), path_(std::move(path)) {}

    /// @brief Construct without a workspace (test-only role check).
    /// @param path Output TSV path.
    explicit FrequencyReport(std::string path)
        : ws_(nullptr), path_(std::move(path)) {}

    std::string name() const override { return "vib_frequency_report"; }

    /// @brief Structural role — Recorder (writes I/O only).
    /// @return `ComponentRole::Recorder`.
    /* role via Recorder base */

    /// @brief Default binding: `vib::Phase::end_of_step` @ 0.
    /// @return Single binding placing this at the end-of-step phase.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::vib::Phase::end_of_step, 0)};
    }

    /// @brief Output TSV path.
    /// @return The configured output file path.
    const std::string& path() const { return path_; }

    /// @brief Write the frequency TSV on the finalize pass.
    ///
    /// Gates on `ctx.step == 6N` so it fires exactly once, after
    /// `Thermochem` has run. No-op on every displacement pass.
    ///
    /// @param ctx Launch context; `ctx.step` is the displacement index.
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    HessianWorkspace* ws_; ///< Non-owning; driver-owned (may be null).
    std::string path_;     ///< Output TSV file path.
};

} // namespace vib
} // namespace cpu
} // namespace ATV
