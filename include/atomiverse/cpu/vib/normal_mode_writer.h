/// @file cpu/vib/normal_mode_writer.h
/// @brief cpu::vib::NormalModeWriter — per-mode XYZ-animation writer.
///
/// Bound at `vib::Phase::end_of_step`, this Recorder fires once on the
/// finalize pass (`ctx.step == 6N`), after `HessianDiagonalizer` has produced
/// the normal modes. For each real vibrational mode it writes a
/// multi-frame XYZ file animating the equilibrium geometry displaced
/// along that normal-mode eigenvector through a sinusoidal sweep —
/// suitable for visualization in any XYZ-trajectory viewer.
///
/// Single responsibility (CLAUDE.md): this component performs I/O only.
/// The eigenvectors are taken from the shared workspace; un-mass-
/// weighting (`dx_i = mode_i / sqrt(m_i)`) is the only arithmetic, and
/// is the standard Cartesian-displacement form PySCF `norm_mode` uses.
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

/// @brief Recorder that writes per-mode XYZ animations.
///
/// `role() == ComponentRole::Recorder` — I/O only; `capabilities()` is
/// empty (it writes no `cpu::Frame` field).
class NormalModeWriter : public cpu::Recorder {
  public:
    /// @brief Construct bound to a driver-owned workspace.
    /// @param ws     Non-owning pointer to the shared Hessian workspace.
    /// @param prefix Output path prefix; mode `k` is written to
    ///               `"<prefix>_mode<k>.xyz"` (default `"mode"`).
    /// @param amplitude Peak Cartesian displacement amplitude [Angstrom]
    ///               of the sinusoidal sweep (default `0.5`).
    /// @param n_frames Number of animation frames per period (default 20).
    explicit NormalModeWriter(HessianWorkspace* ws, std::string prefix = "mode",
                              double amplitude = 0.5, int n_frames = 20)
        : ws_(ws), prefix_(std::move(prefix)), amplitude_(amplitude),
          n_frames_(n_frames) {}

    /// @brief Construct without a workspace (test-only role check).
    /// @param prefix Output path prefix.
    explicit NormalModeWriter(std::string prefix)
        : ws_(nullptr), prefix_(std::move(prefix)), amplitude_(0.5),
          n_frames_(20) {}

    std::string name() const override { return "vib_normal_mode_writer"; }

    /// @brief Structural role — Recorder (writes I/O only).
    /// @return `ComponentRole::Recorder`.
    /* role via Recorder base */

    /// @brief Default binding: `vib::Phase::end_of_step` @ +1.
    ///
    /// Runs after `FrequencyReport` (@ 0) — both gate on the finalize
    /// pass and are independent, but a stable order keeps logs tidy.
    /// @return Single binding placing this at the end-of-step phase.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::vib::Phase::end_of_step, +1)};
    }

    /// @brief Output path prefix.
    /// @return The configured file-name prefix.
    const std::string& prefix() const { return prefix_; }

    /// @brief Output path for a given mode.
    /// @param mode Mode index.
    /// @return `"<prefix>_mode<mode>.xyz"`.
    std::string mode_path(int mode) const {
        return prefix_ + "_mode" + std::to_string(mode) + ".xyz";
    }

    /// @brief Write the per-mode XYZ animations on the finalize pass.
    ///
    /// Gates on `ctx.step == 6N` so it fires exactly once, after
    /// `HessianDiagonalizer` has run. One XYZ file per real mode; imaginary
    /// modes are skipped.
    ///
    /// @param ctx Launch context; `ctx.step` is the displacement index.
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    HessianWorkspace* ws_; ///< Non-owning; driver-owned (may be null).
    std::string prefix_;   ///< Output file-name prefix.
    double amplitude_;     ///< Sweep peak amplitude [Angstrom].
    int n_frames_;         ///< Frames per animation period.
};

} // namespace vib
} // namespace cpu
} // namespace ATV
