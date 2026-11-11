/// @file cuda/dynamics/deform.cuh
/// @brief Deform -- uniaxial strain deformation component (`remap x`).
///
/// Same API and step as `cpu::md::Deform` (LAMMPS `fix deform` `erate` /
/// `trate` on one axis): the deformed axis's cell row and every row's
/// coordinate along that axis scale by `L_target / L_current`, so
/// fractional coordinates are preserved for tilted cells too. The reference
/// length is snapshotted from the live device cell on the first launch and
/// the scale is computed on the device from the current cell; the host only
/// supplies the elapsed time. CUDA-free header.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {
namespace md {

/// @brief Uniaxial constant-strain-rate deformation (mirrors
///        `cpu::md::Deform`).
///
/// Bound at `initial_integrate @ +200`, after the integrator's drift.
class Deform : public Component {
  public:
    /// @brief Per-axis strain mode -- LAMMPS `fix deform` keywords.
    enum class Mode {
        /// Engineering strain `L = L0 (1 + R t)` (LAMMPS `erate`).
        erate,
        /// True strain `L = L0 exp(R t)` (LAMMPS `trate`).
        trate,
    };

    /// @param mode Strain mode.
    /// @param rate Strain rate `R` [1/fs]; positive stretches.
    /// @param dt   MD step [fs] (must match the integrator).
    /// @param axis Deformed axis: 0 = x, 1 = y, 2 = z.
    Deform(Mode mode, double rate, double dt, int axis = 0);
    ~Deform() override;
    Deform(Deform&&) noexcept;
    Deform& operator=(Deform&&) noexcept;

    bool changes_box(int) const override { return true; }
    std::string name() const override { return "deform"; }
    Role role() const override { return 0; }
    /// @brief The strain clock is the device step: replayable.
    bool graph_capturable() const override { return true; }

    /// @brief Write-set: affine remap of coordinates as the box strains.
    std::vector<DefaultBinding> default_bindings() const override;
    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @return Current strain rate `R` [1/fs].
    double rate() const noexcept { return rate_; }

    /// @return Strain mode.
    Mode mode() const noexcept { return mode_; }

  private:
    Mode mode_;
    double rate_, dt_;
    int axis_;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace md
} // namespace cuda
} // namespace ATV
