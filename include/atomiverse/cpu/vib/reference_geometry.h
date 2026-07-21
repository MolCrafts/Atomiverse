/// @file cpu/vib/reference_geometry.h
/// @brief cpu::vib::ReferenceGeometry — caches the equilibrium geometry.
///
/// Bound at `vib::Phase::setup`. On the first displacement pass
/// (`ctx.step == 0`) it copies the frame's Cartesian coordinates into
/// the shared `HessianWorkspace::r0`; on every subsequent pass it is a
/// no-op (idempotent). All later displacement decoding is referenced
/// against this cached `R0`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/vib/hessian_workspace.h"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Structural component that snapshots the equilibrium geometry.
///
/// `role() == None` (structural infrastructure, like `Zero*`); writes no
/// frame field, so `capabilities() == None`.
class ReferenceGeometry : public cpu::Component {
  public:
    /// @brief Construct bound to a driver-owned workspace.
    /// @param ws Non-owning pointer to the shared Hessian workspace.
    explicit ReferenceGeometry(HessianWorkspace* ws) : ws_(ws) {}

    std::string name() const override { return "vib_reference_geometry"; }
    /* role default default role 0 */

    /// @brief Default binding: `vib::Phase::setup` @ 0.
    /// @return Single binding placing this at the setup phase.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::vib::Phase::setup, 0)};
    }

    /// @brief Cache `R0` on the first pass; idempotent thereafter.
    /// @param ctx Launch context; `ctx.step` is the displacement index.
    void launch(const cpu::LaunchContext& ctx) override {
        if (ws_->r0_ready)
            return;
        const int n = ctx.frame.nlocal();
        // Hoisted once before the loop -- the 13.7x rule.
        const StorageReal* __restrict sp_x = ctx.frame.span(kAtom_x);
        const StorageReal* __restrict sp_y = ctx.frame.span(kAtom_y);
        const StorageReal* __restrict sp_z = ctx.frame.span(kAtom_z);
        for (int i = 0; i < n; ++i) {
            ws_->r0[3 * i + 0] = sp_x[i];
            ws_->r0[3 * i + 1] = sp_y[i];
            ws_->r0[3 * i + 2] = sp_z[i];
        }
        ws_->r0_ready = true;
    }

  private:
    HessianWorkspace*
        ws_; ///< Non-owning; outlives this component (driver-owned).
};

} // namespace vib
} // namespace cpu
} // namespace ATV
