/// @file cpu/vib/hessian_accumulator.h
/// @brief cpu::vib::HessianAccumulator — central-difference Hessian-row build.
///
/// Bound at `vib::Phase::accumulate`, after the user-registered force
/// evaluator has written the `kAtom_fx/fy/fz` columns for displacement
/// `s = ctx.step`.
///
/// Displacement `s` is even for the `+h` evaluation and odd for the `-h`
/// evaluation of Cartesian coordinate `c = s/2`:
///   - even `s`: store the `+h` force vector in `HessianWorkspace::f_plus`.
///   - odd  `s`: pair with the just-evaluated `-h` forces and write the
///     central-difference Hessian row
///     `H[c,:] = -(f_plus - f_minus) / (2h)` (eV/Angstrom^2).
///
/// `f = -g`, so `-(f+ - f-)/(2h) = (g+ - g-)/(2h)` is the second
/// derivative `d2E/dR dR`.
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

/// @brief Structural component that assembles Hessian rows by central
///        differencing the evaluator-produced forces.
///
/// Role is the default (required). `cpu::vib::Hessian` identifies this
/// component by `name() == "vib_hessian_accumulator"` at build-time
/// validation — it is the canonical Hessian-producing structural piece.
class HessianAccumulator : public cpu::Component {
  public:
    /// @brief Construct bound to a driver-owned workspace.
    /// @param ws Non-owning pointer to the shared Hessian workspace.
    explicit HessianAccumulator(HessianWorkspace* ws) : ws_(ws) {}

    std::string name() const override { return "vib_hessian_accumulator"; }
    /* role default default role 0 */

    /// @brief Default binding: `vib::Phase::accumulate` @ 0.
    /// @return Single binding placing this at the accumulate phase.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::vib::Phase::accumulate, 0)};
    }

    /// @brief Store `+h` forces or write a central-difference Hessian row.
    /// @param ctx Launch context; `ctx.step` is the displacement index `s`.
    void launch(const cpu::LaunchContext& ctx) override {
        const int n = ws_->n_atoms;
        const int dim = 3 * n;
        const int s = ctx.step;
        if (s >= 2 * dim)
            return; // finalize pass carries no force data.

        // Hoisted once, dominating both mutually-exclusive branches below.
        const AccReal* __restrict sp_fx = ctx.frame.span(kAtom_fx);
        const AccReal* __restrict sp_fy = ctx.frame.span(kAtom_fy);
        const AccReal* __restrict sp_fz = ctx.frame.span(kAtom_fz);

        if (s % 2 == 0) {
            // +h evaluation: snapshot the force vector for later pairing.
            for (int i = 0; i < n; ++i) {
                ws_->f_plus[3 * i + 0] = sp_fx[i];
                ws_->f_plus[3 * i + 1] = sp_fy[i];
                ws_->f_plus[3 * i + 2] = sp_fz[i];
            }
        } else {
            // -h evaluation: write Hessian row for coordinate c = s/2.
            const int coord = s / 2;
            const double inv_2h = 1.0 / (2.0 * kHessianStepAngstrom);
            for (int j = 0; j < n; ++j) {
                const double fmx = sp_fx[j];
                const double fmy = sp_fy[j];
                const double fmz = sp_fz[j];
                ws_->hessian(coord, 3 * j + 0) =
                    -(ws_->f_plus[3 * j + 0] - fmx) * inv_2h;
                ws_->hessian(coord, 3 * j + 1) =
                    -(ws_->f_plus[3 * j + 1] - fmy) * inv_2h;
                ws_->hessian(coord, 3 * j + 2) =
                    -(ws_->f_plus[3 * j + 2] - fmz) * inv_2h;
            }
        }
    }

  private:
    HessianWorkspace* ws_; ///< Non-owning; driver-owned.
};

} // namespace vib
} // namespace cpu
} // namespace ATV
