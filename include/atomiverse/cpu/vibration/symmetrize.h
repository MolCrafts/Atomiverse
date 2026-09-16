/// @file cpu/vibration/symmetrize.h
/// @brief cpu::vib::Symmetrize — symmetric-subspace projection of the Hessian.
///
/// Bound at `vib::Phase::diagonalize` with order `-1` (runs on the final
/// finalize pass, ahead of any future diagonalizer). The row-wise
/// finite-difference Hessian is only approximately symmetric because
/// each row carries independent gradient noise. The standard treatment
/// (Gaussian, PySCF, ASE) is the orthogonal projection onto the
/// symmetric subspace — the least-squares-optimal symmetric
/// approximation of a noisy near-symmetric matrix:
///
/// @code
///   H <- (H + H^T) / 2
/// @endcode
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/vibration/hessian_workspace.h"

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Structural component that symmetrizes the assembled Hessian.
///
/// `role() == None`; reads/writes only the shared workspace, no frame field.
class Symmetrize : public cpu::Component {
  public:
    /// @brief Construct bound to a driver-owned workspace.
    /// @param ws Non-owning pointer to the shared Hessian workspace.
    explicit Symmetrize(HessianWorkspace* ws) : ws_(ws) {}

    std::string name() const override { return "vib_symmetrize"; }
    /* role default default role 0 */

    /// @brief Default binding: `vib::Phase::diagonalize` @ -1 (before any
    ///        future diagonalizer at that phase).
    /// @return Single binding placing this at the diagonalize phase.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::vib::Phase::diagonalize, -1)};
    }

    /// @brief Replace the workspace Hessian with `(H + H^T) / 2`.
    ///
    /// The `diagonalize` phase runs on every displacement pass, but
    /// symmetrization is only meaningful once every Hessian row has been
    /// written — i.e. on the final finalize pass (`ctx.step == 6N`).
    /// Running it on a partially-filled matrix would mix real rows with
    /// still-zero rows. The guard restricts the projection to the
    /// finalize pass.
    /// @param ctx Launch context; `ctx.step` is the displacement index.
    void launch(const cpu::LaunchContext& ctx) override {
        const int dim = 3 * ws_->n_atoms;
        if (ctx.step < 2 * dim)
            return; // not the finalize pass yet.
        for (int i = 0; i < dim; ++i)
            for (int j = i + 1; j < dim; ++j) {
                const double avg =
                    0.5 * (ws_->hessian(i, j) + ws_->hessian(j, i));
                ws_->hessian(i, j) = avg;
                ws_->hessian(j, i) = avg;
            }
    }

  private:
    HessianWorkspace* ws_; ///< Non-owning; driver-owned.
};

} // namespace vib
} // namespace cpu
} // namespace ATV
