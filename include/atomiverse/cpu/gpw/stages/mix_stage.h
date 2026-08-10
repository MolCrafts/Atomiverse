/// @file cpu/gpw/stages/mix_stage.h
/// @brief MixStage — `gpw::Phase::mix`: Pulay DIIS on the KS matrix.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/gpw/internal/diis.h"
#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief DIIS / linear-mix stage for the GPW Driver loop.
///
/// When `use_diis` is true, extrapolates H from the (H, err) history into
/// `H_ext`. When false, copies H unchanged (linear density mixing happens
/// in DensityStage). Honors `skip_update` so a just-converged iteration
/// leaves the raw H for `store_converged_state` (shipping GPW break-before-mix).
class MixStage : public cpu::Component {
  public:
    /// @param use_diis     True → Pulay DIIS; false → pass-through H.
    /// @param diis_depth   History depth for `GpwDiis`.
    /// @param H            Current KS matrix from KsBuild (read).
    /// @param err          DIIS residual from KsBuild (read).
    /// @param H_ext        Extrapolated (or copied) KS matrix (write).
    /// @param skip_update  When true, leave H_ext = H without pushing DIIS.
    MixStage(bool use_diis, int diis_depth, const linalg::RealMatrix& H,
             const linalg::RealMatrix& err, linalg::RealMatrix& H_ext,
             const bool& skip_update)
        : use_diis_(use_diis), diis_depth_(diis_depth), H_(H), err_(err),
          H_ext_(H_ext), skip_update_(skip_update), diis_(diis_depth) {}

    /// @brief Component name.
    /// @return `"gpw_mix_stage"`.
    std::string name() const override { return "gpw_mix_stage"; }

    /// @brief Default binding: `gpw::Phase::mix` @ 0.
    /// @return Single binding vector.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::gpw::Phase::mix, 0)};
    }

    /// @brief DIIS step or H pass-through.
    /// @param ctx Launch context; self-guards on `gpw::Phase::mix`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::gpw::Phase::mix))
            return;
        if (skip_update_) {
            H_ext_ = H_;
            return;
        }
        if (use_diis_)
            H_ext_ = diis_.step(H_, err_);
        else
            H_ext_ = H_;
    }

    /// @brief Clear DIIS history at the start of a run.
    void reset() { diis_ = internal::GpwDiis(diis_depth_); }

  private:
    bool use_diis_;
    int diis_depth_;
    const linalg::RealMatrix& H_;
    const linalg::RealMatrix& err_;
    linalg::RealMatrix& H_ext_;
    const bool& skip_update_;
    internal::GpwDiis diis_;
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
