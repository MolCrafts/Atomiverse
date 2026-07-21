/// @file cpu/async/snapshot_publisher.h
/// @brief `Publish` -- driver-agnostic infrastructure that fills a
/// ring slot from `cpu::Frame` when at least one async subscriber fires.
///
/// Intended to run at `end_of_step @ +10` **after** per-step reductions
/// (`ZeroKinetic(-1000)`, `KineticEnergy(-100)`, `Reduce(-50)`) and
/// **before** async subscribers (default +20), so captured scalars reflect
/// the globally-reduced end-of-step state.
///
/// The phase enum differs across drivers (`md::Phase::end_of_step` vs
/// `minimizer::Phase::end_of_step` vs `evaluate::Phase::finalize`), so the
/// class declares **no default bindings**. Each driver wires it explicitly
/// in its constructor via `on(<its_phase>::end_of_step).add(Publish{}, 10)`.
///
/// Zero async subscribers → 5 ns early return (hot-path throughput budget).
/// With at least one firing subscriber, cost is dominated by the OMP
/// parallel copy over the **static union** of all subscribers'
/// `capture_mask()` flags, computed once and cached (not recomputed per
/// step). The captured field set therefore does not depend on which
/// subscribers fire this step -- it is uniform for every captured step.
///
/// @see .claude/specs/async-analysis-snapshot-ring-01-uniformity.md
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/async/frame_snapshot.h"
#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

class AnalysisExecutor;

/// @brief Copies frame fields into a ring slot for async consumption.
class Publish : public cpu::Component {
  public:
    std::string name() const override { return "publish"; }

    /// @brief No default binding -- drivers wire explicitly at their own
    /// `end_of_step @ +10` (order +10 keeps the publisher after per-step
    /// reductions and before subscribers at +20).
    std::vector<DefaultBinding> default_bindings() const override { return {}; }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  private:
    AnalysisExecutor* exec_ = nullptr;
    /// @brief Static union of all subscribers' capture masks, computed once
    /// on first capturing launch and reused thereafter.
    CaptureMask static_mask_{};
    bool mask_ready_ = false; ///< True once `static_mask_` has been resolved.
};

} // namespace cpu
} // namespace ATV
