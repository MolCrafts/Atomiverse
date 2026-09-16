/// @file cpu/driver/driver_node.h
/// @brief cpu::DriverNode — non-template polymorphic base for nestable loop nodes.
///
/// A Driver is an executable node (build → launch × N → destroy). Type erasure
/// lives at the **node** boundary so a Plan slot can hold a component or a
/// nested driver without agreeing on Phase enums. `Driver<PhaseEnum>`
/// implements this interface; the Plan / Backend dispatch through it.
///
/// @see Driver::run_nested for the status-isolation contract (an inner
///      loop cannot stop its enclosing loop).
#pragma once

namespace ATV {
namespace cpu {

class Runtime;
class Context;
class Frame;

/// @brief Erased base for a nestable CPU loop node.
///
/// Pure virtuals are the three nested lifecycle ops (compile+build_plan /
/// run_nested / destroy_plan), plus a default inner budget for
/// `on(phase).add(driver, budget=-1)`.
class DriverNode {
  public:
    virtual ~DriverNode() = default;

    /// @brief Compile this loop's Plan and build its components on `rt`.
    /// @param rt Parent Backend — one Runtime for the whole driver tree.
    virtual void build_on(Runtime& rt) = 0;

    /// @brief Run this loop on a borrowed Runtime / Context / Frame.
    /// @param ctx Parent Context (re-pointed at this loop's status while nested).
    /// @param f   System state.
    /// @param budget Iteration budget for this nested run.
    /// @return Iterations actually executed (≤ budget).
    virtual int run_nested_erased(Context& ctx, Frame& f, int budget) = 0;

    /// @brief Destroy this loop's components on `rt` (reverse Plan order).
    /// @param rt Parent Backend.
    virtual void destroy_on(Runtime& rt) = 0;

    /// @brief Default nested-run budget when the caller passes budget < 0.
    /// @return Preset-specific default (0 unless overridden).
    virtual int default_budget() const { return 0; }

    /// @brief Does anything in this loop resolve global tags at run time?
    ///
    /// Asked once, before the frame is bound, so `DomainRuntime` can decide
    /// whether the Frame tag index may be released. A nested node has to be
    /// able to answer for its own components: the release happens at
    /// `Runtime::set_frame`, which is one call above the whole tree.
    /// @see cpu::Component::requires_tag_index
    virtual bool requires_tag_index() const { return false; }
};

} // namespace cpu
} // namespace ATV
