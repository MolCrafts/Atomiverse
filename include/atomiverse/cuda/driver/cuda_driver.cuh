/// @file cuda/driver/cuda_driver.cuh
/// @brief cuda::CUDADriver — a loop node: stages + plan + handlers. **Not a category.**
///
/// ## There is no category any more
///
/// There used to be three classes here — `cuda::EvaluateDriver`, `cuda::DynamicsDriver`,
/// `cuda::OptimizeDriver` — one per "task type", and they differed from one another by
/// an `if`. Evaluate ran one pass. Dynamics ran a budget. Optimize stopped when a
/// component had written `outer_status.converged`. Every new method demanded a fourth,
/// and no user could write a fifth, because "when do we stop" was a private `if` inside
/// a driver class.
///
/// Stripped to the bone, a Driver is only three things:
///
///   1. **Stages**      — a typed phase sequence (an enum); the Plan binds nodes to it.
///   2. **Iteration**   — walk the stages, repeatedly.
///   3. **Termination** — when to stop.
///
/// (3) was the one welded into the class hierarchy, and it is the only thing the three
/// classes disagreed about. Take it out and hand it to a policy object, and there is
/// nothing left for a category to be. So:
///
/// @code
///   Driver<PhaseEnum, DeviceTag>       ← plan / dispatch primitive (facade/driver.h)
///     ├─ cpu::CPUDriver<PhaseEnum>     ← the CPU loop node
///     └─ cuda::CUDADriver<PhaseEnum>   ← THE loop node             (this file)
///          └─ {cuda::Evaluate, cuda::md::MD, cuda::minimizer::Minimize}  ← presets
/// @endcode
///
/// Four levels became two. A **preset** is not an architectural layer: it is a class
/// whose *constructor* (or `install_defaults()`) attaches the default handler and
/// infrastructure that a given kind of run wants (`Minimize` attaches `ForceConverged`;
/// `MD` attaches none — a step budget IS its termination). It contributes defaults, not
/// semantics, and a user can override every one of them.
///
/// ## What this class provides
///
///   1. `build()` — installs infrastructure (`install_defaults()`), then validates the
///      component set, then compiles the Plan.
///   2. The loop primitives every preset composes its `run()` from: `setup_pass`,
///      `step_once`, `poll_handlers_`, `teardown_pass`. The CUDA flavour of
///      `setup_pass` materializes the host frame onto the device (H2D) and
///      `teardown_pass` retrieves it back (D2H).
///   3. **This loop's own** `IterationStatus` + `LoopMetrics` — owned here, borrowed by
///      the Context (`cuda::LoopServices`). A nested loop cannot reach the loop above
///      it.
///   4. The termination policy slot: `add_handler(…)`, plural, composed by severity —
///      **Stop > Converged > Continue**. @see cuda/driver/loop_handler.cuh
///
/// Unlike `cpu::CPUDriver`, this layer deliberately provides **no** categorical-setter
/// API (`add_potential`, `add_potential`, …) — `cuda::md::MD` carries its own — and no MPI. The
/// CPU driver all-reduces the handler verdict across ranks (`MPI_MAX` over the
/// severity); the CUDA driver loop is single-rank, so `poll_handlers_` reduces nothing.
/// When a multi-GPU driver loop lands, the reduction goes in `poll_handlers_` and no
/// handler changes.
///
/// User code does not subclass this directly — it constructs a preset.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "atomiverse/core/device.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/driver/backend.cuh"
#include "atomiverse/cuda/driver/backend_traits.cuh"
#include "atomiverse/cuda/driver/loop_handler.cuh"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/driver.h"
#include "atomiverse/facade/iteration_status.h"
#include "atomiverse/facade/loop.h"

namespace ATV {
namespace cuda {

/// @brief CUDA-backend loop node — shared by every CUDA preset.
///
/// @tparam PhaseEnum Phase enum type for the workflow (must end with a `COUNT`
///         sentinel).
template <typename PhaseEnum>
class CUDADriver : public Driver<PhaseEnum, cuda::device_tag> {
  public:
    using Base = Driver<PhaseEnum, cuda::device_tag>;

    /// @brief Install infrastructure, validate the component set, compile the Plan.
    ///
    /// Order matters: `install_defaults()` first (so auto-injected components — and the
    /// preset's default handler — are visible to validation and to the compiled Plan),
    /// `validate_components()` second (so a build with a missing required component
    /// throws before the Plan is compiled), `Base::build()` last.
    void build() {
        install_defaults();
        validate_components();
        Base::build();
    }

    // ── Termination policy ─────────────────────────────────────────────────

    /// @brief Attach a termination policy. Plural — handlers compose.
    ///
    /// `min.add_handler(cuda::ForceConverged{0.01});`
    /// `sim.add_handler(cuda::StopOnNaN{});`
    ///
    /// This is the seam that used to be a class hierarchy.
    /// @see cuda/driver/loop_handler.cuh
    /// @param h A LoopHandler subclass, moved in.
    /// @return Reference to self, for chaining.
    template <typename H>
    CUDADriver& add_handler(H&& h) {
        using D = std::decay_t<H>;
        static_assert(std::is_base_of_v<cuda::LoopHandler, D>,
                      "add_handler requires a cuda::LoopHandler subclass");
        handlers_.push_back(std::make_unique<D>(std::forward<H>(h)));
        return *this;
    }

    /// @brief Whether any handler is attached (a preset injects a default if not).
    bool has_handler() const { return !handlers_.empty(); }

    /// @brief Did this loop terminate because a handler declared success?
    ///
    /// **The loop knows; the algorithm does not.** `CG::converged()` and
    /// `LBFGS::converged()` are deleted: an optimizer that publishes metrics and does
    /// not judge them has no business claiming to know the verdict, and asking it was
    /// how the policy got welded to the algorithm in the first place.
    bool converged() const { return status_.converged; }

    /// @brief Which handler ended the loop (empty if it simply ran out of budget).
    const std::string& stop_reason() const { return status_.reason; }

  protected:
    /// @brief Subclass hook — attach default handlers / auto-inject infrastructure
    ///        derived from declared user state (`cuda::md::MD` injects `cuda::Virial`
    ///        when a component reads Stress; `cuda::minimizer::Minimize` attaches a
    ///        `ForceConverged` handler when the user attached none). Default no-op.
    ///        Called from `build()` before `validate_components()`, so anything it adds
    ///        counts toward validation.
    virtual void install_defaults() {}

    /// @brief Subclass hook — validate the component set and throw on
    ///        missing roles / capabilities.
    ///
    /// Called *before* the Plan is built, so `this->pending_` still
    /// carries the registered components.
    virtual void validate_components() const = 0;

    /// @brief Walk every pending component binding.
    ///
    /// Components with multiple `default_bindings()` entries appear once
    /// per binding — duplicate-tolerant predicates are expected.
    /// @tparam Fn Callable invoked as `fn(const cuda::Component&)`.
    /// @param fn Visitor applied to each pending component.
    template <typename Fn>
    void for_each_pending(Fn&& fn) const {
        for (const auto& pb : this->pending_)
            fn(*pb.component);
    }

    /// @brief Mutable variant of for_each_pending — visitor may mutate the
    /// component (e.g. `install_defaults()` gating per-step virial via
    /// `set_stress_required`). @tparam Fn Callable `fn(cuda::Component&)`.
    template <typename Fn>
    void for_each_pending_mut(Fn&& fn) {
        for (auto& pb : this->pending_)
            fn(*pb.component);
    }

    // ── Driver loop helpers (protected — presets only) ──────────────

    /// @brief One-time setup — backend init, plan load, H2D materialize,
    ///        component build.
    ///
    /// The CUDA flavour transfers the host frame onto the device
    /// (`materialize`, H2D) before building components.
    /// @param frame Host-resident system state, transferred to the device.
    void setup_pass(cpu::Frame& frame) {
        if (!this->backend_)
            this->backend_ = std::make_unique<typename Base::BackendT>();
        this->backend_->init();
        this->backend_->load(this->plan_);
        this->backend_->materialize(frame);
        // Point the Context at THIS loop's status + metrics. The loop owns them; the
        // Context only borrows. A nested driver would re-point them at its own, which
        // is what makes a nested loop physically unable to stop its parent.
        this->backend_->context().loop.status = &status_;
        this->backend_->context().loop.metrics = &metrics_;
        this->backend_->build_components();
        status_.reset();
        metrics_ = LoopMetrics{};
    }

    /// @brief Single iteration through the compiled Plan.
    /// @param iter_index Zero-based iteration index passed to components.
    void step_once(int iter_index) { this->backend_->step(iter_index); }

    /// @brief Tear down — destroy components, retrieve results (D2H),
    ///        shut down the backend.
    /// @param frame Host-resident frame populated from the device.
    void teardown_pass(cpu::Frame& frame) {
        this->backend_->destroy_components();
        this->backend_->retrieve(frame);
        this->backend_->shutdown();
    }

    /// @brief Ask every handler about the iteration just completed, and reduce.
    ///
    /// Composition: **Stop > Converged > Continue** — the most severe verdict wins, so a
    /// run that has gone numerically insane cannot report success by also happening to
    /// satisfy a force tolerance.
    ///
    /// Also folds `status_.stop_requested`, which a *component* may set — components
    /// request stops, they never declare convergence: `IterationStatus::converged` is
    /// written here, from the handlers' reduced verdict, and nowhere else.
    ///
    /// Single-rank: unlike `cpu::CPUDriver::poll_handlers_` there is no `MPI_Allreduce`
    /// of the severity, because the CUDA driver loop has no communicator.
    ///
    /// @param iteration Index of the iteration just completed.
    /// @return The reduced verdict.
    Verdict poll_handlers_(int iteration) {
        const cuda::LoopState st{iteration, this->backend_->frame(),
                                 this->backend_->context(), metrics_};
        Verdict v = status_.stop_requested ? Verdict::Stop : Verdict::Continue;
        for (auto& h : handlers_) {
            const Verdict hv = h->after_iteration(st);
            if (hv != Verdict::Continue && worse_of(v, hv) == hv && v != hv) {
                status_.reason = h->name();
            }
            v = worse_of(v, hv);
        }
        if (v == Verdict::Converged)
            status_.converged = true;
        if (v == Verdict::Stop)
            status_.stop_requested = true;
        return v;
    }

    /// @brief Mutable handle to this loop's iteration status (for the per-iteration
    /// `reset()`). **Owned here, not on the Context.**
    IterationStatus& mutable_outer_status() { return status_; }

    /// @brief Throw a clear exception when a required role or capability
    ///        is absent at `validate_components()` time.
    /// @param what Human-readable name of the missing requirement.
    [[noreturn]] static void throw_missing(const char* what) {
        throw std::runtime_error(std::string("driver build: missing required ")
                                 + what);
    }

  protected:
    /// This loop's own status and metrics. **Not on the Context** — see LoopServices.
    IterationStatus status_{};
    LoopMetrics metrics_{};
    /// Termination policies — plural, composed by severity. @see poll_handlers_
    std::vector<std::unique_ptr<cuda::LoopHandler>> handlers_;
};

} // namespace cuda
} // namespace ATV
