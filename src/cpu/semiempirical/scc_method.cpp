/// @file src/cpu/semiempirical/scc_method.cpp
/// @brief SCC Driver loop: shift → diagonalize → charges → mix → check.

#include "atomiverse/cpu/semiempirical/scc_method.h"
#include "atomiverse/cpu/semiempirical/internal/scc_stages.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/driver/runtime.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

using internal::SccResidual;
using internal::SccStep;

SccOutcome SccDriver::solve(cpu::Runtime* host, int max_iter, double tol,
                            Step shift, Step diagonalize, Step charges,
                            Step accept, Step mix) {
    if (max_iter < 0)
        throw std::invalid_argument("SccDriver: max_iter must be >= 0");
    if (!std::isfinite(tol) || !(tol > 0.0))
        throw std::invalid_argument(
            "SccDriver: tolerance must be finite and positive");
    mutable_outer_status().reset();
    residual_ = 0;
    SccResidual criterion(residual_, tol);

    auto shift_step = std::make_unique<SccStep>(ATV::scc::Phase::shift,
                                                "scc_shift", std::move(shift));
    auto diagonalize_step =
        std::make_unique<SccStep>(ATV::scc::Phase::diagonalize,
                                  "scc_diagonalize", std::move(diagonalize));
    auto charges_step = std::make_unique<SccStep>(
        ATV::scc::Phase::charges, "scc_charges", std::move(charges));
    auto mix_step = std::make_unique<SccStep>(
        ATV::scc::Phase::mix, "scc_mix",
        [&criterion, accept = std::move(accept), mix = std::move(mix)]() {
            if (criterion.accepts())
                accept();
            else
                mix();
        });
    auto check_step =
        std::make_unique<SccStep>(ATV::scc::Phase::check, "scc_check", Step{});

    this->on(ATV::scc::Phase::shift).add_ref(*shift_step);
    this->on(ATV::scc::Phase::diagonalize).add_ref(*diagonalize_step);
    this->on(ATV::scc::Phase::charges).add_ref(*charges_step);
    this->on(ATV::scc::Phase::mix).add_ref(*mix_step);
    this->on(ATV::scc::Phase::check).add_ref(*check_step);
    this->add_criterion_ref(criterion);
    this->build();

    // Stages and the criterion close over this solve. Release the plan before
    // those objects die, including when a step throws. The lambda runs in this
    // member, so it can reach the driver's protected loop state.
    cpu::Frame scratch;
    cpu::Runtime* host_rt = nullptr;
    bool standalone = false;
    cpu::Context* host_ctx = nullptr;
    LoopServices saved_loop{};
    auto release = [&] {
        if (host_ctx)
            host_ctx->loop = saved_loop;
        if (host_rt)
            host_rt->destroy_plan(this->plan());
        if (standalone) {
            teardown_pass();
            this->backend_.reset();
        }
        this->criteria_.clear();
        this->pending_.clear();
        this->owned_components_.clear();
        this->plan_ = {};
    };
    struct OnExit {
        decltype(release) fn;
        ~OnExit() { fn(); }
    } on_exit{release};

    if (host) {
        if (!host->has_frame())
            throw std::logic_error(
                "SccDriver: host Runtime has no Frame; pass it from launch");
        host_rt = host;
        host->build_plan(this->plan());
        host_ctx = &host->context();
        saved_loop = host_ctx->loop;
        host_ctx->loop.status = &this->mutable_outer_status();
        host_ctx->loop.metrics = &this->metrics_;
        this->metrics_ = ConvergenceMetrics{};
    } else {
        setup_pass(scratch);
        standalone = true;
    }

    SccOutcome out;
    for (int iter = 0; iter < max_iter; ++iter) {
        mutable_outer_status().reset();
        residual_ = std::numeric_limits<double>::infinity();
        ++out.iterations;
        if (host) {
            host->step_plan(this->plan(), iter);
            out.residual = residual_;
            if (this->poll_criteria_(iter, host->frame(), *host_ctx)
                != ATV::Verdict::Continue)
                break;
        } else {
            step_once(iter);
            out.residual = residual_;
            if (poll_criteria_(iter) != ATV::Verdict::Continue)
                break;
        }
    }
    out.converged = this->converged();
    return out;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
