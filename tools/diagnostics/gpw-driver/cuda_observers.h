// Shared observation of public/nested CUDA GPW lifecycle, outside CTest.
#pragma once
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
namespace ATV::diagnostics::gpw_driver::cuda_observers {
using Method = cuda::gpw::GpwDft;
inline void require(bool pass, const std::string& label) {
    if (!pass)
        throw std::runtime_error(label);
}
inline void close(double a, double b, double tolerance,
                  const std::string& label) {
    if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > tolerance) {
        std::ostringstream message;
        message.precision(17);
        message << label << ": got " << a << ", expected " << b
                << ", tolerance " << tolerance;
        throw std::runtime_error(message.str());
    }
}

struct Science {
    bool stage_live = false, criterion_live = false, inject = false,
         injected = false;
    int builds = 0, destroys = 0, criterion_builds = 0, criterion_destroys = 0;
    int measurements = 0, polls = 0;
    double last_energy = 0, last_residual = 0;
};
class Measurement final : public cuda::Component {
    Science& s_;
    Method& method_;

  public:
    Measurement(Science& s, Method& method) : s_(s), method_(method) {}
    std::string name() const override { return "gpw_science_measurement"; }
    void build(const cuda::BuildContext& context) override {
        require(!s_.stage_live && context.frame.n_atoms() > 0
                    && context.runtime.stream,
                "measurement build requires real system/stream");
        s_.stage_live = true;
        s_.measurements = 0;
        s_.injected = false;
        ++s_.builds;
    }
    void launch(const cuda::LaunchContext& context) override {
        require(s_.stage_live && s_.criterion_live
                    && context.runtime.loop.metrics,
                "SCF measurement must have full stage/criterion lifecycle");
        const auto& result = method_.last_result();
        require(result.iters == context.step + 1,
                "measurement belongs to current SCF iteration");
        s_.last_energy = result.e_total * units::kHartreeInEv;
        s_.last_residual = context.runtime.loop.metrics->residual;
        require(std::isfinite(s_.last_energy) && std::abs(s_.last_energy) > 1e-4
                    && std::isfinite(s_.last_residual) && s_.last_residual >= 0,
                "measurement needs actual finite energy and DIIS residual");
        ++s_.measurements;
        if (s_.inject) {
            s_.inject = false;
            s_.injected = true;
            throw std::runtime_error("intentional GPW SCF stage failure");
        }
    }
    void destroy(const cuda::DestroyContext&) override {
        require(s_.stage_live, "measurement must be released once");
        s_.stage_live = false;
        ++s_.destroys;
    }
};
class ScientificGuard final : public cuda::Criterion {
    Science& s_;

  public:
    explicit ScientificGuard(Science& s) : s_(s) {}
    std::string name() const override { return "gpw_scientific_guard"; }
    void build(const cuda::BuildContext&) override {
        require(!s_.criterion_live,
                "criterion must be released between physical solves");
        s_.criterion_live = true;
        s_.polls = 0;
        ++s_.criterion_builds;
    }
    Verdict verdict(const cuda::LoopState& state) override {
        require(s_.stage_live && s_.criterion_live
                    && s_.measurements == state.iteration + 1,
                "criterion must follow measured physical iterations");
        close(state.metrics.energy, s_.last_energy, 1e-12,
              "criterion SCF energy/eV");
        close(state.metrics.residual, s_.last_residual, 0,
              "criterion DIIS residual");
        ++s_.polls;
        return Verdict::Continue;
    }
    void destroy(const cuda::DestroyContext&) override {
        require(s_.criterion_live, "criterion must receive matching teardown");
        require(
            s_.injected ? s_.polls == 0 : s_.polls == s_.measurements,
            "criterion must poll exactly the completed physical iterations");
        s_.criterion_live = false;
        ++s_.criterion_destroys;
    }
};
inline void lifecycle(const Science& s, int passes, int iterations) {
    require(!s.stage_live && !s.criterion_live && s.builds == passes
                && s.destroys == passes && s.criterion_builds == passes
                && s.criterion_destroys == passes,
            "every solve must build and destroy mounted measurement/criterion");
    require(s.measurements == iterations && s.polls == iterations,
            "one reference SCF must supply each derivative call");
}

struct ParentProbe {
    cuda::LoopServices loop;
    cuda::ExecServices exec;
    cuda::Frame* frame = nullptr;
    void* stream = nullptr;
    IterationStatus status;
    ConvergenceMetrics metrics;
    int captures = 0, teardowns = 0, moves = 0;
};
inline void restored(const ParentProbe& p, const cuda::Context& context,
                     const cuda::Frame& frame) {
    require(
        p.loop.status == context.loop.status
            && p.loop.metrics == context.loop.metrics
            && p.exec.backend == context.exec.backend && p.frame == &frame
            && p.stream == context.stream,
        "nested GPW must restore parent loop and borrow its execution owner");
    require(p.exec.forces_requested == context.exec.forces_requested
                && p.exec.stress_requested == context.exec.stress_requested,
            "nested GPW must restore parent's physical requests");
    require(p.status.converged == context.loop.status->converged
                && p.status.stop_requested
                       == context.loop.status->stop_requested
                && p.status.reason == context.loop.status->reason,
            "nested convergence must not mutate parent status");
    close(p.metrics.energy, context.loop.metrics->energy, 0,
          "parent energy metric preserved");
    close(p.metrics.residual, context.loop.metrics->residual, 0,
          "parent residual metric preserved");
}
class Capture final : public cuda::Component {
    ParentProbe& p_;

  public:
    explicit Capture(ParentProbe& p) : p_(p) {}
    std::string name() const override { return "gpw_parent_capture"; }
    void launch(const cuda::LaunchContext& c) override {
        p_.loop = c.runtime.loop;
        p_.exec = c.runtime.exec;
        p_.frame = &c.frame;
        p_.stream = c.runtime.stream;
        p_.status = *c.runtime.loop.status;
        p_.metrics = *c.runtime.loop.metrics;
        ++p_.captures;
    }
    void destroy(const cuda::DestroyContext& c) override {
        if (p_.captures) {
            restored(p_, c.runtime, c.frame);
            ++p_.teardowns;
        }
    }
};
class AfterChild final : public cuda::Component {
    ParentProbe& p_;
    Method& method_;
    Science& science_;
    int index_;

  public:
    AfterChild(ParentProbe& p, Method& method, Science& science, int index)
        : p_(p), method_(method), science_(science), index_(index) {}
    std::string name() const override {
        return index_ ? "after_second_gpw" : "after_first_gpw";
    }
    void launch(const cuda::LaunchContext& c) override {
        restored(p_, c.runtime, c.frame);
        const auto& result = method_.last_result();
        require(result.converged && method_.converged()
                    && result.iters == science_.measurements,
                "nested method must own exactly one measured converged SCF");
        bool rejected = false;
        try {
            Method illegal_move(std::move(method_));
        } catch (const std::logic_error&) {
            rejected = true;
        }
        require(rejected, "a live mounted method must reject moving");
        ++p_.moves;
    }
};

} // namespace ATV::diagnostics::gpw_driver::cuda_observers
