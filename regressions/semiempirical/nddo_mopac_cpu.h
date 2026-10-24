/// @file regressions/semiempirical/nddo_mopac_cpu.h
/// @brief Shared CPU acceptance exercises for the MNDO and PM3 oracle cases.
#pragma once
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "nddo_mopac_fixtures.h"
#include <limits>
#include <stdexcept>
#include <utility>
#ifdef ATV_HAS_MD
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#endif
#ifdef ATV_HAS_MINIMIZER
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#endif

namespace atv_reg_nddo_mopac {
struct Offset : ATV::cpu::Potential {
    std::string name() const override { return "nddo_oracle_offset"; }
    void launch(const ATV::cpu::LaunchContext& ctx) override {
        ctx.frame.meta.get(ATV::kEnergy) += .125;
        auto fx = ctx.frame.span(ATV::kAtom_fx);
        for (int i = 0; i < ctx.frame.nlocal(); ++i)
            fx[i] += .25;
    }
};
struct Gate : ATV::cpu::Component {
    int mode, trials = 0;
    explicit Gate(int value) : mode(value) {}
    std::string name() const override { return "nddo_oracle_gate"; }
    void launch(const ATV::cpu::LaunchContext& ctx) override {
        if (ctx.runtime.exec.initializing_forces)
            return;
        if (ctx.runtime.exec.trial_evaluation) {
            ++trials;
            if (mode == 1)
                ctx.frame.meta.get(ATV::kEnergy) =
                    std::numeric_limits<double>::infinity();
            if (mode == 2)
                throw std::runtime_error("injected trial failure");
        } else if (mode == 3)
            throw std::runtime_error("injected later failure");
    }
};
inline bool same_result(const ATV::semiempirical::NDDOResult& a,
                        const ATV::semiempirical::NDDOResult& b,
                        const char* label, bool iterations = false) {
    if (!a.converged || !b.converged
        || a.forces_ev_a.size() != b.forces_ev_a.size()
        || a.mulliken_charges.size() != b.mulliken_charges.size())
        return false;
    bool ok = near(label, "energy", a.total_energy_ev, b.total_energy_ev, 1e-8);
    for (std::size_t i = 0; i < a.forces_ev_a.size(); ++i) {
        char field[40];
        std::snprintf(field, sizeof(field), "force[%zu]", i);
        ok &= near(label, field, a.forces_ev_a[i], b.forces_ev_a[i], 1e-7);
    }
    for (std::size_t i = 0; i < a.mulliken_charges.size(); ++i) {
        char field[40];
        std::snprintf(field, sizeof(field), "charge[%zu]", i);
        ok &= near(label, field, a.mulliken_charges[i], b.mulliken_charges[i],
                   1e-8);
    }
    if (iterations)
        ok &= near(label, "iterations", a.iterations, b.iterations, 0);
    return ok;
}
inline bool same_motion(const ATV::cpu::Frame& a, const ATV::cpu::Frame& b,
                        const char* label) {
    using namespace ATV;
    if (a.nlocal() != b.nlocal())
        return false;
    const auto ax = a.span(kAtom_x), ay = a.span(kAtom_y), az = a.span(kAtom_z);
    const auto bx = b.span(kAtom_x), by = b.span(kAtom_y), bz = b.span(kAtom_z);
    const auto avx = a.span(kAtom_vx), avy = a.span(kAtom_vy),
               avz = a.span(kAtom_vz);
    const auto bvx = b.span(kAtom_vx), bvy = b.span(kAtom_vy),
               bvz = b.span(kAtom_vz);
    bool ok = true;
    for (int i = 0; i < a.nlocal(); ++i) {
        const double lhs[]{ax[i], ay[i], az[i], avx[i], avy[i], avz[i]};
        const double rhs[]{bx[i], by[i], bz[i], bvx[i], bvy[i], bvz[i]};
        for (int d = 0; d < 6; ++d) {
            char field[40];
            std::snprintf(field, sizeof(field), "x/v[%d,%d]", i, d);
            ok &= near(label, field, lhs[d], rhs[d], 1e-10);
        }
    }
    return ok;
}
template <class Fn>
bool must_fail(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    std::fprintf(stderr, "FAIL %s expected exception\n", label);
    return false;
}

template <class Method>
int run_cpu_case(const Pin (&pins)[3]) {
    using namespace ATV;
    try {
        for (const auto& pin : pins) {
            Method method{0, 1, 400, 1e-10, 1e-12};
            auto frame = make_frame(pin);
            for (int i = 0; i < frame.nlocal(); ++i) {
                frame.span(kAtom_fx)[i] = 7;
                frame.span(kAtom_fy)[i] = 8;
                frame.span(kAtom_fz)[i] = 9;
            }
            if (!check_result(method.run(frame), pin, "CPU run", false))
                return 1;
            for (int i = 0; i < frame.nlocal(); ++i)
                if (frame.span(kAtom_fx)[i] != 7 || frame.span(kAtom_fy)[i] != 8
                    || frame.span(kAtom_fz)[i] != 9)
                    return 1;
            frame.meta.get(kEnergy) = 42;
            frame.span(kAtom_charge)[0] = .123;
            if (!check_result(method.run(std::as_const(frame)), pin,
                              "CPU const run", false)
                || frame.meta.get(kEnergy) != 42
                || frame.span(kAtom_charge)[0]
                       != static_cast<StorageReal>(.123))
                return 1;
            const auto expected = method.forces(frame);
            if (!check_result(expected, pin, "CPU forces"))
                return 1;
            cpu::Evaluate evaluate;
            evaluate.on(evaluate::Phase::evaluate).add(Offset{});
            evaluate.on(evaluate::Phase::evaluate).add_ref(method);
            evaluate.build();
            evaluate.run(frame);
            if (!near(pin.name, "additive energy", frame.meta.get(kEnergy),
                      expected.total_energy_ev + .125, 1e-8))
                return 1;
            for (int i = 0; i < frame.nlocal(); ++i)
                if (!near(pin.name, "additive Fx", frame.span(kAtom_fx)[i],
                          expected.forces_ev_a[3 * i] + .25, 1e-7)
                    || !near(pin.name, "additive Fy", frame.span(kAtom_fy)[i],
                             expected.forces_ev_a[3 * i + 1], 1e-7)
                    || !near(pin.name, "additive Fz", frame.span(kAtom_fz)[i],
                             expected.forces_ev_a[3 * i + 2], 1e-7))
                    return 1;
            std::printf("PASS CPU %s external E/HOF/q/full F and "
                        "Frame/Evaluate semantics\n",
                        pin.name);
        }
        const auto& water = pins[2];
        Method retry{0, 1, 1, 1e-10, 1e-12};
        auto frame = make_frame(water);
        if (!must_fail([&] { retry.forces(frame); }, "SCF budget"))
            return 1;
        retry.max_iter = 400;
        if (!check_result(retry.forces(frame), water, "SCF retry"))
            return 1;
        const double old = frame.span(kAtom_x)[0];
        frame.span(kAtom_x)[0] = std::numeric_limits<double>::quiet_NaN();
        if (!must_fail([&] { retry.forces(frame); }, "invalid coordinate"))
            return 1;
        frame.span(kAtom_x)[0] = old;
        if (!check_result(retry.forces(frame), water, "coordinate retry"))
            return 1;

        // Parent completion, rather than a successful trial SCF, accepts history.
        for (int mode = 0; mode < 4; ++mode) {
#ifndef ATV_HAS_MINIMIZER
            if (mode != 3)
                continue;
#endif
            Method method{0, 1, 400, 1e-10, 1e-12},
                control{0, 1, 400, 1e-10, 1e-12};
            auto base = make_frame(water);
            auto control_base = base;
            method.forces(base);
            control.forces(control_base);
            auto proposed = base;
            Gate gate{mode};
            bool failed = false;
            std::string failure_reason;
            if (mode == 3) {
                proposed.span(kAtom_x)[1] += .1;
                cpu::Evaluate parent;
                parent.on(evaluate::Phase::evaluate).add_ref(method);
                parent.on(evaluate::Phase::finalize).add_ref(gate);
                parent.build();
                try {
                    parent.run(proposed);
                } catch (const std::exception& error) {
                    failed = true;
                    failure_reason = error.what();
                }
            }
#ifdef ATV_HAS_MINIMIZER
            else {
                cpu::minimizer::Minimize parent;
                // The acceptance path uses LBFGS's normal search budget.
                // Rejection/error probes intentionally stop after three trials.
                cpu::minimizer::LBFGS optimizer{5, .001, mode == 0 ? 10 : 3};
                optimizer.line_search(cpu::minimizer::LineSearchMode::Armijo);
                parent.set_optimizer(std::move(optimizer));
                parent.on(minimizer::Phase::force).add_ref(method);
                parent.on(minimizer::Phase::force).add_ref(gate, 10);
                parent.build();
                try {
                    parent.run(proposed, 1);
                } catch (const std::exception& error) {
                    failed = true;
                    failure_reason = error.what();
                }
                if (gate.trials == 0) {
                    std::fprintf(stderr,
                                 "FAIL CPU %s history mode=%d trial-count "
                                 "got=0 want=>0; threw=%d reason=%s\n",
                                 water.name, mode, failed,
                                 failure_reason.c_str());
                    return 1;
                }
                if (mode
                    && !same_motion(proposed, base, "rejected trial origin"))
                    return 1;
                if (!mode
                    && !(proposed.meta.get(kEnergy) < base.meta.get(kEnergy))) {
                    std::fprintf(stderr,
                                 "FAIL CPU %s accepted step energy got=%.17g "
                                 "want<%.17g; threw=%d reason=%s\n",
                                 water.name, proposed.meta.get(kEnergy),
                                 base.meta.get(kEnergy), failed,
                                 failure_reason.c_str());
                    return 1;
                }
            }
#endif
            if (failed != (mode != 0) || !method.has_density()) {
                std::fprintf(stderr,
                             "FAIL CPU %s history mode=%d threw got=%d want=%d "
                             "accepted-density got=%d want=1 reason=%s\n",
                             water.name, mode, failed, mode != 0,
                             method.has_density(), failure_reason.c_str());
                return 1;
            }
            if (!mode)
                control.forces(proposed);
            auto next = make_frame(water);
            next.span(kAtom_x)[1] += .02;
            auto next_control = next;
            method.forces(next);
            control.forces(next_control);
            if (!same_result(method.last_result(), control.last_result(),
                             "accepted history", true))
                return 1;
        }
#ifdef ATV_HAS_MD
        Method chunked{0, 1, 400, 1e-10, 1e-12}, whole{0, 1, 400, 1e-10, 1e-12};
        cpu::md::MD chunks, once;
        chunks.set_integrator(cpu::md::VelocityVerlet{.01});
        once.set_integrator(cpu::md::VelocityVerlet{.01});
        chunks.on(md::Phase::force).add_ref(chunked);
        once.on(md::Phase::force).add_ref(whole);
        chunks.build();
        once.build();
        auto a = make_frame(water), b = a;
        chunks.run(a, 0);
        if (chunked.has_density()
            || !check_result(chunked.last_result(), water, "MD initial"))
            return 1;
        chunks.run(a, 2);
        chunks.run(a, 2);
        once.run(b, 4);
        if (!same_motion(a, b, "MD chunks")
            || !same_result(chunked.last_result(), whole.last_result(),
                            "MD chunks"))
            return 1;
        a.span(kAtom_x)[1] += .03;
        a.span(kAtom_vx)[1] = .002;
        a.span(kAtom_charge)[0] += .2;
        a.box = cpu::Box::cubic(20);
        auto edited = a;
        Method fresh{0, 1, 400, 1e-10, 1e-12};
        fresh.forces(edited);
        chunks.run(a, 0);
        if (chunked.has_density() || !same_motion(a, edited, "MD host edits")
            || !same_result(chunked.last_result(), fresh.last_result(),
                            "MD host edits"))
            return 1;
        a = make_frame(pins[1]);
        chunks.run(a, 0);
        if (chunked.has_density()
            || !check_result(chunked.last_result(), pins[1], "MD N/type edit"))
            return 1;
#endif
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL CPU %s: %s\n", pins[0].name, error.what());
        return 1;
    }
    return 0;
}
} // namespace atv_reg_nddo_mopac
