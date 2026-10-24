/// @file regressions/semiempirical/am1_water_force/reg_am1_water_force.cpp
/// @brief AM1 water: literature ΔHf-scale energy + force self-consistency.
///
/// Literature / textbook anchors:
/// - Dewar, Zoebisch, Healy, Stewart, J. Am. Chem. Soc. 107, 3902 (1985)
///   (AM1 parameterization). Water experimental ΔHf(gas) ≈ −57.8 kcal/mol;
///   AM1 reproduces small-molecule thermochemistry to a few kcal/mol.
/// - Force check: the published force — an analytic integral derivative at
///   FIXED converged density — must match a RE-CONVERGED central
///   finite difference of the total energy. It does because the
///   density-response term of dE/dR vanishes at SCF stationarity, and NDDO's
///   orthonormal (zero-differential-overlap) basis carries no
///   basis-derivative "Pulay" term.
///
/// Units: energy compared after converting total electronic+nuclear energy to
/// a heat-of-formation proxy is NOT used here — we pin the **force** identity
/// and that the total energy is negative and finite (bound molecule).
///
/// Exit non-zero on failure.

#include "atomiverse/cpu/semiempirical/rm1_solver.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "harness/registry.h"
#ifdef ATV_HAS_MD
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#endif
#ifdef ATV_HAS_MINIMIZER
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#endif

namespace atv_reg_semiempirical_am1_water_force {

namespace {

ATV::cpu::Frame make_water() {
    // Experimental-like geometry (Å), slightly off-equilibrium so |F| ≠ 0.
    ATV::cpu::Frame f(3);
    f.span(ATV::kAtom_type)[0] = 8;
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_x)[1] = 1.05;
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_x)[2] = -0.26;
    f.span(ATV::kAtom_y)[2] = 1.0;
    f.span(ATV::kAtom_z)[2] = 0.0;
    f.box = ATV::cpu::Box::cubic(20.0);
    return f;
}

// Component-author boundary used by the real initialization/trial requests.
// Keep this in the existing scientific regression, not a new API-shape case.
struct NddoRequestMode : ATV::cpu::Component {
    int& mode;
    bool clear;
    NddoRequestMode(int& request, bool reset) : mode(request), clear(reset) {}
    std::string name() const override { return "nddo_history_request"; }
    void launch(const ATV::cpu::LaunchContext& ctx) override {
        ctx.runtime.exec.initializing_forces = !clear && mode == 1;
        ctx.runtime.exec.trial_evaluation = !clear && mode == 2;
    }
};

// Inject a later failure or a rejected trial through the real parent pipeline.
struct NddoEvaluationGate : ATV::cpu::Component {
    int& mode;
    int& trials;
    NddoEvaluationGate(int& control, int& count)
        : mode(control), trials(count) {}
    std::string name() const override { return "nddo_evaluation_gate"; }
    void launch(const ATV::cpu::LaunchContext& ctx) override {
        if (ctx.runtime.exec.initializing_forces)
            return;
        if (ctx.runtime.exec.trial_evaluation) {
            ++trials;
            if (mode == 1)
                ctx.frame.meta.get(ATV::kEnergy) =
                    std::numeric_limits<double>::infinity();
            if (mode == 2)
                throw std::runtime_error("injected NDDO trial failure");
        } else if (mode == 3) {
            throw std::runtime_error("injected failure after NDDO publication");
        }
    }
};

ATV::cpu::semiempirical::AM1Result run_am1(ATV::cpu::Frame frame) {
    ATV::cpu::semiempirical::AM1 solver;
    solver.build(frame);
    return solver.run(frame);
}

double total_energy_ev(ATV::cpu::Frame frame) {
    return run_am1(std::move(frame)).total_energy_ev;
}

} // namespace

int run() {
    using namespace ATV;

    auto frame = make_water();

    // Public overwrite contract for both NDDO methods. These are supplemental
    // integration assertions within the existing scientifically pinned case.
    const auto check_frame = [&]<typename Method>() {
        auto f = make_water();
        Method method;
        f.meta.get(kEnergy) = 17.0;
        f.meta.get(kEnergyElectronic) = 19.0;
        for (int i = 0; i < 3; ++i) {
            f.span(kAtom_charge)[i] = 8.0;
            f.span(kAtom_fx)[i] = 3.0;
        }
        method.run(std::as_const(f));
        if (f.meta.get(kEnergy) != 17.0 || f.span(kAtom_charge)[0] != 8.0)
            return false;
        method.run(f);
        for (int i = 0; i < 3; ++i)
            if (f.span(kAtom_charge)[i]
                    != method.last_result().mulliken_charges[i]
                || f.span(kAtom_fx)[i] != 3.0)
                return false;
        if (f.meta.get(kEnergy) != method.last_result().total_energy_ev
            || f.meta.get(kEnergyElectronic)
                   != method.last_result().total_energy_ev)
            return false;
        method.forces(f);
        const auto energy = f.meta.get(kEnergy);
        const auto forces = method.last_result().forces_ev_a;
        // Accumulation uses the same publication path as a parent phase.
        method.publish(f, true, true);
        if (std::abs(f.meta.get(kEnergy) - 2 * energy) > 1.e-7)
            return false;
        for (int i = 0; i < 3; ++i)
            if (std::abs(f.span(kAtom_fx)[i] - 2 * forces[3 * i]) > 1.e-6)
                return false;
        // A changed iteration budget must take effect on an already-built
        // object. Failed solves publish nothing; correcting it permits retry.
        f.span(kAtom_x)[1] += 0.2;
        const auto saved_energy = f.meta.get(kEnergy);
        const auto saved_force = f.span(kAtom_fx)[0];
        const auto saved_charge = f.span(kAtom_charge)[0];
        method.max_iter = 1;
        bool failed = false;
        try {
            method.forces(f);
        } catch (const std::runtime_error&) {
            failed = true;
        }
        if (!failed || method.last_result().converged
            || f.meta.get(kEnergy) != saved_energy
            || f.span(kAtom_fx)[0] != saved_force
            || f.span(kAtom_charge)[0] != saved_charge)
            return false;
        method.max_iter = 400;
        method.forces(f);
        // A failed first build must not leave an unusable non-null engine.
        Method retry;
        auto invalid = f;
        invalid.span(kAtom_type)[0] = 0;
        failed = false;
        try {
            retry.forces(invalid);
        } catch (const std::exception&) {
            failed = true;
        }
        if (!failed)
            return false;
        retry.forces(f);
        return retry.last_result().converged;
    };
    if (!check_frame.operator()<cpu::semiempirical::AM1>()
        || !check_frame.operator()<cpu::semiempirical::RM1>()) {
        std::fprintf(stderr,
                     "FAIL: NDDO Frame overwrite/accumulation contract\n");
        return 1;
    }

    const auto check_history = [&]<typename Method>() {
        Method normal{0, 1, 400, 1.e-10, 1.e-12};
        Method probed{0, 1, 400, 1.e-10, 1.e-12};
        cpu::Evaluate plain, with_probes;
        int mode = 0;
        plain.on(evaluate::Phase::evaluate).add_ref(normal);
        with_probes.on(evaluate::Phase::evaluate)
            .add(NddoRequestMode{mode, false}, -1);
        with_probes.on(evaluate::Phase::evaluate).add_ref(probed);
        with_probes.on(evaluate::Phase::evaluate)
            .add(NddoRequestMode{mode, true}, 1);
        plain.build();
        with_probes.build();
        // Initial preparation can produce forces without accepting any guess.
        for (mode = 1; mode <= 2; ++mode) {
            auto f = make_water();
            with_probes.run(f);
            if (!probed.last_result().converged || probed.has_density())
                return false;
        }
        mode = 0;
        for (int step = 0; step < 6; ++step) {
            auto a = make_water();
            a.span(kAtom_x)[1] += 0.015 * step;
            auto b = a;
            plain.run(a);
            with_probes.run(b);
            if (normal.last_result().iterations
                    != probed.last_result().iterations
                || std::abs(a.meta.get(kEnergy) - b.meta.get(kEnergy)) > 1.e-10)
                return false;
            for (int i = 0; i < 3; ++i)
                if (std::abs(a.span(kAtom_fx)[i] - b.span(kAtom_fx)[i]) > 1.e-9
                    || std::abs(a.span(kAtom_fy)[i] - b.span(kAtom_fy)[i])
                           > 1.e-9
                    || std::abs(a.span(kAtom_fz)[i] - b.span(kAtom_fz)[i])
                           > 1.e-9)
                    return false;
            for (mode = 1; mode <= 2; ++mode) {
                auto trial = b;
                trial.span(kAtom_x)[1] += 0.12 * mode;
                with_probes.run(trial);
            }
            mode = 0;
            auto failed = b;
            failed.span(kAtom_y)[2] += 0.2;
            probed.max_iter = 1;
            bool rejected = false;
            try {
                with_probes.run(failed);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            probed.max_iter = 400;
            if (!rejected || probed.last_result().converged
                || !probed.has_density())
                return false;
        }
        return true;
    };
    if (!check_history.operator()<cpu::semiempirical::AM1>()
        || !check_history.operator()<cpu::semiempirical::RM1>()) {
        std::fprintf(stderr, "FAIL: NDDO initialization/trial/failure changed "
                             "accepted history\n");
        return 1;
    }

    const auto check_transaction = [&]<typename Method>(const char* name) {
        const auto same = [&](const Method& a, const Method& b,
                              const char* phase) {
            const auto& x = a.last_result();
            const auto& y = b.last_result();
            if (x.iterations != y.iterations
                || std::abs(x.total_energy_ev - y.total_energy_ev) > 1.e-10) {
                std::fprintf(stderr,
                             "FAIL %s %s iterations got=%d want=%d tol=0; "
                             "energy got=%.17g want=%.17g tol=1e-10 eV\n",
                             name, phase, x.iterations, y.iterations,
                             x.total_energy_ev, y.total_energy_ev);
                return false;
            }
            for (std::size_t i = 0; i < x.forces_ev_a.size(); ++i)
                if (std::abs(x.forces_ev_a[i] - y.forces_ev_a[i]) > 1.e-9) {
                    std::fprintf(stderr,
                                 "FAIL %s %s force[%zu] got=%.17g "
                                 "want=%.17g tol=1e-9 eV/A\n",
                                 name, phase, i, x.forces_ev_a[i],
                                 y.forces_ev_a[i]);
                    return false;
                }
            return true;
        };
        for (int failure : {3, 1, 2, 0}) {
#ifndef ATV_HAS_MINIMIZER
            if (failure != 3)
                continue;
#endif
            Method method{0, 1, 400, 1.e-10, 1.e-12};
            Method control{0, 1, 400, 1.e-10, 1.e-12};
            auto base = make_water();
            method.forces(base);
            control.forces(base);
            auto evaluated = base;
            int trials = 0, mode = failure;
            bool threw = false;
            if (failure == 3) {
                evaluated.span(kAtom_x)[1] += 0.1;
                cpu::Evaluate evaluation;
                evaluation.on(evaluate::Phase::evaluate).add_ref(method);
                evaluation.on(evaluate::Phase::finalize)
                    .add(NddoEvaluationGate{mode, trials});
                evaluation.build();
                try {
                    evaluation.run(evaluated);
                } catch (const std::runtime_error&) {
                    threw = true;
                }
            }
#ifdef ATV_HAS_MINIMIZER
            else {
                cpu::minimizer::Minimize optimization;
                cpu::minimizer::LBFGS optimizer;
                optimizer.line_search(cpu::minimizer::LineSearchMode::Armijo);
                optimization.set_optimizer(std::move(optimizer));
                optimization.on(minimizer::Phase::force).add_ref(method);
                optimization.on(minimizer::Phase::force)
                    .add(NddoEvaluationGate{mode, trials}, 10);
                optimization.build();
                try {
                    optimization.run(evaluated, 1);
                } catch (const std::runtime_error&) {
                    threw = true;
                }
                if (trials == 0) {
                    std::fprintf(
                        stderr,
                        "FAIL %s actual line-search trials got=0 want=>0\n",
                        name);
                    return false;
                }
                if (failure != 0)
                    for (int i = 0; i < 3; ++i)
                        for (const auto key : {kAtom_x, kAtom_y, kAtom_z})
                            if (evaluated.span(key)[i] != base.span(key)[i]) {
                                std::fprintf(stderr,
                                             "FAIL %s rejected origin atom=%d "
                                             "got=%.17g want=%.17g tol=0 A\n",
                                             name, i,
                                             double(evaluated.span(key)[i]),
                                             double(base.span(key)[i]));
                                return false;
                            }
            }
#endif
            if (threw != (failure != 0)) {
                std::fprintf(
                    stderr,
                    "FAIL %s transaction mode=%d threw got=%d want=%d\n", name,
                    failure, threw, failure != 0);
                return false;
            }
            // The successful final geometry must already be the accepted
            // guess at run return. Failed parent/trial calls keep base's guess.
            if (failure == 0)
                control.forces(evaluated);
            auto next = base;
            next.span(kAtom_y)[2] += 0.035;
            auto reference = next;
            method.forces(next);
            control.forces(reference);
            if (!same(method, control,
                      failure == 0 ? "accepted-final" : "failed-parent-retry"))
                return false;
        }
        return true;
    };
    if (!check_transaction.operator()<cpu::semiempirical::AM1>("AM1")
        || !check_transaction.operator()<cpu::semiempirical::RM1>("RM1"))
        return 1;

#ifdef ATV_HAS_MD
    // An edited host Frame must seed the next first half kick with fresh forces,
    // while preserving the parent Runtime and the caller's velocity. These
    // fresh-solver comparisons supplement the external pins in this case.
    const auto check_input_updates = [&]<typename Method>(const char* name) {
        auto frame = make_water();
        Method method{0, 1, 400, 1.e-10, 1.e-12};
        cpu::md::MD dynamics;
        dynamics.set_integrator(cpu::md::VelocityVerlet{.01});
        dynamics.on(md::Phase::force).add_ref(method);
        dynamics.build();
        frame.span(kAtom_mass)[0] = 16;
        frame.span(kAtom_mass)[1] = frame.span(kAtom_mass)[2] = 1;
        dynamics.run(frame, 2);
        if (!method.has_density()) {
            std::fprintf(stderr,
                         "FAIL %s chunk baseline accepted history absent\n",
                         name);
            return false;
        }
        const auto compare = [&](const cpu::Frame& got, const cpu::Frame& want,
                                 const char* phase, bool trajectory) {
            const auto scalar = [&](const char* field, int atom, double a,
                                    double b, double tol) {
                if (std::abs(a - b) <= tol)
                    return true;
                std::fprintf(
                    stderr,
                    "FAIL %s chunk %s %s[%d] got=%.17g want=%.17g tol=%g\n",
                    name, phase, field, atom, a, b, tol);
                return false;
            };
            if (!scalar("energy", -1, got.meta.get(kEnergy),
                        want.meta.get(kEnergy), 1.e-8))
                return false;
            for (int i = 0; i < got.nlocal(); ++i) {
                if (!scalar("charge", i, got.span(kAtom_charge)[i],
                            want.span(kAtom_charge)[i], 1.e-8))
                    return false;
                const AccReal gf[]{got.span(kAtom_fx)[i], got.span(kAtom_fy)[i],
                                   got.span(kAtom_fz)[i]};
                const AccReal wf[]{want.span(kAtom_fx)[i],
                                   want.span(kAtom_fy)[i],
                                   want.span(kAtom_fz)[i]};
                for (int d = 0; d < 3; ++d)
                    if (!scalar("force", 3 * i + d, gf[d], wf[d], 1.e-7))
                        return false;
                if (trajectory) {
                    const StorageReal g[]{
                        got.span(kAtom_x)[i],  got.span(kAtom_y)[i],
                        got.span(kAtom_z)[i],  got.span(kAtom_vx)[i],
                        got.span(kAtom_vy)[i], got.span(kAtom_vz)[i]};
                    const StorageReal w[]{
                        want.span(kAtom_x)[i],  want.span(kAtom_y)[i],
                        want.span(kAtom_z)[i],  want.span(kAtom_vx)[i],
                        want.span(kAtom_vy)[i], want.span(kAtom_vz)[i]};
                    for (int d = 0; d < 6; ++d)
                        if (!scalar("x/v", 6 * i + d, g[d], w[d], 1.e-10))
                            return false;
                }
            }
            return true;
        };
        frame.span(kAtom_x)[1] += .03;
        frame.span(kAtom_vx)[1] = .002;
        frame.span(kAtom_charge)[0] += .1;
        frame.box = cpu::Box::cubic(21);
        auto reference = frame;
        Method fresh{0, 1, 400, 1.e-10, 1.e-12};
        cpu::md::MD restart;
        restart.set_integrator(cpu::md::VelocityVerlet{.01});
        restart.on(md::Phase::force).add_ref(fresh);
        restart.build();
        dynamics.run(frame, 0);
        restart.run(reference, 0);
        if (method.has_density()
            || !compare(frame, reference, "numeric-init", true)) {
            std::fprintf(
                stderr,
                "FAIL %s chunk numeric-init accepted history got=%d want=0\n",
                name, method.has_density());
            return false;
        }
        dynamics.run(frame, 2);
        restart.run(reference, 2);
        if (!compare(frame, reference, "numeric-continued", true))
            return false;
        // Replace atom count and species within the same Frame object. The
        // standard NDDO/VV infrastructure updates its maps without re-running
        // initialization components or inheriting the old water density.
        frame = cpu::Frame(2);
        frame.box = cpu::Box::cubic(21);
        for (int i = 0; i < 2; ++i) {
            frame.span(kAtom_type)[i] = 1;
            frame.span(kAtom_mass)[i] = 1;
            frame.span(kAtom_tag)[i] = i + 1;
        }
        frame.span(kAtom_x)[1] = .8;
        frame.span(kAtom_vx)[0] = .003;
        auto hydrogen = frame;
        Method hydrogen_method{0, 1, 400, 1.e-10, 1.e-12};
        hydrogen_method.forces(hydrogen);
        dynamics.run(frame, 0);
        if (method.has_density()
            || !compare(frame, hydrogen, "identity-init", true)) {
            std::fprintf(
                stderr,
                "FAIL %s chunk identity-init accepted history got=%d want=0\n",
                name, method.has_density());
            return false;
        }
        return true;
    };
    if (!check_input_updates.operator()<cpu::semiempirical::AM1>("AM1")
        || !check_input_updates.operator()<cpu::semiempirical::RM1>("RM1"))
        return 1;
#endif

    // A persistent plan must remain valid when the scientific engine is
    // replaced, resized, or moved. Compare to the same method's fresh solve;
    // these are supplemental lifetime checks, not independent oracle pins.
    const auto check_lifetime = [&]<typename Method>() {
        Method method{0, 1, 400, 1.e-10, 1.e-12};
        const auto method_name = method.qm_engine_name();
        const char* phase = "initial";
        const auto exact = [&](const char* field, auto got, auto want) {
            if (got == want)
                return true;
            std::fprintf(stderr,
                         "FAIL: %s lifetime %s %s got=%.17g want=%.17g tol=0\n",
                         method_name.c_str(), phase, field,
                         static_cast<double>(got), static_cast<double>(want));
            return false;
        };
        auto water = make_water();
        method.forces(water);
        const auto reference = method.last_result();
        const auto* stage = method.plan().slots[0].ops[0].component;
        const auto* criterion = method.plan().lifecycle_only[0];
        const auto same_water = [&](const Method& candidate) {
            const auto& r = candidate.last_result();
            if (!exact("result.converged", r.converged, true)
                || !exact("driver.converged", candidate.converged(), true)
                || !exact("force_count", r.forces_ev_a.size(),
                          reference.forces_ev_a.size()))
                return false;
            if (std::abs(r.total_energy_ev - reference.total_energy_ev)
                > 1.e-8) {
                std::fprintf(stderr,
                             "FAIL: %s lifetime %s energy got=%.17g want=%.17g "
                             "tol=1e-8 eV\n",
                             method_name.c_str(), phase, r.total_energy_ev,
                             reference.total_energy_ev);
                return false;
            }
            for (std::size_t i = 0; i < r.forces_ev_a.size(); ++i) {
                if (std::abs(r.forces_ev_a[i] - reference.forces_ev_a[i])
                    > 1.e-7) {
                    std::fprintf(stderr,
                                 "FAIL: %s lifetime %s force[%zu] got=%.17g "
                                 "want=%.17g tol=1e-7 eV/A\n",
                                 method_name.c_str(), phase, i,
                                 r.forces_ev_a[i], reference.forces_ev_a[i]);
                    return false;
                }
            }
            return exact("stage_identity",
                         candidate.plan().slots[0].ops[0].component == stage,
                         true)
                   && exact("criterion_identity",
                            candidate.plan().lifecycle_only[0] == criterion,
                            true);
        };
        phase = "reset_history";
        method.reset_history();
        if (!exact("has_density", method.has_density(), false))
            return false;
        method.forces(water);
        if (!same_water(method)
            || !exact("iterations", method.last_result().iterations,
                      reference.iterations))
            return false;
        phase = "build_hydrogen";
        cpu::Frame hydrogen(2);
        hydrogen.span(kAtom_type)[0] = hydrogen.span(kAtom_type)[1] = 1;
        hydrogen.span(kAtom_x)[1] = 0.8;
        method.build(hydrogen);
        method.forces(hydrogen);
        if (!exact("result.converged", method.last_result().converged, true)
            || !exact("force_count", method.last_result().forces_ev_a.size(),
                      6u))
            return false;
        phase = "rebuild_water";
        method.forces(water); // same engine, rebuilt atom identity/size
        if (!same_water(method))
            return false;
        phase = "move";
        Method moved(std::move(method));
        moved.forces(water);
        if (!same_water(moved))
            return false;
        // Mount, release, then resume through the standalone boundary.
        phase = "mount";
        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate).add_ref(moved);
        eval.build();
        eval.run(water);
        if (!same_water(moved))
            return false;
        phase = "standalone_after_mount";
        moved.forces(water);
        if (!same_water(moved))
            return false;
        for (const int budget : {0, 1}) {
            phase = budget == 0 ? "zero_budget" : "short_budget";
            cpu::Evaluate capped;
            capped.on(evaluate::Phase::evaluate).add_ref(moved, 0, budget);
            capped.build();
            const auto charge = water.span(kAtom_charge)[0];
            bool failed = false;
            try {
                capped.run(water);
            } catch (const std::runtime_error&) {
                failed = true;
            }
            if (!exact("threw", failed, true)
                || !exact("result.converged", moved.last_result().converged,
                          false)
                || !exact("driver.converged", moved.converged(), false)
                || !exact("has_density", moved.has_density(), true)
                || !exact("max_iter", moved.max_iter, 400)
                || !exact("charge[0]", water.span(kAtom_charge)[0], charge)
                || !exact("energy", water.meta.get(kEnergy), 0.0))
                return false;
            // Neither the parent's failed status nor its cap can leak into
            // the next ordinary solve of the same method.
            phase = budget == 0 ? "retry_nested_after_zero"
                                : "retry_nested_after_short";
            eval.run(water);
            if (!same_water(moved))
                return false;
            phase = budget == 0 ? "retry_standalone_after_zero"
                                : "retry_standalone_after_short";
            moved.forces(water);
            if (!same_water(moved))
                return false;
        }
        phase = "invalid_geometry";
        auto invalid = water;
        invalid.span(kAtom_x)[0] = std::numeric_limits<double>::quiet_NaN();
        bool failed = false;
        try {
            moved.forces(invalid);
        } catch (const std::exception&) {
            failed = true;
        }
        if (!exact("threw", failed, true)
            || !exact("result.converged", moved.last_result().converged, false)
            || !exact("driver.converged", moved.converged(), false)
            || !exact("has_density", moved.has_density(), true))
            return false;
        phase = "retry_after_invalid_geometry";
        moved.forces(water);
        return same_water(moved);
    };
    if (!check_lifetime.operator()<cpu::semiempirical::AM1>()
        || !check_lifetime.operator()<cpu::semiempirical::RM1>()) {
        std::fprintf(stderr, "FAIL: NDDO persistent plan/engine lifetime\n");
        return 1;
    }

    // Public Evaluate path: AM1 mounted as a nested method.
    {
        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate).add(cpu::semiempirical::AM1{});
        eval.build();
        eval.run(frame);

        const double e = frame.meta.get(kEnergy);
        if (!(e < 0.0) || !std::isfinite(e)) {
            std::fprintf(stderr, "FAIL: AM1 water energy not bound (E=%g eV)\n",
                         e);
            return 1;
        }

        double f2 = 0.0;
        double net[3] = {0, 0, 0};
        for (int i = 0; i < 3; ++i) {
            const double fx = frame.span(kAtom_fx)[i];
            const double fy = frame.span(kAtom_fy)[i];
            const double fz = frame.span(kAtom_fz)[i];
            f2 += fx * fx + fy * fy + fz * fz;
            net[0] += fx;
            net[1] += fy;
            net[2] += fz;
        }
        if (f2 < 1.0e-10) {
            std::fprintf(stderr,
                         "FAIL: forces vanished on distorted water (f2=%g)\n",
                         f2);
            return 1;
        }
        for (int d = 0; d < 3; ++d) {
            if (std::abs(net[d]) > 1.0e-5) {
                std::fprintf(stderr, "FAIL: net force component %d = %g eV/Å\n",
                             d, net[d]);
                return 1;
            }
        }
        std::printf("Evaluate path: E=%.6f eV  |F|=%.6f eV/Å  netF ok\n", e,
                    std::sqrt(f2));
    }

    // MOPAC-style heat of formation (Dewar AM1; openmopac calc_of_hof).
    {
        const auto r = run_am1(make_water());
        const double hof = r.heat_of_formation_kcal_mol;
        // Expt. H2O(g) ΔHf ≈ −57.8 kcal/mol; AM1 is typically within a few
        // kcal/mol at near-experimental geometry.
        if (!std::isfinite(hof) || std::abs(hof + 59.0) > 8.0) {
            std::fprintf(stderr,
                         "FAIL: AM1 water ΔHf = %g kcal/mol outside "
                         "[-67, -51] (expect ~-59)\n",
                         hof);
            return 1;
        }
        std::printf("ΔHf(H2O) = %.3f kcal/mol (AM1, MOPAC-style)\n", hof);
    }

    // Fixed-density force vs re-SCF energy finite difference (literature
    // identity: variational SCF gradient ≡ energy derivative).
    {
        auto f0 = make_water();
        cpu::semiempirical::AM1 solver;
        solver.build(f0);
        solver.forces(f0); // one SCF + analytic forces [eV/Å]
        const auto& forces = solver.last_result().forces_ev_a;

        constexpr double h = 5.0e-4;
        double max_err = 0.0;
        for (int i = 0; i < 3; ++i) {
            for (int d = 0; d < 3; ++d) {
                auto fp = make_water();
                auto fm = make_water();
                StorageReal* cp = d == 0   ? fp.span(kAtom_x)
                                  : d == 1 ? fp.span(kAtom_y)
                                           : fp.span(kAtom_z);
                StorageReal* cm = d == 0   ? fm.span(kAtom_x)
                                  : d == 1 ? fm.span(kAtom_y)
                                           : fm.span(kAtom_z);
                cp[i] =
                    static_cast<StorageReal>(static_cast<double>(cp[i]) + h);
                cm[i] =
                    static_cast<StorageReal>(static_cast<double>(cm[i]) - h);
                const double fd =
                    -(total_energy_ev(fp) - total_energy_ev(fm)) / (2.0 * h);
                const double an = forces[static_cast<std::size_t>(3 * i + d)];
                max_err = std::max(max_err, std::abs(an - fd));
            }
        }
        constexpr double tol = 5.0e-3; // eV/Å
        if (max_err > tol) {
            std::fprintf(stderr,
                         "FAIL: max |F_analytic - F_FD| = %g > %g eV/Å\n",
                         max_err, tol);
            return 1;
        }
        std::printf("PASS: AM1 water force (max|ΔF|=%.3e eV/Å) [Dewar AM1 "
                    "JACS 1985 method; SCF gradient identity]\n",
                    max_err);
    }
    return 0;
}

} // namespace atv_reg_semiempirical_am1_water_force

ATV_REGRESSION("semiempirical/am1", "am1_water_force",
               atv_reg_semiempirical_am1_water_force::run);
