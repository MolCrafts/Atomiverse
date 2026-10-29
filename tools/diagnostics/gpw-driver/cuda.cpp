// Opt-in complete GPW execution checks. Never registered with CTest.
// CPU twins qualify the original scientific envelope; standalone CUDA methods
// supply nested accumulation references. No new independent external oracle.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"
#include "cuda_observers.h"
#include "fixtures.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ATV;
namespace {
using namespace ATV::diagnostics::gpw_driver::cuda_observers;
using Method = cuda::gpw::GpwDft;
using Six = std::array<double, 6>;
namespace fixture = ATV::diagnostics::gpw_driver;
using fixture::Case;
using fixture::cases;
using fixture::configure;
using fixture::cutoff;
using fixture::forces;
using fixture::geometry;
using fixture::kinds;
using fixture::mark;
using fixture::xc;
constexpr double twin_energy_ha = 5e-6;
constexpr double twin_force_eva = 5e-5 * units::kHartreePerBohrInEvPerAngstrom;
constexpr double twin_stress_eva3 =
    5e-5 * units::kHartreePerBohr3InEvPerAngstrom3;
constexpr double nested_energy_ev = 2e-7;
constexpr double nested_force_eva = 2e-7;
constexpr double nested_virial_ev = 2e-6;
std::unique_ptr<Method> moved_method(const Case& c, int index) {
    auto source = std::make_unique<Method>(kinds(), cutoff[index], xc(c));
    configure(*source, c);
    auto frame = geometry(false);
    source->build(frame);
    const auto f = forces(frame);
    const auto v = frame.meta.get(kStress);
    const auto accepted = source->run(std::as_const(frame));
    require(accepted.converged, "pre-move SCF convergence");
    require(forces(frame) == f && frame.meta.get(kStress) == v
                && frame.meta.get(kEnergy) == 31415.125,
            "const run must preserve caller Frame channels");
    auto destination = std::make_unique<Method>(std::move(*source));
    source.reset();
    close(destination->last_result().e_total, accepted.e_total, 0,
          "move retains physical result");
    return destination;
}
struct ChildReference {
    double energy_ev = 0;
    std::vector<double> force;
    Six virial{};
};
struct Reference {
    std::array<ChildReference, 2> child;
    double energy_ev = 0;
    std::vector<double> force;
    Six virial{};
};
Reference standalone(const Case& c, bool shifted) {
    Reference out;
    for (int index = 0; index < 2; ++index) {
        auto frame = geometry(shifted);
        auto solver = moved_method(c, index);
        Science science;
        Measurement measurement(science, *solver);
        ScientificGuard guard(science);
        solver->on(gpw::Phase::check).add_ref(measurement);
        solver->add_criterion_ref(guard);
        int passes = 0;
        if (std::string(c.name) == "rks" && !shifted && index == 0) {
            const auto before = frame;
            science.inject = true;
            bool failed = false;
            try {
                solver->forces(frame);
            } catch (const std::runtime_error& error) {
                failed = std::string(error.what())
                         == "intentional GPW SCF stage failure";
            }
            require(failed && science.injected,
                    "standalone injected stage must fail");
            require(forces(frame) == forces(before)
                        && frame.meta.get(kStress) == before.meta.get(kStress)
                        && frame.meta.get(kEnergy) == before.meta.get(kEnergy),
                    "failed standalone solve must not publish partial results");
            ++passes;
        }
        const auto result = solver->forces(frame);
        require(result.converged, "standalone force SCF convergence");
        lifecycle(science, ++passes, result.iters);
        auto& child = out.child[index];
        child.energy_ev = frame.meta.get(kEnergy);
        child.force = forces(frame);
        const auto sigma = solver->stress(frame);
        require(solver->last_result().converged,
                "standalone stress convergence");
        lifecycle(science, ++passes, solver->last_result().iters);
        require(forces(frame) == child.force, "stress must preserve forces");
        for (int k = 0; k < 6; ++k) {
            close(frame.meta.get(kStress)[k], sigma[k] * frame.box.volume(),
                  1e-10, "standalone Frame virial conversion");
            child.virial[k] = frame.meta.get(kStress)[k];
        }
        // Preserve the established public GPW CPU/CUDA scientific gates.
        cpu::gpw::GpwDft cpu(kinds(), cutoff[index], xc(c));
        configure(cpu, c);
        auto twin = geometry(shifted);
        cpu.build(twin);
        const auto cpu_result = cpu.forces(twin);
        require(cpu_result.converged, "CPU twin force convergence");
        close(result.e_total, cpu_result.e_total, twin_energy_ha,
              "CPU/CUDA energy/Ha");
        const auto cpu_force = forces(twin);
        for (std::size_t i = 0; i < child.force.size(); ++i)
            close(child.force[i], cpu_force[i], twin_force_eva,
                  "CPU/CUDA full force/eV/A");
        const auto cpu_sigma = cpu.stress(twin);
        require(cpu.last_result().converged, "CPU twin stress convergence");
        for (int k = 0; k < 6; ++k)
            close(sigma[k], cpu_sigma[k], twin_stress_eva3,
                  "CPU/CUDA stress/eV/A3");
        out.energy_ev += child.energy_ev;
        if (out.force.empty())
            out.force.assign(child.force.size(), 0);
        for (std::size_t i = 0; i < out.force.size(); ++i)
            out.force[i] += child.force[i];
        for (int k = 0; k < 6; ++k)
            out.virial[k] += child.virial[k];
        solver.reset();
    }
    double peak = 0;
    for (double value : out.force)
        peak = std::max(peak, std::abs(value));
    require(peak > 1e-3, "nested force reference must exercise nonzero forces");
    return out;
}
void run_case(const Case& c) {
    const std::array<Reference, 2> reference = {standalone(c, false),
                                                standalone(c, true)};
    require(std::abs(reference[0].energy_ev - reference[1].energy_ev) > 1e-6,
            "changed geometry must alter physical energy");
    double e_error = 0, f_error = 0, v_error = 0;
    for (int request = 0; request < 3; ++request) {
        const bool force = request > 0, stress = request == 2;
        auto first = moved_method(c, 0), second = moved_method(c, 1);
        Science science_a, science_b;
        Measurement measure_a(science_a, *first), measure_b(science_b, *second);
        ScientificGuard guard_a(science_a), guard_b(science_b);
        first->on(gpw::Phase::check).add_ref(measure_a);
        first->add_criterion_ref(guard_a);
        second->on(gpw::Phase::check).add_ref(measure_b);
        second->add_criterion_ref(guard_b);
        ParentProbe probe;
        Capture capture(probe);
        AfterChild after_first(probe, *first, science_a, 0),
            after_second(probe, *second, science_b, 1);
        cuda::Evaluate parent(force, stress);
        parent.on(evaluate::Phase::evaluate).add_ref(capture, -10);
        parent.on(evaluate::Phase::evaluate).add_ref(*first, 0);
        parent.on(evaluate::Phase::evaluate).add_ref(after_first, 10);
        parent.on(evaluate::Phase::evaluate).add_ref(*second, 20);
        parent.on(evaluate::Phase::evaluate).add_ref(after_second, 30);
        parent.build();
        const bool inject = std::string(c.name) == "rks" && request == 0;
        if (inject) {
            auto frame = geometry(false);
            const auto before = frame;
            science_a.inject = true;
            bool failed = false;
            try {
                parent.run(frame);
            } catch (const std::runtime_error& e) {
                failed = std::string(e.what())
                         == "intentional GPW SCF stage failure";
            }
            require(failed && science_a.injected && probe.teardowns == 1,
                    "nested exception must restore parent before cleanup");
            require(!science_a.stage_live && !science_a.criterion_live
                        && science_a.builds == 1 && science_a.destroys == 1
                        && science_a.criterion_builds == 1
                        && science_a.criterion_destroys == 1
                        && science_a.measurements == 1 && science_a.polls == 0,
                    "failed first child must release its measured stage and "
                    "unpolled criterion");
            // The parent builds both children before launching its Plan.
            // The second child must be torn down even though the first
            // child's failure prevents its first physical iteration.
            require(!science_b.stage_live && !science_b.criterion_live
                        && science_b.builds == 1 && science_b.destroys == 1
                        && science_b.criterion_builds == 1
                        && science_b.criterion_destroys == 1
                        && science_b.measurements == 0 && science_b.polls == 0
                        && !science_b.injected && probe.moves == 0,
                    "unlaunched second child must release its prepared stage "
                    "and criterion without measuring or polling");
            require(forces(frame) == forces(before)
                        && frame.meta.get(kStress) == before.meta.get(kStress)
                        && frame.meta.get(kEnergy) == before.meta.get(kEnergy),
                    "failed parent must preserve caller outputs");
        }
        for (int shifted = 0; shifted < 2; ++shifted) {
            auto frame = geometry(shifted != 0);
            const auto before = frame;
            parent.run(frame);
            const auto& expected = reference[shifted];
            close(frame.meta.get(kEnergy), expected.energy_ev, nested_energy_ev,
                  "two nested energies must accumulate");
            e_error = std::max(e_error, std::abs(frame.meta.get(kEnergy)
                                                 - expected.energy_ev));
            const auto f = forces(frame);
            if (!force)
                require(f == forces(before),
                        "energy-only parent must preserve forces");
            else
                for (std::size_t i = 0; i < f.size(); ++i) {
                    close(f[i], expected.force[i], nested_force_eva,
                          "two nested force vectors must accumulate");
                    f_error =
                        std::max(f_error, std::abs(f[i] - expected.force[i]));
                }
            if (!stress)
                require(frame.meta.get(kStress) == before.meta.get(kStress),
                        "unrequested virial must be preserved");
            else
                for (int k = 0; k < 6; ++k) {
                    close(frame.meta.get(kStress)[k], expected.virial[k],
                          nested_virial_ev,
                          "two nested virials must accumulate");
                    v_error =
                        std::max(v_error, std::abs(frame.meta.get(kStress)[k]
                                                   - expected.virial[k]));
                }
            lifecycle(science_a, shifted + 1 + (inject ? 1 : 0),
                      first->last_result().iters);
            lifecycle(science_b, shifted + 1 + (inject ? 1 : 0),
                      second->last_result().iters);
        }
        require(probe.moves == 4 && probe.teardowns == (inject ? 3 : 2),
                "every child and parent teardown must be observed");
    }
    // Exhaustion after a successful solve must replace the old verdict. Public
    // derivatives fail closed and leave every caller channel unchanged.
    auto limited = moved_method(c, 0);
    limited->max_iter = 1;
    auto frame = geometry(false);
    const auto before = frame;
    const auto result = limited->run(std::as_const(frame));
    require(!result.converged && result.iters == 1,
            "budget exhaustion must replace accepted verdict");
    for (int derivative = 0; derivative < 2; ++derivative) {
        bool failed = false;
        try {
            if (derivative)
                limited->stress(frame);
            else
                limited->forces(frame);
        } catch (const std::runtime_error&) {
            failed = true;
        }
        require(failed && !limited->last_result().converged
                    && limited->last_result().iters == 1,
                "unconverged derivative must reject with its own solve status");
        require(frame.meta.get(kEnergy) == before.meta.get(kEnergy)
                    && frame.meta.get(kStress) == before.meta.get(kStress)
                    && forces(frame) == forces(before),
                "unconverged derivative must preserve caller outputs");
    }
    {
        cuda::Evaluate limited_parent(false, false);
        limited_parent.set_evaluator_ref(*limited);
        limited_parent.build();
        limited_parent.run(frame);
        require(!limited->last_result().converged
                    && limited->last_result().iters == 1,
                "nested budget must retain its exhausted SCF verdict");
        close(frame.meta.get(kEnergy), result.e_total * units::kHartreeInEv,
              nested_energy_ev, "nested and standalone truncated energy/eV");
        require(forces(frame) == forces(before)
                    && frame.meta.get(kStress) == before.meta.get(kStress),
                "nested exhausted energy-only call preserves derivatives");
    }
    std::printf("PASS GPW_CUDA_DRIVER %s modes=3 geometries=2 children=2 "
                "dE_eV=%.6g dF_eV_A=%.6g dVirial_eV=%.6g\n",
                c.name, e_error, f_error, v_error);
    std::fflush(stdout);
}
} // namespace
int main() {
    try {
        for (const auto& c : cases)
            run_case(c);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL GPW_CUDA_DRIVER %s\n", e.what());
        return 1;
    }
}
