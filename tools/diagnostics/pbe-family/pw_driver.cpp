// Complete physical CUDA PW nesting checks. Opt-in executable, never CTest.
// Standalone public methods supply the integration references; this diagnostic
// qualifies dispatch, requests and accumulation, not an external science pin.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/planewave/pw_solver.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ATV;
namespace {
using Method = cuda::pw::PW;
using Six = std::array<double, 6>;
constexpr double kEnergyToleranceEv = 1e-8;
constexpr double kForceToleranceEvA = 2e-8;
constexpr double kVirialToleranceEv = 1e-7;
constexpr double kCutoff[2] = {6.0, 7.0};

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}
void close(double got, double expected, double tolerance,
           const std::string& message) {
    require(std::isfinite(got) && std::isfinite(expected)
                && std::abs(got - expected) <= tolerance,
            message + ": got " + std::to_string(got) + ", expected "
                + std::to_string(expected));
}
std::vector<double> forces(const cpu::Frame& frame) {
    std::vector<double> result;
    for (auto column : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < frame.nlocal(); ++i)
            result.push_back(frame.span(column)[i]);
    return result;
}
void mark_outputs(cpu::Frame& frame) {
    int marker = 0;
    for (auto column : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < frame.nlocal(); ++i)
            frame.span(column)[i] = 10.25 + marker++;
    frame.meta.get(kEnergy) = 31415.125;
    frame.meta.get(kStress) = {1, 2, 3, 4, 5, 6};
}
cpu::Frame hydrogen(bool spin, bool thermal, bool shifted) {
    cpu::Frame frame(spin || thermal ? 3 : 2);
    Mat3 cell;
    cell(0, 0) = 4.2;
    cell(0, 1) = .3;
    cell(0, 2) = .2;
    cell(1, 0) = 0;
    cell(1, 1) = 4;
    cell(1, 2) = .1;
    cell(2, 0) = cell(2, 1) = 0;
    cell(2, 2) = 3.8;
    frame.box.set_cell(cell);
    constexpr double position[3][3] = {
        {1.2, 1.1, 1.0}, {2.1, 1.6, 1.3}, {1.6, 2.25, 1.8}};
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_x)[i] = position[i][0];
        frame.span(kAtom_y)[i] = position[i][1];
        frame.span(kAtom_z)[i] = position[i][2];
    }
    if (shifted)
        frame.span(kAtom_x)[1] += .017;
    mark_outputs(frame);
    return frame;
}
std::unique_ptr<Method> method(bool spin, bool thermal, int index) {
    auto potential = std::make_shared<cpu::pw::GthPseudopotential>(
        molrs::Element::H, "GTH-PBE-q1");
    // Exercise an actual move before mounting, then destroy the source. The
    // subsequent physical SCFs must use valid stage/criterion references.
    auto source = std::make_unique<Method>(potential, kCutoff[index], 1,
                                           cpu::xc::XCType::PBE);
    source->spin_polarized = spin;
    source->multiplicity = spin ? 2 : 1;
    source->smearing = thermal ? electronic::Smearing::fermi_dirac
                               : electronic::Smearing::none;
    source->degauss = .02;
    source->extra_bands = 3;
    source->tol = 1e-9;
    source->max_iter = 400;
    source->mix_beta = .3;
    auto moved = std::make_unique<Method>(std::move(*source));
    source.reset();
    return moved;
}
// Audit actual SCF measurements after installing stages/criteria on a moved
// standalone method. A pending binding that never reaches the Plan cannot
// supply the measured iteration sequence, and a criterion without lifecycle
// preparation rejects its first poll rather than passing a count-only check.
struct ScienceAudit {
    cuda::Frame* frame = nullptr;
    void* stream = nullptr;
    std::vector<double> energy_ev, residual;
    int stage_builds = 0, stage_destroys = 0;
    int criterion_builds = 0, criterion_destroys = 0, polls = 0;
    bool stage_live = false, criterion_live = false;
    bool inject_once = false, injected = false;
};
class ScienceStage final : public cuda::Component {
    ScienceAudit& audit_;
    Method& method_;

  public:
    ScienceStage(ScienceAudit& audit, Method& method)
        : audit_(audit), method_(method) {}
    std::string name() const override { return "pw_standalone_science_stage"; }
    void build(const cuda::BuildContext& context) override {
        require(!audit_.stage_live && context.frame.n_atoms() > 0
                    && context.runtime.exec.backend && context.runtime.stream,
                "standalone stage build needs the real physical execution");
        audit_.stage_live = true;
        audit_.frame = &context.frame;
        audit_.stream = context.runtime.stream;
        audit_.energy_ev.clear();
        audit_.residual.clear();
        audit_.injected = false;
        ++audit_.stage_builds;
    }
    void launch(const cuda::LaunchContext& context) override {
        require(audit_.stage_live && audit_.criterion_live
                    && &context.frame == audit_.frame
                    && context.runtime.stream == audit_.stream
                    && context.runtime.loop.metrics,
                "standalone stage must run inside its prepared SCF execution");
        const auto& result = method_.last_result();
        require(result.iterations == context.step + 1,
                "measurement must belong to the current physical iteration");
        const double energy = result.energy_ha * units::kHartreeInEv;
        const double residual = context.runtime.loop.metrics->residual;
        require(std::isfinite(energy) && std::abs(energy) > 1e-4
                    && std::isfinite(residual) && residual >= 0,
                "stage must observe finite nontrivial SCF energy and density "
                "residual");
        audit_.energy_ev.push_back(energy);
        audit_.residual.push_back(residual);
        if (audit_.inject_once) {
            audit_.inject_once = false;
            audit_.injected = true;
            throw std::runtime_error("intentional standalone PW stage failure");
        }
    }
    void destroy(const cuda::DestroyContext& context) override {
        require(audit_.stage_live && &context.frame == audit_.frame
                    && context.runtime.stream == audit_.stream,
                "standalone stage must be destroyed on its original execution");
        audit_.stage_live = false;
        ++audit_.stage_destroys;
    }
};
class ScienceCriterion final : public cuda::Criterion {
    ScienceAudit& audit_;

  public:
    explicit ScienceCriterion(ScienceAudit& audit) : audit_(audit) {}
    std::string name() const override { return "pw_standalone_science_guard"; }
    void build(const cuda::BuildContext& context) override {
        require(!audit_.criterion_live && context.frame.n_atoms() > 0
                    && context.runtime.exec.backend && context.runtime.stream,
                "standalone criterion must receive its real build lifecycle");
        audit_.criterion_live = true;
        audit_.polls = 0;
        ++audit_.criterion_builds;
    }
    Verdict verdict(const cuda::LoopState& state) override {
        require(audit_.criterion_live && audit_.stage_live
                    && &state.frame == audit_.frame
                    && state.runtime.stream == audit_.stream,
                "a pending criterion must be built before polling");
        require(audit_.energy_ev.size()
                        == static_cast<std::size_t>(state.iteration + 1)
                    && audit_.residual.size() == audit_.energy_ev.size(),
                "criterion needs the actual stage measurement from every "
                "iteration");
        close(state.metrics.energy, audit_.energy_ev.back(), 1e-12,
              "criterion energy must match the physical SCF stage/eV");
        close(state.metrics.residual, audit_.residual.back(), 0,
              "criterion density residual must match the physical SCF stage");
        ++audit_.polls;
        // A stop-only scientific guard supplements, and never replaces, the
        // method's residual convergence policy.
        return Verdict::Continue;
    }
    void destroy(const cuda::DestroyContext& context) override {
        require(
            audit_.criterion_live && &context.frame == audit_.frame
                && context.runtime.stream == audit_.stream,
            "standalone criterion must receive its matching destroy lifecycle");
        require(audit_.injected
                    ? audit_.polls == 0
                    : audit_.polls == static_cast<int>(audit_.energy_ev.size()),
                "criterion must judge all completed physical iterations");
        audit_.criterion_live = false;
        ++audit_.criterion_destroys;
    }
};
void check_science_audit(const ScienceAudit& audit,
                         const cuda::pw::PWResult& result, double tolerance,
                         int expected_passes) {
    require(
        !audit.stage_live && !audit.criterion_live
            && audit.stage_builds == expected_passes
            && audit.stage_destroys == expected_passes
            && audit.criterion_builds == expected_passes
            && audit.criterion_destroys == expected_passes,
        "every standalone solve must prepare and release stage and criterion");
    require(
        result.converged && audit.polls == result.iterations
            && audit.energy_ev.size()
                   == static_cast<std::size_t>(result.iterations)
            && !audit.residual.empty(),
        "standalone lifecycle must cover the real converged SCF trajectory");
    close(audit.energy_ev.back(), result.energy_ha * units::kHartreeInEv, 1e-12,
          "last measured energy must be the accepted SCF energy/eV");
    require(
        audit.residual.front() > 1e-12 && audit.residual.back() < tolerance,
        "observer must track a genuine nonzero-to-converged density residual");
}
struct Reference {
    std::array<cuda::pw::PWResult, 2> child;
    double energy_ev = 0;
    std::vector<double> force;
    Six virial{};
};
Reference standalone(bool spin, bool thermal, bool shifted) {
    Reference reference;
    for (int index = 0; index < 2; ++index) {
        auto frame = hydrogen(spin, thermal, shifted);
        auto solver = method(spin, thermal, index);
        ScienceAudit audit;
        ScienceStage science(audit, *solver);
        ScienceCriterion criterion(audit);
        // Both bindings are added after construction AND an actual move.
        solver->on(pw::Phase::check).add_ref(science);
        solver->add_criterion_ref(criterion);
        int completed_passes = 0;
        if (!spin && !thermal && !shifted && index == 0) {
            audit.inject_once = true;
            const auto old_force = forces(frame);
            const auto old_virial = frame.meta.get(kStress);
            const double old_energy = frame.meta.get(kEnergy);
            bool failed = false;
            try {
                solver->forces(frame);
            } catch (const std::runtime_error& failure) {
                failed = std::string(failure.what())
                         == "intentional standalone PW stage failure";
            }
            require(
                failed && audit.injected && !audit.stage_live
                    && !audit.criterion_live && audit.stage_builds == 1
                    && audit.stage_destroys == 1 && audit.criterion_builds == 1
                    && audit.criterion_destroys == 1,
                "failed standalone stage must unwind both physical lifecycles");
            require(frame.meta.get(kEnergy) == old_energy
                        && frame.meta.get(kStress) == old_virial
                        && forces(frame) == old_force,
                    "failed standalone solve must preserve caller outputs");
            ++completed_passes;
        }
        reference.child[index] = solver->forces(frame);
        check_science_audit(audit, reference.child[index], solver->tol,
                            ++completed_passes);
        require(reference.child[index].converged,
                "standalone force reference convergence");
        const auto force = forces(frame);
        reference.energy_ev += frame.meta.get(kEnergy);
        if (reference.force.empty())
            reference.force.assign(force.size(), 0.0);
        for (std::size_t i = 0; i < force.size(); ++i)
            reference.force[i] += force[i];
        const auto stress = solver->stress(frame);
        require(solver->last_result().converged,
                "standalone stress reference convergence");
        check_science_audit(audit, solver->last_result(), solver->tol,
                            ++completed_passes);
        close(solver->last_result().energy_ha, reference.child[index].energy_ha,
              1e-9, "force/stress reference SCF identity/Ha");
        require(forces(frame) == force, "standalone stress preserves forces");
        for (int k = 0; k < 6; ++k) {
            close(frame.meta.get(kStress)[k], stress[k] * frame.box.volume(),
                  1e-10, "standalone virial conversion");
            reference.virial[k] += frame.meta.get(kStress)[k];
        }
        if (thermal) {
            close(reference.child[index].n_elec_occ, frame.nlocal(), 1e-8,
                  "reference thermal electron count");
            require(std::abs(reference.child[index].e_entropy) > 1e-8,
                    "reference must exercise the thermal correction");
        }
        // Borrowed diagnostic stages outlive the method's final teardown.
        solver.reset();
    }
    const auto peak = [](const auto& values) {
        double result = 0;
        for (double value : values)
            result = std::max(result, std::abs(value));
        return result;
    };
    require(peak(reference.force) > 1e-3, "force fixture must be nonzero");
    require(peak(reference.virial) > 1e-3, "virial fixture must be nonzero");
    return reference;
}
struct Probe {
    cuda::LoopServices loop;
    cuda::ExecServices exec;
    cuda::Frame* frame = nullptr;
    void* stream = nullptr;
    IterationStatus status;
    ConvergenceMetrics metrics;
    const Reference* reference = nullptr;
    int captures = 0, after_first = 0, after_second = 0;
    int moves_rejected = 0, teardown_restored = 0;
};
void restored(Probe& probe, cuda::Context& context, cuda::Frame& frame) {
    require(context.loop.status == probe.loop.status
                && context.loop.metrics == probe.loop.metrics,
            "child must restore the parent's loop service pointers");
    require(&frame == probe.frame && context.stream == probe.stream
                && context.exec.backend == probe.exec.backend,
            "child must retain the parent's Frame/stream/Backend");
    require(context.exec.forces_requested == probe.exec.forces_requested
                && context.exec.stress_requested == probe.exec.stress_requested
                && context.exec.trial_evaluation == probe.exec.trial_evaluation
                && context.exec.initializing_forces
                       == probe.exec.initializing_forces,
            "child must restore evaluation requests");
    require(context.loop.status->converged == probe.status.converged
                && context.loop.status->stop_requested
                       == probe.status.stop_requested
                && context.loop.status->reason == probe.status.reason,
            "child convergence must not mutate parent status");
    close(context.loop.metrics->energy, probe.metrics.energy, 0,
          "parent energy measurement must be preserved");
    close(context.loop.metrics->residual, probe.metrics.residual, 0,
          "parent residual measurement must be preserved");
}
class ParentCapture final : public cuda::Component {
    Probe& probe_;

  public:
    explicit ParentCapture(Probe& probe) : probe_(probe) {}
    std::string name() const override { return "pw_parent_capture"; }
    void launch(const cuda::LaunchContext& context) override {
        auto& runtime = context.runtime;
        require(runtime.loop.status && runtime.loop.metrics
                    && runtime.exec.backend && runtime.stream,
                "parent must have real execution services");
        probe_.loop = runtime.loop;
        probe_.exec = runtime.exec;
        probe_.status = *runtime.loop.status;
        probe_.metrics = *runtime.loop.metrics;
        probe_.frame = &context.frame;
        probe_.stream = runtime.stream;
        ++probe_.captures;
    }
    void destroy(const cuda::DestroyContext& context) override {
        // This runs before the owning Backend clears its Context. It checks
        // restoration on normal teardown and on an injected SCF exception.
        if (probe_.captures) {
            restored(probe_, context.runtime, context.frame);
            ++probe_.teardown_restored;
        }
    }
};
class AfterChild final : public cuda::Component {
    Probe& probe_;
    Method& child_;
    int index_;

  public:
    AfterChild(Probe& probe, Method& child, int index)
        : probe_(probe), child_(child), index_(index) {}
    std::string name() const override {
        return index_ == 0 ? "after_first_pw" : "after_second_pw";
    }
    void launch(const cuda::LaunchContext& context) override {
        restored(probe_, context.runtime, context.frame);
        require(probe_.reference, "standalone reference must be selected");
        const auto& expected = probe_.reference->child[index_];
        const auto& got = child_.last_result();
        require(got.converged && child_.converged() && got.iterations > 0,
                "nested child must converge through its own Driver");
        close(got.energy_ha, expected.energy_ha, 1e-10,
              "nested child's own reference energy/Ha");
        require(got.n_kpts == expected.n_kpts
                    && got.n_kpts_local == expected.n_kpts_local,
                "nested child k-point metadata");
        close(got.n_elec_occ, expected.n_elec_occ, 1e-8,
              "nested occupation metadata");
        close(got.e_entropy, expected.e_entropy, 1e-9,
              "nested thermal correction/Ry");
        const double energy_before_move = got.energy_ha;
        bool rejected = false;
        try {
            Method illegal_move(std::move(child_));
        } catch (const std::logic_error&) {
            rejected = true;
        }
        require(rejected, "moving a method bound to a live parent must reject");
        close(child_.last_result().energy_ha, energy_before_move, 0,
              "rejected move preserves the physical result");
        ++probe_.moves_rejected;
        ++(index_ == 0 ? probe_.after_first : probe_.after_second);
    }
};
class ThrowOnce final : public cuda::Component {
  public:
    bool armed = true;
    std::string name() const override { return "pw_injected_failure"; }
    void launch(const cuda::LaunchContext&) override {
        if (armed) {
            armed = false;
            throw std::runtime_error("intentional nested PW stage failure");
        }
    }
};
struct ErrorEnvelope {
    double energy = 0, force = 0, virial = 0;
};
void check_outputs(const cpu::Frame& frame, const Reference& expected,
                   const std::vector<double>& force_marker,
                   const Six& virial_marker, bool force_requested,
                   bool stress_requested, ErrorEnvelope& error) {
    close(frame.meta.get(kEnergy), expected.energy_ev, kEnergyToleranceEv,
          "two child energies must accumulate exactly once");
    error.energy = std::max(
        error.energy, std::abs(frame.meta.get(kEnergy) - expected.energy_ev));
    const auto got = forces(frame);
    if (!force_requested)
        require(got == force_marker, "energy-only evaluation preserves forces");
    else
        for (std::size_t i = 0; i < got.size(); ++i) {
            close(got[i], expected.force[i], kForceToleranceEvA,
                  "two child fixed-point forces must accumulate");
            error.force =
                std::max(error.force, std::abs(got[i] - expected.force[i]));
        }
    if (!stress_requested)
        require(frame.meta.get(kStress) == virial_marker,
                "unrequested virial must be preserved");
    else
        for (int k = 0; k < 6; ++k) {
            close(frame.meta.get(kStress)[k], expected.virial[k],
                  kVirialToleranceEv,
                  "two child extensive virials must accumulate");
            error.virial =
                std::max(error.virial, std::abs(frame.meta.get(kStress)[k]
                                                - expected.virial[k]));
        }
}
void run_case(bool spin, bool thermal) {
    const std::array<Reference, 2> reference = {
        standalone(spin, thermal, false), standalone(spin, thermal, true)};
    require(std::abs(reference[0].energy_ev - reference[1].energy_ev) > 1e-6,
            "geometry update must change the physical energy");
    ErrorEnvelope error;
    int move_checks = 0, teardown_checks = 0;
    for (int request = 0; request < 3; ++request) {
        const bool with_force = request > 0, with_stress = request == 2;
        auto first = method(spin, thermal, 0);
        auto second = method(spin, thermal, 1);
        Probe probe;
        ParentCapture capture(probe);
        AfterChild after_first(probe, *first, 0),
            after_second(probe, *second, 1);
        ThrowOnce thrower;
        const bool inject_failure = !spin && !thermal && request == 0;
        if (inject_failure)
            first->on(pw::Phase::check).add_ref(thrower);
        cuda::Evaluate parent(with_force, with_stress);
        parent.on(evaluate::Phase::evaluate).add_ref(capture, -10);
        parent.on(evaluate::Phase::evaluate).add_ref(*first, 0);
        parent.on(evaluate::Phase::evaluate).add_ref(after_first, 10);
        parent.on(evaluate::Phase::evaluate).add_ref(*second, 20);
        parent.on(evaluate::Phase::evaluate).add_ref(after_second, 30);
        parent.build();
        auto frame = hydrogen(spin, thermal, false);
        if (inject_failure) {
            probe.reference = &reference[0];
            const auto f = forces(frame);
            const auto v = frame.meta.get(kStress);
            const double e = frame.meta.get(kEnergy);
            bool failed = false;
            try {
                parent.run(frame);
            } catch (const std::runtime_error& failure) {
                failed = std::string(failure.what())
                         == "intentional nested PW stage failure";
            }
            require(failed && !thrower.armed,
                    "intentional stage failure must escape");
            require(
                frame.meta.get(kEnergy) == e && forces(frame) == f
                    && frame.meta.get(kStress) == v,
                "failed parent evaluation must not publish partial outputs");
            require(probe.teardown_restored == 1,
                    "parent loop must be restored during failure cleanup");
        }
        for (int shifted = 0; shifted < 2; ++shifted) {
            // Reuse both method instances and the same outer Driver, applying
            // the caller's geometry edit between full materializations.
            frame = hydrogen(spin, thermal, shifted != 0);
            probe.reference = &reference[shifted];
            const auto f = forces(frame);
            const auto v = frame.meta.get(kStress);
            parent.run(frame);
            check_outputs(frame, reference[shifted], f, v, with_force,
                          with_stress, error);
        }
        require(probe.after_first == 2 && probe.after_second == 2
                    && probe.moves_rejected == 4,
                "every nested child and rejected move must be observed twice");
        require(probe.teardown_restored == (inject_failure ? 3 : 2),
                "every parent teardown must restore its loop");
        move_checks += probe.moves_rejected;
        teardown_checks += probe.teardown_restored;
    }
    std::printf("PASS PW_DRIVER %s %s modes=3 geometries=2 children=2 "
                "move_rejections=%d loop_teardowns=%d "
                "dE_eV=%.6g dF_eV_A=%.6g dVirial_eV=%.6g\n",
                thermal ? "fd" : "fixed", spin ? "UKS" : "RKS", move_checks,
                teardown_checks, error.energy, error.force, error.virial);
    std::fflush(stdout);
}
} // namespace
int main() {
    try {
        for (const bool thermal : {false, true})
            for (const bool spin : {false, true})
                run_case(spin, thermal);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL PW_DRIVER %s\n", error.what());
        return 1;
    }
}
