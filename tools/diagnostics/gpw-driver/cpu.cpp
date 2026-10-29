// Opt-in actual CPU GPW solves. Never registered with CTest.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/component/zero.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif
using namespace ATV;
namespace {
int output_rank = 0;
using Method = cpu::gpw::GpwDft;
using Six = std::array<double, 6>;
constexpr double kParityHa = 1e-4;
constexpr double kEnergyEv = 1e-5;
constexpr double kForceEvA = 1e-3;
constexpr double kVirialEv = 1e-4;
void require(bool value, const std::string& label) {
    if (!value)
        throw std::runtime_error(label);
}
void close(double value, double target, double tolerance,
           const std::string& label) {
    require(std::isfinite(value) && std::isfinite(target)
                && std::abs(value - target) <= tolerance,
            label + ": got " + std::to_string(value) + " target "
                + std::to_string(target));
}
struct Case {
    const char* name;
    bool spin, kpoints, plus_u;
};
constexpr Case cases[] = {
    {"gamma-rks", false, false, false}, {"gamma-uks", true, false, false},
    {"k-rks", false, true, false},      {"k-uks", true, true, false},
    {"k-rks-u", false, true, true},     {"k-uks-u", true, true, true}};
cpu::Frame hydrogen(bool spin, bool shifted = false) {
    cpu::Frame f(spin ? 3 : 2);
    constexpr double a = units::kBohrInAngstrom;
    f.box = cpu::Box::orthorhombic(12 * a, 11 * a, 10 * a);
    constexpr double xyz[3][3] = {
        {5.45, 5.15, 4.4}, {6.55, 5.65, 4.8}, {5.3, 6.7, 5.5}};
    for (int i = 0; i < f.nlocal(); ++i) {
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_mass)[i] = 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_x)[i] = xyz[i][0] * a;
        f.span(kAtom_y)[i] = xyz[i][1] * a;
        f.span(kAtom_z)[i] = xyz[i][2] * a;
        f.span(kAtom_fx)[i] = 10.25 + i;
        f.span(kAtom_fy)[i] = 20.25 + i;
        f.span(kAtom_fz)[i] = 30.25 + i;
    }
    if (shifted)
        f.span(kAtom_x)[1] += .017;
    f.meta.get(kEnergy) = 123.75;
    f.meta.get(kEnergyElectronic) = 91.25;
    f.meta.get(kStress) = {1, 2, 3, 4, 5, 6};
    return f;
}
std::vector<double> forces(const cpu::Frame& f) {
    std::vector<double> values;
    for (auto key : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < f.nlocal(); ++i)
            values.push_back(f.span(key)[i]);
    return values;
}
bool same_outputs(const cpu::Frame& a, const cpu::Frame& b) {
    return forces(a) == forces(b) && a.meta.get(kEnergy) == b.meta.get(kEnergy)
           && a.meta.get(kEnergyElectronic) == b.meta.get(kEnergyElectronic)
           && a.meta.get(kStress) == b.meta.get(kStress);
}
std::unique_ptr<Method> method(const Case& c, bool ot, double cutoff = 60) {
    auto m = std::make_unique<Method>(
        std::vector<cpu::gpw::Kind>{
            {molrs::Element::H,
             cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
             cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")}},
        cutoff, cpu::xc::XCType::PBE);
    m->n_occ = c.spin ? 0 : 1;
    m->n_alpha = c.spin ? 2 : 0;
    m->n_beta = c.spin ? 1 : 0;
    m->max_iter = 240;
    // A fixed-P strain derivative is first-order sensitive to the accepted
    // density. Match cold and warm Gamma states more tightly than a bare
    // energy comparison; the virial gate and physical fixtures stay fixed.
    m->tol = c.kpoints ? 1e-9 : 1e-11;
    if (!c.kpoints)
        m->residual_tol = 1e-7;
    m->diis_depth = 6;
    m->eps_default = 1e-10;
    m->mix = .5;
    m->use_ot = ot;
    if (c.kpoints)
        m->k_mesh[0] = 2;
    if (c.plus_u)
        m->plus_u.emplace_back(molrs::Element::H, 0, .5);
    return m;
}
struct Audit {
    cpu::Frame* frame = nullptr;
    cpu::Runtime* runtime = nullptr;
    std::vector<double> energy, residual;
    bool stage_live = false, criterion_live = false, inject = false,
         destroy_error = false;
    int builds = 0, destroys = 0, criterion_builds = 0, criterion_destroys = 0,
        polls = 0;
};
class Measure final : public cpu::Component {
    Audit& a_;

  public:
    explicit Measure(Audit& a) : a_(a) {}
    std::string name() const override { return "gpw_actual_iteration_measure"; }
    void build(const cpu::BuildContext& ctx) override {
        require(!a_.stage_live && ctx.runtime.exec.runtime,
                "actual stage build");
        a_.stage_live = true;
        a_.frame = &ctx.frame;
        a_.runtime = ctx.runtime.exec.runtime;
        a_.energy.clear();
        a_.residual.clear();
        ++a_.builds;
    }
    void launch(const cpu::LaunchContext& ctx) override {
        require(a_.stage_live && a_.criterion_live && a_.frame == &ctx.frame
                    && a_.runtime == ctx.runtime.exec.runtime,
                "actual stage execution ownership");
        const auto* m = ctx.runtime.loop.metrics;
        require(m && std::isfinite(m->energy) && std::abs(m->energy) > 1e-3
                    && std::isfinite(m->residual) && m->residual >= 0,
                "physical SCF energy/residual");
        require(a_.energy.size() == static_cast<std::size_t>(ctx.step),
                "one measurement per actual SCF budget step");
        a_.energy.push_back(m->energy);
        a_.residual.push_back(m->residual);
        if (a_.inject) {
            a_.inject = false;
            throw std::runtime_error("intentional GPW launch failure");
        }
    }
    void destroy(const cpu::DestroyContext&) override {
        require(a_.stage_live, "actual stage destroy");
        a_.stage_live = false;
        ++a_.destroys;
        if (a_.destroy_error) {
            a_.destroy_error = false;
            throw std::runtime_error("intentional GPW destroy failure");
        }
    }
};
class Guard final : public cpu::Criterion {
    Audit& a_;

  public:
    explicit Guard(Audit& a) : a_(a) {}
    std::string name() const override { return "gpw_actual_iteration_guard"; }
    void launch(const cpu::LaunchContext&) override {}
    void build(const cpu::BuildContext&) override {
        require(!a_.criterion_live, "criterion build once");
        a_.criterion_live = true;
        a_.polls = 0;
        ++a_.criterion_builds;
    }
    Verdict verdict(const cpu::LoopState& state) override {
        require(a_.criterion_live && a_.stage_live && a_.frame == &state.frame
                    && a_.runtime == state.runtime.exec.runtime,
                "criterion shares actual execution");
        require(a_.energy.size()
                    == static_cast<std::size_t>(state.iteration + 1),
                "criterion observes every measured SCF step");
        close(state.metrics.energy, a_.energy.back(), 0,
              "criterion physical energy/eV");
        close(state.metrics.residual, a_.residual.back(), 0,
              "criterion physical residual");
        ++a_.polls;
        return Verdict::Continue;
    }
    void destroy(const cpu::DestroyContext&) override {
        require(a_.criterion_live, "criterion destroy after build");
        a_.criterion_live = false;
        ++a_.criterion_destroys;
    }
};
void attach(Method& m, Measure& measure, Guard& guard) {
    m.on(gpw::Phase::check).add_ref(measure, 10);
    m.add_criterion_ref(guard);
}
void audited(const Method& m, const Audit& a) {
    require(!a.stage_live && !a.criterion_live && a.builds == a.destroys
                && a.criterion_builds == a.criterion_destroys,
            "every actual solve releases stage and criterion");
    require(a.polls == m.last_result().iters
                && a.energy.size()
                       == static_cast<std::size_t>(m.last_result().iters),
            "all real SCF steps must pass through the Driver");
}
void standalone_case(const Case& c) {
    auto f = hydrogen(c.spin);
    auto before = f;
    auto source = method(c, true);
    source->build(f);
    const auto initial = source->run(std::as_const(f));
    require(initial.converged, "pre-move OT convergence");
    require(same_outputs(f, before), "const run publication");
    auto moved = std::make_unique<Method>(std::move(*source));
    source.reset();
    close(moved->last_result().e_total, initial.e_total, 0,
          "move retains accepted result");
    Audit a;
    Measure measure(a);
    Guard guard(a);
    attach(*moved, measure, guard);
    if (!c.spin && !c.kpoints) {
        a.inject = true;
        bool caught = false;
        try {
            moved->run(std::as_const(f));
        } catch (const std::runtime_error& e) {
            caught = std::string(e.what()) == "intentional GPW launch failure";
        }
        require(caught && same_outputs(f, before) && !a.stage_live
                    && !a.criterion_live,
                "failed standalone solve unwinds and preserves Frame");
        close(moved->last_result().e_total, initial.e_total, 0,
              "failed SCF preserves accepted method result");
    }
    const auto solved = moved->forces(f);
    require(solved.converged, "OT force convergence");
    audited(*moved, a);
    auto reference = method(c, false);
    reference->build(f);
    const auto reference_result = reference->run(std::as_const(f));
    require(reference_result.converged, "DIIS comparison convergence");
    close(solved.e_total, reference_result.e_total, kParityHa,
          "OT/DIIS stationary energy/Ha");
    const auto force = forces(f);
    const auto sigma = moved->stress(f);
    audited(*moved, a);
    require(moved->last_result().converged && forces(f) == force,
            "stress uses one converged solve and preserves forces");
    for (int k = 0; k < 6; ++k)
        close(f.meta.get(kStress)[k], sigma[k] * f.box.volume(), 1e-10,
              "stress/virial publication");
    if (!c.spin && !c.kpoints) {
        auto plus = f, minus = f;
        constexpr double h = 1e-4;
        plus.span(kAtom_x)[1] += h;
        minus.span(kAtom_x)[1] -= h;
        moved->run(plus);
        require(moved->last_result().converged, "plus force FD");
        audited(*moved, a);
        moved->run(minus);
        require(moved->last_result().converged, "minus force FD");
        audited(*moved, a);
        require(std::abs(force[1]) > 1e-4, "nonzero independent ionic force");
        close(force[1],
              -(plus.meta.get(kEnergy) - minus.meta.get(kEnergy)) / (2 * h),
              kForceEvA, "OT full-SCF force derivative/eV/A");
    }
    auto shifted = hydrogen(c.spin, true);
    const auto reused = moved->run(std::as_const(shifted));
    audited(*moved, a);
    auto fresh = method(c, true);
    fresh->build(shifted);
    const auto fresh_result = fresh->run(std::as_const(shifted));
    require(reused.converged && fresh_result.converged,
            "changed geometry converges");
    close(reused.e_total, fresh_result.e_total, 1e-8,
          "changed geometry reuse/Ha");
    require(std::abs(reused.e_total - initial.e_total) > 1e-7,
            "geometry update has a physical effect");
    // DIIS must preserve the same newly mounted stage/criterion contract.
    Audit diis_a;
    Measure diis_m(diis_a);
    Guard diis_g(diis_a);
    attach(*reference, diis_m, diis_g);
    reference->run(std::as_const(shifted));
    require(reference->last_result().converged,
            "DIIS mounted stage convergence");
    audited(*reference, diis_a);
    moved.reset();
    reference.reset();
    if (output_rank == 0)
        std::printf(
            "PASS GPW_CPU_DRIVER %s E_Ha=%.15g shifted_E_Ha=%.15g "
            "iterations=%d shifted_iterations=%d OT_DIIS_dE_Ha=%.6g stages=%d "
            "residual=%.9g geometry_reuse=checked\n",
            c.name, solved.e_total, reused.e_total, solved.iters, reused.iters,
            std::abs(solved.e_total - reference_result.e_total), a.builds,
            a.residual.back());
    std::fflush(stdout);
}
class Parent final : public cpu::Driver<evaluate::Phase> {
    bool forces_, stress_;

  public:
    Parent(bool f, bool s, bool initialize = true) : forces_(f), stress_(s) {
        if (initialize) {
            on(evaluate::Phase::prepare).add(cpu::ZeroEnergy{}, -1000);
            if (f)
                on(evaluate::Phase::prepare).add(cpu::ZeroForces{}, -999);
            if (s)
                on(evaluate::Phase::prepare).add(cpu::ZeroStress{}, -998);
        }
    }
    void run(cpu::Frame& f) {
        try {
            setup_pass(f);
            EvaluationTransaction::Scope history(
                this->backend_->evaluation_transaction());
            step_once(0);
            teardown_pass();
            history.accept();
        } catch (...) {
            if (this->backend_)
                this->backend_->unwind_failed_run();
            this->mutation_lock_ = nullptr;
            throw;
        }
    }

  protected:
    bool phase_wants_forces(evaluate::Phase) const override { return forces_; }
    bool phase_wants_stress(evaluate::Phase) const override { return stress_; }
    void validate_components() const override { require_energy_or_force(); }
};
struct ParentState {
    cpu::LoopServices loop;
    cpu::Frame* frame = nullptr;
    IterationStatus status;
    ConvergenceMetrics metrics;
    int restored = 0, moves_rejected = 0, mutations_rejected = 0,
        reentries_rejected = 0;
};
class Capture final : public cpu::Component {
    ParentState& s_;
    bool stress_;

  public:
    Capture(ParentState& s, bool stress) : s_(s), stress_(stress) {}
    std::string name() const override { return "gpw_parent_capture"; }
    bool requires_system_stress() const override { return stress_; }
    void launch(const cpu::LaunchContext& ctx) override {
        s_.loop = ctx.runtime.loop;
        s_.frame = &ctx.frame;
        s_.status = *ctx.runtime.loop.status;
        s_.metrics = *ctx.runtime.loop.metrics;
    }
};
class Restored final : public cpu::Component {
    ParentState& s_;
    std::array<Method*, 2> methods_;

  public:
    Restored(ParentState& s, Method& first, Method& second)
        : s_(s), methods_{&first, &second} {}
    std::string name() const override { return "gpw_parent_restored"; }
    void launch(const cpu::LaunchContext& ctx) override {
        require(s_.frame == &ctx.frame
                    && s_.loop.status == ctx.runtime.loop.status
                    && s_.loop.metrics == ctx.runtime.loop.metrics,
                "parent Frame/loop pointer restoration");
        require(ctx.runtime.loop.status->converged == s_.status.converged
                    && ctx.runtime.loop.status->stop_requested
                           == s_.status.stop_requested
                    && ctx.runtime.loop.status->reason == s_.status.reason,
                "child cannot mutate parent convergence");
        close(ctx.runtime.loop.metrics->energy, s_.metrics.energy, 0,
              "parent physical metrics preserved");
        Method& method = *methods_[s_.restored % 2];
        const double energy = method.last_result().e_total;
        const int iterations = method.last_result().iters;
        bool rejected = false;
        try {
            Method illegal(std::move(method));
        } catch (const std::logic_error&) {
            rejected = true;
        }
        require(rejected, "moving a method bound to a live parent must reject");
        ++s_.moves_rejected;
        rejected = false;
        try {
            method.on(gpw::Phase::check).add_ref(*this);
        } catch (const std::logic_error&) {
            rejected = true;
        }
        require(rejected, "user callbacks cannot change a running child Plan");
        ++s_.mutations_rejected;
        rejected = false;
        try {
            method.run(std::as_const(ctx.frame));
        } catch (const std::logic_error&) {
            rejected = true;
        }
        require(rejected,
                "a mounted child cannot reenter standalone execution");
        ++s_.reentries_rejected;
        close(method.last_result().e_total, energy, 0,
              "rejected operations preserve accepted physical energy");
        require(method.last_result().iters == iterations,
                "rejected operations preserve the accepted SCF iteration");
        ++s_.restored;
    }
};
struct Reference {
    double energy = 0;
    std::vector<double> force;
    Six virial{};
    double residual = 0;
};
Reference standalone_sum(const Case& c, bool ot, bool shifted) {
    Reference r;
    for (double cutoff : {60., 70.}) {
        auto f = hydrogen(c.spin, shifted);
        auto m = method(c, ot, cutoff);
        m->build(f);
        Audit a;
        Measure measure(a);
        Guard guard(a);
        attach(*m, measure, guard);
        m->forces(f);
        require(m->last_result().converged,
                "nested standalone force reference");
        r.energy += f.meta.get(kEnergy);
        auto force = forces(f);
        if (r.force.empty())
            r.force.assign(force.size(), 0);
        for (std::size_t i = 0; i < force.size(); ++i)
            r.force[i] += force[i];
        m->stress(f);
        audited(*m, a);
        require(m->last_result().converged,
                "nested standalone stress reference");
        require(a.residual.back() < m->residual_tol,
                "cold derivative reference meets its declared residual");
        r.residual = std::max(r.residual, a.residual.back());
        for (int k = 0; k < 6; ++k)
            r.virial[k] += f.meta.get(kStress)[k];
        m.reset();
    }
    return r;
}
void nested_case(bool spin, bool ot) {
    Case c{spin ? "gamma-uks" : "gamma-rks", spin, false, false};
    const std::array<Reference, 2> ref = {standalone_sum(c, ot, false),
                                          standalone_sum(c, ot, true)};
    double max_energy_error = 0, max_force_error = 0, max_virial_error = 0,
           accepted_residual = 0;
    int moves_rejected = 0, mutations_rejected = 0, reentries_rejected = 0;
    for (int mode = 0; mode < 3; ++mode) {
        auto first = method(c, ot), second = method(c, ot, 70);
        Audit a, b;
        Measure measure(a), measure_second(b);
        Guard guard(a), guard_second(b);
        attach(*first, measure, guard);
        attach(*second, measure_second, guard_second);
        Parent parent(mode > 0, mode == 2);
        ParentState state;
        Capture capture(state, mode == 2);
        Restored restored(state, *first, *second);
        parent.on(evaluate::Phase::evaluate).add_ref(capture, -30);
        parent.on(evaluate::Phase::evaluate).add_ref(*first, -20);
        parent.on(evaluate::Phase::evaluate).add_ref(restored, -10);
        parent.on(evaluate::Phase::evaluate).add_ref(*second, 0);
        parent.on(evaluate::Phase::evaluate).add_ref(restored, 10);
        parent.build();
        if (!spin && !ot && mode == 0) {
            auto failed = hydrogen(false);
            a.destroy_error = true;
            bool caught = false;
            try {
                parent.run(failed);
            } catch (const std::runtime_error& e) {
                caught =
                    std::string(e.what()) == "intentional GPW destroy failure";
            }
            require(caught && !a.stage_live && !a.criterion_live,
                    "normal nested teardown exception must escape without "
                    "terminate");
        }
        for (int shifted = 0; shifted < 2; ++shifted) {
            auto f = hydrogen(spin, shifted);
            const auto fmark = forces(f);
            const auto smark = f.meta.get(kStress);
            parent.run(f);
            require(first->last_result().converged
                        && second->last_result().converged,
                    "nested physical convergence");
            audited(*first, a);
            audited(*second, b);
            require(a.residual.back() < first->residual_tol
                        && b.residual.back() < second->residual_tol,
                    "warm derivative states meet their declared residuals");
            accepted_residual = std::max(
                {accepted_residual, a.residual.back(), b.residual.back()});
            close(f.meta.get(kEnergy), ref[shifted].energy, kEnergyEv,
                  "two child energies/eV");
            max_energy_error =
                std::max(max_energy_error,
                         std::abs(f.meta.get(kEnergy) - ref[shifted].energy));
            if (mode > 0) {
                const auto got = forces(f);
                for (std::size_t i = 0; i < got.size(); ++i) {
                    close(got[i], ref[shifted].force[i], kForceEvA,
                          "two child forces/eV/A");
                    max_force_error =
                        std::max(max_force_error,
                                 std::abs(got[i] - ref[shifted].force[i]));
                }
            } else
                require(forces(f) == fmark,
                        "unrequested nuclear forces preserved");
            if (mode == 2)
                for (int k = 0; k < 6; ++k) {
                    close(f.meta.get(kStress)[k], ref[shifted].virial[k],
                          kVirialEv, "two child extensive virials/eV");
                    max_virial_error = std::max(
                        max_virial_error, std::abs(f.meta.get(kStress)[k]
                                                   - ref[shifted].virial[k]));
                }
            else
                require(f.meta.get(kStress) == smark,
                        "unrequested virial preserved");
        }
        require(state.restored == 4, "all nested loop boundaries observed");
        require(state.moves_rejected == 4 && state.mutations_rejected == 4
                    && state.reentries_rejected == 4,
                "every physical nested child retains its live binding");
        moves_rejected += state.moves_rejected;
        mutations_rejected += state.mutations_rejected;
        reentries_rejected += state.reentries_rejected;
    }
    if (output_rank == 0)
        std::printf(
            "PASS GPW_CPU_DRIVER nested-%s-%s modes=3 geometries=2 "
            "children=2 live_moves=%d live_mutations=%d live_reentries=%d "
            "dE_eV=%.9g dF_eV_A=%.9g dVirial_eV=%.9g "
            "accepted_residual=%.9g reference_residual=%.9g\n",
            ot ? "ot" : "diis", spin ? "uks" : "rks", moves_rejected,
            mutations_rejected, reentries_rejected, max_energy_error,
            max_force_error, max_virial_error, accepted_residual,
            std::max(ref[0].residual, ref[1].residual));
    std::fflush(stdout);
}
void budgets() {
    const auto c = cases[0];
    for (bool ot : {false, true})
        for (int cap : {1, 2, 4}) {
            auto f = hydrogen(false);
            auto m = method(c, ot);
            m->max_iter = cap;
            m->build(f);
            Audit a;
            Measure measure(a);
            Guard guard(a);
            attach(*m, measure, guard);
            m->run(std::as_const(f));
            audited(*m, a);
            require(m->last_result().iters <= cap,
                    "standalone hard budget includes polish");
            if (cap == 1)
                require(!m->last_result().converged,
                        "one SCF iteration cannot retain convergence");
            auto nested = method(c, ot);
            Audit b;
            Measure bm(b);
            Guard bg(b);
            attach(*nested, bm, bg);
            Parent parent(false, false, false);
            parent.on(evaluate::Phase::evaluate).add_ref(*nested, 0, cap);
            parent.build();
            const auto before = f;
            parent.run(f);
            audited(*nested, b);
            require(nested->last_result().iters <= cap
                        && nested->max_iter == 240,
                    "nested cap does not mutate max_iter");
            require(forces(f) == forces(before)
                        && f.meta.get(kStress) == before.meta.get(kStress),
                    "budgeted energy-only child respects requests");
            close(f.meta.get(kEnergy),
                  before.meta.get(kEnergy)
                      + nested->last_result().e_total * units::kHartreeInEv,
                  1e-10, "budgeted result publication");
        }
    for (int cap : {0, 1}) {
        auto f = hydrogen(false), before = f;
        auto m = method(c, true);
        Audit a;
        Measure measure(a);
        Guard guard(a);
        attach(*m, measure, guard);
        Parent parent(cap == 1, false, false);
        parent.on(evaluate::Phase::evaluate).add_ref(*m, 0, cap);
        parent.build();
        bool caught = false;
        try {
            parent.run(f);
        } catch (const std::exception&) {
            caught = true;
        }
        require(
            caught && same_outputs(f, before) && m->max_iter == 240,
            "zero/failed force cap must not publish or mutate configuration");
        if (cap == 0)
            require(a.builds == 0,
                    "zero budget must reject before actual SCF work");
        else {
            require(m->last_result().iters == 1 && !m->last_result().converged,
                    "failed derivative keeps its own exhausted solve");
            audited(*m, a);
        }
    }
    auto f = hydrogen(false);
    auto fixed = method(c, false);
    fixed->build(f);
    const auto accepted = fixed->run(std::as_const(f));
    require(accepted.converged, "warm fixed-density seed");
    fixed->max_iter = 0;
    Parent parent(false, false, false);
    parent.on(evaluate::Phase::evaluate).add_ref(*fixed);
    parent.build();
    parent.run(f);
    require(fixed->last_result().converged && fixed->last_result().iters == 0
                && fixed->max_iter == 0,
            "existing zero-SCF warm mode preserved");
    if (output_rank == 0)
        std::printf("PASS GPW_CPU_DRIVER budgets standalone_nested=1,2,4 "
                    "zero_rejection=checked warm_zero=checked "
                    "derivative_failure=checked\n");
    std::fflush(stdout);
}
} // namespace
int main(int argc, char** argv) {
#ifdef ATV_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    struct Finalize {
        bool own;
        ~Finalize() {
            if (own)
                MPI_Finalize();
        }
    } finalize{!initialized};
    if (!initialized)
        MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &output_rank);
#endif
    bool kpoints_only = false;
    if (argc == 2 && std::string(argv[1]) == "--kpoints-only")
        kpoints_only = true;
    else if (argc != 1)
        return 2;
    int failed = 0;
    for (const auto& c : cases)
        try {
            if (kpoints_only && !c.kpoints)
                continue;
            standalone_case(c);
        } catch (const std::exception& e) {
            ++failed;
            if (output_rank == 0)
                std::fprintf(stderr, "FAIL GPW_CPU_DRIVER %s: %s\n", c.name,
                             e.what());
        }
    if (!kpoints_only) {
        try {
            budgets();
        } catch (const std::exception& e) {
            ++failed;
            if (output_rank == 0)
                std::fprintf(stderr, "FAIL GPW_CPU_DRIVER budgets: %s\n",
                             e.what());
        }
        for (bool ot : {false, true})
            for (bool spin : {false, true})
                try {
                    nested_case(spin, ot);
                } catch (const std::exception& e) {
                    ++failed;
                    if (output_rank == 0)
                        std::fprintf(
                            stderr, "FAIL GPW_CPU_DRIVER nested-%s-%s: %s\n",
                            ot ? "ot" : "diis", spin ? "uks" : "rks", e.what());
                }
    }
#ifdef ATV_HAS_MPI
    int global_failed = failed;
    MPI_Allreduce(&failed, &global_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    failed = global_failed;
#endif
    return failed ? 1 : 0;
}
