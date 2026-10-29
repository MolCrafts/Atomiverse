// Complete public/nested fixed-LSDA vertical. Opt-in, never CTest.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "general_trace.h"
#include "k8_pins.h"
#include "qe-2026-10-28/pins.h"
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif
#include <array>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ATV;
using Method = cpu::pw::PW;
using Result = cpu::pw::PWResult;
using Six = std::array<double, 6>;
using Forces = std::array<std::array<double, 3>, 2>;
struct Input {
    int mesh = 1;
    bool rotate = false, mpi = false;
} input;
double reference_energy() {
    return input.mesh == 1 ? pw_analytic_qe::energy_ha
                           : pw_analytic_k8_qe::energy_ha;
}
Forces reference_forces() {
    auto f = input.mesh == 1 ? pw_analytic_qe::forces_ha_bohr
                             : pw_analytic_k8_qe::forces_ha_bohr;
    if (input.rotate)
        for (auto& row : f)
            row = {-row[1], row[0], row[2]};
    return f;
}
Six reference_stress() {
    auto s = input.mesh == 1 ? pw_analytic_qe::stress_ev_ang3
                             : pw_analytic_k8_qe::stress_ev_ang3;
    if (input.rotate)
        s = {s[1], s[0], s[2], -s[3], -s[5], s[4]};
    return s;
}

constexpr double energy_gate_ha = 1e-5, force_gate_ha_bohr = 1e-4,
                 stress_gate_ev_ang3 = 2e-3;
void require(bool value, const std::string& label) {
    if (!value)
        throw std::runtime_error(label);
}
void close(double a, double b, double tolerance, const std::string& label) {
    if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > tolerance) {
        std::ostringstream message;
        message << std::setprecision(17) << label << ": got " << a
                << " expected " << b << " gate " << tolerance;
        throw std::runtime_error(message.str());
    }
}
Forces forces(const cpu::Frame& f) {
    Forces values{};
    constexpr auto columns = std::array{kAtom_fx, kAtom_fy, kAtom_fz};
    for (int i = 0; i < 2; ++i)
        for (int d = 0; d < 3; ++d)
            values[i][d] = f.span(columns[d])[i];
    return values;
}
cpu::Frame silicon() {
    cpu::Frame f(2);
    Mat3 h;
    constexpr double columns[3][3] = {
        {.10, 2.70, 2.65}, {2.75, .05, 2.70}, {2.65, 2.75, .15}};
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            h(r, c) = columns[c][r];
    f.box.set_cell(h);
    constexpr double p[2][3] = {{.2, .3, .4}, {1.57, 1.66, 1.75}};
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 14;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = 28.0855;
        f.span(kAtom_x)[i] = p[i][0];
        f.span(kAtom_y)[i] = p[i][1];
        f.span(kAtom_z)[i] = p[i][2];
        f.span(kAtom_fx)[i] = 10 + i;
        f.span(kAtom_fy)[i] = 20 + i;
        f.span(kAtom_fz)[i] = 30 + i;
    }
    f.meta.get(kEnergy) = 1234;
    f.meta.get(kStress) = {1, 2, 3, 4, 5, 6};
    if (input.rotate) {
        Mat3 q;
        q(0, 1) = -1;
        q(1, 0) = 1;
        q(2, 2) = 1;
        f.box.set_cell(q * f.box.cell());
        for (int i = 0; i < 2; ++i) {
            const auto r = q
                           * Vec3{f.span(kAtom_x)[i], f.span(kAtom_y)[i],
                                  f.span(kAtom_z)[i]};
            f.span(kAtom_x)[i] = r[0];
            f.span(kAtom_y)[i] = r[1];
            f.span(kAtom_z)[i] = r[2];
        }
    }
    return f;
}
void geometry(const cpu::Frame& a, const cpu::Frame& b) {
    require(a.nlocal() == b.nlocal(), "SCF execution atom count");
    for (int i = 0; i < 2; ++i) {
        require(a.span(kAtom_type)[i] == b.span(kAtom_type)[i], "SCF species");
        for (auto col : {kAtom_x, kAtom_y, kAtom_z})
            close(a.span(col)[i], b.span(col)[i], 0, "SCF geometry snapshot");
    }
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            close(a.box.cell()(r, c), b.box.cell()(r, c), 0,
                  "SCF cell snapshot");
}
std::unique_ptr<Method> method() {
    trace::reset();
    auto m =
        std::make_unique<Method>(std::make_shared<cpu::pw::GthPseudopotential>(
                                     molrs::Element::Si, "GTH-PBE-q4"),
                                 12, input.mesh, cpu::xc::XCType::LSDA);
    m->tol = 1e-10;
    m->max_iter = 200;
    m->mix_beta = .2;
    return m;
}
struct Audit {
    const cpu::Frame* expected_geometry = nullptr;
    const cpu::Frame* actual = nullptr;
    cpu::Frame* parent_frame = nullptr;
    cpu::Runtime* runtime = nullptr;
    bool stage_live = false, guard_live = false;
    int builds = 0, destroys = 0, guard_builds = 0, guard_destroys = 0,
        polls = 0;
    std::vector<double> energies, residuals, energy_changes;
};
class Measure final : public cpu::Component {
    Audit& a_;
    Method& m_;

  public:
    Measure(Audit& a, Method& m) : a_(a), m_(m) {}
    std::string name() const override {
        return "pw_analytic_actual_scf_measure";
    }
    void build(const cpu::BuildContext& c) override {
        require(!a_.stage_live && a_.expected_geometry
                    && c.runtime.exec.runtime,
                "SCF stage build");
        geometry(c.frame, *a_.expected_geometry);
        if (a_.parent_frame)
            require(&c.frame == a_.parent_frame,
                    "nested SCF uses actual parent Frame");
        a_.stage_live = true;
        a_.actual = &c.frame;
        a_.runtime = c.runtime.exec.runtime;
        ++a_.builds;
    }
    void launch(const cpu::LaunchContext& c) override {
        require(a_.stage_live && a_.guard_live && a_.actual == &c.frame
                    && a_.runtime == c.runtime.exec.runtime,
                "SCF stage execution ownership");
        geometry(c.frame, *a_.expected_geometry);
        const auto* v = c.runtime.loop.metrics;
        const auto& r = m_.last_result();
        require(v && r.iterations == c.step + 1
                    && a_.energies.size() == std::size_t(c.step),
                "one collector sample per actual SCF iteration");
        require(std::isfinite(v->energy) && std::abs(v->energy) > 1e-3
                    && std::isfinite(v->residual) && v->residual >= 0,
                "actual finite SCF metrics");
        close(v->energy, r.energy_ha * units::kHartreeInEv, 1e-11,
              "measured SCF energy/eV");
        if (a_.energies.empty())
            require(std::isinf(v->energy_change) && v->energy_change > 0,
                    "first SCF delta energy is explicitly unavailable");
        else
            close(v->energy_change, std::abs(v->energy - a_.energies.back()), 0,
                  "actual SCF delta energy");
        a_.energy_changes.push_back(v->energy_change);
        a_.energies.push_back(v->energy);
        a_.residuals.push_back(v->residual);
    }
    void destroy(const cpu::DestroyContext& c) override {
        require(a_.stage_live && a_.actual == &c.frame
                    && a_.runtime == c.runtime.exec.runtime,
                "SCF stage destroy");
        a_.stage_live = false;
        ++a_.destroys;
    }
};
class Guard final : public cpu::Criterion {
    Audit& a_;

  public:
    explicit Guard(Audit& a) : a_(a) {}
    std::string name() const override { return "pw_analytic_actual_scf_guard"; }
    void launch(const cpu::LaunchContext&) override {}
    void build(const cpu::BuildContext&) override {
        require(!a_.guard_live, "guard build");
        a_.guard_live = true;
        ++a_.guard_builds;
    }
    Verdict verdict(const cpu::LoopState& s) override {
        require(a_.guard_live && a_.stage_live && a_.actual == &s.frame
                    && a_.runtime == s.runtime.exec.runtime,
                "guard uses same SCF Frame/runtime");
        require(a_.energies.size() == std::size_t(s.iteration + 1),
                "guard observes each measured iteration");
        close(s.metrics.energy, a_.energies.back(), 0, "guard energy");
        close(s.metrics.residual, a_.residuals.back(), 0, "guard residual");
        if (std::isinf(a_.energy_changes.back()))
            require(std::isinf(s.metrics.energy_change),
                    "guard first delta energy");
        else
            close(s.metrics.energy_change, a_.energy_changes.back(), 0,
                  "guard delta energy");
        ++a_.polls;
        return Verdict::Continue; // Does not replace convergence policy.
    }
    void destroy(const cpu::DestroyContext&) override {
        require(a_.guard_live, "guard destroy");
        a_.guard_live = false;
        ++a_.guard_destroys;
    }
};
void complete(const char* label, const Audit& a, const Result& r, double tol) {
    require(!a.stage_live && !a.guard_live && a.builds == 1 && a.destroys == 1
                && a.guard_builds == 1 && a.guard_destroys == 1,
            "exactly one completed SCF lifecycle");
    require(r.converged && r.iterations > 2 && a.polls == r.iterations
                && a.energies.size() == std::size_t(r.iterations)
                && a.residuals.size() == a.energies.size()
                && a.energy_changes.size() == a.energies.size(),
            "complete collector matches returned iterations");
    require(a.residuals.front() > tol && a.residuals.back() < tol,
            "real nonzero-to-converged residual");
    close(a.energies.back(), r.energy_ha * units::kHartreeInEv, 1e-11,
          "accepted measured energy");
    const bool derivative = std::string(label) != "public-run";
    const bool stress = derivative && std::string(label) != "public-forces";
    require(trace::counts.extract == int(derivative)
                && trace::counts.assembly == int(stress)
                && trace::counts.legacy_stress == 0
                && trace::counts.davidson
                       == r.n_kpts_local * (r.iterations + int(derivative)),
            "exact local SCF/final sweep budget, no strained solves");
    std::printf("BAND_AUDIT %s local_k=%d global_k=%d Davidson=%d extract=%d "
                "assembly=%d legacy=%d\n",
                label, r.n_kpts_local, r.n_kpts, trace::counts.davidson,
                trace::counts.extract, trace::counts.assembly,
                trace::counts.legacy_stress);
    std::printf("SCF_AUDIT %s solves=1 builds=1 destroys=1 guard_builds=1 "
                "guard_destroys=1 iterations=%d polls=%d residual_first=%.16g "
                "residual_last=%.16g\n",
                label, r.iterations, a.polls, a.residuals.front(),
                a.residuals.back());
}
void metadata(const Result& a, const Result& b) {
    require(a.converged == b.converged && a.iterations == b.iterations
                && a.n_kpts == b.n_kpts && a.n_kpts_local == b.n_kpts_local,
            "reference verdict/iteration/k metadata preserved");
    close(a.energy, b.energy, 1e-10, "reference energy/Ry");
    close(a.energy_ha, b.energy_ha, 1e-10, "reference energy/Ha");
    close(a.magnetization, b.magnetization, 1e-12, "reference magnetization");
    close(a.e_fermi, b.e_fermi, 1e-12, "reference mu");
    close(a.n_elec_occ, b.n_elec_occ, 1e-12, "reference occupation metadata");
    close(a.e_entropy, b.e_entropy, 1e-12, "reference entropy");
}
void qe_energy(const Result& r) {
    close(r.energy_ha, reference_energy(), energy_gate_ha,
          "actual QE VWN energy");
}
void qe_forces(const Forces& f) {
    const double conv = units::convert(1., units::Force::HartreePerBohr,
                                       units::Force::eVPerAngstrom);
    for (int i = 0; i < 2; ++i)
        for (int d = 0; d < 3; ++d)
            close(f[i][d] / conv, reference_forces()[i][d], force_gate_ha_bohr,
                  "actual QE each force/HaBohr");
}
void qe_stress(const Six& s) {
    for (int i = 0; i < 6; ++i)
        close(s[i], reference_stress()[i], stress_gate_ev_ang3,
              "actual QE each stress/eVA3");
}
struct Reference {
    Result result;
    Forces force;
    Six sigma;
};
Reference standalone() {
    Reference ref;
    for (int request = 0; request < 3; ++request) {
        auto f = silicon();
        const auto original_forces = forces(f);
        const auto original_stress = f.meta.get(kStress);
        auto m = method();
        Audit a;
        a.expected_geometry = &f;
        Measure stage(a, *m);
        Guard guard(a);
        m->on(pw::Phase::check).add_ref(stage, 10);
        m->add_criterion_ref(guard);
        m->build(f);
        Result result;
        if (request == 0)
            result = m->run(f);
        else if (request == 1) {
            result = m->forces(f);
            ref.force = forces(f);
            qe_forces(ref.force);
            const double conversion = units::convert(
                1., units::Force::HartreePerBohr, units::Force::eVPerAngstrom);
            for (int i = 0; i < 2; ++i)
                std::printf("PUBLIC_FORCE_HaBohr atom=%d %.17g %.17g %.17g\n",
                            i, ref.force[i][0] / conversion,
                            ref.force[i][1] / conversion,
                            ref.force[i][2] / conversion);
        } else {
            ref.sigma = m->stress(f);
            result = m->last_result();
            qe_stress(ref.sigma);
            std::printf("PUBLIC_STRESS_eVA3");
            for (auto value : ref.sigma)
                std::printf(" %.17g", value);
            std::printf("\n");
            for (int v = 0; v < 6; ++v)
                close(f.meta.get(kStress)[v], ref.sigma[v] * f.box.volume(),
                      1e-10, "public stress extensive virial");
        }
        complete(request == 0   ? "public-run"
                 : request == 1 ? "public-forces"
                                : "public-stress",
                 a, result, m->tol);
        qe_energy(result);
        close(f.meta.get(kEnergy),
              result.energy * cpu::pw::qe_units::rydberg_to_ev, 1e-10,
              "public Frame energy");
        if (request != 1)
            require(forces(f) == original_forces,
                    "unrequested forces preserved");
        if (request != 2)
            require(f.meta.get(kStress) == original_stress,
                    "unrequested virial preserved");
        if (request == 0)
            ref.result = result;
        else
            metadata(ref.result, result);
        m.reset();
        require(a.builds == 1 && a.destroys == 1,
                "method teardown must not repeat SCF lifecycle");
    }
    return ref;
}
struct ParentState {
    cpu::LoopServices loop;
    cpu::ExecServices exec;
    IterationStatus status;
    ConvergenceMetrics metrics;
};
class Before final : public cpu::Component {
    Audit& a_;
    ParentState& p_;

  public:
    Before(Audit& a, ParentState& p) : a_(a), p_(p) {}
    std::string name() const override { return "pw_analytic_parent_before"; }
    void launch(const cpu::LaunchContext& c) override {
        require(c.runtime.loop.status && c.runtime.loop.metrics,
                "parent loop services");
        a_.parent_frame = &c.frame;
        p_.loop = c.runtime.loop;
        p_.exec = c.runtime.exec;
        p_.status = *c.runtime.loop.status;
        p_.metrics = *c.runtime.loop.metrics;
    }
};
class Collect final : public cpu::Component {
    Audit& a_;
    ParentState& p_;
    Method& m_;
    const Reference& ref_;

  public:
    int calls = 0;
    Collect(Audit& a, ParentState& p, Method& m, const Reference& r)
        : a_(a), p_(p), m_(m), ref_(r) {}
    std::string name() const override { return "pw_analytic_parent_collect"; }
    bool requires_system_stress() const override { return true; }
    void launch(const cpu::LaunchContext& c) override {
        require(&c.frame == a_.parent_frame
                    && c.runtime.exec.runtime == p_.exec.runtime
                    && c.runtime.loop.status == p_.loop.status
                    && c.runtime.loop.metrics == p_.loop.metrics,
                "nested restores actual parent services");
        require(c.runtime.exec.forces_requested
                    && c.runtime.exec.stress_requested
                    && c.runtime.loop.status->converged == p_.status.converged
                    && c.runtime.loop.status->stop_requested
                           == p_.status.stop_requested
                    && c.runtime.loop.status->reason == p_.status.reason,
                "nested preserves parent status and combined request");
        close(c.runtime.loop.metrics->energy, p_.metrics.energy, 0,
              "parent energy metric preserved");
        close(c.runtime.loop.metrics->energy_change, p_.metrics.energy_change,
              0, "parent delta energy preserved");
        close(c.runtime.loop.metrics->residual, p_.metrics.residual, 0,
              "parent residual metric preserved");
        const auto& r = m_.last_result();
        complete("nested-force-and-stress", a_, r, m_.tol);
        metadata(ref_.result, r);
        qe_energy(r);
        close(c.frame.meta.get(kEnergy),
              r.energy * cpu::pw::qe_units::rydberg_to_ev, 1e-10,
              "one nested child energy publication");
        const auto f = forces(c.frame);
        qe_forces(f);
        Six sigma;
        for (int i = 0; i < 2; ++i)
            for (int d = 0; d < 3; ++d)
                close(f[i][d], ref_.force[i][d], 2e-8,
                      "nested/public force parity");
        for (int v = 0; v < 6; ++v) {
            close(c.frame.meta.get(kStress)[v],
                  ref_.sigma[v] * c.frame.box.volume(), 1e-7,
                  "nested/public extensive virial parity");
            sigma[v] = c.frame.meta.get(kStress)[v] / c.frame.box.volume();
        }
        qe_stress(sigma);
        ++calls;
    }
};
void nested(const Reference& ref) {
    auto f = silicon();
    auto child = method();
    Audit a;
    a.expected_geometry = &f;
    Measure stage(a, *child);
    Guard guard(a);
    child->on(pw::Phase::check).add_ref(stage, 10);
    child->add_criterion_ref(guard);
    child->build(f);
    ParentState state;
    Before before(a, state);
    Collect collect(a, state, *child, ref);
    {
        cpu::Evaluate parent;
        parent.on(evaluate::Phase::evaluate).add_ref(before, -10);
        parent.on(evaluate::Phase::evaluate).add_ref(*child, 0);
        parent.on(evaluate::Phase::evaluate).add_ref(collect, 10);
        parent.set_serial_on_replica();
        parent.build();
        parent.run(f);
    }
    require(collect.calls == 1,
            "exactly one complete nested publication collected");
    child.reset();
    require(a.builds == 1 && a.destroys == 1,
            "nested method teardown repeats no lifecycle");
}
} // namespace
int main(int argc, char** argv) {
    try {
        for (int argument = 1; argument < argc; ++argument) {
            const std::string option = argv[argument];
            if (option == "--mesh" && argument + 1 < argc)
                input.mesh = std::stoi(argv[++argument]);
            else if (option == "--rotate-z90")
                input.rotate = true;
            else if (option == "--mpi")
                input.mpi = true;
            else
                throw std::invalid_argument(
                    "--mesh 1|2 [--rotate-z90] [--mpi]");
        }
        require(input.mesh == 1 || input.mesh == 2,
                "diagnostic mesh must be 1 or2");
#ifdef ATV_HAS_MPI
        if (input.mpi)
            MPI_Init(&argc, &argv);
        struct MpiLifetime {
            bool active;
            ~MpiLifetime() {
                if (active)
                    MPI_Finalize();
            }
        } mpi_lifetime{input.mpi};
#else
        require(!input.mpi, "diagnostic --mpi requires an MPI build");
#endif
        const auto ref = standalone();
        nested(ref);
        std::printf("PASS PW_ANALYTIC_STRESS RKS_fixed_LSDA_Si2 public=3 "
                    "nested_combined=1 external=QE7.6_Slater_VWN5 mesh=%d "
                    "rotated=%d E_Ha=%.16g "
                    "gates_Ha_HaBohr_eVA3=1e-5,1e-4,2e-3\n",
                    input.mesh, int(input.rotate), ref.result.energy_ha);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL PW_ANALYTIC_STRESS %s\n", e.what());
        return 1;
    }
}
