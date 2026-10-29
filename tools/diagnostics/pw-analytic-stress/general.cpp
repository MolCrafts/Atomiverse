#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/zero.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "general_fixture.h"
#include "general_trace.h"
#include <cmath>
#include <cstdio>
using namespace ATV;
namespace {
void require(bool x, const char* m) {
    if (!x)
        throw std::runtime_error(m);
}
void close(double x, double y, double gate, const char* m) {
    if (!std::isfinite(x) || std::abs(x - y) > gate) {
        std::fprintf(stderr, "GATE %s got %.17g expected %.17g bound %.17g\n",
                     m, x, y, gate);
        throw std::runtime_error(m);
    }
}
class Parent final : public cpu::Driver<evaluate::Phase> {
  public:
    Parent() {
        set_serial_on_replica();
        on(evaluate::Phase::prepare).add(cpu::ZeroEnergy{}, -1000);
        on(evaluate::Phase::prepare).add(cpu::ZeroForces{}, -999);
        on(evaluate::Phase::prepare).add(cpu::ZeroStress{}, -998);
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
    void validate_components() const override { require_energy_or_force(); }
};
class StressRead final : public cpu::Component {
  public:
    std::string name() const override { return "pw_general_stress_consumer"; }
    bool requires_system_stress() const override { return true; }
    void launch(const cpu::LaunchContext&) override {}
};
void check_result(const pbe_periodic_qe::Pin& p, const cpu::pw::PWResult& r) {
    require(r.converged, "actual reference convergence");
    close(r.energy_ha, p.energy_ha, pbe_periodic_qe::energy_tolerance_ha,
          "original actual QE energy");
    close(r.n_elec_occ, p.spin ? 3. : 8., 1e-8,
          "original occupied electron count");
    require(r.e_entropy < -1e-5, "original nonzero FD entropy");
}
void run(const pbe_periodic_qe::Pin& p) {
    auto f = fixture::frame(p);
    auto m = fixture::method(p);
    trace::Counter s(*m);
    m->on(pw::Phase::check).add_ref(s, 10);
    trace::reset();
    const auto sigma = m->stress(f);
    const auto reference = m->last_result();
    trace::closed_budget(s, p.spin ? 2 : 1);
    const auto standalone = trace::counts;
    check_result(p, reference);
    double se = 0;
    for (int k = 0; k < 6; ++k) {
        const auto target =
            p.stress_ha_bohr3[k] * units::kHartreePerBohr3InEvPerAngstrom3;
        close(sigma[k], target, pbe_periodic_qe::stress_tolerance_ev_ang3,
              "original each QE stress");
        close(f.meta.get(kStress)[k], sigma[k] * f.box.volume(), 1e-10,
              "standalone extensive virial");
        se = std::max(se, std::abs(sigma[k] - target));
    }
    close(f.meta.get(kEnergy),
          reference.energy * cpu::pw::qe_units::rydberg_to_ev, 1e-9,
          "accepted QE-unit energy publication");
    auto nf = fixture::frame(p);
    auto nm = fixture::method(p);
    trace::Counter ns(*nm);
    nm->on(pw::Phase::check).add_ref(ns, 10);
    StressRead read;
    trace::reset();
    {
        Parent parent;
        parent.on(evaluate::Phase::evaluate).add_ref(*nm, 0);
        parent.on(evaluate::Phase::evaluate).add_ref(read, 10);
        parent.build();
        parent.run(nf);
    }
    trace::closed_budget(ns, p.spin ? 2 : 1);
    const auto nested = trace::counts;
    const auto result = nm->last_result();
    check_result(p, result);
    close(result.energy_ha, reference.energy_ha, 1e-8,
          "unchanged reference energy");
    close(result.e_entropy, reference.e_entropy, 1e-12,
          "reference entropy preserved");
    close(result.magnetization, reference.magnetization, 1e-12,
          "reference spin metadata preserved");
    double fe = 0;
    for (int i = 0; i < p.nat; ++i)
        for (int d = 0; d < 3; ++d) {
            const auto value = (d == 0   ? nf.span(kAtom_fx)[i]
                                : d == 1 ? nf.span(kAtom_fy)[i]
                                         : nf.span(kAtom_fz)[i])
                               / units::kHartreePerBohrInEvPerAngstrom;
            close(value, p.forces_ha_bohr[i][d],
                  pbe_periodic_qe::force_tolerance_ha_bohr,
                  "original each QE force");
            fe = std::max(fe, std::abs(value - p.forces_ha_bohr[i][d]));
            std::printf("GENERAL_FORCE %s atom=%d axis=%d Ha_Bohr=%.17g\n",
                        p.id, i, d, value);
        }
    for (int k = 0; k < 6; ++k)
        close(nf.meta.get(kStress)[k], sigma[k] * nf.box.volume(), 1e-7,
              "nested extensive virial parity");
    close(nf.meta.get(kEnergy),
          result.energy * cpu::pw::qe_units::rydberg_to_ev, 1e-9,
          "nested accepted energy publication");
    std::printf("GENERAL_SIGMA %s", p.id);
    for (auto v : sigma)
        std::printf(" %.17g", v);
    std::printf(
        "\nPASS PW_ANALYTIC_GENERAL_QE %s E_Ha=%.17g dE_Ha=%.17g "
        "dF_Ha_Bohr=%.17g "
        "dSigma_eV_A3=%.17g standalone_iters=%d nested_iters=%d "
        "standalone_Davidson=%d nested_Davidson=%d standalone_extract=%d "
        "nested_extract=%d standalone_assembly=%d nested_assembly=%d "
        "legacy_calls=%d reference_SCF_calls=2\n",
        p.id, reference.energy_ha, std::abs(reference.energy_ha - p.energy_ha),
        fe, se, reference.iterations, result.iterations, standalone.davidson,
        nested.davidson, standalone.extract, nested.extract,
        standalone.assembly, nested.assembly,
        standalone.legacy_stress + nested.legacy_stress);
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2)
            run(pbe_periodic_qe::pins.at(std::stoul(argv[1])));
        else
            for (const auto& p : pbe_periodic_qe::pins)
                run(p);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL PW_ANALYTIC_GENERAL_QE %s\n", e.what());
        return 1;
    }
}
