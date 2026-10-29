// Shared opt-in ensemble vertical; no copied SCF or band loop.
#pragma once
#include "atomiverse/cpu/planewave/qe_units.h"
#include "multi_k_fd_audit.h"
#include "multi_k_fd_fixture.h"
#include <array>
#include <memory>
#include <string>
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif
namespace pw_ensemble_diagnostic {
using namespace ATV;
using namespace pw_analytic_diagnostic;
using Pin = pw_multi_k_fd_qe::Pin;
using Six = std::array<double, 6>;
inline int rank = 0, ranks = 1;
inline std::vector<std::array<double, 3>> forces(const cpu::Frame& f) {
    std::vector<std::array<double, 3>> out(f.nlocal());
    for (int i = 0; i < f.nlocal(); ++i)
        out[i] = {f.span(kAtom_fx)[i], f.span(kAtom_fy)[i],
                  f.span(kAtom_fz)[i]};
    return out;
}
inline void physics(const Pin& p, const Result& r, const Recipe& recipe) {
    require(r.converged, "SCF accepted reference convergence");
    close(r.energy_ha, p.energy_ha, pw_multi_k_fd_qe::energy_gate_ha,
          "actual matched QE free energy");
    if (recipe.fixed()) {
        close(r.magnetization, p.magnetization, 1e-8,
              "fixed reference magnetization");
        close(r.e_fermi, 0, 0, "fixed reference Fermi marker");
        close(r.n_elec_occ, 0, 0, "fixed reference optional count preserved");
        close(r.e_entropy, 0, 0, "fixed reference entropy");
        const auto& a = trace::shared_bands;
        if (recipe.empty_beta()) {
            require(a.fixed_na == 2 && a.fixed_nb == 0 && a.beta_layout_zero
                        && a.beta_density_exact_zero
                        && a.empty_beta_blocks == r.n_kpts_local,
                    "producer declared canonical empty beta and exact density "
                    "manifold");
            std::printf(
                "EMPTY_BETA_STATE typed_Na=%d typed_Nb=%d beta_blocks=%d "
                "beta_rows=0 current_reference_orbital_beta_exact0=1\n",
                a.fixed_na, a.fixed_nb, a.empty_beta_blocks);
        }

        const double na =
            p.spin ? (p.electrons + p.magnetization) / 2 : p.electrons;
        const double nb = p.spin ? (p.electrons - p.magnetization) / 2 : 0;
        close(a.fcounts[0], na, 1e-8, "actual sum wf alpha");
        close(a.fcounts[1], nb, 1e-8, "actual sum wf beta");
        close(a.rho_counts[0], na, 1e-8, "actual rho G0 alpha");
        close(a.rho_counts[1], nb, 1e-8, "actual rho G0 beta");
        close(a.aux_ne, p.electrons, 1e-8, "actual fixed aux count");
        close(a.aux_m, p.magnetization, 1e-8, "actual fixed aux magnetization");
        close(a.aux_entropy, 0, 0, "actual fixed auxiliary entropy");
        std::printf("FIXED_REFERENCE %s reference_optional_Ne=%.17g fNa=%.17g "
                    "fNb=%.17g rhoNa=%.17g rhoNb=%.17g auxNe=%.17g auxM=%.17g "
                    "auxS=%.17g max_local_Hres=%.17g\n",
                    p.id, r.n_elec_occ, a.fcounts[0], a.fcounts[1],
                    a.rho_counts[0], a.rho_counts[1], a.aux_ne, a.aux_m,
                    a.aux_entropy, a.completion_residual);
        return;
    }
    if (recipe.mp()) {
        const auto& a = trace::shared_bands;
        std::printf(
            "MP_STATE sum_wf_alpha=%.17g sum_wf_beta=%.17g rhoG0_alpha=%.17g "
            "rhoG0_beta=%.17g auxNe=%.17g auxM=%.17g auxS=%.17g auxMu_Ha=%.17g "
            "max_local_Hres_Ha=%.17g target_Ha=%.17g order=%d width_Ha=%.17g\n",
            a.fcounts[0], a.fcounts[1], a.rho_counts[0], a.rho_counts[1],
            a.aux_ne, a.aux_m, a.aux_entropy, a.mp_mu, a.completion_residual,
            a.mp_target_ha, a.mp_order, a.mp_width_ha);
    }
    close(r.n_elec_occ, p.electrons, 1e-8, "actual electron count");
    require(std::isfinite(r.e_fermi) && std::isfinite(r.magnetization)
                && std::isfinite(r.e_entropy),
            "finite reference metadata");
    if (recipe.gaussian())
        std::printf("GAUSSIAN_REFERENCE %s Ne=%.17g M=%.17g entropy_Ry=%.17g\n",
                    p.id, r.n_elec_occ, r.magnetization, r.e_entropy);
    if (recipe.mp()) {
        const auto& a = trace::shared_bands;
        require(a.mp_active && a.mp_order == 1, "MP1 fixture context");
        require(a.mp_min_f < -1e-4 && a.mp_max_f > (p.spin ? 1 : 2) + 1e-4,
                "predeclared nontrivial signed MP occupations");
        close(a.fcounts[0] + a.fcounts[1], p.electrons, 1e-8,
              "MP signed sum wf count");
        close(a.rho_counts[0] + a.rho_counts[1], p.electrons, 1e-8,
              "MP signed rhoG0 count");
        if (p.spin)
            close(a.fcounts[0] - a.fcounts[1], a.aux_m, 1e-8,
                  "MP free auxiliary spin");
        else {
            close(a.aux_m, 0, 0, "RKS auxiliary M0");
            close(r.magnetization, 0, 0, "RKS reference M0");
            close(a.fcounts[1], 0, 0, "one physical RKS channel");
            std::printf("MP_RKS_COUNT degeneracy=2 physical_channels=1 "
                        "total_Ne=%.17g M=0\n",
                        a.fcounts[0]);
        }
        std::printf("MP_REFERENCE %s Ne=%.17g M=%.17g minusTS_Ry=%.17g "
                    "min_f=%.17g max_f=%.17g auxS=%.17g auxMu_Ha=%.17g\n",
                    p.id, r.n_elec_occ, r.magnetization, r.e_entropy,
                    a.mp_min_f, a.mp_max_f, a.aux_entropy, a.mp_mu);
    }
    require((recipe.mp() ? std::abs(r.e_entropy) : -r.e_entropy)
                > pw_multi_k_fd_qe::minimum_abs_entropy_ry,
            "predeclared nonzero ensemble entropy");
    if (p.spin)
        require(std::abs(r.magnetization)
                    > pw_multi_k_fd_qe::minimum_abs_magnetization,
                "predeclared nonzero spin");
}
struct Reference {
    Result result;
    Six sigma;
    trace::Counts counts;
};
struct ParentState {
    cpu::LoopServices loop;
    cpu::ExecServices exec;
    IterationStatus status;
    ConvergenceMetrics metrics;
    int before_calls = 0, after_calls = 0;
};
class Before final : public cpu::Component {
    Audit& a_;
    ParentState& p_;

  public:
    Before(Audit& a, ParentState& p) : a_(a), p_(p) {}
    std::string name() const override { return "pw_multik_fd_before"; }
    void launch(const cpu::LaunchContext& c) override {
        require(c.runtime.loop.status && c.runtime.loop.metrics,
                "live parent loop services");
        a_.parent_frame = &c.frame;
        p_.loop = c.runtime.loop;
        p_.exec = c.runtime.exec;
        p_.status = *c.runtime.loop.status;
        p_.metrics = *c.runtime.loop.metrics;
        ++p_.before_calls;
        c.frame.meta.get(kEnergy) = 123;
        c.frame.meta.get(kEnergyElectronic) = 456;
        c.frame.meta.get(kStress) = {1, 2, 3, 4, 5, 6};
        for (int i = 0; i < c.frame.nlocal(); ++i) {
            c.frame.span(kAtom_fx)[i] = 10 + i;
            c.frame.span(kAtom_fy)[i] = 20 + i;
            c.frame.span(kAtom_fz)[i] = 30 + i;
        }
    }
};
class Collect final : public cpu::Component {
    Audit& a_;
    ParentState& p_;
    Method& m_;
    const Pin& pin_;
    const Reference& ref_;
    const Recipe recipe_;

  public:
    double max_force_error = 0, max_stress_error = 0;
    trace::Counts measured;
    Collect(Audit& a, ParentState& p, Method& m, const Pin& pin,
            const Reference& ref, const Recipe& recipe)
        : a_(a), p_(p), m_(m), pin_(pin), ref_(ref), recipe_(recipe) {}
    std::string name() const override { return "pw_multik_fd_collect"; }
    bool requires_system_stress() const override { return true; }
    void launch(const cpu::LaunchContext& c) override {
        require(&c.frame == a_.parent_frame
                    && c.runtime.exec.runtime == p_.exec.runtime
                    && c.runtime.loop.status == p_.loop.status
                    && c.runtime.loop.metrics == p_.loop.metrics,
                "nested restores actual parent context");
        require(c.runtime.exec.forces_requested
                    && c.runtime.exec.stress_requested
                    && c.runtime.loop.status->converged == p_.status.converged
                    && c.runtime.loop.status->stop_requested
                           == p_.status.stop_requested
                    && c.runtime.loop.status->reason == p_.status.reason,
                "parent status and combined request restored");
        require(c.runtime.loop.metrics->energy == p_.metrics.energy
                    && c.runtime.loop.metrics->energy_change
                           == p_.metrics.energy_change
                    && c.runtime.loop.metrics->residual == p_.metrics.residual,
                "parent metrics restored");
        require(trace::shared_bands.force_calls == 1
                    && trace::shared_bands.same_inputs,
                "force/stress sharedC/f andEwald");
        const auto& r = m_.last_result();
        complete("nested-force-and-stress", a_, r, m_.tol,
                 recipe_.empty_beta() ? 1
                 : pin_.spin          ? 2
                                      : 1);
        metadata(ref_.result, r);
        physics(pin_, r, recipe_);
        const double e = r.energy * cpu::pw::qe_units::rydberg_to_ev;
        close(c.frame.meta.get(kEnergy), 123 + e, 1e-9,
              "exactly one nested total energy contribution");
        close(c.frame.meta.get(kEnergyElectronic), 456 + e, 1e-9,
              "exactly one nested electronic energy contribution");
        auto f = forces(c.frame);
        for (int i = 0; i < pin_.nat; ++i)
            for (int d = 0; d < 3; ++d) {
                const double value = (f[i][d] - ((d + 1) * 10 + i))
                                     / units::kHartreePerBohrInEvPerAngstrom;
                close(value, pin_.forces_ha_bohr[i][d],
                      pw_multi_k_fd_qe::force_gate_ha_bohr,
                      "matched QE each total force");
                max_force_error =
                    std::max(max_force_error,
                             std::abs(value - pin_.forces_ha_bohr[i][d]));
                std::printf("%s %s atom=%d axis=%d HaBohr=%.17g\n",
                            recipe_.force_prefix(), pin_.id, i, d, value);
            }
        std::printf("%s %s", recipe_.stress_prefix(), pin_.id);
        for (int k = 0; k < 6; ++k) {
            close(c.frame.meta.get(kStress)[k],
                  k + 1 + ref_.sigma[k] * c.frame.box.volume(), 1e-7,
                  "exactly one nested virial contribution");
            const double sigma =
                (c.frame.meta.get(kStress)[k] - (k + 1)) / c.frame.box.volume();
            const double want = pin_.stress_ha_bohr3[k]
                                * units::kHartreePerBohr3InEvPerAngstrom3;
            close(sigma, want, pw_multi_k_fd_qe::stress_gate_ev_ang3,
                  "matched QE each stress");
            max_stress_error =
                std::max(max_stress_error, std::abs(sigma - want));
            std::printf(" %.17g", sigma);
        }
        std::printf("\n");
        measured = trace::counts;
        ++p_.after_calls;
    }
};
inline void run(const Pin& p, const Recipe& recipe = {}) {
    auto f = pw_multi_k_fd::frame(p, recipe);
    auto old_forces = forces(f);
    auto m = pw_multi_k_fd::method(p, recipe);
    Audit a;
    a.expected_geometry = &f;
    Measure measure(a, *m);
    Guard guard(a);
    m->on(pw::Phase::check).add_ref(measure, 10);
    m->add_criterion_ref(guard);
    trace::reset();
    Reference ref;
    ref.sigma = m->stress(f);
    ref.result = m->last_result();
    ref.counts = trace::counts;
    complete("standalone-stress", a, ref.result, m->tol,
             recipe.empty_beta() ? 1
             : p.spin            ? 2
                                 : 1);
    physics(p, ref.result, recipe);
    require(a.actual != &f, "standalone SCF uses its geometry copy");
    require(forces(f) == old_forces,
            "stress-only public request preserves force output");
    close(f.meta.get(kEnergy),
          ref.result.energy * cpu::pw::qe_units::rydberg_to_ev, 1e-9,
          "standalone accepted energy publication");
    for (int k = 0; k < 6; ++k) {
        close(f.meta.get(kStress)[k], ref.sigma[k] * f.box.volume(), 1e-10,
              "standalone extensive virial publication");
        close(ref.sigma[k],
              p.stress_ha_bohr3[k] * units::kHartreePerBohr3InEvPerAngstrom3,
              pw_multi_k_fd_qe::stress_gate_ev_ang3,
              "standalone matched QE stress");
    }
    auto nf = pw_multi_k_fd::frame(p, recipe);
    auto child = pw_multi_k_fd::method(p, recipe);
    Audit na;
    na.expected_geometry = &nf;
    Measure nested_measure(na, *child);
    Guard nested_guard(na);
    child->on(pw::Phase::check).add_ref(nested_measure, 10);
    child->add_criterion_ref(nested_guard);
    ParentState state;
    Before before(na, state);
    Collect collect(na, state, *child, p, ref, recipe);
    trace::reset();
    {
        cpu::Evaluate parent;
        parent.set_serial_on_replica();
        parent.on(evaluate::Phase::evaluate).add_ref(before, -10);
        parent.on(evaluate::Phase::evaluate).add_ref(*child, 0);
        parent.on(evaluate::Phase::evaluate).add_ref(collect, 10);
        parent.build();
        parent.run(nf);
    }
    require(state.before_calls == 1 && state.after_calls == 1,
            "exactly one before/after nested publication");
    const auto result = child->last_result();
    const double contribution =
        result.energy * cpu::pw::qe_units::rydberg_to_ev;
    close(nf.meta.get(kEnergy), 123 + contribution, 1e-9,
          "returned parent publishes total energy once");
    close(nf.meta.get(kEnergyElectronic), 456 + contribution, 1e-9,
          "returned parent publishes electronic energy once");
    for (int k = 0; k < 6; ++k)
        close(nf.meta.get(kStress)[k], k + 1 + ref.sigma[k] * nf.box.volume(),
              1e-7, "returned parent publishes virial once");
    child.reset();
    m.reset();
    require(a.builds == 1 && a.destroys == 1 && na.builds == 1
                && na.destroys == 1,
            "method teardown does not repeat SCF lifecycle");
    std::string format =
        std::string("PASS ") + recipe.pass_prefix()
        + " %s rank=%d ranks=%d mesh=%d local_k=%d nspin=%d "
          "E_Ha=%.17g Ne=%.17g M=%.17g entropy_Ry=%.17g dE_Ha=%.17g "
          "dF_HaBohr=%.17g dSigma_eVA3=%.17g standalone_iters=%d "
          "nested_iters=%d "
          "standalone_Davidson=%d nested_Davidson=%d standalone_extract=%d "
          "nested_extract=%d standalone_assembly=%d nested_assembly=%d "
          "legacy=%d "
          "energy_publications=1 virial_publications=1 "
          "reference_metadata=preserved "
        + std::string(recipe.gaussian() || recipe.fixed() || recipe.mp()
                          ? "same_C_f=1 "
                          : "")
        + std::string(recipe.empty_beta()
                          ? "active_spin_channels=1 reference_optional_Ne=0 "
                          : "")
        + "SCF_calls=2\n";
    std::printf(format.c_str(), p.id, rank, ranks, recipe.kmesh,
                result.n_kpts_local, p.spin ? 2 : 1, result.energy_ha,
                result.n_elec_occ, result.magnetization, result.e_entropy,
                std::abs(result.energy_ha - p.energy_ha),
                collect.max_force_error, collect.max_stress_error,
                ref.result.iterations, result.iterations, ref.counts.davidson,
                collect.measured.davidson, ref.counts.extract,
                collect.measured.extract, ref.counts.assembly,
                collect.measured.assembly,
                ref.counts.legacy_stress + collect.measured.legacy_stress);
}
} // namespace pw_ensemble_diagnostic

namespace pw_ensemble_diagnostic {
template <class Run>
int entry(int argc, char** argv, std::size_t count, Run run,
          const char* prefix) {
    bool mpi = false;
    int index = -1;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--mpi")
                mpi = true;
            else if (index < 0)
                index = std::stoi(arg);
            else
                return 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "invalid diagnostic arguments: %s\n", e.what());
        return 2;
    }
#ifdef ATV_HAS_MPI
    if (mpi) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    }
    struct Finalize {
        bool active;
        ~Finalize() {
            if (active)
                MPI_Finalize();
        }
    } finalize{mpi};
#else
    if (mpi)
        return 2;
#endif
    try {
        if (index >= 0)
            run(std::size_t(index));
        else
            for (std::size_t i = 0; i < count; ++i)
                run(i);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s rank=%d %s\n", prefix, rank, e.what());
        return 1;
    }
    return 0;
}
} // namespace pw_ensemble_diagnostic
