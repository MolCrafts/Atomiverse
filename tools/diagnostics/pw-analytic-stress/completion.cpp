// Opt-in strict auxiliary completion; GNU link wrapping is test-only.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/pw_final_state.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "general_fixture.h"
#include <cstdio>
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif
#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>
using namespace ATV;
using namespace ATV::cpu::pw;
using namespace ATV::cpu::pw::internal;
namespace {
void require(bool b, const char* m) {
    if (!b)
        throw std::runtime_error(m);
}
bool force_short = false, in_aux = false;
int short_calls = 0;
int rank_id = 0, world_size = 1, forced_rank = 0;
AccReal aux_target = 0, observed_residual = 0;
} // namespace
void real_d(DavidsonSolver*, PWWorkspace&) asm(
    "__real__ZN3ATV3cpu2pw8internal14DavidsonSolver4stepERNS2_11PWWorkspaceE");
void wrap_d(DavidsonSolver*, PWWorkspace&) asm(
    "__wrap__ZN3ATV3cpu2pw8internal14DavidsonSolver4stepERNS2_11PWWorkspaceE");
void wrap_d(DavidsonSolver* solver, PWWorkspace& w) {
    if (force_short && in_aux && rank_id == forced_rank) {
        DavidsonSolver short_solver(6, aux_target, 1);
        short_solver.prepare(w);
        real_d(&short_solver, w);
        ++short_calls;
        std::vector<std::complex<AccReal>> hc(w.n_g);
        for (size_t b = 0; b < w.eigenvalues.size(); ++b) {
            const auto* c = w.wfc.data() + b * w.n_g;
            pw_apply_h(w, c, hc.data());
            AccReal v = 0;
            for (int g = 0; g < w.n_g; ++g)
                v += std::norm(hc[g] - w.eigenvalues[b] * c[g]);
            observed_residual = std::max(observed_residual, std::sqrt(v));
        }
    } else
        real_d(solver, w);
}
FinalBandState real_f(const PWProblem&, const PW&, const PWResult&, int) asm(
    "__real__ZN3ATV3cpu2pw8internal19extract_final_bandsERKNS2_"
    "9PWProblemERKNS1_2PWERKNS1_8PWResultEi");
FinalBandState wrap_f(const PWProblem&, const PW&, const PWResult&, int) asm(
    "__wrap__ZN3ATV3cpu2pw8internal19extract_final_bandsERKNS2_"
    "9PWProblemERKNS1_2PWERKNS1_8PWResultEi");
FinalBandState wrap_f(const PWProblem& p, const PW& m, const PWResult& r,
                      int n) {
    require(!force_short || (p.n_elec == 8 && (n == 6 || m.extra_bands == 2)),
            "short solver six-band fixture");
    aux_target = m.tol * .1;
    in_aux = true;
    try {
        auto s = real_f(p, m, r, n);
        in_aux = false;
        return s;
    } catch (...) {
        in_aux = false;
        throw;
    }
}
class Parent final : public cpu::Driver<evaluate::Phase> {
  public:
    Parent() { set_serial_on_replica(); }
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
    std::string name() const override {
        return "shadow_general_stress_consumer";
    }
    bool requires_system_stress() const override { return true; }
    void launch(const cpu::LaunchContext&) override {}
};

namespace pw_frozen {
inline ATV::cpu::Frame silicon() {
    using namespace ATV;
    cpu::Frame frame(2);
    Mat3 h;
    h(0, 0) = .10;
    h(1, 0) = 2.70;
    h(2, 0) = 2.65;
    h(0, 1) = 2.75;
    h(1, 1) = .05;
    h(2, 1) = 2.70;
    h(0, 2) = 2.65;
    h(1, 2) = 2.75;
    h(2, 2) = .15;
    frame.box.set_cell(h);
    constexpr double p[2][3] = {{.2, .3, .4}, {1.57, 1.66, 1.75}};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 14;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_mass)[i] = 28.0855;
        frame.span(kAtom_x)[i] = p[i][0];
        frame.span(kAtom_y)[i] = p[i][1];
        frame.span(kAtom_z)[i] = p[i][2];
    }
    return frame;
}
} // namespace pw_frozen
using C = std::complex<AccReal>;
std::vector<C> snapshot(const PWWorkspace& w) {
    std::vector<C> v;
    auto add = [&](const auto& a) {
        for (auto x : a)
            v.emplace_back(x);
    };
    add(w.rho_G);
    add(w.rho_G_old);
    add(w.V_H_G);
    add(w.V_xc_G);
    add(w.V_eff_G);
    add(w.V_loc_G);
    add(w.wfc);
    add(w.T_G);
    add(w.eigenvalues);
    return v;
}

void run_public_parent() {
    require(world_size == 1, "public parent fixture runs on one MPI rank");
    const auto& pin = pbe_periodic_qe::pins.at(0);
    auto f = fixture::frame(pin);
    auto m = fixture::method(pin);
    f.meta.get(kEnergy) = 123;
    f.meta.get(kEnergyElectronic) = 456;
    f.meta.get(kStress).fill(7);
    for (int i = 0; i < f.nlocal(); ++i) {
        f.span(kAtom_fx)[i] = 8;
        f.span(kAtom_fy)[i] = 9;
        f.span(kAtom_fz)[i] = 10;
    }
    Parent parent;
    StressRead read;
    parent.on(evaluate::Phase::evaluate).add_ref(*m, 0);
    parent.on(evaluate::Phase::evaluate).add_ref(read, 10);
    parent.build();
    force_short = true;
    bool caught = false;
    try {
        parent.run(f);
    } catch (const std::runtime_error& e) {
        caught = true;
        std::printf("EXPECTED_AUX_FAILURE %s\n", e.what());
    }
    require(caught && short_calls == 1 && observed_residual > aux_target,
            "real one-sweep incomplete eigenproblem must reject");
    require(f.meta.get(kEnergy) == 123 && f.meta.get(kEnergyElectronic) == 456,
            "failed aux must not publish energy");
    for (auto x : f.meta.get(kStress))
        require(x == 7, "failed aux must not publish stress");
    for (int i = 0; i < f.nlocal(); ++i)
        require(f.span(kAtom_fx)[i] == 8 && f.span(kAtom_fy)[i] == 9
                    && f.span(kAtom_fz)[i] == 10,
                "failed aux must not publish forces");
    force_short = false;
    parent.run(f);
    const auto& r = m->last_result();
    require(r.converged, "same parent retry converges");
    require(std::abs(r.energy_ha - pin.energy_ha)
                < pbe_periodic_qe::energy_tolerance_ha,
            "same parent original QE E gate");
    for (int i = 0; i < f.nlocal(); ++i)
        for (int a = 0; a < 3; ++a) {
            const auto x = (a == 0   ? f.span(kAtom_fx)[i] - 8
                            : a == 1 ? f.span(kAtom_fy)[i] - 9
                                     : f.span(kAtom_fz)[i] - 10)
                           / units::kHartreePerBohrInEvPerAngstrom;
            require(std::abs(x - pin.forces_ha_bohr[i][a])
                        < pbe_periodic_qe::force_tolerance_ha_bohr,
                    "same parent original QE fullF gate");
        }
    for (int k = 0; k < 6; ++k) {
        const auto x = (f.meta.get(kStress)[k] - 7) / f.box.volume();
        require(std::abs(x
                         - pin.stress_ha_bohr3[k]
                               * units::kHartreePerBohr3InEvPerAngstrom3)
                    < pbe_periodic_qe::stress_tolerance_ev_ang3,
                "same parent original QE six stress gate");
    }
    require(
        std::abs(f.meta.get(kEnergy) - 123 - r.energy * qe_units::rydberg_to_ev)
            < 1e-9,
        "same parent retry publishes accepted energy exactly once");
    require(std::abs(f.meta.get(kEnergyElectronic) - 456
                     - r.energy * qe_units::rydberg_to_ev)
                < 1e-9,
            "same parent retry publishes electronic energy exactly once");
    std::printf("PASS AUX_COMPLETION_PARENT budget=1 short_calls=%d "
                "actual_residual_Ha=%.17g target_Ha=%.17g unchanged=1 "
                "same_parent_retry=1 E_Ha=%.17g\n",
                short_calls, observed_residual, aux_target, r.energy_ha);
}
void run_prepared(int mesh) {
    forced_rank = mesh == 1 ? 0 : (world_size > 1 ? 1 : 0);
    auto f = pw_frozen::silicon();
    auto pp =
        std::make_shared<GthPseudopotential>(molrs::Element::Si, "GTH-PBE-q4");
    PW m{pp, 12, mesh, cpu::xc::XCType::LSDA};
    m.tol = 1e-10;
    PWProblem p;
    setup_pw_problem(f, *pp, 12, mesh, p);
    p.ws.V_H_G.fill(7);
    p.ws.V_xc_G.fill(-5);
    p.ws.rho_G_old.fill(2);
    p.ws.V_eff_G.fill(-3);
    p.ws.T_G.fill(11);
    p.ws.wfc = xt::ones<C>({std::size_t(1), std::size_t(1)});
    p.ws.eigenvalues = xt::ones<AccReal>({std::size_t(1)});
    PWResult ref;
    ref.energy = 17;
    ref.energy_ha = 8.5;
    ref.iterations = 7;
    ref.converged = false;
    const auto before = snapshot(p.ws);
    const auto* fft = p.ws.fft.get();
    const auto* binding = p.ws.gvec;
    FinalBandState output;
    output.auxiliary_mu_ha = 123;
    force_short = true;
    bool caught = false;
    try {
        output = extract_final_bands(p, m, ref, 6);
    } catch (const std::runtime_error& e) {
        caught = true;
        std::printf("EXPECTED_AUX_MPI_FAILURE rank=%d message=%s\n", rank_id,
                    e.what());
    }
    int n = caught ? 1 : 0;
#ifdef ATV_HAS_MPI
    MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    require(n == world_size, "all ranks must reject remote incomplete solve");
    require(output.auxiliary_mu_ha == 123 && output.local_blocks.empty(),
            "no partial auxiliary output published");
    require(before == snapshot(p.ws) && fft == p.ws.fft.get()
                && binding == p.ws.gvec && ref.energy == 17 && !ref.converged,
            "accepted inputs unchanged after failed extract");
    require(rank_id != forced_rank
                || (short_calls == 1 && observed_residual > aux_target),
            "real same-class max_iter1 residual must exceed advertised target");
    force_short = false;
    auto state = extract_final_bands(p, m, ref, 6);
    require(before == snapshot(p.ws) && state.reference_result.energy == 17
                && !state.reference_result.converged,
            "normal retry preserves reference false verdict");
    require(state.max_local_eigen_residual_ha <= m.tol * .1,
            "normal retry actual strict completion");
    std::printf(
        "PASS AUX_COMPLETION_MPI mesh=%d rank=%d ranks=%d forced_rank=%d "
        "all_reject=%d short_calls=%d failed_residual_Ha=%.17g "
        "target_Ha=%.17g retry_residual_Ha=%.17g unchanged=1 "
        "no_partial_output=1 reference_converged=0 SCF_calls=0\n",
        mesh, rank_id, world_size, forced_rank, n, short_calls,
        observed_residual, aux_target, state.max_local_eigen_residual_ha);
}

int main(int argc, char** argv) {
#ifdef ATV_HAS_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif
    int rc = 0;
    try {
        bool prepared = false;
        int mesh = 2;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--prepared")
                prepared = true;
            else if (arg == "--mesh" && i + 1 < argc)
                mesh = std::stoi(argv[++i]);
            else
                throw std::invalid_argument(
                    "usage: completion [--prepared [--mesh 1|2]]");
        }
        require(mesh == 1 || mesh == 2, "prepared mesh must be1or2");
        if (prepared)
            run_prepared(mesh);
        else
            run_public_parent();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL AUX_COMPLETION rank=%d %s\n", rank_id,
                     e.what());
        rc = 1;
    }
#ifdef ATV_HAS_MPI
    MPI_Finalize();
#endif
    return rc;
}
