/// @file regressions/multiscale/qmmm_oniom_h2_ar/reg_qmmm_oniom_h2_ar.cpp
/// @brief Two-layer ONIOM: H2 (DFTB2 high) + classical low on H2+Ar.
///
/// \f$ E = E_{\mathrm{high}}(\mathrm{H_2}) + E_{\mathrm{low}}(\mathrm{real})
///         - E_{\mathrm{low}}(\mathrm{H_2}) \f$
/// Identity is checked to 1e-10 eV. See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/column.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"
#include "atomiverse/cpu/multiscale/qmmm_mechanical.h"
#include "atomiverse/cpu/multiscale/qmmm_oniom.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
#include "harness/registry.h"

namespace atv_reg_multiscale_qmmm_oniom_h2_ar {

using namespace ATV;

namespace {

struct HistoryRequest : cpu::Component {
    int& mode;
    bool clear;
    HistoryRequest(int& m, bool reset) : mode(m), clear(reset) {}
    std::string name() const override { return "qm_history_request"; }
    void launch(const cpu::LaunchContext& ctx) override {
        ctx.runtime.exec.initializing_forces = !clear && mode == 1;
        ctx.runtime.exec.trial_evaluation = !clear && mode == 2;
    }
};

struct AdditionalPotential : cpu::Potential {
    std::string name() const override { return "qm_additional_potential"; }
    void launch(const cpu::LaunchContext& ctx) override {
        ctx.frame.meta.get(kEnergy) += 0.75;
        if (ctx.forces_requested)
            ctx.frame.span(kAtom_fx)[3] += 0.125;
    }
};

cpu::Frame make_water_mm() {
    cpu::Frame frame(4);
    const int species[] = {8, 1, 1, 18};
    const double xyz[][3] = {
        {0, 0, 0}, {0.96, 0, 0}, {-0.24, 0.93, 0.1}, {5.0, 0.3, 0.2}};
    frame.blocks.at("atoms").insert(
        "qm_z", cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                                kDynamicFieldId});
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_type)[i] = species[i];
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_mass)[i] = 1;
        frame.span(cpu::kAtomQmZ)[i] = i == 3 ? 0 : species[i];
        frame.span(kAtom_x)[i] = xyz[i][0];
        frame.span(kAtom_y)[i] = xyz[i][1];
        frame.span(kAtom_z)[i] = xyz[i][2];
    }
    frame.box = cpu::Box::cubic(30);
    return frame;
}

template <class Method>
bool check_nddo_qmmm_history() {
    Method normal{0, 1, 400, 1e-10, 1e-12};
    Method probed{0, 1, 400, 1e-10, 1e-12};
    const auto fail = [&](const char* quantity, int step, double got,
                          double want, double tolerance) {
        std::fprintf(
            stderr, "FAIL: %s QM/MM %s step=%d got=%.16e want=%.16e tol=%.3e\n",
            normal.qm_engine_name().c_str(), quantity, step, got, want,
            tolerance);
        return false;
    };
    const std::vector<cpu::LJTypeCoeff> lj = {
        {1, 1, 0.01, 1.2}, {8, 8, 0.02, 2.5}, {18, 18, 0.02, 3.0}};
    cpu::QmmmMechanical plain_qm{lj, 10}, probe_qm{lj, 10};
    plain_qm.set_engine_ref(normal);
    probe_qm.set_engine_ref(probed);
    cpu::Evaluate plain, with_probes;
    int mode = 0;
    plain.on(evaluate::Phase::evaluate).add(AdditionalPotential{}, -2);
    plain.on(evaluate::Phase::evaluate).add_ref(plain_qm);
    with_probes.on(evaluate::Phase::evaluate).add(AdditionalPotential{}, -2);
    with_probes.on(evaluate::Phase::evaluate)
        .add(HistoryRequest{mode, false}, -1);
    with_probes.on(evaluate::Phase::evaluate).add_ref(probe_qm);
    with_probes.on(evaluate::Phase::evaluate)
        .add(HistoryRequest{mode, true}, 1);
    plain.set_threads(1);
    with_probes.set_threads(1);
    plain.build();
    with_probes.build();
    for (mode = 1; mode <= 2; ++mode) {
        auto initial = make_water_mm();
        with_probes.run(initial);
        if (!probed.last_result().converged)
            return fail("probe converged", -mode, 0, 1, 0);
        if (probed.has_density())
            return fail("probe committed density", -mode, 1, 0, 0);
    }
    for (int step = 0; step < 4; ++step) {
        mode = 0;
        auto a = make_water_mm();
        a.span(kAtom_x)[1] += 0.015 * step;
        auto b = a;
        plain.run(a);
        with_probes.run(b);
        if (normal.last_result().iterations != probed.last_result().iterations)
            return fail("iterations", step, probed.last_result().iterations,
                        normal.last_result().iterations, 0);
        if (std::abs(a.meta.get(kEnergy) - b.meta.get(kEnergy)) > 1e-10)
            return fail("energy", step, b.meta.get(kEnergy),
                        a.meta.get(kEnergy), 1e-10);
        const double expected_energy =
            probe_qm.last_qm_energy() + probe_qm.last_mm_energy() + 0.75;
        if (std::abs(b.meta.get(kEnergy) - expected_energy) > 1e-10)
            return fail("accumulated energy", step, b.meta.get(kEnergy),
                        expected_energy, 1e-10);
        const AccReal* af[] = {a.span(kAtom_fx), a.span(kAtom_fy),
                               a.span(kAtom_fz)};
        const AccReal* bf[] = {b.span(kAtom_fx), b.span(kAtom_fy),
                               b.span(kAtom_fz)};
        double net_fx = 0;
        for (int i = 0; i < 4; ++i) {
            net_fx += bf[0][i];
            for (int d = 0; d < 3; ++d)
                if (std::abs(af[d][i] - bf[d][i]) > 1e-9) {
                    std::fprintf(stderr, "atom=%d axis=%d ", i, d);
                    return fail("force", step, bf[d][i], af[d][i], 1e-9);
                }
        }
        if (std::abs(net_fx - 0.125) > 1e-9)
            return fail("net x force", step, net_fx, 0.125, 1e-9);
        if (step == 0) {
            // Same off-equilibrium water as the live OpenMOPAC pins. Compare
            // the adapter to the independently invoked method at the QM geometry.
            cpu::Frame bare(3);
            bare.box = a.box;
            for (int i = 0; i < 3; ++i) {
                bare.span(kAtom_type)[i] = a.span(kAtom_type)[i];
                bare.span(kAtom_x)[i] = a.span(kAtom_x)[i];
                bare.span(kAtom_y)[i] = a.span(kAtom_y)[i];
                bare.span(kAtom_z)[i] = a.span(kAtom_z)[i];
            }
            Method isolated{0, 1, 400, 1e-10, 1e-12};
            const auto& reference = isolated.forces(bare);
            if (std::abs(probe_qm.last_qm_energy() - reference.total_energy_ev)
                > 1e-9)
                return fail("QM vs standalone energy", step,
                            probe_qm.last_qm_energy(),
                            reference.total_energy_ev, 1e-9);
            // Closed-form three-pair LJ oracle, Lorentz–Berthelot mixing.
            // All pairs lie inside 10 Å and need no periodic image shift.
            double classical_force[4][3] = {};
            double classical_energy = 0;
            for (int i = 0; i < 3; ++i) {
                const double displacement[] = {
                    a.span(kAtom_x)[3] - a.span(kAtom_x)[i],
                    a.span(kAtom_y)[3] - a.span(kAtom_y)[i],
                    a.span(kAtom_z)[3] - a.span(kAtom_z)[i]};
                const double distance =
                    std::sqrt(displacement[0] * displacement[0]
                              + displacement[1] * displacement[1]
                              + displacement[2] * displacement[2]);
                const double epsilon = std::sqrt((i == 0 ? 0.02 : 0.01) * 0.02);
                const double sigma = 0.5 * ((i == 0 ? 2.5 : 1.2) + 3.0);
                const double sixth = std::pow(sigma / distance, 6);
                const double cutoff_sixth = std::pow(sigma / 10, 6);
                classical_energy +=
                    4 * epsilon
                    * (sixth * sixth - sixth - cutoff_sixth * cutoff_sixth
                       + cutoff_sixth);
                const double radial_derivative =
                    24 * epsilon / distance * (sixth - 2 * sixth * sixth);
                for (int d = 0; d < 3; ++d) {
                    const double component =
                        radial_derivative * displacement[d] / distance;
                    classical_force[i][d] += component;
                    classical_force[3][d] -= component;
                }
            }
            if (std::abs(probe_qm.last_mm_energy() - classical_energy) > 1e-12)
                return fail("classical LJ energy", step,
                            probe_qm.last_mm_energy(), classical_energy, 1e-12);
            for (int i = 0; i < 4; ++i)
                for (int d = 0; d < 3; ++d) {
                    const double expected =
                        classical_force[i][d]
                        + (i < 3 ? reference.forces_ev_a[3 * i + d] : 0)
                        + (i == 3 && d == 0 ? 0.125 : 0);
                    if (std::abs(bf[d][i] - expected) > 1e-9) {
                        std::fprintf(stderr, "atom=%d axis=%d ", i, d);
                        return fail("QM/MM vs standalone force", step, bf[d][i],
                                    expected, 1e-9);
                    }
                }
        }
        // Even the explicit cold-start policy must not erase accepted history
        // during a trial or initial-force preparation.
        probe_qm.reset_history_each_launch() = true;
        for (mode = 1; mode <= 2; ++mode) {
            auto trial = b;
            trial.span(kAtom_y)[2] += 0.1 * mode;
            with_probes.run(trial);
        }
        probe_qm.reset_history_each_launch() = false;
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
        if (!rejected)
            return fail("failed SCF rejected", step, 0, 1, 0);
        if (probed.last_result().converged)
            return fail("failed SCF converged", step, 1, 0, 0);
        if (!probed.has_density())
            return fail("failed SCF preserved density", step, 0, 1, 0);
    }
    // Same borrowed method, a different parent whose lifetime ends first.
    auto replay = make_water_mm();
    {
        cpu::Evaluate another;
        another.on(evaluate::Phase::evaluate).add_ref(probe_qm);
        another.set_threads(1);
        another.build();
        another.run(replay);
    }
    if (!probed.last_result().converged)
        return fail("new parent retry converged", 4, 0, 1, 0);
    return true;
}

} // namespace

int run() {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_x)[2] = 5.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 18;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = (i < 2) ? 1.0 : 40.0;
        f.span(kAtom_charge)[i] = 0;
        f.span(kAtom_y)[i] = f.span(kAtom_z)[i] = 0;
    }
    f.box = cpu::Box::cubic(30);
    f.blocks.at("atoms").insert(
        "qm_z", cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                                kDynamicFieldId});
    f.span(cpu::kAtomQmZ)[0] = 1;
    f.span(cpu::kAtomQmZ)[1] = 1;
    f.span(cpu::kAtomQmZ)[2] = 0;

    const double eps = 0.01, sigma = 3.4, cut = 12.0;
    cpu::QmmmOniom oniom{{{1, 1, eps, sigma}, {18, 18, eps, sigma}}, cut};
    cpu::semiempirical::DFTB2 dftb{cpu::semiempirical::DftbSkSet::MioSketch};

    oniom.set_engine(std::move(dftb));

    // Keep a second instance for last_* after Evaluate takes ownership.
    cpu::QmmmOniom probe{{{1, 1, eps, sigma}, {18, 18, eps, sigma}}, cut};
    probe.set_engine(
        cpu::semiempirical::DFTB2{cpu::semiempirical::DftbSkSet::MioSketch});

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(probe);
    eval.set_threads(1);
    eval.build();
    eval.run(f);

    const double E = f.meta.get(kEnergy);
    const double eh = probe.last_high_energy();
    const double elr = probe.last_low_real_energy();
    const double elm = probe.last_low_model_energy();
    const double recon = eh + elr - elm;

    std::printf("qmmm_oniom_h2_ar E=%.10e high=%.10e low_real=%.10e "
                "low_model=%.10e recon=%.10e\n",
                E, eh, elr, elm, recon);

    if (!std::isfinite(E) || !std::isfinite(eh) || !std::isfinite(elr)
        || !std::isfinite(elm)) {
        std::printf("FAIL: non-finite\n");
        return 1;
    }
    if (std::fabs(E - recon) > 1e-9) {
        std::printf("FAIL: ONIOM identity |E-recon|=%.3e\n",
                    std::fabs(E - recon));
        return 1;
    }
    if (!(std::fabs(elm) > 1e-12)) {
        std::printf("FAIL: E_low(model) should be nonzero with H–H LJ\n");
        return 1;
    }
    if (!(std::fabs(elr - elm) > 1e-6)) {
        std::printf("FAIL: E_low(real) must differ from E_low(model) "
                    "(cross terms)\n");
        return 1;
    }
    // Rebuilding the same QM adapter must not leave DFTB2 attached to a dead
    // Runtime. A second caller borrows the engine and releases it on return.
    eval.run(f);
    {
        cpu::Evaluate another;
        another.on(evaluate::Phase::evaluate).add_ref(probe);
        another.set_threads(1);
        another.build();
        another.run(f);
    }
    if (!check_nddo_qmmm_history<cpu::semiempirical::AM1>()
        || !check_nddo_qmmm_history<cpu::semiempirical::RM1>()) {
        std::fprintf(stderr,
                     "FAIL: NDDO QM/MM history, retry or accumulation\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

} // namespace atv_reg_multiscale_qmmm_oniom_h2_ar

ATV_REGRESSION("multiscale/oniom", "qmmm_oniom_h2_ar",
               atv_reg_multiscale_qmmm_oniom_h2_ar::run);
