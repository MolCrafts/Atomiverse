#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "operators.h"
#include "pins.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#ifdef ATV_HAS_MD
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#endif
#ifdef ATV_HAS_MINIMIZER
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#endif

using namespace ATV;
using cpu::pw::PW;

namespace {
void controls(PW& method) {
    method.tol = 1e-9;
    method.max_iter = 200;
    method.mix_beta = .2;
    method.smearing = ATV::electronic::Smearing::fermi_dirac;
    method.degauss = .02; // Ry; match the independently executed QE input.
    method.extra_bands = 2;
}
void energy(const PW& method, const cpu::Frame& frame) {
    if (!method.last_result().converged)
        throw std::runtime_error("SCF did not converge");
    nc_example::near(method.last_result().energy_ha, nc_example::energy_ha,
                     nc_example::energy_tol_ha, "QE Mermin F [Ha]");
    nc_example::near(frame.meta.get(kEnergy),
                     method.last_result().energy
                         * cpu::pw::qe_units::rydberg_to_ev,
                     1e-11, "Frame energy publication [eV]");
}
void force(const PW& method, const cpu::Frame& frame) {
    energy(method, frame);
    const auto fx = frame.span(kAtom_fx), fy = frame.span(kAtom_fy),
               fz = frame.span(kAtom_fz);
    for (int i = 0; i < 2; ++i)
        for (int d = 0; d < 3; ++d) {
            const double component = d == 0 ? fx[i] : d == 1 ? fy[i] : fz[i];
            nc_example::near(component / units::kHartreePerBohrInEvPerAngstrom,
                             nc_example::forces_ha_bohr[i][d],
                             nc_example::force_tol_ha_bohr,
                             "QE Cartesian force [Ha/Bohr]");
        }
    std::printf("PASS E=%.12f Ha, %d iterations, all6 nonzero external force "
                "components\n",
                method.last_result().energy_ha,
                method.last_result().iterations);
}
// Internal consistency supplement at a moved geometry; absolute science
// qualification remains the independent QE pin at the reference geometry.
void same_as_fresh(const PW& method, const cpu::Frame& frame) {
    auto copy = frame;
    PW fresh(method.potentials, method.ecut_ry, method.k_mesh, method.xc);
    controls(fresh);
    fresh.forces(copy);
    if (!method.last_result().converged || !fresh.last_result().converged)
        throw std::runtime_error("workflow/fresh SCF convergence");
    nc_example::near(method.last_result().energy_ha,
                     fresh.last_result().energy_ha, nc_example::energy_tol_ha,
                     "workflow/fresh energy supplement [Ha]");
    const auto fx = frame.span(kAtom_fx), fy = frame.span(kAtom_fy),
               fz = frame.span(kAtom_fz);
    const auto qx = copy.span(kAtom_fx), qy = copy.span(kAtom_fy),
               qz = copy.span(kAtom_fz);
    for (int i = 0; i < 2; ++i)
        for (int d = 0; d < 3; ++d) {
            const double got = d == 0 ? fx[i] : d == 1 ? fy[i] : fz[i];
            const double want = d == 0 ? qx[i] : d == 1 ? qy[i] : qz[i];
            nc_example::near(got / units::kHartreePerBohrInEvPerAngstrom,
                             want / units::kHartreePerBohrInEvPerAngstrom,
                             nc_example::force_tol_ha_bohr,
                             "workflow/fresh force supplement [Ha/Bohr]");
        }
}
void mark_forces(cpu::Frame& frame) {
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_fx)[i] = 11 + i;
        frame.span(kAtom_fy)[i] = 21 + i;
        frame.span(kAtom_fz)[i] = 31 + i;
    }
}
void preserved_forces(const cpu::Frame& frame) {
    for (int i = 0; i < 2; ++i) {
        nc_example::near(frame.span(kAtom_fx)[i], 11 + i, 0, "unrequested Fx");
        nc_example::near(frame.span(kAtom_fy)[i], 21 + i, 0, "unrequested Fy");
        nc_example::near(frame.span(kAtom_fz)[i], 31 + i, 0, "unrequested Fz");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "si2";
        const auto start = std::chrono::steady_clock::now();
        auto provider = std::make_shared<const nc_example::PublishedSi>();
        nc_example::check_operators(*provider);
        if (mode == "operators")
            return 0;
        if (mode != "si2" && mode != "lifecycle" && mode != "md"
            && mode != "minimize")
            throw std::invalid_argument(
                "usage: nc-extension [operators|si2|lifecycle|md|minimize]");
        cpu::Frame reference(2);
        reference.box = cpu::Box::cubic(6);
        const double xyz[2][3] = {{2, 2, 2}, {3.8, 3.2, 3}};
        for (int i = 0; i < 2; ++i) {
            reference.span(kAtom_atomic_number)[i] = 14;
            reference.span(kAtom_mass)[i] = 28.0855;
            reference.span(kAtom_x)[i] = xyz[i][0];
            reference.span(kAtom_y)[i] = xyz[i][1];
            reference.span(kAtom_z)[i] = xyz[i][2];
        }
        PW method({provider}, 12, 1, cpu::xc::XCType::PBE);
        controls(method);
        if (method.potentials.front().get() != provider.get())
            throw std::runtime_error(
                "PW replaced the external physics instance");
        auto frame = reference;
        if (mode == "si2") {
            mark_forces(frame);
            method.run(frame);
            energy(method, frame);
            preserved_forces(frame);
            frame.meta.get(kEnergy) = 123;
            method.run(static_cast<const cpu::Frame&>(frame));
            nc_example::near(frame.meta.get(kEnergy), 123, 0,
                             "const-Frame energy preservation");
            preserved_forces(frame);
            nc_example::near(method.last_result().energy_ha,
                             nc_example::energy_ha, nc_example::energy_tol_ha,
                             "QE const-Frame energy [Ha]");
            method.forces(frame);
            force(method, frame);
            // The same method (and provider pointer), nested in the existing
            // one-pass workflow. Repeated evaluation must not double forces.
            {
                cpu::Evaluate evaluator;
                evaluator.on(evaluate::Phase::evaluate).add_ref(method);
                evaluator.build();
                evaluator.run(frame);
                force(method, frame);
                evaluator.run(frame);
                force(method, frame);
            }
        } else if (mode == "md") {
#ifdef ATV_HAS_MD
            method.forces(frame);
            force(method, frame);      // Absolute QE pin before any motion.
            constexpr double dt = .02; // fs; two real Verlet steps.
            for (int i = 0; i < 2; ++i) {
                frame.span(kAtom_vx)[i] = 0;
                frame.span(kAtom_vy)[i] = 0;
                frame.span(kAtom_vz)[i] = 0;
            }
            std::array<std::array<double, 3>, 2> initial_force{};
            for (int i = 0; i < 2; ++i)
                initial_force[i] = {frame.span(kAtom_fx)[i],
                                    frame.span(kAtom_fy)[i],
                                    frame.span(kAtom_fz)[i]};
            // Poison the existing force columns: the host must evaluate its
            // initial force, rather than drifting with an old caller result.
            mark_forces(frame);
            {
                cpu::md::MD dynamics;
                dynamics.set_integrator(cpu::md::VelocityVerlet{dt});
                dynamics.on(md::Phase::force).add_ref(method);
                dynamics.build();
                if (dynamics.run(frame, 1) != 1)
                    throw std::runtime_error("MD first-step budget");
                const auto x = frame.span(kAtom_x), y = frame.span(kAtom_y),
                           z = frame.span(kAtom_z);
                for (int i = 0; i < 2; ++i)
                    for (int d = 0; d < 3; ++d) {
                        const double got = d == 0 ? x[i] : d == 1 ? y[i] : z[i];
                        const double want =
                            xyz[i][d]
                            + .5 * dt * dt * units::UnitSystem::lj().ftm2v
                                  * initial_force[i][d]
                                  / frame.span(kAtom_mass)[i];
                        nc_example::near(got, want, 2e-12,
                                         "MD initial-force Verlet drift [Ang]");
                    }
                if (dynamics.run(frame, 1) != 1)
                    throw std::runtime_error("MD continuation budget");
                // Mutate the continuing host's physical inputs, then take a
                // real step. The existing runtime must refresh the method.
                frame.span(kAtom_y)[1] += .001;
                frame.box = cpu::Box::cubic(6.05);
                if (dynamics.run(frame, 1) != 1)
                    throw std::runtime_error("MD changed-input budget");
            }
            same_as_fresh(method, frame);
            std::printf("PASS MD initial-force, continuation and changed-input "
                        "force state\n");
#else
            throw std::invalid_argument("md mode requires the MD module");
#endif
        } else if (mode == "minimize") {
#ifdef ATV_HAS_MINIMIZER
            frame = reference;
            method.forces(frame);
            force(method, frame);
            const double before = frame.meta.get(kEnergy);
            {
                // Deliberately overlong first trial compresses the Si bond.
                // Armijo must backtrack; the accepted state is checked fresh.
                cpu::minimizer::LBFGS search{5, 1.0, 4};
                search.line_search(cpu::minimizer::LineSearchMode::Armijo, 1e-4,
                                   .9, .35);
                cpu::minimizer::Minimize optimizer;
                optimizer.set_optimizer_ref(search);
                optimizer.on(minimizer::Phase::force).add_ref(method);
                optimizer.build(); // Existing ForceConverged criterion.
                if (optimizer.run(frame, 1) != 1)
                    throw std::runtime_error("Minimize one-step budget");
                double displacement = 0;
                for (int i = 0; i < 2; ++i)
                    for (int d = 0; d < 3; ++d) {
                        const double x = d == 0   ? frame.span(kAtom_x)[i]
                                         : d == 1 ? frame.span(kAtom_y)[i]
                                                  : frame.span(kAtom_z)[i];
                        displacement =
                            std::max(displacement, std::abs(x - xyz[i][d]));
                    }
                if (!(displacement > 1e-8 && displacement < .35 * (1 - 1e-8))
                    || !(frame.meta.get(kEnergy) < before))
                    throw std::runtime_error("Minimize must move downhill "
                                             "after rejecting first trial");
            }
            same_as_fresh(method, frame);
            std::printf("PASS Minimize rejected-trial/accepted-state force "
                        "supplement\n");
#else
            throw std::invalid_argument("minimize mode requires MINIMIZER");
#endif
        } else {
            method.run(frame);
            energy(method, frame);
            const auto accepted = method.last_result();
            PW peer({provider}, 12, 1, cpu::xc::XCType::PBE);
            controls(peer);
            if (peer.potentials.front().get()
                != method.potentials.front().get())
                throw std::runtime_error(
                    "independent methods did not share provider");
            auto changed = reference;
            changed.span(kAtom_x)[1] += .01;
            changed.box = cpu::Box::cubic(6.2);
            peer.run(changed);
            if (!peer.last_result().converged
                || !std::isfinite(peer.last_result().energy_ha))
                throw std::runtime_error("changed geometry/cell SCF");
            nc_example::near(method.last_result().energy_ha, accepted.energy_ha,
                             0, "independent accepted electronic state");
            mark_forces(frame);
            frame.meta.get(kEnergy) = 123;
            frame.span(kAtom_atomic_number)[1] =
                8; // provider describes Si only.
            bool rejected = false;
            try {
                method.run(frame);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error(
                    "species mismatch was silently accepted");
            preserved_forces(frame);
            nc_example::near(frame.meta.get(kEnergy), 123, 0,
                             "species rejection output preservation");
            nc_example::near(method.last_result().energy_ha, accepted.energy_ha,
                             0,
                             "species rejection accepted-state preservation");
            frame.span(kAtom_atomic_number)[1] = 14;
            method.forces(frame);
            force(method, frame);
            // Refresh the same peer object back to the original cell/geometry.
            peer.run(reference);
            energy(peer, reference);
            method.forces(frame);
            force(method, frame);
        }
        if (method.potentials.front().get() != provider.get())
            throw std::runtime_error(
                "workflow replaced physical provider identity");
        const double wall = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - start)
                                .count();
        std::printf("PASS nc-extension %s whole=%.6f seconds (bound60)\n",
                    mode.c_str(), wall);
        if (wall > 60)
            throw std::runtime_error("whole acceptance section exceeds60s");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
